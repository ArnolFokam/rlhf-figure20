import functools
import logging
import argparse
import os
import copy
from random import shuffle
from tqdm import tqdm

import jax
import flax
import optax
import orbax
import einops
import orbax.checkpoint
import jax.numpy as jnp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax.metrics.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

from data import JaxDataloader, prompts_datasets as datasets
from model import LMBackboneWithScalarHeadParams, RegressionHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="HH & PM vs Specialized Skills Experiments (RLHF)"
    )
    parser.add_argument(
        "--saved_pm_path",
        type=str,
        default="logs/pm_sentiment_sentiment_1024_0/model",
        help="Directory of the saved reward model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    return parser.parse_args()


# Function to create training state
def create_policy_train_state(args, rng):
    backbone = FlaxAutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=jnp.bfloat16,
    )
    backbone.params = backbone.to_bf16(backbone.params)
    head = RegressionHead(
        head_input_size=backbone.config.hidden_size,
        param_dtype=jnp.bfloat16,
    )

    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.min_response_length,
        min_new_tokens=args.max_response_length,
        temperature=args.temperature,
        top_k=args.generation_topk,
        top_p=args.generation_topp,
        pad_token_id=args.pad_token_id,
        do_sample=True,
    )

    # Function that generates the responses
    def _policy_generate(params: LMBackboneWithScalarHeadParams, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        prompt_length = input_ids.shape[1]
        return backbone.generate(
            params=params.backbone_params["params"],
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
        ).sequences[:, prompt_length:]

    # Function that computes the values of the responses
    def _policy_forward(params: LMBackboneWithScalarHeadParams, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        position_ids = attention_mask.cumsum(1) - attention_mask
        output = backbone.module.apply(
            variables=params.backbone_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # shape: [batch_size, length, hidden_size]
        value_latents = output.hidden_states[-1]

        # shape: [batch_size, length, 1]
        values = head.apply(variables=params.head_params, x=value_latents)

        return output.logits, values

    # Initialize policy optimizer
    optimizer = optax.MultiSteps(
        optax.adamw(
            learning_rate=args.ppo_learning_rate,
            b1=0.9,
            b2=0.98,
            eps=1e-8,
            weight_decay=0.01,
        ),
        every_k_schedule=args.ppo_gradient_accumulation_steps,
    )

    # Initialize policy state
    head_params = head.init(rng, jnp.ones(backbone.config.hidden_size)[None, None, :])
    head_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), head_params)
    state = TrainState.create(
        apply_fn=_policy_forward,
        params=LMBackboneWithScalarHeadParams(
            backbone_params=flax.core.FrozenDict({"params": backbone.params}),
            head_params=flax.core.FrozenDict(head_params),
        ),
        tx=optimizer,
    )

    return _policy_forward, _policy_generate, state


# Function to get the reward model
def get_reward_fn(args, _):
    backbone = FlaxAutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=jnp.bfloat16,
    )
    backbone.params = backbone.to_bf16(backbone.params)
    head = RegressionHead(
        head_input_size=backbone.config.hidden_size,
        param_dtype=jnp.bfloat16,
    )

    # Function that computes the reward
    def _get_reward(params: LMBackboneWithScalarHeadParams, x):
        query_responses_ids, attention_mask = x["input_ids"], x["attention_mask"]
        position_ids = attention_mask.cumsum(1) - attention_mask
        reward_latents = backbone.module.apply(
            variables=params.backbone_params,
            input_ids=query_responses_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
        last_reward_latents = reward_latents[:, -1, :]
        reward = head.apply(variables=params.head_params, x=last_reward_latents)
        return reward

    # load the saved model weights
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state_params = orbax_checkpointer.restore(args.saved_pm_path)["reward_model"][
        "params"
    ]
    params = LMBackboneWithScalarHeadParams(
        backbone_params=flax.core.FrozenDict(
            {"params": state_params["backbone_params"]["params"]}
        ),
        head_params=flax.core.FrozenDict(
            {"params": state_params["head_params"]["params"]}
        ),
    )
    logging.info(f"Loaded pretrained reward model from {args.saved_pm_path}")

    return functools.partial(_get_reward, params)


# Function to update the policy
def get_update_fn(args, rng):

    # Initialize policy model with reference policy
    rng, policy_init_rng = jax.random.split(rng, 2)
    policy_forward, policy_generate, policy_state = create_policy_train_state(
        args, policy_init_rng
    )
    initial_policy_ref_params = copy.deepcopy(policy_state.params)

    # Initialize reward model
    rng, reward_init_rng = jax.random.split(rng, 2)
    reward_fn = get_reward_fn(args, reward_init_rng)

    # Compute number of mini-batches for
    # PPO update and metrics normalization
    args.num_ppo_mini_batches = (
        args.prompts_batch_size // args.ppo_gradient_accumulation_steps
    )

    # Function to computer advantages and returns
    def _compute_advantages_and_returns(rewards, values, dones, gamma, lamda):

        def _gae_at_t(carry, inp):
            gae, next_value = carry
            done, reward, value = inp

            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * lamda * (1 - done) * gae

            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _gae_at_t,
            (
                jnp.zeros((values.shape[0],), dtype=jnp.bfloat16),
                jnp.zeros((values.shape[0],), dtype=jnp.bfloat16),
            ),
            xs=(
                dones.T,
                rewards.T,
                values.T,
            ),  # transpose [batch, time] to [time, batch]
            reverse=True,
        )

        # retunrs (advantages, returns)
        return advantages.T, advantages.T + values

    @jax.jit
    def update_fn(policy_state, x, update_rng):

        # generate responses
        responses = policy_generate(
            params=policy_state.params,
            x=x,
        )

        # get original query response pairs
        query_responses_ids = jnp.concatenate([x["input_ids"], responses], axis=1)
        attention_mask = jnp.concatenate(
            [x["attention_mask"], jnp.ones_like(responses, dtype=jnp.bfloat16)], axis=1
        )
        context_length = x["input_ids"].shape[1]

        # computer reward from function
        scores = reward_fn(
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            }
        ).squeeze()

        #### PPO UPDATE ####

        # get logprobs of action over responses (reference policy)
        ref_logits, _ = policy_forward(
            params=initial_policy_ref_params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            },  # but perform pass over query/response
        )
        ref_all_logprobs = jax.nn.log_softmax(
            ref_logits[:, context_length:] / args.temperature, axis=-1
        )
        ref_logprobs = jnp.take_along_axis(
            ref_all_logprobs, responses[..., None], axis=-1
        ).squeeze()

        # get logprobs of action over responses (active policy)
        logits, values = policy_forward(
            params=policy_state.params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            },  # but perform pass over query/response
        )
        all_logprobs = jax.nn.log_softmax(
            logits[:, context_length:] / args.temperature, axis=-1
        )
        logprobs = jnp.take_along_axis(
            all_logprobs, responses[..., None], axis=-1
        ).squeeze()

        # Compute KL-Div penalty between active and reference policy
        kl_div = logprobs - ref_logprobs
        rewards = -args.ppo_kl_coeff * kl_div

        # Compute advantage and returns
        rewards = rewards.at[:, -1].add(
            scores
        )  # add score of query/response to last step
        values = values[:, context_length:].squeeze()  # extract values from response

        dones = jnp.zeros_like(rewards, dtype=jnp.bfloat16)
        dones = dones.at[:, -1].set(1.0)  # set last step as done

        advantages, returns = _compute_advantages_and_returns(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=args.ppo_gamma,
            lamda=args.ppo_lamda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create transition batch for PPO update
        transition_batch = (
            query_responses_ids,
            attention_mask,
            advantages,
            logprobs,
            returns,
            values,
        )

        # Function to update the policy parameters
        def _update_epoch(carry, _):

            def _update_minibatch(carry, inp):
                policy_state, metrics = carry
                (
                    query_responses_ids,
                    attention_mask,
                    advantages,
                    logprobs,
                    returns,
                    values,
                ) = inp

                def _loss_fn(params):
                    logits_new, values_new = policy_state.apply_fn(
                        params,
                        x={
                            "input_ids": query_responses_ids,
                            "attention_mask": attention_mask,
                        },
                    )

                    # Calculate critic loss
                    values_new = values_new[:, context_length:].squeeze()
                    values_new_clipped = values + jnp.clip(
                        values_new - values,
                        -args.ppo_epsilon_clip,
                        args.ppo_epsilon_clip,
                    )
                    value_losses = jnp.square(values_new - returns)
                    value_losses_clip = jnp.square(values_new_clipped - returns)
                    critic_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clip).mean()
                    )

                    # Calculate actor loss
                    responses = query_responses_ids[:, context_length:]
                    logprobs_new = jax.nn.log_softmax(
                        logits_new[:, context_length:] / args.temperature, axis=-1
                    )
                    logprobs_new = jnp.take_along_axis(
                        logprobs_new, responses[..., None], axis=-1
                    ).squeeze()
                    logprobs_diff = logprobs_new - logprobs
                    ratio = jnp.exp(logprobs_diff)
                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1 - args.ppo_epsilon_clip,
                            1 + args.ppo_epsilon_clip,
                        )
                        * advantages
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

                    # Get total loss
                    loss = actor_loss + args.ppo_vf_coef * critic_loss

                    # Get metrics
                    current_metrics = dict(
                        actor_loss=actor_loss,
                        critic_loss=critic_loss,
                        total_loss=loss,
                        ratio=ratio.mean(),
                        approxkl=0.5 * (logprobs_diff**2).mean(),
                        value_losses=value_losses.mean(),
                        actor_loss1=actor_loss1.mean(),
                    )

                    return loss, current_metrics

                # Update policy parameters with minibatch
                loss_grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (_, current_metrics), grads = loss_grad_fn(policy_state.params)
                policy_state = policy_state.apply_gradients(grads=grads)

                # merge metrics from current minibatch
                metrics = jax.tree_util.tree_map(
                    lambda x, y: x + y, metrics, current_metrics
                )

                return (policy_state, metrics), None

            policy_state, metrics, rng = carry
            rng, permutation_rng = jax.random.split(rng, 2)
            permutation = jax.random.permutation(
                permutation_rng, args.prompts_batch_size
            )

            # Shuffle batch per epoch update
            shuffled_transition_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), transition_batch
            )

            # Split the batch into multiple mini-batches
            mini_batches = jax.tree_util.tree_map(
                lambda x: einops.rearrange(
                    x, "(m b) l -> m b l", m=args.num_ppo_mini_batches
                ),
                shuffled_transition_batch,
            )

            # Update the policy with each mini-batch
            (policy_state, metrics), _ = jax.lax.scan(
                f=_update_minibatch,
                init=(policy_state, metrics),
                xs=mini_batches,
            )

            return (policy_state, metrics, rng), None

        # Multiple optimization epochs
        metrics = dict(
            actor_loss=0.0,
            critic_loss=0.0,
            total_loss=0.0,
            ratio=0.0,
            approxkl=0.0,
            value_losses=0.0,
            actor_loss1=0.0,
        )
        (policy_state, metrics, _), _ = jax.lax.scan(
            f=_update_epoch,
            init=(policy_state, metrics, update_rng),
            xs=None,
            length=args.ppo_num_epochs_per_batch,
        )

        # Normalize metrics w.r.t number of epochs and mini-batches
        metrics = jax.tree_util.tree_map(
            lambda x: x / (args.ppo_num_epochs_per_batch * args.num_ppo_mini_batches),
            metrics,
        )

        # Add metrics concerning the rewards
        metrics.update(
            dict(
                mean_score=scores.mean(),
                kl_div_initial=kl_div.mean(),
            )
        )

        # Get samples and their scores
        samples = dict(
            prompts=x["input_ids"],
            responses=responses,
            scores=scores,
        )

        return policy_state, metrics, samples

    return policy_state, update_fn


if __name__ == "__main__":

    passed_args = parse_args()

    args = argparse.Namespace(
        logs_dir="logs",
        model_name="distilbert/distilgpt2",
        experiment_name="rlhf_sentiment",
        prompts_dataset="sentiment",
        prompts_batch_size=64,
        max_query_length=128,
        min_response_length=128,
        max_response_length=128,
        generation_topk=0.0,
        generation_topp=1.0,
        num_episodes=5000,
        temperature=0.7,
        saved_pm_path="logs/pm_sentiment_sentiment_1024_0/model",
        log_every_n_episodes=10,
        seed=0,
        # ppo training params
        ppo_gamma=1.00,
        ppo_lamda=0.95,
        ppo_vf_coef=0.10,
        ppo_kl_coeff=0.15,
        ppo_epsilon_clip=0.2,
        ppo_learning_rate=1e-5,
        ppo_num_epochs_per_batch=4,
        ppo_gradient_accumulation_steps=1,
    )

    args.experiment_name = f"{args.experiment_name}_{args.saved_pm_path.split('/')[-2]}_{args.seed}"
    args.saved_pm_path = passed_args.saved_pm_path
    args.seed = passed_args.seed

    logging.info(f"Performing RLHF with seed {args.seed}")

    # Ensure reproducibility
    rng = jax.random.PRNGKey(args.seed)

    # Initialize logging
    logs_dir = f"{args.logs_dir}/{args.experiment_name}"
    summary_writer = SummaryWriter(logs_dir)
    summary_writer.hparams(vars(args))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token
    args.pad_token_id = tokenizer.pad_token_id

    # Initialize training DataLoader
    rng, train_iter_rng = jax.random.split(rng, 2)
    train_prompts_dataloader = JaxDataloader(
        rng=train_iter_rng,
        dataset=datasets[args.prompts_dataset](
            args=args,
            tokenizer=tokenizer,
        ),
        batch_size=args.prompts_batch_size,
        shuffle=True,
    )
    train_prompts_dataloader_iter = iter(train_prompts_dataloader)

    # Initialize update function
    policy_state, rlhf_update = get_update_fn(args, rng)

    # RLHF training loop
    for episode in tqdm(
        range(1, args.num_episodes + 1), desc="Episode ", position=0, leave=True
    ):

        # Indefinitely loop over the prompts dataset
        try:
            batch_prompts = next(train_prompts_dataloader_iter)
        except StopIteration:
            train_prompts_dataloader_iter = iter(train_prompts_dataloader)
            batch_prompts = next(train_prompts_dataloader_iter)

        # Update the RLHF policy
        rng, rlhf_update_rng = jax.random.split(rng, 2)
        policy_state, metrics, samples = rlhf_update(
            x=batch_prompts,
            policy_state=policy_state,
            update_rng=rlhf_update_rng,
        )

        if episode % args.log_every_n_episodes == 0:

            # Log metrics
            for k, v in metrics.items():
                summary_writer.scalar(f"{k}", v, step=episode)

            # Log samples
            for i in range(min(5, len(samples["scores"]))):
                prompt = tokenizer.decode(
                    samples["prompts"][i], skip_special_tokens=True
                )
                response = tokenizer.decode(
                    samples["responses"][i], skip_special_tokens=True
                )
                summary_writer.text(
                    f'Episode {episode} - Sample {i} - Score: {float(samples["scores"][i]):.2f}',
                    f"Prompt: \n{prompt}\n\nResponse: \n{response}",
                    step=episode,
                )

    # Save the model
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {"rlhf_model": policy_state, "args": vars(args)}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        os.path.abspath(f"{logs_dir}/model/"), ckpt, save_args=save_args, force=True
    )
