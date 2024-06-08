import functools
import logging
import argparse
import os
import copy
from tqdm import tqdm
from typing import NamedTuple
from dataclasses import dataclass

import jax
import flax
import optax
import orbax
import einops
import orbax.checkpoint
import jax.numpy as jnp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

from model import LMBackboneWithScalarHeadParams, RegressionHead
from data import JaxDataloader, prompts_datasets as datasets


# Function to create training state
def create_policy_train_state(args, rng):
    backbone = FlaxAutoModelForCausalLM.from_pretrained(args.model_name, dtype=jnp.float16)
    backbone.params = backbone.to_bf16(backbone.params)
    head = RegressionHead(head_input_size=backbone.config.hidden_size)

    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.min_response_length,
        min_new_tokens=args.max_response_length,
        temperature=args.temperature,
        top_k=args.generation_topk,
        top_p=args.generation_topp,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
    )

    # Function that generates the responses
    def policy_generate(params: LMBackboneWithScalarHeadParams, x):
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
    def policy_forward(params: LMBackboneWithScalarHeadParams, x):
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
        every_k_schedule=args.ppo_gradient_accumulation_steps
    )

    # Initialize policy state
    state = TrainState.create(
        apply_fn=policy_forward,
        params=LMBackboneWithScalarHeadParams(
            backbone_params=flax.core.FrozenDict({"params": backbone.params}),
            head_params=flax.core.FrozenDict(
                head.init(rng, jnp.ones(backbone.config.hidden_size)[None, None, :])
            ),
        ),
        tx=optimizer,
    )

    return policy_forward, policy_generate, state

# Function to get the reward model
def get_reward_fn(args, rng):
    backbone = FlaxAutoModelForCausalLM.from_pretrained(args.model_name, dtype=jnp.float16)
    backbone.params = backbone.to_bf16(backbone.params)
    head = RegressionHead(head_input_size=backbone.config.hidden_size)

    # Function that computes the reward
    def get_reward(params: LMBackboneWithScalarHeadParams, x):
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
    state_params = orbax_checkpointer.restore(args.saved_pm_path)["reward_model"]["params"]
    params = LMBackboneWithScalarHeadParams(
        backbone_params=flax.core.FrozenDict({"params": state_params["backbone_params"]["params"]}),
        head_params=flax.core.FrozenDict({"params": state_params["head_params"]["params"]}),
    )
    logging.info(f"Loaded pretrained reward model from {args.saved_pm_path}")

    return functools.partial(get_reward, params)
    
# Function to computer advantages and returns
def compute_advantages_and_returns(rewards, values, dones, gamma, lamda):

    def gae_at_t(carry, inp):
        gae, next_value = carry
        done, reward, value = inp

        delta = reward + gamma * next_value * (1 - done) - value 
        gae = delta + gamma * lamda * (1 - done) * gae

        return (gae, value), gae

    _, advantages = jax.lax.scan(
        gae_at_t,
        (jnp.zeros((values.shape[0],)), jnp.zeros((values.shape[0],))),
        xs=(dones.T, rewards.T, values.T),
        reverse=True
    )

    # retunrs (advantages, returns)
    return advantages.T, advantages.T + values

    

if __name__ == "__main__":
    # args = parse_args()
    
    args = argparse.Namespace(
        logs_dir="logs",
        model_name="distilbert/distilgpt2",
        experiment_name="rlhf_sentiment",
        prompts_dataset="sentiment",
        prompts_batch_size=8,
        max_query_length=768,
        min_response_length=128,
        max_response_length=128,
        generation_topk=0.0,
        generation_topp=1.0,
        num_episodes=100,
        temperature=0.7,
        saved_pm_path="logs/pm_sentiment_sentiment_1024_0/model",
        seed=0,

        # ppo training params
        ppo_gamma=0.99,
        ppo_lamda=0.95,
        ppo_vf_coef=0.5,
        ppo_cliprange=0.2,
        ppo_kl_coeff=0.001,
        ppo_learning_rate=1e-4,
        ppo_num_epochs_per_batch=4,
        ppo_gradient_accumulation_steps=2,
    )

    logging.info(f"Performing RLHF with seed {args.seed}")

    # Ensure reproducibility
    rng = jax.random.PRNGKey(args.seed)

    # Initialize logging
    logs_dir = f"{args.logs_dir}/{args.experiment_name}"
    summary_writer = SummaryWriter(logs_dir)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token

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

    # Initialize policy model with reference policy
    rng, policy_state_rng = jax.random.split(rng, 2)
    policy_forward, policy_generate, policy_state = create_policy_train_state(args, policy_state_rng)
    initial_policy_ref_params = copy.deepcopy(policy_state.params)

    # Initialize reward model
    reward_fn = get_reward_fn(args, rng)

    @jax.jit
    def rlhf_update(policy_state):
        # generate responses
        responses = policy_generate(
            params=policy_state.params,
            x=batch_prompts,
        )

        # get original query response pairs
        query_responses_ids = jnp.concatenate([batch_prompts["input_ids"], responses], axis=1)
        attention_mask = jnp.concatenate([batch_prompts["attention_mask"], jnp.ones_like(responses)], axis=1)
        context_length = batch_prompts["input_ids"].shape[1]

        # computer reward from function
        scores = reward_fn(x={
            "input_ids": query_responses_ids,
            "attention_mask": attention_mask,
        }).squeeze()

        #### PPO UPDATE ####

        # get logprobs of action over responses (active policy)
        logits, values = policy_forward(
            params=policy_state.params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            }, # but perform pass over query/response
        )
        all_logprobs = jax.nn.log_softmax(logits[:, context_length:] / args.temperature, axis=-1)
        logprobs = jnp.take_along_axis(all_logprobs, responses[..., None], axis=-1).squeeze()
        
        # get logprobs of action over responses (reference policy)
        ref_logits, _ = policy_forward(
            params=initial_policy_ref_params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            }, # but perform pass over query/response
        )        
        ref_all_logprobs = jax.nn.log_softmax(ref_logits[:, context_length:] / args.temperature, axis=-1)
        ref_logprobs = jnp.take_along_axis(ref_all_logprobs, responses[..., None], axis=-1).squeeze()
        
        # Compute KL-Div penalty between active and reference policy
        rewards = - args.ppo_kl_coeff * (logprobs - ref_logprobs)

        # Compute advantage and returns
        rewards = rewards.at[:, -1].add(scores) # add score of query/response to last step
        values = values[:, context_length:].squeeze()  # extract values from response     
        
        dones = jnp.zeros_like(rewards)
        dones = dones.at[:, -1].set(1.0) # set last step as done

        advantages, returns = compute_advantages_and_returns(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=args.ppo_gamma,
            lamda=args.ppo_lamda,
        )

        def ppo_step(
            policy_state,
            mb_advantages,
            mb_returns,
            mb_values,
            mb_query_responses,
            mb_logprobs,
            args,
        ):
            def loss(params):
                logits, vpred_temp = policy_state.apply_fn(
                    params, 
                    {
                        "input_ids": mb_query_responses,
                        "attention_mask": (mb_query_responses != tokenizer.pad_token_id).astype(jnp.int32),
                    }
                )
                # vpred_temp: [local_micro_batch_size, query_length + response_length, 1]
                vpred = jnp.squeeze(vpred_temp[:, args.max_query_length- 1 : -1, :], axis=-1)
                # vpred: [local_micro_batch_size, response_length]
                vpredclipped = jnp.clip(
                    vpred,
                    mb_values - args.ppo_cliprange,
                    mb_values + args.ppo_cliprange,
                )
                vf_losses1 = jnp.square(vpred - mb_returns)
                vf_losses2 = jnp.square(vpredclipped - mb_returns)
                vf_loss = 0.5 * jnp.maximum(vf_losses1, vf_losses2).mean()
                vf_clipfrac = (vf_losses2 > vf_losses1).astype(jnp.float32).mean()

                logits = logits[:, args.max_query_length - 1 : -1, :]
                logits /= args.temperature
                responses = mb_query_responses[:, args.max_query_length :]
                new_logprobs = optax.softmax_cross_entropy_with_integer_labels(logits, responses)

                logprobs_diff = new_logprobs - mb_logprobs
                ratio = jnp.exp(logprobs_diff)
                pg_losses1 = -mb_advantages * ratio
                pg_losses2 = -mb_advantages * jnp.clip(ratio, 1.0 - args.ppo_cliprange, 1.0 + args.ppo_cliprange)
                pg_loss = jnp.maximum(pg_losses1, pg_losses2).mean()
                pg_clipfrac = (pg_losses2 > pg_losses1).astype(jnp.float32).mean()

                pd = jax.nn.softmax(logits, axis=-1)
                entropy = jax.nn.logsumexp(logits, axis=-1) - jnp.sum(pd * logits, axis=-1)

                approxkl = 0.5 * ((logprobs_diff) ** 2).mean()
                loss = pg_loss + args.ppo_vf_coef * vf_loss

                current_rl_stats = dict(
                    approxkl=approxkl,
                    entropy=entropy.mean(),
                    ratio=ratio.mean(),
                    pg_loss=pg_loss,
                    pg_clipfrac=pg_clipfrac,
                    vf_loss1=vf_losses1.mean(),
                    vf_loss=vf_loss,
                    vf_clipfrac=vf_clipfrac,
                    loss=loss,
                )
                return loss, current_rl_stats

            (_, current_rl_stats), grads = jax.value_and_grad(loss, has_aux=True)(policy_state.params)
            policy_state = policy_state.apply_gradients(grads=grads)

            return policy_state

        def ppo_single_microbatch(carry, inp):
            policy_state, = carry
            mb_advantages, mb_returns, mb_values, mb_query_responses, mb_logprobs = inp

            policy_state = ppo_step(
                policy_state=policy_state,
                mb_advantages=mb_advantages,
                mb_returns=mb_returns,
                mb_values=mb_values,
                mb_query_responses=mb_query_responses,
                mb_logprobs=mb_logprobs,
                args=args,
            )
            return (policy_state,), None

        def ppo_single_epoch(carry, inp):
            policy_state, key = carry
            key, _ = jax.random.split(key, 2)
            perm = jax.random.permutation(key, args.prompts_batch_size)
            
            mbs_advantages = einops.rearrange(
                advantages[perm],
                "(c m) l -> c m l",
                c=args.ppo_gradient_accumulation_steps,
            )
            mbs_returns = einops.rearrange(
                returns[perm],
                "(c m) l -> c m l",
                c=args.ppo_gradient_accumulation_steps,
            )
            mbs_values = einops.rearrange(values[perm], "(c m) l -> c m l", c=args.ppo_gradient_accumulation_steps)
            mbs_query_responses = einops.rearrange(
                query_responses_ids[perm],
                "(c m) l -> c m l",
                c=args.ppo_gradient_accumulation_steps,
            )
            mbs_logprobs = einops.rearrange(
                logprobs[perm],
                "(c m) l -> c m l",
                c=args.ppo_gradient_accumulation_steps,
            )
            (policy_state, ), _ = jax.lax.scan(
                f=ppo_single_microbatch,
                init=(policy_state,),
                xs=(
                    mbs_advantages,
                    mbs_returns,
                    mbs_values,
                    mbs_query_responses,
                    mbs_logprobs,
                ),
            )
            return (policy_state, key), None

        # multiple training epochs
        (policy_state, _), _ = jax.lax.scan(
            f=ppo_single_epoch,
            init=(policy_state, rng),
            xs=None,
            length=args.ppo_num_epochs_per_batch,
        )

        return policy_state

    # perform RLHF with PPO
    for episode in tqdm(range(1, args.num_episodes + 1), desc="Episode ", position=0, leave=True):
        
        # Indefinitely loop over the prompts dataset
        try:
            batch_prompts = next(train_prompts_dataloader_iter)
        except StopIteration:
            train_prompts_dataloader_iter = iter(train_prompts_dataloader)
            batch_prompts = next(train_prompts_dataloader_iter)

        # Update RLHF policy
        policy_state = rlhf_update(policy_state)

    # Save the model
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {"rlhf_model": policy_state, "args": vars(args)}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(os.path.abspath(f"{logs_dir}/model/"), ckpt, save_args=save_args, force=True)

       





 