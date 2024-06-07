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


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune an LLM with RLHF")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of training episodes")
    parser.add_argument("--prompts_dataset", type=str, default="sentiment", help="Prompt dataset for RLHF", choices=["sentiment"])
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2", help="Model name")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=100, help="Gradient accumulation steps")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--experiment_name", type=str, default="base_hh_rlfh_data", help="Experiment name")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--temperature", type=int, default=1.0, help="Temperature for lob prob on actions (tokens)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--ppo_num_epochs", type=int, default=5, help="Number of optimization epochs for ppo")
    parser.add_argument("--ppo_batch_size", type=int, default=5, help="Batch size for ppo")
    parser.add_argument("--ppo_cliprange_value", type=float, default=0.2, help="Clip range for value function")# TODO; fix this
    parser.add_argument("--ppo_cliprange", type=float, default=0.2, help="Clip range for policy gradient loss")# TODO: fix this
    parser.add_argument("--ppo_vf_coef", type=float, default=0.1, help="Value function coefficient for PPO loss")
    return parser.parse_args()


# NamedTuple for preference model parameters
class PolicyModelParams(NamedTuple):
    backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes

class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = jnp.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult

def train_step(
    policy_state,
    mb_advantages,
    mb_returns,
    mb_values,
    mb_query_responses,
    mb_logprobs,
    args,
):
    def loss(params):
        output, vpred_temp = policy_state.apply_fn(params, mb_query_responses)
        # vpred_temp: [local_micro_batch_size, query_length + response_length, 1]
        vpred = jnp.squeeze(vpred_temp[:, args.max_seq_len- 1 : -1, :], axis=-1)
        # vpred: [local_micro_batch_size, response_length]
        vpredclipped = jnp.clip(
            vpred,
            mb_values - args.ppo_cliprange_value,
            mb_values + args.ppo_cliprange_value,
        )
        vf_losses1 = jnp.square(vpred - mb_returns)
        vf_losses2 = jnp.square(vpredclipped - mb_returns)
        vf_loss = 0.5 * jnp.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).astype(jnp.float32).mean()

        logits = output.logits[:, args.max_seq_len - 1 : -1, :]
        logits /= args.temperature
        responses = mb_query_responses[:, args.max_seq_len :]
        new_logprobs = -optax.softmax_cross_entropy_with_integer_labels(logits, responses)

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

    grad_fn = jax.value_and_grad(loss, has_aux=True)
    (loss, current_rl_stats), grads = grad_fn(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)

    return policy_state

if __name__ == "__main__":
    args = parse_args()
    
    args = argparse.Namespace(
        model_name="distilbert/distilgpt2",
        logs_dir="logs",
        experiment_name="rlhf_sentiment",
        num_episodes=100,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        prompts_dataset="sentiment",
        ppo_num_epochs="5",
        ppo_batch_size=8,
        train_batch_size=8,
        temperature=1.0,
        max_seq_len=512,
        gamma=0.99,
        lamda=0.95,
        ppo_cliprange_value=0.2,
        ppo_cliprange=0.2,
        ppo_vf_coef=0.1,
        seed=0,
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
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    train_prompts_dataloader_iter = iter(train_prompts_dataloader)

    # Initialize policy state
    backbone = FlaxAutoModelForCausalLM.from_pretrained(args.model_name, dtype=jnp.float16)
    backbone.params = backbone.to_bf16(backbone.params)
    head = RegressionHead(head_input_size=backbone.config.hidden_size)

    def policy_forward(params: LMBackboneWithScalarHeadParams, x):
        input_ids = x
        attention_mask = input_ids != tokenizer.pad_token_id
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

        return output, values

    def policy_generate(
        params: LMBackboneWithScalarHeadParams,
        prompts,
    ):
        input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
        output = backbone.generate(
            params=params.backbone_params["params"],
            input_ids=input_ids,
            generation_config=GenerationConfig(
                min_new_tokens=256,
                max_new_tokens=256,
                do_sample=True,
                top_k=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            ),
            attention_mask=attention_mask,
            return_dict_in_generate=True,
        )
        context_length = input_ids.shape[1]
        return jnp.concatenate((input_ids, output.sequences[:, context_length:]), axis=1)

    optimizer = optax.MultiSteps(
        optax.adamw(
            learning_rate=args.learning_rate,
            b1=0.9,
            b2=0.98,
            eps=1e-8,
            weight_decay=0.01,
        ),
        every_k_schedule=args.gradient_accumulation_steps
    )
    # TODO: seperate gradient accummulation of ppo and policy reward
    policy_state = TrainState.create(
        apply_fn=policy_forward, 
        params=LMBackboneWithScalarHeadParams(
            backbone_params=flax.core.FrozenDict({"params": backbone.params}),
            head_params=flax.core.FrozenDict(head.init(rng, jnp.ones(backbone.config.hidden_size)[None, None, :],)),
        ), 
        tx=optimizer,
    )

    # Initialize reference policy
    ref_policy_params = copy.deepcopy(policy_state.params)

    # Initialize reward state
    r_backbone = FlaxAutoModelForCausalLM.from_pretrained(args.model_name, dtype=jnp.float16)
    r_backbone.params = backbone.to_bf16(r_backbone.params)
    r_head = RegressionHead(head_input_size=r_backbone.config.hidden_size)

    def reward_forward(params: LMBackboneWithScalarHeadParams, query_responses_ids):
        # mask out padding tokens
        attention_mask = query_responses_ids != tokenizer.pad_token_id
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

    reward_forward_p = functools.partial(
        reward_forward,
        params=LMBackboneWithScalarHeadParams(
            backbone_params=flax.core.FrozenDict({"params": r_backbone.params}),
            head_params=flax.core.FrozenDict(
                r_head.init(
                    rng,
                    jnp.ones(r_backbone.config.hidden_size)[None, None, :],
                )
            ),
        )
    )

    kl_ctl = AdaptiveKLController(
        init_kl_coef=0.15, 
        hparams=AdaptiveKLParams(target=6.0, horizon=10000)
    )

    # perform RLHF with PPO
    for episode in tqdm(range(1, args.num_episodes + 1), desc="Episode ", position=0, leave=True):
        
        # Indefinitely loop over the prompts dataset
        try:
            batch_prompts = next(train_prompts_dataloader_iter)
        except StopIteration:
            train_prompts_dataloader_iter = iter(train_prompts_dataloader)
            batch_prompts = next(train_prompts_dataloader_iter)

        # generate responses
        query_responses = policy_generate(
            params=policy_state.params,
            prompts=batch_prompts,
        )

        # query_responses: [local_batch_size, query_length + response_length]
        responses = query_responses[:, args.max_seq_len:]

        # values
        output, full_values = policy_forward(policy_state.params, query_responses)

        # values: [local_batch_size, response_length]
        values = full_values[:, args.max_seq_len - 1 : -1].squeeze(-1)

        # logits: [local_batch_size, response_length, vocab_size]
        logits = output.logits[:, args.max_seq_len - 1 : -1, :] / args.temperature

        # all_logprobs: [local_batch_size, response_length, vocab_size]
        all_logprobs = jax.nn.log_softmax(logits, axis=-1)

        # logprobs: [local_batch_size, response_length]
        logprobs = jnp.take_along_axis(all_logprobs, responses[..., None], -1).squeeze(-1)

        # reference values
        ref_output, ref_full_values = policy_forward(ref_policy_params, query_responses)

        # values: [local_batch_size, response_length]
        ref_values = ref_full_values[:, args.max_seq_len - 1 : -1].squeeze(-1)

        # logits: [local_batch_size, response_length, vocab_size]
        ref_logits = ref_output.logits[:, args.max_seq_len - 1 : -1, :] / args.temperature

        # all_logprobs: [local_batch_size, response_length, vocab_size]
        ref_all_logprobs = jax.nn.log_softmax(logits, axis=-1)

        # logprobs: [local_batch_size, response_length]
        ref_logprobs = jnp.take_along_axis(all_logprobs, responses[..., None], -1).squeeze(-1)

        # TODO: check response postprocessing

        # get the reward
        scores = reward_forward_p(query_responses_ids=query_responses).flatten()

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = -jnp.array([[kl_ctl.value]] * responses.shape[0]) * kl
        rewards = non_score_reward
        rewards = rewards.at[:, -1].add(scores)

        # 6. compute advantages and returns
        def compute_gae_once(carry, inp):
            advantages = carry
            nextdone, nextvalues, curvalues, reward = inp
            nextnonterminal = 1.0 - nextdone

            delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
            advantages = delta + args.gamma * args.lamda * nextnonterminal * advantages
            return advantages, advantages

        extended_values = jnp.concatenate((values, jnp.zeros((args.ppo_batch_size, 1))), axis=1)
        dones = jnp.zeros_like(rewards)
        dones = dones.at[:, -1].set(1.0)

        advantages = jnp.zeros((args.ppo_batch_size,))
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones.T, extended_values[:, 1:].T, extended_values[:, :-1].T, rewards.T),
            reverse=True,
        )

        advantages = advantages.T
        returns = advantages + values

        def ppo_single_microbatch(carry, inp):
            policy_state, = carry
            mb_advantages, mb_returns, mb_values, mb_query_responses, mb_logprobs = inp

            policy_state = train_step(
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
            key, subkey = jax.random.split(key, 2)
            perm = jax.random.permutation(key, args.ppo_batch_size)
            # advantages, returns, values, query_responses, logprobs = inp
            mbs_advantages = einops.rearrange(
                advantages[perm],
                "(c m) l -> c m l",
                c=args.gradient_accumulation_steps,
            )
            mbs_returns = einops.rearrange(
                returns[perm],
                "(c m) l -> c m l",
                c=args.gradient_accumulation_steps,
            )
            mbs_values = einops.rearrange(values[perm], "(c m) l -> c m l", c=args.gradient_accumulation_steps)
            mbs_query_responses = einops.rearrange(
                query_responses[perm],
                "(c m) l -> c m l",
                c=args.gradient_accumulation_steps,
            )
            mbs_logprobs = einops.rearrange(
                logprobs[perm],
                "(c m) l -> c m l",
                c=args.gradient_accumulation_steps,
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

        key = jax.random.PRNGKey(args.seed)
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        (policy_state, _), _ = jax.lax.scan(
            f=ppo_single_epoch,
            init=(policy_state, key),
            xs=None,
            length=args.ppo_num_epochs,
        )