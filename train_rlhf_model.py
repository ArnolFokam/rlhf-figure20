from email import policy
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
    @jax.jit
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
    @jax.jit
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
        ppo_learning_rate=1e-4,
        ppo_gradient_accumulation_steps=100,
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

    # perform RLHF with PPO
    for episode in tqdm(range(1, args.num_episodes + 1), desc="Episode ", position=0, leave=True):
        
        # Indefinitely loop over the prompts dataset
        try:
            batch_prompts = next(train_prompts_dataloader_iter)
        except StopIteration:
            train_prompts_dataloader_iter = iter(train_prompts_dataloader)
            batch_prompts = next(train_prompts_dataloader_iter)

        # Get the prompts
        human_prompts = batch_prompts.pop("query")

        # generate responses
        responses = policy_generate(
            params=policy_state.params,
            x=batch_prompts,
        )
        responses_text = tokenizer.batch_decode(responses, skip_special_tokens=True)

        # construct query/response pairs for reward model
        query_responses_r =  [
            "\n\nHuman: " + q + "\n\nAssistant: " + r for q, r in zip(human_prompts, responses_text)
        ]
        tokenized_query_responses_r = tokenizer(
            query_responses_r,
            truncation=True,
            return_tensors="np",
            padding="max_length",
            max_length=args.max_query_length + args.max_response_length,
        )

        # computer reward from function
        rewards = reward_fn(x=tokenized_query_responses_r).squeeze()

        #### PPO UPDATE ####
        
        # get original query response pairs
        query_responses_ids = jnp.concatenate([batch_prompts["input_ids"], responses], axis=1)
        attention_mask = jnp.concatenate([batch_prompts["attention_mask"], jnp.ones_like(responses)], axis=1)
        
        # get logprobs of action over sequence of state (active policy)
        logprobs, values = policy_forward(
            params=policy_state.params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            },
        )

        # get logprobs of action over sequence of state (reference policy)
        ref_logprobs, _ = policy_forward(
            params=initial_policy_ref_params,
            x={
                "input_ids": query_responses_ids,
                "attention_mask": attention_mask,
            },
        )

        pass
        
       





 