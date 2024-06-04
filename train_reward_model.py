import logging
import argparse
from tqdm import tqdm
from typing import NamedTuple

import jax
import flax
import optax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from model import RegressionHead
from data import JaxDataloader, datasets


def parse_args():
    parser = argparse.ArgumentParser(description="HH & PM vs Specialized Skills Experiments")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="distilbert/distilgpt2", help="Model name or path")
    parser.add_argument("--experiment_name", type=str, default="base_hh_rlfh_data", help="Experiment name")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Log every n steps")
    parser.add_argument("--train_dataset", type=str, default="mix", help="Training dataset", choices=["hh-rlhf", "sentiment", "mix"])
    parser.add_argument("--eval_dataset", type=str, default="sentiment", help="Training dataset", choices=["hh-rlhf", "sentiment", "mix"])
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


# Function to create training state
def create_train_state(args, rng):
    backbone = FlaxAutoModelForCausalLM.from_pretrained(args.model_name)
    head = RegressionHead(head_input_size=backbone.config.hidden_size)

    def get_reward(params: PreferenceModelParams, x):
        input_ids, attention_mask = x
        position_ids = attention_mask.cumsum(1) - attention_mask
        reward_latents = backbone.module.apply(
            variables=params.backbon_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
        last_reward_latents = reward_latents[:, -1, :]
        reward = head.apply(variables=params.head_params, x=last_reward_latents)
        return reward

    optimizer = optax.adamw(
        learning_rate=args.learning_rate,
        b1=0.9,
        b2=0.98,
        eps=1e-8,
        weight_decay=0.01,
    )

    state = TrainState.create(
        apply_fn=get_reward,
        params=PreferenceModelParams(
            backbon_params=flax.core.FrozenDict({"params": backbone.params}),
            head_params=flax.core.FrozenDict(head.init(rng, jnp.ones(backbone.config.hidden_size)[None, None, :],)),
        ),
        tx=optimizer,
    )

    return state

# NamedTuple for preference model parameters
class PreferenceModelParams(NamedTuple):
    backbon_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict

# Training step
@jax.jit
def train_step(state, batch):
    chosen_input_ids = batch["chosen_input_ids"]
    rejected_input_ids = batch["rejected_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    def loss_fn(params):
        chosen_reward = state.apply_fn(params, (chosen_input_ids, chosen_attention_mask))
        rejected_reward = state.apply_fn(params, (rejected_input_ids, rejected_attention_mask))
        loss = -jnp.mean(jnp.log(jax.nn.sigmoid(chosen_reward - rejected_reward)))
        accuracy = (chosen_reward > rejected_reward).astype('float32').mean()
        return loss, accuracy

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "accuracy": accuracy}

# Training function for a single epoch
def train_single_epoch(epoch, dataloader, state, args, writer):
    with tqdm(total=len(dataloader), desc="Training ", leave=False) as progress_bar_train:
        for step, batch in enumerate(dataloader):
            state, train_metrics = train_step(state, batch)
            progress_bar_train.update(1)
            progress_bar_train.set_description(
                f"Training ({epoch}/{args.num_epochs}) | Loss: {round(train_metrics['loss'].mean(), 3)} | Accuracy: {train_metrics['accuracy']}"
            )

            if (epoch * len(dataloader) + step) % args.log_every_n_steps == 0:
                for key, value in train_metrics.items():
                    writer.add_scalar(f"train/{key}", round(float(value), 3), epoch * len(dataloader) + step)

            if step == 10:
                break

    return state

# Evaluation step
@jax.jit
def eval_step(state, batch):
    chosen_input_ids = batch["chosen_input_ids"]
    rejected_input_ids = batch["rejected_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    def loss_fn(params):
        chosen_reward = state.apply_fn(params, (chosen_input_ids, chosen_attention_mask))
        rejected_reward = state.apply_fn(params, (rejected_input_ids, rejected_attention_mask))
        loss = -jnp.mean(jnp.log(jax.nn.sigmoid(chosen_reward - rejected_reward)))
        accuracy = (chosen_reward > rejected_reward).astype('float32').mean()
        return loss, accuracy

    loss, accuracy = loss_fn(state.params)
    return {"accuracy": accuracy, "loss": loss}

# Evaluation function
def evaluate(dataloader, state, args, writer):
    eval_metrics = []
    with tqdm(total=len(dataloader), desc="Evaluation ", leave=False) as progress_bar_eval:
        for batch in dataloader:
            eval_metric = eval_step(state, batch)
            eval_metrics.append(eval_metric)
            progress_bar_eval.update(1)

        eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.array(x), *eval_metrics)
        eval_metrics = jax.tree.map(jnp.mean, eval_metrics)

        progress_bar_eval.set_description(
            f"Evaluation | Loss: {eval_metrics['loss']} | Accuracy: {eval_metrics['accuracy']}"
        )

        for key, value in eval_metrics.items():
            writer.add_scalar(f"eval/{key}", round(float(value), 3), 0)


if __name__ == "__main__":

    args = parse_args()

    logging.info(f"Train and evaluate preference model with seed {args.seed}")

    # Ensure reproducibility
    rng = jax.random.PRNGKey(args.seed)

    # Initialize logging
    summary_writer = SummaryWriter(f"logs/{args.experiment_name}_{args.seed}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token # TODO: fix this to a better padding token

    # Initialize datasets
    train_dataset = datasets[args.train_dataset](
        args=args,
        tokenizer=tokenizer, 
        split="train",
    )
    eval_dataset = datasets[args.eval_dataset](
        args=args,
        tokenizer=tokenizer,
        split="test",
    )

    # Initialize training DataLoader
    rng, train_iter_rng = jax.random.split(rng, 2)
    train_dataloader = JaxDataloader(
        rng=train_iter_rng,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    # Initialize model state
    rng, train_state_rng = jax.random.split(rng, 2)
    state = create_train_state(args, train_state_rng)

    # Train model
    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epoch ", position=0, leave=True):
        state = train_single_epoch(args=args, state=state, epoch=epoch, dataloader=train_dataloader, writer=summary_writer)

    # Evaluate model
    eval_dataloader = JaxDataloader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
    )

    eval_acc = evaluate(args=args, state=state, dataloader=eval_dataloader, writer=summary_writer)
