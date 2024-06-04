import logging
import argparse
from tqdm import tqdm
from typing import NamedTuple

import jax
import flax
import optax
import jax.numpy as jnp
import flax.linen as nn
from datasets import load_dataset
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from flax.training.common_utils import get_metrics
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


class HHPreferencesDatasets:
    def __init__(self,  tokenizer, max_seq_len, split) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.dataset = dataset.map(
            self._preprocess_sequence_pairs,
            num_proc=4,
            batched=True,
            remove_columns=dataset.column_names,
        )

        pass

    def _preprocess_sequence_pairs(self, examples):
        chosen_tokenized = self.tokenizer(
            examples["chosen"],
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        rejected_tokenized = self.tokenizer(
            examples["rejected"],
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        return {
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "chosen_attention_mask": chosen_tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
            "rejected_attention_mask": rejected_tokenized["attention_mask"],
        }

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
# Custom dataset class for the training data
class SentimentPreferencesDataset(HHPreferencesDatasets):
    
    def __init__(self,  tokenizer, max_seq_len, split) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if split == "train":
            slice = ":4000"
        elif split == "test":
            slice = "4000:5300"
        else:
            raise ValueError("split must be either 'train' or 'test'")

        dataset = load_dataset("OEvortex/SentimentSynth", split=f"train[{slice}]")
    
        self.dataset = dataset.map(
            self._preprocess_sequence_pairs,
            num_proc=4,
            remove_columns=dataset.column_names,
        )

        pass

    def _preprocess_sequence_pairs(self, examples):
        # This piece of code only works when the processing is not batched.
        examples['chosen'] = '\n\nHuman: '+examples['prompt']+'\n\nAssistant: '+examples['chosen']
        examples['rejected'] = '\n\nHuman: '+examples['prompt']+'\n\nAssistant: '+examples['rejected']
        
        examples = super()._preprocess_sequence_pairs(examples)

        return {k:v[0] for k, v in examples.items()}


# Custom DataLoader class for JAX
class JaxDataloader:
    def __init__(self, dataset, batch_size, rng=None, shuffle=False):
        if shuffle:
            assert isinstance(rng, jax.Array), "rng must be provided if shuffle is True"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng

    def __iter__(self):
        steps_per_epoch = len(self.dataset) // self.batch_size
        if self.shuffle:
            _, self.rng = jax.random.split(self.rng)
            batch_idx = jax.random.permutation(self.rng, len(self.dataset))
        else:
            batch_idx = jnp.arange(len(self.dataset))

        batch_idx = batch_idx[:steps_per_epoch * self.batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, self.batch_size))

        for idx in batch_idx:
            batch = self.dataset[idx]
            batch = {k: jnp.array(v) for k, v in batch.items()}
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

# Regression head class for preference model
class RegressionHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)

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
def evaluate(epoch, dataloader, state, args, writer):
    eval_metrics = []
    with tqdm(total=len(dataloader), desc="Evaluation ", leave=False) as progress_bar_eval:
        for batch in dataloader:
            eval_metric = eval_step(state, batch)
            eval_metrics.append(eval_metric)
            progress_bar_eval.update(1)

        eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.array(x), *eval_metrics)
        eval_metrics = jax.tree.map(jnp.mean, eval_metrics)

        progress_bar_eval.set_description(
            f"Evaluation ({epoch}/{args.num_epochs}) | Loss: {eval_metrics['loss']} | Accuracy: {eval_metrics['accuracy']}"
        )

        for key, value in eval_metrics.items():
            writer.add_scalar(f"eval/{key}", round(float(value), 3), 0)

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

    return state

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
    train_dataset = SentimentPreferencesDataset(
        tokenizer=tokenizer, 
        max_seq_len=args.max_seq_len, 
        split="train",
    )
    test_dataset = SentimentPreferencesDataset(
        tokenizer=tokenizer, 
        max_seq_len=args.max_seq_len, 
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
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
    )

    eval_acc = evaluate(args=args, state=state, epoch=epoch, dataloader=eval_dataloader, writer=summary_writer)
