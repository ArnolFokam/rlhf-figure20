import logging
import argparse

import jax
import jax.numpy as jnp
import flax.linen as nn
from datasets import load_dataset
from flax.training.common_utils import shard
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('RLFH')

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="sshleifer/tiny-gpt2",
        help="The name of the hugging face language model to use for training the reward model",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed of the run for reproducibility",
    )

    parser.add_argument(
        "--max_seq_len", 
        type=int, 
        default=512,
        help="Max sequence length for the preference modelling input",
    )

    return parser.parse_args()


class PMHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == self.head_input_size

        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1 / jnp.sqrt(self.head_input_size + 1)),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        reward_gain = self.param("reward_gain", nn.initializers.ones_init(), ())
        reward_bias = self.param("reward_bias", nn.initializers.zeros_init(), ())
        x = x * reward_gain + reward_bias
        return x


def get_data_loader(rng, dataset, batch_size, shuffle=False):
    """get data loader from dataset"""
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


def train_reward_model(args: argparse.Namespace):
    """training module for the reward model"""

    logging.info(f"Training reward model with language model: {args.model_name}")

    # add tokenizer for query/response input
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        padding_side="right",
    )

    # manually add the padding token for reward model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # initialize the preference model
    pm_backbone =  FlaxAutoModelForCausalLM.from_pretrained(args.model_name)
    pm_head = PMHead(head_input_size=pm_backbone.config.hidden_size)

    key = jax.random.PRNGKey(args.seed)

    # load dataset
    raw_hh_dataset = load_dataset("Anthropic/hh-rlhf")
    
    # create training dataset
    def preprocess_sequence_pairs(examples):
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        chosen_tokenized = tokenizer(chosen, padding="max_length", max_length=args.max_seq_len, truncation=True, return_tensors="np")
        rejected_tokenized = tokenizer(rejected, padding="max_length", max_length=args.max_seq_len, truncation=True, return_tensors="np")
        return {
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "chosen_attention_mask": chosen_tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
            "rejected_attention_mask": rejected_tokenized["attention_mask"],
        }

    tokenized_hh_dataset = raw_hh_dataset.map(
        preprocess_sequence_pairs, 
        batched=True, 
        num_proc=4, 
        remove_columns=raw_hh_dataset["train"].column_names,
    )

    train_iter, test_iter = get_data_loader(tokenized_hh_dataset["train"]), get_data_loader(tokenized_hh_dataset["test"])


if __name__ == "__main__":
    args = parse_args()
    train_reward_model(args)
