import functools
import logging
import argparse

import jax
import flax
import optax
import jax.numpy as jnp
import flax.linen as nn
from datasets import load_dataset
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, get_metrics
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

    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=64,
        help="Batch size for the training dataset",
    )

    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=32,
        help="Batch size for the evaluation dataset",
    )

    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5,
        help="Learning rate for the training of the reward model",
    )

    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=1,
        help="Number of epochs for training the reward model",
    )

    return parser.parse_args()


class PMHead(nn.Module):
    head_input_size: int

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == self.head_input_size

        return nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1 / jnp.sqrt(self.head_input_size + 1)),
            bias_init=nn.initializers.zeros_init(),
        )(x)


@flax.struct.dataclass
class RewardModelParams:
    """Parameters for the reward model."""

    lm_backbone_params: flax.core.FrozenDict
    head_params: flax.core.FrozenDict


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

@functools.partial(jax.pmap, axis_name="batch")
def train_step(state, batch):
    """Preference model training step"""

    chosen_input_ids, rejected_input_ids, chosen_attention_mask, rejected_attention_mask = batch
    labels = jnp.zeros(chosen_input_ids.shape[0], dtype=jnp.int32)

    def loss_function(params):
        logits = state.apply_fn(
            params, 
            chosen_input_ids, 
            rejected_input_ids,
            chosen_attention_mask,
            rejected_attention_mask,
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        accuracy = (logits.argmax(axis=1) == labels).astype("float32").mean()
        return loss, accuracy

    (loss, accuracy), grads = jax.value_and_grad(loss_function, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, "batch")

    state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")

    return state, {"loss": loss, "accuracy": accuracy}


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

    # split the dataset into training and evaluation
    train_iter_key1, train_iter_key2, key = jax.random.split(key, 3)
    train_iter, test_iter = (
        get_data_loader(
            rng=train_iter_key1,
            dataset=tokenized_hh_dataset["train"],
            batch_size=args.train_batch_size,
            shuffle=True,
        ), 
        get_data_loader(
            rng=train_iter_key2,
            dataset=tokenized_hh_dataset["test"],
            batch_size=args.eval_batch_size,
        ),
    )

    # initialize model
    pm_backbone =  FlaxAutoModelForCausalLM.from_pretrained(args.model_name)
    pm_head = PMHead(head_input_size=pm_backbone.config.hidden_size)

    def get_reward(
        params: RewardModelParams,
        chosen_input_ids: jnp.ndarray,
        rejected_input_ids: jnp.ndarray,
        chosen_attention_mask: jnp.ndarray,
        rejected_attention_mask: jnp.ndarray,
    ):
        """Reward model forward pass"""

        def get_logits(
            input_ids: jnp.ndarray,
            attention_mask: jnp.ndarray,
        ):
            # shape: [batch_size, length, hidden_size]
            position_ids = jnp.cumsum(attention_mask, axis=1) - attention_mask
            hidden_states = pm_backbone.module.apply(
                variables=params.lm_backbone_params,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            ).hidden_states[-1]

            # shape: [batch_size, hidden_size]
            last_hidden_states = hidden_states[:, -1, :]

            # shape: [batch_size, 1]
            return pm_head.apply(variables=params.head_params, x=last_hidden_states)

        chosen_reward = get_logits(chosen_input_ids, chosen_attention_mask)
        rejected_reward = get_logits(rejected_input_ids, rejected_attention_mask)

        return jnp.concatenate([chosen_reward, rejected_reward], axis=1)


    init_key, key = jax.random.split(key)

    optimizer = optax.adamw(
        learning_rate=optax.linear_schedule(
            init_value=args.learning_rate, 
            end_value=0, 
            transition_steps=len(tokenized_hh_dataset["train"]) // args.train_batch_size * args.num_epochs
        ), 
        b1=0.9, 
        b2=0.98, 
        eps=1e-8, 
        weight_decay=0.01,
    )
    training_state = TrainState.create(
        apply_fn=get_reward,
        params=RewardModelParams(
            lm_backbone_params=flax.core.FrozenDict({
                "params": pm_backbone.params,
            }),
            head_params=flax.core.FrozenDict(
                pm_head.init(
                    init_key,
                    jnp.ones(pm_backbone.config.hidden_size)[None, None, :],
                )
            ),
        ),
        tx=optimizer,
    )

    logging.info(f"<============= Training started =============>")

    # training loop
    for global_step, training_batch in enumerate(train_iter):

        training_state, train_metrics = train_step(training_state, training_batch)

        train_metrics = get_metrics([train_metrics])

        for key, value in train_metrics.items():
            print(f"train/{key}", value, global_step)



if __name__ == "__main__":
    args = parse_args()
    train_reward_model(args)
