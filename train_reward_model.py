import argparse
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('RLFH')

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="openai-community/gpt2",
        help="The name of the hugging face language model to use for training the reward model",
    )

    return parser.parse_args()




def train_reward_model(args: argparse.Namespace):
    """training module for the reward model"""

    logging.info(f"Training reward model with language model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    text = "Hello, my dog is cute"
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

if __name__ == "__main__":
    args = parse_args()
    train_reward_model(args)
