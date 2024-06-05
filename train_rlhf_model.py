import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune an LLM with RLHF")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    pass