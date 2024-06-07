import jax
import jax.numpy as jnp
from datasets import load_dataset, concatenate_datasets


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
            for k, v in batch.items():
                if not isinstance(v[0], str):
                    batch[k] = jnp.array(v)
                else:
                    batch[k] = v
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


# Custom dataset class for the Antropic/HH-RLHF preferences data
class HHPreferencesDatasets:
    def __init__(self,  args, tokenizer, split) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.dataset = dataset.map(
            lambda x: self.preprocess_sequence_pairs(x, tokenizer, self.max_seq_len),
            num_proc=4,
            batched=True,
            remove_columns=dataset.column_names,
        )

    @staticmethod
    def preprocess_sequence_pairs(examples, tokenizer, max_seq_len):
        chosen_tokenized = tokenizer(
            examples["chosen"],
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        rejected_tokenized = tokenizer(
            examples["rejected"],
            padding="max_length",
            max_length=max_seq_len,
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


# Custom dataset class for the Sentiment preferences data
class SentimentPreferencesDataset(HHPreferencesDatasets):
    
    def __init__(self,  args, tokenizer, split) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        
        if split == "train":
            slice = ":4000"
        elif split == "test":
            slice = "4000:5300"
        else:
            raise ValueError("split must be either 'train' or 'test'")

        dataset = load_dataset("OEvortex/SentimentSynth", split=f"train[{slice}]")
    
        self.dataset = dataset.map(
            lambda x: self.preprocess_sequence_pairs(x, tokenizer, self.max_seq_len),
            num_proc=4,
            remove_columns=dataset.column_names,
        )

    @staticmethod
    def preprocess_sequence_pairs(examples, tokenizer, max_seq_len):
        # This piece of code only works when the processing is not batched.
        examples['chosen'] = '\n\nHuman: '+examples['prompt']+'\n\nAssistant: '+examples['chosen']
        examples['rejected'] = '\n\nHuman: '+examples['prompt']+'\n\nAssistant: '+examples['rejected']
        
        examples = HHPreferencesDatasets.preprocess_sequence_pairs(examples, tokenizer, max_seq_len)

        return {k:v[0] for k, v in examples.items()}


# Custom dataset class for the HH-RLF/Sentiment preferences data
class HHSentimentMixPreferencesDataset(HHPreferencesDatasets):
    
    def __init__(self,  args, tokenizer, split) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.seed = args.seed
        
        # load sentiment dataset
        if split == "train":
            slice = ":4000"
        elif split == "test":
            slice = "4000:5300"
        else:
            raise ValueError("split must be either 'train' or 'test'")
        sent_dataset = load_dataset("OEvortex/SentimentSynth", split=f"train[{slice}]")
        sent_dataset = sent_dataset.map(
            lambda x: self.sentiment_preprocess_sequence_pairs(x, tokenizer, self.max_seq_len),
            num_proc=4,
            remove_columns=sent_dataset.column_names,
        )

        # load HH-RLF dataset
        hh_dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        hh_dataset = hh_dataset.map(
           lambda x: self.hh_preprocess_sequence_pairs(x, tokenizer, self.max_seq_len),
            num_proc=4,
            batched=True,
            remove_columns=hh_dataset.column_names,
        )

        self.dataset = concatenate_datasets([sent_dataset, hh_dataset]).shuffle(seed=self.seed)
        pass

    @staticmethod
    def hh_preprocess_sequence_pairs(examples, tokenizer, max_seq_len):
        return HHPreferencesDatasets.preprocess_sequence_pairs(examples, tokenizer, max_seq_len)

    @staticmethod
    def sentiment_preprocess_sequence_pairs(examples, tokenizer, max_seq_len):
        return SentimentPreferencesDataset.preprocess_sequence_pairs(examples, tokenizer, max_seq_len)


# custom dataset class for Sentiment prompts data
class SentimentsPromptsDatasets:
    def __init__(self, args, tokenizer) -> None:
        self.max_query_length = args.max_query_length
        dataset = load_dataset("OEvortex/SentimentSynth", split="train")
        self.dataset = dataset.map(
            lambda x: self.preprocess_sequence(x, tokenizer, self.max_query_length),
            num_proc=4,
            batched=True,
            remove_columns=dataset.column_names,
        )

    @staticmethod
    def preprocess_sequence(examples, tokenizer, max_query_length):
        prompt_tokenized = tokenizer(
            examples["prompt"],
            padding="max_length",
            max_length=max_query_length,
            return_tensors="np",
            truncation=True,
        )
        return {
            "query": examples["prompt"],
            "input_ids": prompt_tokenized["input_ids"],
            "attention_mask": prompt_tokenized["attention_mask"],
        }

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


pm_datasets = {
    "hh-rlhf": HHPreferencesDatasets,
    "sentiment": SentimentPreferencesDataset,
    "mix": HHSentimentMixPreferencesDataset,
}

prompts_datasets = {
    "sentiment": SentimentsPromptsDatasets,
}
