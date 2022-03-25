import datasets
import pandas as pd
import torch


def pad_tokens(self, batch, data_args):
        tokens = batch["input_ids"]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            batch["input_ids"] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            batch["input_ids"] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        batch["mask"] = torch.cat((torch.ones(data_args.prefix_length), mask), dim=0)  # adding prefix mask
        return batch


def load_dataset(dir_path):
    preprocessed_dataset = datasets.load_from_disk('{}/preprocess_data.arrow'.format(dir_path))
    # # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # preprocessed_dataset.set_transform(_tokenize(tokenizer))
    return preprocessed_dataset