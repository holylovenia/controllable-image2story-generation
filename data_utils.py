from datasets import Dataset
from PIL import Image
from transformers import (
    AutoTokenizer
)

import datasets
import json
import os
import pandas as pd


def _tokenize(batch, tokenizer):
    input = tokenizer(text=batch["caption"], return_tensors="pt")
    batch["input_ids"] = input["input_ids"]
    return batch


def load_dataset(dir_path, model_args):
    preprocessed_dataset = datasets.load_from_disk('{}/preprocess_data.arrow'.format(dir_path))
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    preprocessed_dataset.set_transform(_tokenize(tokenizer))
    return preprocessed_dataset