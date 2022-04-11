from functools import cache
from args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from datasets import load_from_disk, load_metric, set_caching_enabled, DatasetDict
from data_utils import load_dataset
from models import ClipCaptionModel
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
    AdamW,
    DataCollatorWithPadding,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import logging
import numpy as np
import os
import sys
import torch
import transformers

from datasets import load_dataset

import pandas as pd
import string



set_caching_enabled(True)
logger = logging.getLogger(__name__)


#####
# Main Functions
#####
def run(model_args, data_args, training_args):

    os.makedirs(data_args.cache_dir_path, exist_ok=True)

    bookcorpus = load_dataset("bookcorpusopen")
    bookcorpus["train"] = bookcorpus["train"].shard(num_shards=5, index=data_args.shard_index)
    print("BookCorpusOpen length: {}".format(len(bookcorpus["train"])))

    print("Loading genre source...")
    preprocessed_source_path = "{}/preprocessed_smashwords_april_2021.csv".format(data_args.cache_dir_path)
    if os.path.isfile(preprocessed_source_path):
        source = pd.read_csv(preprocessed_source_path)
    else:
        source = pd.read_csv("~/datasets/books1/smashwords_april_2021.csv")
        source["Plain_Title"] = source["Title"].apply(
            lambda title: str(title).lower().translate(str.maketrans("", "", string.punctuation)).replace("\s+", " "))
        source.to_csv(preprocessed_source_path)

    def search_genre_by_title(book):
        _title = book["title"].replace(".epub.txt", "")
        _title = " ".join(_title.split("-"))
        result = source.loc[source["Plain_Title"] == _title]
        if result is not None and len(result) > 0:
            book["genre"] = result["Categories"].replace("»", "|")
        else:
            book["genre"] = None
        return book

    print("Searching genre by title...")
    bookcorpus = bookcorpus.map(
        search_genre_by_title,
        num_proc=data_args.preprocessing_num_workers,
        batched=False,
        writer_batch_size=data_args.writer_batch_size,
        desc="genre search",
        load_from_cache_file=True,
        cache_file_names={
            "train": "{}/bookcorpus_genre_{:02d}.arrow".format(data_args.cache_dir_path, data_args.shard_index)
        },
    )

    bookcorpus.save_to_disk("{}/bookcorpusopen_genre_{:02d}.arrow".format(data_args.cache_dir_path, data_args.shard_index))


#####
# Entry Point
#####
def main():

    ###
    # Parsing & Initialization
    ###
    
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    
    # Init logging
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    run(model_args, data_args, training_args)

if __name__ == '__main__':
    main()