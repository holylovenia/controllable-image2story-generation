from cgi import test
from functools import cache
from args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from datasets import concatenate_datasets, load_dataset, load_from_disk, set_caching_enabled, Dataset, DatasetDict
from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from tqdm import tqdm

import logging
import numpy as np
import os
import pandas as pd
import sys
import transformers


set_caching_enabled(True)
logger = logging.getLogger(__name__)


#####
# Main Functions
#####
def run(model_args, data_args, training_args):

    os.makedirs(data_args.cache_dir_path, exist_ok=True)

    def is_bookcorpusopen_genre_ready(data_args):
        for shard_index in range(data_args.num_shards):
            shard_path = "{}/bookcorpusopen_genre_{:02d}.arrow".format(data_args.cache_dir_path, shard_index)
            if not os.path.exists(shard_path):
                return False
        return True

    if not is_bookcorpusopen_genre_ready(data_args):
        logger.info(f"Not all data has genre. Files cached at {data_args.cache_dir_path}.")
        return

    # Load all dataset shards and concatenate them into a single dataset
    dataset_shards = []
    for shard_index in range(data_args.num_shards):
        shard_path = "{}/bookcorpusopen_genre_{:02d}.arrow".format(data_args.cache_dir_path, shard_index)
        _shard = load_from_disk(shard_path)
        dataset_shards.append(_shard["train"])
    bookcorpusopen = DatasetDict()
    bookcorpusopen["train"] = concatenate_datasets(dataset_shards)
    
    def convert_book_to_chunks(book, resulting_dict, min_chunk_words=30, max_chunk_words=60):
        chunks = book["text"].replace("\s+", "\n").split("\n")

        for i, chunk in enumerate(chunks):
            chunk_length = len(chunk.split(" "))
            if chunk_length >= min_chunk_words and chunk_length <= max_chunk_words:
                resulting_dict["title"].append(book["title"])
                resulting_dict["chunk"].append(chunk)

                _genre = book["genre"][0].replace("»", ",") if book["genre"] is not None else ""
                _genre = str(set([g.strip() for g in _genre.split(",")]))
                resulting_dict["genre"].append(_genre)

        return resulting_dict

    print("Filter chunks from books...")
    data = {
        "title": [],
        "genre": [],
        "chunk": [],
    }
    for i, book in tqdm(enumerate(bookcorpusopen["train"]), total=len(bookcorpusopen["train"])):
        data = convert_book_to_chunks(book, data)

    df = pd.DataFrame(data)
    bookcorpus_chunks = Dataset.from_pandas(df).shuffle(seed=training_args.seed)

    print('Splitting dataset to train, valid, test')
    bookcorpus_chunks_train = bookcorpus_chunks.train_test_split(test_size=0.2, train_size=0.8)
    bookcorpus_chunks_train, bookcorpus_chunks_test = bookcorpus_chunks_train["train"], bookcorpus_chunks_train["test"]
    bookcorpus_chunks_test = bookcorpus_chunks_test.train_test_split(test_size=0.5)
    
    preprocessed_datasets = DatasetDict({
        "train": bookcorpus_chunks_train,
        "valid": bookcorpus_chunks_test["train"],
        "test": bookcorpus_chunks_test["test"],
    })

    print('Saving to disk...')
    preprocessed_datasets.save_to_disk("{}/bookcorpusopen_chunked.arrow".format(data_args.cache_dir_path))



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