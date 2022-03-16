from args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from datasets import load_from_disk, load_metric, set_caching_enabled, DatasetDict
from data_utils import load_dataset
from transformers import (
    set_seed,
    CLIPProcessor,
    CLIPModel,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import datasets
import logging
import os
import sys
import transformers


set_caching_enabled(True)
logger = logging.getLogger(__name__)


#####
# Main Functions
#####
def run(model_args, data_args, training_args):
    ###
    # Prepare Processor & Model    
    ###
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    ###
    # Prepare Dataset
    ###
    raw_datasets = DatasetDict()
    print('Loading train dataset...')
    raw_datasets["train"] = load_dataset(data_args.train_manifest_path, data_args.preprocessing_num_workers, 
                                    data_args.image_column_name, data_args.text_column_name)
    print('Loading validation dataset...')
    raw_datasets["valid"] = load_dataset(data_args.valid_manifest_path, data_args.preprocessing_num_workers, 
                                    data_args.image_column_name, data_args.text_column_name)
    print('Loading test dataset...')
    raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.preprocessing_num_workers, 
                                    data_args.image_column_name, data_args.text_column_name)

    print('Preprocess dataset...')

    # Load model and processor
    clip_model = CLIPModel.from_pretrained(model_args.model_name_or_path)
    clip_processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)

    # Preprocess image sample and label text
    print('Vectorize dataset...')

    def prepare_dataset_with_clip_embeddings(batch):
        batch["input"] = clip_processor(text=batch["caption"], images=batch["image_path"], return_tensors="pt", padding=True)
        batch["clip_embeddings"] = clip_model.get_image_features(**batch["input"])
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset_with_clip_embeddings,
            remove_columns=raw_datasets["valid"].column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess datasets",
            load_from_cache_file=True,
            cache_file_names={
                "train": "{}/train_clip.arrow".format(cache_dir_path),
                "valid": "{}/valid_clip.arrow".format(cache_dir_path),
                "test": "{}/test_clip.arrow".format(cache_dir_path),
            }
        )

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}.")
        return

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