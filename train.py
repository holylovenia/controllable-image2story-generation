from functools import cache
from args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from datasets import load_from_disk, load_metric, set_caching_enabled, DatasetDict
from data_utils import load_dataset
from models import ClipCaptionModel
from transformers import (
    set_seed,
    GPT2Tokenizer,
    GPT2LMHeadModel,
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

    if data_args.cache_dir_path is None:
        data_args.cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(data_args.cache_dir_path, exist_ok=True)

    ###
    # Prepare Dataset
    ###
    preprocessed_datasets = DatasetDict()
    print('Loading train, validation, test dataset...')
    preprocessed_datasets = load_dataset(data_args.cache_dir_path)

    print('Preprocess dataset...')

    # Load model and processor
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)

    # Preprocess image sample and label text
    print('Vectorize dataset...')

    def tokenize(batch):
        batch = tokenizer(text=batch["caption"], return_tensors="pt")
        return batch

    with training_args.main_process_first(desc="dataset tokenization"):
        preprocessed_datasets = preprocessed_datasets.map(
            tokenize,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["image", "image_path", "caption", "id", "image_id"],
            batched=False,
            writer_batch_size=data_args.writer_batch_size,
            desc="preprocess datasets",
            load_from_cache_file=True,
            cache_file_names={
                "train": "{}/train_tokenized.arrow".format(data_args.cache_dir_path),
                "valid": "{}/valid_tokenized.arrow".format(data_args.cache_dir_path),
                "test": "{}/test_tokenized.arrow".format(data_args.cache_dir_path),
            }
        )

    print(preprocessed_datasets["valid"].features)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {preprocessed_datasets.cache_files}.")
        return

    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')

    print('Load model...')
    print('Model ID', model_args.model_name_or_path)
    model = ClipCaptionModel(model_args.prefix_length, clip_length=model_args.prefix_length_clip, prefix_size=model_args.prefix_dim)

    # Initialize Trainer
    trainer = Trainer(
        train_dataset=preprocessed_datasets["train"],
        eval_dataset=preprocessed_datasets["valid"],
        model=model,
        args=training_args,
        compute_metrics=datasets.load_metric("bleu"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(preprocessed_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Evaluation Phase ***")
    metrics = trainer.evaluate(eval_dataset=preprocessed_datasets["valid"])
    metrics["eval_samples"] = len(preprocessed_datasets["valid"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


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