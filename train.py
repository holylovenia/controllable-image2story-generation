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

    def pad_tokens(batch, max_seq_len=70): # max length from the data is 67
        tokens = batch["input_ids"]
        padding = max_seq_len - tokens.shape[1]
        if padding > 0:
            tokens = torch.cat((tokens,
                torch.zeros((tokens.shape[0], padding), dtype=torch.int64) - 1), dim=1)
            batch["input_ids"] = tokens
        elif padding < 0:
            batch["input_ids"] = tokens[:max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        batch["mask"] = torch.cat((torch.ones((mask.shape[0], model_args.prefix_length)), mask), dim=1)  # adding prefix mask
        return batch

    def tokenize(batch):
        batch["input_ids"] = tokenizer.encode(text=batch["caption"], return_tensors="pt")
        batch = pad_tokens(batch)
        if model_args.normalize_prefix:
            batch["clip_embeddings"] = torch.Tensor(batch["clip_embeddings"]).float()
            batch["clip_embeddings"] = batch["clip_embeddings"] / batch["clip_embeddings"].norm(2, -1)
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

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {preprocessed_datasets.cache_files}.")
        return
        
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')
    def train(datasets: dict, model: ClipCaptionModel, model_args, training_args, output_prefix: str = "coco", device = torch.device('cuda:0')):
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
            
        model.to(device)
        model.train()
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
        train_dataloader = torch.utils.data.DataLoader(datasets["train"], batch_size=training_args.per_device_train_batch_size,
                                shuffle=True, drop_last=training_args.dataloader_drop_last)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.num_train_epochs * len(train_dataloader)
        )
        
        # save_config(model_args)
        for epoch in range(int(training_args.num_train_epochs)):
            print(f">>> Training epoch {epoch}")
            sys.stdout.flush()
            progress = tqdm(total=len(train_dataloader), desc=output_prefix)
            for idx, data in enumerate(train_dataloader):

                input_ids = torch.stack(data["input_ids"][0], dim=1).to(device)
                masks = torch.stack(data["mask"][0], dim=1).to(device).float()
                prefixes = torch.stack(data["clip_embeddings"][0], dim=1).to(device).float()

                model.zero_grad()
                outputs = model(input_ids, masks, prefixes)
                logits = outputs.logits[:, model_args.prefix_length - 1: -1]
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids.flatten(), ignore_index=0)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.set_postfix({"loss": loss.item()})
                progress.update()
                if (idx + 1) % 10000 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(training_args.output_dir, f"{output_prefix}_latest.pt"),
                    )
            progress.close()
            if epoch % training_args.save_steps == 0 or epoch == training_args.num_train_epochs - 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(training_args.output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
        return model

    print('Load model...')
    model = ClipCaptionModel(model_args.prefix_length, clip_length=model_args.prefix_length_clip, prefix_size=model_args.prefix_dim)

    # Initialize training
    train(preprocessed_datasets, model, model_args, training_args)
    
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