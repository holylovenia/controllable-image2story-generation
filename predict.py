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
import pandas as pd
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
            remove_columns=["image", "caption", "id", "image_id"], # "image_path"
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
    # Prepare Data Collator and Predictor
    ###
    print('Preparing Predictor...')
    def predict(datasets: dict, model: ClipCaptionModel, model_args, training_args, output_prefix: str = "coco", device = torch.device('cuda:0')):
        model.to(device)
        model.eval()

        test_dataloader = torch.utils.data.DataLoader(datasets["test"], batch_size=training_args.per_device_eval_batch_size,
                                shuffle=True, drop_last=training_args.dataloader_drop_last)

        sys.stdout.flush()
        test_results = {
            "id": [],
            "gold_caption": [],
            "generated_text_0": [],
            "generated_text_1": [],
            "generated_text_2": [],
            "generated_text_3": [],
            "generated_text_4": [],
        }
        for idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=output_prefix):
            input_ids = torch.stack(data["input_ids"][0], dim=1).to(device)
            masks = torch.stack(data["mask"][0], dim=1).to(device).float()
            prefixes = torch.stack(data["clip_embeddings"][0], dim=1).to(device).float()
            prefix_embeddings = model.get_prefix_projections(prefixes)

            def generate_text_using_beam_search(model, tokenizer, beam_size: int = 5,
                                prompt=None, embeddings=None, entry_length=70, temperature=1., stop_token: str = '.'):

                stop_token_index = tokenizer.encode(stop_token)[0]
                tokens = None
                scores = None
                device = next(model.parameters()).device
                seq_lengths = torch.ones(beam_size, device=device)
                is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
                
                with torch.no_grad():
                    if embeddings is not None:
                        generated_text_embeddings = embeddings
                    else:
                        if tokens is None:
                            tokens = torch.tensor(tokenizer.encode(prompt))
                            tokens = tokens.unsqueeze(0).to(device)
                            generated_text_embeddings = model.decoder.transformer.wte(tokens)

                    for i in range(entry_length):
                        outputs = model.decoder(inputs_embeds=generated_text_embeddings)
                        logits = outputs.logits
                        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                        logits = logits.softmax(-1).log()
                        if scores is None:
                            scores, next_tokens = logits.topk(beam_size, -1)
                            generated_text_embeddings = generated_text_embeddings.expand(beam_size, *generated_text_embeddings.shape[1:])
                            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                            if tokens is None:
                                tokens = next_tokens
                            else:
                                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                                tokens = torch.cat((tokens, next_tokens), dim=1)
                        else:
                            logits[is_stopped] = -float(np.inf)
                            logits[is_stopped, 0] = 0
                            scores_sum = scores[:, None] + logits
                            seq_lengths[~is_stopped] += 1
                            scores_sum_average = scores_sum / seq_lengths[:, None]
                            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                            next_tokens_source = next_tokens // scores_sum.shape[1]
                            seq_lengths = seq_lengths[next_tokens_source]
                            next_tokens = next_tokens % scores_sum.shape[1]
                            next_tokens = next_tokens.unsqueeze(1)
                            tokens = tokens[next_tokens_source]
                            tokens = torch.cat((tokens, next_tokens), dim=1)
                            generated_text_embeddings = generated_text_embeddings[next_tokens_source]
                            scores = scores_sum_average * seq_lengths
                            is_stopped = is_stopped[next_tokens_source]
                        next_token_embed = model.decoder.transformer.wte(next_tokens.squeeze()).view(generated_text_embeddings.shape[0], 1, -1)
                        generated_text_embeddings = torch.cat((generated_text_embeddings, next_token_embed), dim=1)
                        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                        if is_stopped.all():
                            break
                scores = scores / seq_lengths
                output_list = tokens.cpu().numpy()
                output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
                order = scores.argsort(descending=True)
                output_texts = [output_texts[i] for i in order]
                return output_texts

            test_results["id"].append(data["image_path"][0])
            test_results["gold_caption"].append(tokenizer.batch_decode(sequences=input_ids)[0].replace("!", ""))
            generated_texts = generate_text_using_beam_search(model, tokenizer, embeddings=prefix_embeddings)
            for i, text in enumerate(generated_texts):
                test_results["generated_text_{}".format(i)].append(text)

        test_df = pd.DataFrame.from_dict(test_results)
        test_df.to_csv(os.path.join(training_args.output_dir, "predict.csv"))

    print('Load model...')
    model = ClipCaptionModel(model_args.prefix_length, clip_length=model_args.prefix_length_clip, prefix_size=model_args.prefix_dim)
    if os.path.isdir(training_args.output_dir) and training_args.do_eval:
        model.load_state_dict(torch.load(os.path.join(training_args.output_dir, 'coco_latest.pt')))

    # Initialize training
    predict(preprocessed_datasets, model, model_args, training_args)
    
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