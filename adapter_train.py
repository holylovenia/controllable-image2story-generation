from functools import cache
from adapter_args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from datasets import load_from_disk, load_metric, set_caching_enabled, DatasetDict
from data_utils import load_dataset
from itertools import chain
from models import ClipCaptionModel
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
    AdamW,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2Tokenizer,
    GPT2Model,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import logging
import math
import numpy as np
import os
import sys
import torch
import transformers
from torch.utils.data import Dataset

from ppcm_models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
from utils.helper import load_model_recursive

set_caching_enabled(True)
logger = logging.getLogger(__name__)


class BookcorpusopenGenreAdapterDataset(Dataset):
    def __init__(self, data_args, split, tokenizer, genre=None, adapter_id=-1,
                         sample_row=100, match_up_to_n_genres=None, truncate=True, 
                         max_seq_len=512, add_special_tokens=True,
                         *args, **kwargs):
        super(BookcorpusopenGenreAdapterDataset, self).__init__(*args, **kwargs)
        """
        Args:
            adapter_id: int, adapter_id for the genre we want the adapter to be trained with
        """
        
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.truncate = truncate
        self.max_seq_len = max_seq_len
        self.adapter_id = adapter_id
        self.preprocessing_num_workers = data_args.preprocessing_num_workers
        self.dataset = self.load_bookcorpusopen(split, genre, 
                                                match_up_to_n_genres,
                                                sample_row)

    def load_bookcorpusopen(self, split, genre='Fiction', 
                            match_up_to_n_genres=None, sample_row=None):
        """
        Load bookcorpusopen from pyarrow file.
        
        Further improvement:
        Group, concat, and truncate entries based on the adapter_id after tokenization
            
        Args:
            split: string, {train, valid, test}
            genre: string, genre that we want the adapter to be trained with, e.g. 'Fiction'
            match_up_to_n_genres: int, how many of the firsts bookcorpusopen genres entries 
                                    is considered as a genre to match with the genre input.
                                    None defaults to use all bookcorpusopen genres to match.
            sample_row: int, set the int number to sample the dataset, 
                        None means using all the datasets samples available
            match_up_to_n_genres
            
        Returns:
            dataset: tokenized huggingface dataset format from one of the bookcorpusopen split, 
                        with the adapter_id attached, and without any adapter_id = -1
        """

        def genre_match(entry_genres_string_list, genre, match_up_to_n_genres):
            """
            True to the genre that match to match_up_to_n_genres genres from the entry_genres
            else false
            """
            story_genre_list = [genre[1:-1] for genre in entry_genres_string_list[1:-1].split(', ')]
            story_genre_stringlist = ", ".join(story_genre_list[:match_up_to_n_genres])
            
            return genre.lower() in story_genre_stringlist.lower()
        
        def map_tokenization(batch):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized = self.tokenizer(batch[self.data_args.bookcorpusopen_story_column_name], 
                                          truncation=self.truncate,
                                          max_length=self.max_seq_len,
                                          add_special_tokens=self.add_special_tokens)
            return tokenized
        
        # Main data processing function that will concatenate all texts 
        # from our dataset and generate chunks of max_seq_len.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported 
            # it instead of this drop, you can customize this part to your needs.
            if total_length >= self.max_seq_len:
                total_length = (total_length // self.max_seq_len) * self.max_seq_len
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + self.max_seq_len] \
                    for i in range(0, total_length, self.max_seq_len)]
                for k, t in concatenated_examples.items()
            }
            return result
        
        # load bookcorpusopen from arrow file
        datasets = DatasetDict()
        print('Loading train, validation, test dataset...')
        datasets = load_from_disk(self.data_args.dataset_path)
        print('Loaded')
        
        # Select rows sampled and filter for the matching genres
        sample_row = len(datasets[split]) if sample_row == None else sample_row
        
        dataset = datasets[split].select(np.arange(0,sample_row,1))\
                                .filter(lambda x: genre_match(x['genre'], genre, match_up_to_n_genres)\
                                        , num_proc=self.preprocessing_num_workers)

        
        # Tokenize with huggingface datasets mapping function
        tokenized_dataset = dataset.map(
            map_tokenization,
            remove_columns=self.data_args.bookcorpusopen_story_column_name,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=True
        )
        print(split, 'split tokenized')
        
        group_concatted_dataset = tokenized_dataset.map(
                                        group_texts,
                                        batched=True,
                                        num_proc=self.preprocessing_num_workers,
                                        load_from_cache_file=True,
                                        desc=f"Grouping texts in chunks of {self.max_seq_len}",
                                    )
                                
        return group_concatted_dataset

    def __getitem__(self, index):
            
        forward_inputs = {}
        forward_inputs['task_id'] = self.adapter_id
        forward_inputs['input_ids'] = [self.dataset[index]['input_ids']]
        forward_inputs["labels"] = forward_inputs["input_ids"].copy()
        
        return forward_inputs

    def __len__(self):
        return self.dataset.num_rows


def run(model_args, data_args, training_args):
    
    model_args.model_path = f'ppcm_models/dialoGPT/{model_args.model_size}/'

    config = GPT2Config.from_json_file(os.path.join(model_args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_path)

    ## Load either Adapters' checkpoint, or just finetuned DialoGPT
    if(model_args.load_checkpoint_adapter != ""):
        print("Loading ADAPTERS")
        model = load_model_recursive(GPT2LMHeadModel(config), model_args.load_checkpoint_adapter, \
                                     model_args, verbose=True)
    else:
        model = load_model_recursive(GPT2LMHeadModel(config), \
                                     model_args.model_path+f"{model_args.model_size}_ft.pkl", \
                                     model_args, verbose=True)

    ## Load GPT2 instead of DialoGPT
    print('Load pretrained GPT2')
    pt_gpt2_model = GPT2Model.from_pretrained('gpt2-medium')

    model.transformer.wte.weight = pt_gpt2_model.wte.weight
    model.transformer.wpe.weight = pt_gpt2_model.wpe.weight

    layers = np.arange(0,len(pt_gpt2_model.h),1)
    for layer in layers:
        model.transformer.h[layer].ln_1.weight = pt_gpt2_model.h[layer].ln_1.weight
        model.transformer.h[layer].attn.c_attn.weight = pt_gpt2_model.h[layer].attn.c_attn.weight
        model.transformer.h[layer].attn.c_proj.weight = pt_gpt2_model.h[layer].attn.c_proj.weight
        model.transformer.h[layer].ln_2.weight = pt_gpt2_model.h[layer].ln_2.weight
        model.transformer.h[layer].mlp.c_fc.weight = pt_gpt2_model.h[layer].mlp.c_fc.weight
        model.transformer.h[layer].mlp.c_proj.weight = pt_gpt2_model.h[layer].mlp.c_proj.weight
    print('GPT2 pretrained params loaded to previous DialoGPT adapter')

    for n, p in model.named_parameters():
        if "adapter" not in str(n):
            p.requires_grad = False
    parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)]
    print('GPT2 param frozen, Adapter is trainable and initialized with AdamW')


    # Load the preprocessed dataset splits
    dataset_dict = {}
    for split in ['train', 'valid', 'test']:
        dataset_dict[split] = BookcorpusopenGenreAdapterDataset(
                                        data_args, split, tokenizer, genre=data_args.genre,
                                        adapter_id=data_args.adapter_id, sample_row=data_args.sample_row,
                                        match_up_to_n_genres=data_args.match_up_to_n_genres,
                                        max_seq_len=model_args.max_seq_len)
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    print('Preparing Trainer...')
     # Initialize Trainer
    trainer = Trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['valid'],
        model=model,
        data_collator=default_data_collator,
        args=training_args,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )
    
    ### Save path
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    run_name = 'GPT2{}_adapterid{}_genre{}_matched{}_sample{}_maxseqlen{}_bs{}_lr{}_{}epoch_wd{}_ws{}'\
                    .format(model_args.model_size,
                            data_args.adapter_id,
                            data_args.genre,
                            data_args.match_up_to_n_genres,
                            data_args.sample_row,
                            model_args.max_seq_len,
                            training_args.per_device_train_batch_size,
                            training_args.learning_rate,
                            training_args.num_train_epochs,
                            training_args.weight_decay,
                            training_args.warmup_steps)
    training_args.output_dir = training_args.output_dir + run_name
    
    ###
    # Training Phase
    ###
    if training_args.do_train:
        print('*** Training Phase ***')
        checkpoint = None

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        ### Saving
        trainer.save_model() # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset_dict["train"])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    if training_args.do_eval:
        print("*** Evaluation Phase ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset_dict["valid"])
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": "GPT2"+model_args.model_size, "tasks": "text-generation"}
    data_args.dataset_name = "Bookcorpusopen"
    data_args.dataset_config_name = "Chunked"
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    
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
        
    model_args.model_name_or_path = 'GPT2-'+model_args.model_size

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