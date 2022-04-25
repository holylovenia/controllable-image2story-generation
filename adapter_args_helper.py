from dataclasses import dataclass, field
from transformers import (
    TrainingArguments,
)
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_size: Optional[str] = field(
        default="medium", metadata={"help": "The size of pretrained GPT2 model."}
    )
    load_checkpoint_adapter: Optional[str] = field(
        default="", metadata={"help": "Path to adapter checkpoint."}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    dataset_path: Optional[str] = field(
        default='/home/bryan/datasets/bookcorpusopen/bookcorpusopen_chunked.arrow',
        metadata={"help": "Dataset path."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    bookcorpusopen_story_column_name: Optional[str] = field(
        default="chunk",
        metadata={"help": "The name of the dataset column containing the story data."},
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertraining to the training pipeline.
    """
    output_dir: Optional[str] = field(
        default="./save",
        metadata={"help": "Output directory"},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Evaluation accumulation steps"}
    )