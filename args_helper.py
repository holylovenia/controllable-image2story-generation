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
    model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-base-patch32", metadata={"help": "The path of the HuggingFace model."}
    )
    prefix_length: Optional[int] = field(
        default=10,
        metadata={"help": "Prefix length"},
    )
    prefix_length_clip: Optional[int] = field(
        default=10,
        metadata={"help": "Prefix length clip"},
    )
    prefix_dim: Optional[int] = field(
        default=512,
        metadata={"help": "Prefix dimension"},
    )
    normalize_prefix: Optional[bool] = field(
        default=True,
        metadata={"help": "Normalize prefix."},
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    train_manifest_path: Optional[str] = field(
        default="./data/MS-COCO/annotations/captions_train2017.json", metadata={"help": "The path of the training dataset to use."}
    )
    valid_manifest_path: Optional[str] = field(
        default="./data/MS-COCO/annotations/captions_val2017.json", metadata={"help": "The path of the validation dataset to use."}
    )
    test_manifest_path: Optional[str] = field(
        default="./data/MS-COCO/annotations/captions_val2017.json", metadata={"help": "The path of the testing dataset to use."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only run preprocessing."},
    )
    image_column_name: Optional[str] = field(
        default="image_id",
        metadata={"help": "The name of the dataset column containing the image path. Defaults to 'image_id'"},
    )
    text_column_name: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text_path'"},
    )
    cache_dir_name: Optional[str] = field(
        default="cache",
        metadata={"help": "Name of cache directory"},
    )
    cache_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of cache directory"},
    )
    writer_batch_size: Optional[str] = field(
        default=50,
        metadata={"help": "Number of rows per write operation for the cache file writer. This value is a good trade-off between memory usage during the processing, and processing speed. Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running .map()."},
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