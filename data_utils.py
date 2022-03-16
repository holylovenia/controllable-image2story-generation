from email.mime import image
from datasets import load_dataset
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader

import json
import os
import pandas as pd


def pil_loader(batch):
    with open(batch["image_path"], "rb") as f:
        img = Image.open(f)
        batch["img"] = img.convert("RGB")
        return batch


def load_dataset(data_split, num_proc):
    base_path = "./data/MS-COCO"
    annotation_dir_path = "{}/annotations/captions_{}2017.json".format(base_path, data_split)
    img_dir_path = "{}/{}2017/".format(base_path, data_split)

    with open(annotation_dir_path, "r") as f:
        annotations = json.loads(f.readline())["annotations"]
    df = pd.json_normalize(annotations)
    df["image_path"] = df["image_id"].apply(lambda img_id: os.path.join(img_dir_path, f"{int(img_id):012d}.jpg"))

    batches = Dataset.from_pandas(df)
    batches = batches.map(pil_loader, num_proc=num_proc)
    return batches