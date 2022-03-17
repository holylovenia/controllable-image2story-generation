from datasets import Dataset
from PIL import Image

import json
import os
import pandas as pd


def pil_loader(batch):
    with open(batch["image_path"][0], "rb") as f:
        img = Image.open(f)
        batch["image"] = [img.convert("RGB")]
        return batch

def load_dataset(manifest_path, num_proc, image_column_name, text_column_name):
    base_path = os.path.join(*manifest_path.split("/")[:3])
    img_dir_name = manifest_path.split("/")[-1].split("_")[-1].replace(".json", "")
    img_dir_path = "{}/{}/".format(base_path, img_dir_name)

    with open(manifest_path, "r") as f:
        annotations = json.loads(f.readline())["annotations"]
    df = pd.json_normalize(annotations)
    df["image_path"] = df[image_column_name].apply(lambda img_id: os.path.join(img_dir_path, f"{int(img_id):012d}.jpg"))

    batches = Dataset.from_pandas(df)
    batches.set_transform(pil_loader)
    return batches