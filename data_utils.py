from datasets import Dataset
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
)

import json
import os
import pandas as pd


class EmbeddingsDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def _extract_embeddings(self, batch):
        with open(batch["image_path"][0], "rb") as f:
            img = Image.open(f)
            batch["image"] = [img.convert("RGB")]
            input = self.processor(text=batch["caption"], images=batch["image"], return_tensors="pt")
            batch["input_ids"], batch["pixel_values"] = input["input_ids"], input["pixel_values"]
            batch["clip_embeddings"] = self.model.get_image_features(batch["pixel_values"], return_dict=True)
            return batch

def load_dataset(manifest_path, num_proc, image_column_name, text_column_name):
    base_path = os.path.join(*manifest_path.split("/")[:3])
    img_dir_name = manifest_path.split("/")[-1].split("_")[-1].replace(".json", "")
    img_dir_path = "{}/{}/".format(base_path, img_dir_name)

    with open(manifest_path, "r") as f:
        annotations = json.loads(f.readline())["annotations"]
    df = pd.json_normalize(annotations)
    df["image_path"] = df[image_column_name].apply(lambda img_id: os.path.join(img_dir_path, f"{int(img_id):012d}.jpg"))

    batches = EmbeddingsDataset.from_pandas(df)
    batches.set_transform(batches._extract_embeddings)
    return batches