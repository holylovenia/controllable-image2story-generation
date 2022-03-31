from torch.utils.data import DataLoader

import datasets
import pandas as pd


def load_dataset(dir_path):
    preprocessed_dataset = datasets.load_from_disk('{}/preprocess_data.arrow'.format(dir_path))
    return preprocessed_dataset