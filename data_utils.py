from torch.utils.data import DataLoader

import datasets
import pandas as pd


# class DataLoader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.collate_fn = self._collate_fn

#     def _collate_fn(self, batch):

#         batch_input_ids, batch_mask, batch_token_type_ids, batch_labels = [], [], [], []
#         for data in batch:
#             batch_input_ids.append(np.array(data['input_ids']))
#             batch_mask.append(np.array(data['attention_mask']))
#             if 'token_type_ids' in data:
#                 batch_token_type_ids.append(np.array(data['token_type_ids']))
#             batch_labels.append(data['label'])

#         return np.array(batch_input_ids), np.array(batch_mask), np.array(batch_token_type_ids) if len(batch_token_type_ids) > 0 else None, np.array(batch_labels)



def load_dataset(dir_path):
    preprocessed_dataset = datasets.load_from_disk('{}/preprocess_data.arrow'.format(dir_path))
    return preprocessed_dataset