import torch

from .generic_tokenizer import GenericTokenizer
import pandas as pd
import numpy as np
from typing import Tuple, Set, List, Dict

from event_prediction import data_utils
import datasets


class Simple(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)


    def normalize(self, dataset: datasets.Dataset) -> datasets.Dataset:
        dataset = self.data_processor.normalize_data(dataset)
        for col in self.data_processor.get_numeric_columns():
            dataset[col], buckets = data_utils.convert_to_binary_string(dataset[col], self.numeric_bucket_amount)
        # def batch_iterator(batch_size=1024):
        #     len_dataset = len(dataset)
        #     for i in range(0, len_dataset, batch_size):
        #         rows = dataset.loc[i: i + batch_size]
        #         rows = rows.astype(str).values.tolist()
        #         rows = [" ".join(x) for x in rows]
        #         yield rows

        # dataset = ' '.join([' '.join(x) for x in dataset.astype(str).values.tolist()])
        return datasets.Dataset.from_pandas(dataset)


    def pretokenize(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset


    def model(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset


    def post_process(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset


    def tokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


    def encode(self, data: List[str]) -> torch.Tensor:
        pass


    def decode(self, data: torch.Tensor) -> List[str]:
        pass
