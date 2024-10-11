import torch

from .generic_tokenizer import GenericTokenizer
import pandas as pd
import numpy as np
from typing import Tuple, Set, List, Dict

from event_prediction import data_utils


class Multi(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)
        tokenizer_dict = {}
        for col in self.data_processor.get_numeric_columns():
            tokenizer_dict[col] = NumericTokenizer()

        for col in self.data_processor.get_categorical_columns():
            tokenizer_dict[col] = CategoricalTokenizer()


    def normalize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = self.data_processor.normalize_data(dataset)


    def pretokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


    def model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


    def post_process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


    def tokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


    def encode(self, data: List[str]) -> torch.Tensor:
        pass


    def decode(self, data: torch.Tensor) -> List[str]:
        pass
