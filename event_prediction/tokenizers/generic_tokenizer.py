import logging

import pandas as pd

from event_prediction import data_utils, get_data_processor
import torch
from typing import Set, List, Dict

log = logging.getLogger(__name__)


class GenericTokenizer:
    def __init__(self, tokenizer_cfgs, data_cfgs):
        self.special_tokens_dict = {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }

        self.tokenizer_type = data_cfgs.data_processor
        self.data_processor = get_data_processor(data_cfgs)
        self.numeric_bucket_type = tokenizer_cfgs.numeric_bucket_type
        self.numeric_bucket_amount = tokenizer_cfgs.numeric_bucket_amount
        self.normalization_type = tokenizer_cfgs.normalization_type

        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = set()
        self.total_tokens = 0

        self.buckets = {}

        self.is_train = True

        # Properties expected by huggingface Trainer
        self.pad_token = self.special_tokens_dict["pad_token"]
        self.bos_token = self.special_tokens_dict["bos_token"]
        self.eos_token = self.special_tokens_dict["eos_token"]
        self.unk_token = self.special_tokens_dict["unk_token"]

        # todo should we assign the special tokens with ids here?
        # self.bos_token_id = None
        # self.eos_token_id = None
        self.unk_token_id = self.add_token(
            self.unk_token
        )  # init with the unknown token
        self.is_initialized = False

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def normalize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Normalizes all the data in the table
        This includes:
            1. bucketing numeric values if doing that (maybe preprocessing?)
            2. adding any new values to the table (such as converting dollars, adding total minutes, etc)
        """
        dataset = self.data_processor.normalize_data(dataset)

        # todo normalize and bucket by user?
        # if self.normalization_type is not None:
        #     updated = normalize_numeric(data[self.numeric_columns], self.normalization_type)
        #     data[self.numeric_columns].replace(updated[self.numeric_columns])

        if self.numeric_bucket_type is not None:
            for col_name in self.data_processor.get_numeric_columns():
                col = dataset[col_name]
                if self.is_train:
                    updated, buckets = data_utils.bucket_numeric(
                        col, self.numeric_bucket_type, self.numeric_bucket_amount
                    )
                    self.buckets[col_name] = list(buckets)
                else:
                    updated, _ = data_utils.bucket_numeric(
                        col, "uniform", self.buckets[col_name]
                    )  # eval always is uniform bc passing in buckets
                    updated.fillna(0, inplace=True)
                updated = updated.astype(str)
                dataset[col_name] = updated

            # Static data is aggregated from a parent column (like the avg of Amount) so categorize into the buckets of the parent
            for static_col_dict in self.data_processor.get_static_numeric_columns():
                col_name = static_col_dict["name"]
                parent_name = static_col_dict["parent"]
                col = dataset[col_name]
                updated, _ = data_utils.bucket_numeric(
                    col, "uniform", self.buckets[parent_name]
                )
                updated.fillna(0, inplace=True)
                updated = updated.astype(str)
                dataset[col_name] = updated

        return dataset

    def pretokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """This is where the different tokenizers will convert the tables into 'sentences' of 'words'
        The words outputted from here can be:
            1. Composite tokens with goal of predicting next composite token
            2. atomic tokens with the goal of predicting the next set of atomic tokens
            3. The tokens can actually be embedding vectors
        so the goal here is to create those sentences to be passed.  The output here should be agnostic to the problem (of tabular data)
        """
        raise NotImplementedError()

    def model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Tokenization here consists of taking the previous 'sentences' and doing actual tokenization such as:
        1. BPE
        2. Word Piece
        3. Word Level
        4. Unigram
        """
        # raise NotImplementedError()
        self.add_special_tokens()
        for col_name in self.get_token_cols():
            self.add_all_tokens(set(dataset[col_name].values.tolist()))
        return dataset

    def post_process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # todo consider splitting into training/test sets
        raise NotImplementedError()

    def training_complete(self):
        self.is_initialized = True

    def get_token_cols(self):
        raise NotImplementedError()

    def _encode_val(self, data: str) -> int:
        return self.token_to_id.get(data, self.unk_token_id)  # todo (unknown tokens?)

    def encode(self, data: List[str]) -> torch.Tensor:
        # if not self.is_initialized:
        #     raise ValueError("Must create token ids first")
        assert isinstance(data, list), f"Data must be a list of string tokens, instead got type {type(data)}"
        output = torch.zeros(len(data), dtype=torch.long)
        for i in range(len(data)):
            val = self._encode_val(data[i])
            output[i] = val
        return output

    def _decode_val(self, data: int) -> str:
        return self.id_to_token.get(data, self.unk_token)

    def decode(self, data: torch.Tensor) -> List[str]:
        # if not self.is_initialized:
        #     raise ValueError("Must create token ids first")
        output = []
        for i in data:
            val = self._decode_val(i)
            output.append(val)
        return output

    def add_special_tokens(self):
        for key, val in self.special_tokens_dict.items():
            #  todo make preordered?
            self.add_token(val)
        self.update_special_token_ids()

    def add_all_tokens(self, dataset: Set[str]):
        for val in dataset:
            self.add_token(val)

    def add_token(self, val: str) -> int:
        if val in self.vocab:
            return self._encode_val(val)
        self.vocab.add(val)
        self.id_to_token[self.total_tokens] = val
        self.token_to_id[val] = self.total_tokens
        self.total_tokens += 1
        return self.total_tokens - 1

    def add_special_token(self, st: str) -> int:
        self.special_tokens_dict[st] = st
        return self.add_token(st)

    def update_special_token_ids(self):
        # self.bos_token_id = self.encode([self.special_tokens_dict['bos_token']]).item()
        # self.eos_token_id = self.encode([self.special_tokens_dict['eos_token']]).item()
        for key, value in self.special_tokens_dict.items():
            setattr(self, key, value)
            setattr(self, f"{key}_id", self._encode_val(value))

    def get_metrics(self) -> Dict:
        output = {}
        output["is_initialized"] = self.is_initialized
        output["vocab_size"] = len(self.vocab)
        output["Tokenizer type"] = self.tokenizer_type

        return output

    def save(self, file_name: str, tokenizer_dir: str):
        output = {}
        # output["vocab"] = list(self.vocab)
        # output["id_to_token"] = self.id_to_token
        output["token_to_id"] = self.token_to_id
        output["buckets"] = self.buckets
        output["special_tokens"] = self.special_tokens_dict
        output["is_initialized"] = self.is_initialized

        path = data_utils.save_json(output, tokenizer_dir, f"{file_name}.json")
        log.info(f"Saved tokenizer to {path}")

    def load(self, data: Dict):
        try:
            # self.vocab = set(data["vocab"])
            # self.id_to_token = data["id_to_token"]
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {value: key for key, value in self.token_to_id.items()}
            self.vocab = set(self.token_to_id.keys())
            self.total_tokens = len(self.vocab)
            self.buckets = data["buckets"]
            self.special_tokens_dict = data["special_tokens"]
            self.is_initialized = data["is_initialized"]
        except KeyError as e:
            log.error(
                f"Error loading tokenizer because JSON file didn't include one of the expected tokenizer fields: {e}"
            )
            raise e
        self.update_special_token_ids()

    def load_vocab_from_file(self, file_name: str, tokenizer_dir: str):
        log.info(f"Loading tokenizer from {file_name}.json")
        data = data_utils.read_json(tokenizer_dir, f"{file_name}.json")
        self.load(data)

    def tokenize(self, dataset):
        raise NotImplementedError()
