"""Initialize event_predictions"""

from event_prediction import utils, tokenizer_utils
from event_prediction.data import data_utils, get_data_processor, data_preparation
from event_prediction.models import model_utils, trainer_utils, ModelTrainerInterface
from event_prediction.tokenizers import get_tokenizer, GenericTokenizer

from transformers import AutoTokenizer
import datasets

__all__ = [
    "utils",
    "data_utils",
    "tokenizer_utils",
    "model_utils",
    "trainer_utils",
    "get_data_processor",
    "get_tokenizer",
    "ModelTrainerInterface"
]

# def load_tokenizer_and_data(tokenizer_dir, data_dir, tokenizer_cfg, data_cfg) -> (GenericTokenizer, str):
#
#     tokenizer = get_tokenizer(tokenizer_cfg, data_cfg)
#     tokenizer.eval()
#     tokenizer_data = data_utils.read_json(tokenizer_dir, data_cfg.name)
#     tokenizer.load(tokenizer_data)
#     dataset = data_utils.load_dataset(data_cfg.name, data_dir)
#     return tokenizer, dataset


def load_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)


def get_data(filepath: str):
    return datasets.load_from_disk(filepath)
# /cmlscratch/astein0/event_prediction/tokenizer/ibm_fraud_transaction_small_simple
# /cmlscratch/astein0/event_prediction/tokenizer/ibm_fraud_transaction_small_simple