import hydra
import event_prediction
from event_prediction import get_data_processor, data_preparation, data_utils
import logging
from typing import Dict
import os
from hydra.utils import get_original_cwd


log = logging.getLogger(__name__)


def tokenize_dataset(cfg, setup=None) -> Dict:
    log.info(f"STARTING TRAINING")

    log.info(f"GET TOKENIZER")
    tokenized_name = cfg.tokenizer_name
    tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    log.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = event_prediction.load_tokenizer(tokenizer_path)
    log.info(f"TOKENIZER LOAD COMPLETE")

    log.info(f"GET DATA")
    data_processor = get_data_processor(cfg.data)
    dataset = data_utils.get_data_from_raw(cfg.data, cfg.data_dir)
    tokenized_data, col_to_id_dict = data_preparation.process_data_and_tokenize(dataset, cfg, data_processor, tokenizer)
    # tokenized_data = data_preparation.split_data_by_column(tokenized_data, "User")
    log.info(f"DATASET SPLIT")

    data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
    save_name = cfg.data.name
    if cfg.data.save_name is not None:
        save_name = cfg.data.save_name
    filepath = os.path.join(data_dir, save_name)
    data_preparation.save_dataset(tokenized_data, filepath)
    # tokenized_data.save_to_disk(filepath, num_proc=threads)
    log.info(f"DATASET SAVED to {filepath}")
    log.info(f"Index cols: {data_processor.get_index_columns()}")
    log.info(f"Label cols: {data_processor.get_label_columns()}")
    log.info(f"Num cols: {len(data_processor.get_data_cols())} | {data_processor.get_data_cols()}")
    log.info("Column keys: " + " ".join([f"{k}: {v}" for k, v in col_to_id_dict.items()]))

    log.info(f"DATASET SAVE COMPLETE")
    return col_to_id_dict



@hydra.main(
    config_path="event_prediction/config",
    config_name="pretrain",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, tokenize_dataset, job_name="tokenize_dataset")


if __name__ == "__main__":
    launch()
