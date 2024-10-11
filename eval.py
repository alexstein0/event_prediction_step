import logging

import hydra
from hydra.utils import get_original_cwd
import torch
import event_prediction
from event_prediction import utils, data_preparation, ModelTrainerInterface, get_data_processor
import os
from typing import Dict
import time

from datasets import DatasetDict

log = logging.getLogger(__name__)


def main_eval(cfg, setup=None) -> Dict:
    log.info(f"------------- STARTING EVAL -------------")
    initial_time = time.time()
    metrics = {}

    # GET TOKENIZER
    log.info(f"------------- GETTING TOKENIZER -------------")
    section_timer = time.time()
    tokenized_name = cfg.tokenizer_name
    tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    log.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = event_prediction.load_tokenizer(tokenizer_path)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- TOKENIZER LOAD COMPLETE -------------")

    log.info(f"------------- GET DATA -------------")
    section_timer = time.time()
    data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
    if cfg.tokenized_data_name is None:
        log.info("No tokenized data name provided, exiting")
        return {}
    filepath = os.path.join(data_dir, cfg.tokenized_data_name)
    log.info(f"Loading data from: {filepath}")
    tokenized_data = event_prediction.get_data(filepath)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET LOAD COMPLETE -------------")

    # PROCESS DATA
    log.info(f"------------- PROCESS TOKENIZED DATASET -------------")
    section_timer = time.time()
    tokenized_data = data_preparation.split_data_by_column(tokenized_data, "User")
    prev_static_info = data_preparation.load_static_info(os.path.join(get_original_cwd(), cfg.model_dir), cfg.model_save_name)

    test_user_list = prev_static_info["test_ids"]

    if cfg.dryrun:
        test_user_list = tokenized_data.keys()

    tokenized_data = DatasetDict({key: tokenized_data[key] for key in test_user_list})
    dataloaders = {"test": data_preparation.prepare_validation_dataloader(tokenized_data, tokenizer, cfg)}
    val_loader = dataloaders["test"]

    log.info(f"Num val users: {len(test_user_list)}, Num val loader batches: {len(val_loader)}")
    log.info(f"Number of Columns: {val_loader.dataset.num_columns}")
    log.info(f"Columns: {get_data_processor(cfg.data).get_data_cols()}")
    log.info(f"Ordered: {'true' if not cfg.model.randomize_order else 'random'}")
    sample_size = val_loader.dataset.data.shape[1]
    log.info(f"Sequence length (rows): {cfg.model.seq_length}")
    log.info(f"Elements per sample (columns * sequence length): {sample_size}")
    log.info(f"Batch Size: {val_loader.batch_size}")
    log.info(f"Total tokens in dataloaders (n_batches * batch_sz * seq_length * columns): "
             f"{len(val_loader) * val_loader.batch_size * cfg.model.seq_length * sample_size:,}")
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET PROCESSED -------------")

    log.info(f"------------- INITIALIZING MODEL -------------")
    section_timer = time.time()
    classification = True
    if classification:
        # todo add collator
        model_interface = ModelTrainerInterface(cfg, tokenizer, dataloaders, prev_static_info=prev_static_info, train_eval=False)

    else:
        # todo this doesnt work
        data_collator = datacollator.TransDataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mask_prob
        )
        model_interface = ModelTrainerInterface(cfg.model, tokenizer, dataloaders, num_cols=num_cols, label_ids=label_ids, train_eval=False)

    model = model_interface.model
    model_size = sum(t.numel() for t in model.parameters())
    vocab_size = len(tokenizer.vocab)
    log.info(f"Model Name: {model.name_or_path}")
    log.info(f"Model size: {model_size/1000**2:.1f}M parameters")
    log.info(f"Vocab size: {vocab_size}")
    label_col = model_interface.loc_to_col.get(model_interface.label_col_position)
    if label_col is None:
        label_col = model_interface.loc_to_col[max(model_interface.loc_to_col.keys())]
    log.info(f"Label Column is: {label_col}")
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- LOADING COMPLETE -------------")
    log.info(f"------------- BEGIN EVAL -------------")
    section_timer = time.time()
    eval_metrics = model_interface.test()

    metrics.update(eval_metrics)
    for k, v in model_interface.prev_static_info.items():
        metrics[f"{k}"] = v
    for k, v in model_interface.get_static_info().items():
        metrics[f"eval_{k}"] = v

    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- EVAL COMPLETE -------------")

    return metrics


@hydra.main(config_path="event_prediction/config", config_name="eval", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_eval, job_name="eval")


if __name__ == "__main__":
    launch()
