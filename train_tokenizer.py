import logging
import time
from typing import Dict
import datasets
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex, processors
from transformers import PreTrainedTokenizerFast
import os
import hydra
from hydra.utils import get_original_cwd

import event_prediction
from event_prediction import utils, data_utils, get_data_processor, data_preparation

log = logging.getLogger(__name__)


def main_process_data(cfg, setup=None) -> Dict:
    log.info(f"------------- STARTING TOKENIZER TRAINING -------------")
    initial_time = time.time()

    log.info(f"------------- GET DATA -------------")
    section_timer = time.time()
    log.info(f"Retrieving dataset from: {cfg.data}")
    dataset = data_utils.get_data_from_raw(cfg.data, cfg.data_dir)
    data_processor = get_data_processor(cfg.data)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET LOAD COMPLETE -------------")

    log.info(f"------------- PROCESS DATASET -------------")
    section_timer = time.time()
    dataset, col_to_id_dict = data_preparation.preprocess_dataset(dataset, data_processor, cfg.tokenizer.numeric_bucket_amount)
    log.info(f"DATASET converting to huggingface")
    dataset = data_preparation.convert_to_huggingface(dataset, data_processor)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET PROCESSED -------------")

    log.info(f"------------- TRAINING TOKENIZER -------------")
    section_timer = time.time()
    unk_token = "[UNK]"
    tokenizer = Tokenizer(models.WordLevel(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.WordLevelTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=['[PAD]',
                        '[BOS]',
                        '[EOS]',
                        '[MASK]',
                        unk_token,
                        # '[ROW]'
                        ]
        # special_tokens=list(set(special_token_args.values()))
    )

    def data_generator(batch_size=1024):

        len_dataset = len(dataset)
        for i in range(0, len_dataset, batch_size):
            rows = dataset[i: i + batch_size]
            yield rows['text']

    tokenizer.train_from_iterator(data_generator(), trainer=trainer, length=len(dataset))

    # # todo does this work?
    # processed_data_dir = os.path.join(cfg.processed_data_dir, cfg.data.name)
    # tokenized_data.save_to_disk(processed_data_dir)

    # Wrap into fast codebase
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # model_max_length=cfg.data.seq_length,
        # **special_token_args,
    )

    wrapped_tokenizer.add_special_tokens(
        {'pad_token': '[PAD]',
         'unk_token': '[UNK]',
         'bos_token': '[BOS]',
         'eos_token': '[EOS]',
         'mask_token': '[MASK]',
         # 'eos_token': '[ROW]'
         }
    )
    log.info(f"Vocab_size: {tokenizer.get_vocab_size()}")

    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- TOKENIZER TRAIN COMPLETE -------------")

    log.info(f"------------- SAVING TOKENIZER -------------")
    section_timer = time.time()
    tok_name = f"{cfg.data.name}_{cfg.tokenizer.name}"
    files1 = wrapped_tokenizer.save_pretrained(tok_name)
    log.info(f"Saved tokenizer to {os.getcwd()}/{tok_name}")
    if cfg.tokenizer_dir is not None:
        tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tok_name)
        files2 = wrapped_tokenizer.save_pretrained(tokenizer_path)
        log.info(f"Also saved tokenizer to {tokenizer_path}")

    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- TOKENIZER TRAIN COMPLETE -------------")

    return {}


@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="process-data")


if __name__ == "__main__":
    launch()

