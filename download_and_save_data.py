import hydra
import event_prediction
from event_prediction import data_utils
import logging
from typing import Dict
from hydra.utils import get_original_cwd
import os

log = logging.getLogger(__name__)


def download_data(cfg, setup=None) -> Dict:
    dataset = data_utils.download_and_save_data(cfg.data, cfg.data_dir, True, True)
    log.info(f"Dataset has {len(dataset)} samples with the following columns\n{str(' '.join(list(dataset.columns)))}")

    data_dir = os.path.join(get_original_cwd(), cfg.data_dir)
    path = data_utils.save_small(dataset, data_dir, cfg.data.name)

    log.info(dataset.loc[0])
    return {}


@hydra.main(
    config_path="event_prediction/config",
    config_name="pre_process_data",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, download_data, job_name="download_data")


if __name__ == "__main__":
    launch()
