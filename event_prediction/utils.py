"""System utilities."""
"""this code is originally courtesy of Jonas Geiping"""
import socket
import sys

import os
import csv
import yaml
import psutil
import pynvml

import multiprocess  # hf uses this for some reason
import collections

import torch
import torch._inductor.config
import pandas as pd
# import transformers

import json
import random
import numpy as np
import time
import datetime
import tempfile

import logging
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
import transformers

import wandb
from typing import Dict

log = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "0"

def main_launcher(cfg, main_fn, job_name=""):
    """This is boiler-plate code for a launcher."""
    launch_time = time.time()
    # Set definitive random seed:
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # TODO
    # Decide GPU and possibly connect to distributed setup
    setup, kWh_counter = system_startup(cfg)
    # Initialize wanDB
    if cfg.wandb.enabled:
        _initialize_wandb(setup, cfg)
    log.info("--------------------------------------------------------------")
    log.info(f"--------------Launching {job_name} run! ---------------------")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    metrics = main_fn(cfg, setup)
    metrics = collect_system_metrics(cfg, metrics, kWh_counter, setup)

    log.info("-------------------------------------------------------------")
    log.info(f"Finished running job {cfg.name} with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}")
    if is_main_process():
        metrics = flatten(metrics)
        dump_metrics(cfg, metrics)
        # Export to wandb:
        # if cfg.wandb.enabled:
        #     wandb_log(metrics)
        # if cfg.wandb.enabled:
        #     for k, v in metrics.items():
        #         wandb.run.summary[k] = v

        # if torch.cuda.is_available():
        #     max_alloc = f"{torch.cuda.max_memory_allocated(setup['device'])/float(1024**3):,.3f} GB"
        #     max_reserved = f"{torch.cuda.max_memory_reserved(setup['device'])/float(1024**3):,.3f} GB"
        #     log.info(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        #     log.info(f"{metrics['kWh']:.2e} kWh of electricity used for GPU(s) during job.")
    log.info("-----------------Shutdown complete.--------------------------")


def system_startup(cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode.

    Set all required and interesting environment variables.
    """
    torch.backends.cudnn.benchmark = cfg.impl.benchmark
    torch.backends.cuda.enable_flash_sdp(cfg.impl.enable_flash_sdp) if cfg.impl.enable_flash_sdp is not None else 0
    torch.backends.cuda.enable_math_sdp(cfg.impl.enable_math_sdp) if cfg.impl.enable_math_sdp is not None else 0
    torch.backends.cuda.enable_mem_efficient_sdp(cfg.impl.enable_mem_efficient_sdp) if cfg.impl.enable_mem_efficient_sdp is not None else 0
    torch.set_float32_matmul_precision(cfg.impl.matmul_precision)

    if cfg.impl.sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(cfg.impl.sharing_strategy)

    if cfg.impl.tf32_allowed:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    multiprocess.set_start_method("forkserver")
    if cfg.impl.local_staging_dir is not None:
        tmp_path = os.path.join(cfg.impl.local_staging_dir, "tmp")
        os.makedirs(tmp_path, exist_ok=True)
        os.environ["TMPDIR"] = tmp_path
        tempfile.tempdir = None  # Force temporary directory regeneration
    if cfg.impl.enable_huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if cfg.impl.add_env_variables is not None:
        # Note that for any environment variables added here, they have to be able to change behavior at runtime
        # for example, the torchdynamo settings are read at import and cannot be changed at runtime here
        for env_var, string_val in cfg.impl.add_env_variables.items():
            os.environ[str(env_var)] = str(string_val)
        log.info(os.environ)

    # allowed_cpus_available = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))  # covering both affinity and phys.
    # if cfg.impl.is_mac:
    #     allowed_cpus_available = 1  # when running on mac
    # else:
    #     allowed_cpus_available = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))  # covering both affinity and phys.
    allowed_cpus_available = get_cpus()
    # cfg.impl["num_threads"] = allowed_cpus_available

    try:
        ram = psutil.Process().rlimit(psutil.RLIMIT_RSS)[0] / (2**30)
    except:
        log.warning("Cannot find process")
        ram = 0

    # Distributed launch?
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend=cfg.impl.dist_backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run = os.environ.get("TORCHELASTIC_RUN_ID", "unknown")
        threads_per_gpu = max(1, min(allowed_cpus_available // max(1, torch.cuda.device_count()), cfg.impl.threads))
        log.info(
            f"Distributed worker initialized on rank {global_rank} (local rank {local_rank}) "
            f"with {world_size} total processes. OMP Threads set to {threads_per_gpu}. Run ID is {run}."
        )
        log.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    else:
        threads_per_gpu = max(1, min(allowed_cpus_available, cfg.impl.threads))
        global_rank = local_rank = 0

    torch.set_num_threads(threads_per_gpu)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_gpu)
    cfg.impl.local_rank = local_rank

    # datasets will automatically disable tokenizer parallelism when needed:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_RS_NUM_CPUS"] = str(threads_per_gpu)
    max_dataset_memory = f"{psutil.virtual_memory().total // 2 // max(torch.cuda.device_count(), 1)}"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory

    # Construct setup dictionary:
    dtype = getattr(torch, cfg.impl.default_precision)  # :> dont mess this up
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}. CUDA: {torch.version.cuda}.")

        # Populate kwH counter:
        pynvml.nvmlInit()
        miilijoule_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(device.index))
        kWh_counter = dict(initial_value=miilijoule_start * 1e-6 / 3600)  # kilojoule per hour
    else:
        kWh_counter = dict(initial_value=float("NaN"))
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]

    if local_rank == 0:
        log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
        log.info(f"CPUs: {allowed_cpus_available}, GPUs: {torch.cuda.device_count()} (ram: {ram}GB) on {socket.gethostname()}.")

    # 100% reproducibility?
    if cfg.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        if is_main_process():
            log.info(f"Seeding with random seed {cfg.seed} on rank 0.")
        set_random_seed(cfg.seed + 10 * global_rank)

    return setup, kWh_counter


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def num_processes():
    num_procs = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    return num_procs


def find_pretrained_checkpoint(cfg, downstream_classes=None):
    """Load a checkpoint either locally or from the internet."""
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.name, "checkpoints")
    if cfg.eval.checkpoint == "latest":
        # Load the latest local checkpoint
        all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
        checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
        checkpoint_name = max(checkpoint_paths, key=os.path.getmtime)
    elif cfg.eval.checkpoint == "smallest":
        # Load maybe the local checkpoint with smallest loss
        all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
        checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
        checkpoint_losses = [float(path[-5:]) for path in checkpoint_paths]
        checkpoint_name = checkpoint_paths[np.argmin(checkpoint_losses)]
    elif not os.path.isabs(cfg.eval.checkpoint) and not cfg.eval.checkpoint.startswith("hf://"):
        # Look locally for a checkpoint with this name
        checkpoint_name = os.path.join(local_checkpoint_folder, cfg.eval.checkpoint)
    elif cfg.eval.checkpoint.startswith("hf://"):
        # Download this checkpoint directly from huggingface
        model_name = cfg.eval.checkpoint.split("hf://")[1].removesuffix("-untrained")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        cfg_arch = transformers.AutoConfig.from_pretrained(model_name)
        model_file = cfg.eval.checkpoint
        checkpoint_name = None
    else:
        # Look for this name as an absolute path
        checkpoint_name = cfg.eval.checkpoint

    if checkpoint_name is not None:
        # Load these checkpoints locally, might not be a huggingface model
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
        with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
            cfg_arch = OmegaConf.create(json.load(file))  # Could have done pure hydra here, but wanted interop

        # Use merge from default config to build in new arguments
        # with hydra.initialize(config_path="config/arch"):
        # cfg_default = OmegaConf.load(os.path.join(cfg.original_cwd, "cramming/config/arch/bert-base.yaml"))
        # cfg_arch = OmegaConf.merge(cfg_default, cfg_arch)

        # Optionally modify parts of the arch at eval time. This is not guaranteed to be a good idea ...
        # All mismatched parameters will be randomly initialized ...
        if cfg.eval.arch_modifications is not None:
            cfg_arch = OmegaConf.merge(cfg_arch, cfg.eval.arch_modifications)
        model_file = os.path.join(checkpoint_name, "model.safetensors")

        print(cfg_arch)

    log.info(f"Loading from checkpoint {model_file}...")
    return tokenizer, cfg_arch, model_file


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table-{table_name}.csv")
    fieldnames = list(kwargs.keys())
    # Read or write header
    log.info(f"Saving to table: {fname}")
    try:
        df = pd.read_csv(fname)
        # Convert the new_data dictionary to a DataFrame
        new_data_df = pd.DataFrame([kwargs])

        # Concatenate the new row to the existing DataFrame
        updated_df = pd.concat([df, new_data_df], ignore_index=True, sort=False)

    except Exception as e:
        updated_df = pd.DataFrame([kwargs])

    # Write the updated DataFrame back to a CSV file
    updated_df.to_csv(fname, index=False)
    save_to_json(out_dir, table_name, dryrun, **kwargs)


def save_to_json(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"json-{table_name}.json")
    # Read or write header
    log.info(f"Saving to table: {fname}")
    with open(fname, 'a') as file:
        json.dump(kwargs, file)
        file.write('\n')


def save_static_info(static_info: Dict, path: str):
    static_file = os.path.join(path, "static_info.json")
    if not os.path.isfile(static_file):
        with open(static_file, "w") as f:
            json.dump(static_info, f)
        return True
    return False


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.name}"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        metric = " ".join(metric.split("_"))
        try:
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
        if isinstance(val, int):
            log.info(f"{metric:30s} {val:6d}")
        elif isinstance(val, float):
            log.info(f"{metric:30s} {val:6.4f}")
        else:
            log.info(f"{metric:30s} {val}")

    with open(f"{filepath}.yaml", "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)

    with open(f"{filepath}.json", "w") as json_file:
        json.dump(sanitized_metrics, json_file, indent=2)


def _initialize_wandb(setup, cfg):
    if is_main_process():
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        settings = wandb.Settings(start_method="thread")
        settings.update({"git_root": get_original_cwd()})
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=settings,
            name=cfg.name,
            mode="disabled" if cfg.dryrun else None,
            tags=cfg.wandb.tags if len(cfg.wandb.tags) > 0 else None,
            config=config_dict,
        )
        run.summary["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
        run.summary["numGPUs"] = torch.cuda.device_count()


def wandb_log(stats):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=stats["step"] if "step" in stats else None)


def flatten(d, parent_key="", sep="_"):
    """Straight-up from https://stackoverflow.com/a/6027615/3775820."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def collect_memory_usage(metrics, device=None):
    try:
        mem_info = torch.cuda.mem_get_info()
        metrics["Usage"] = mem_info[0] / float(1 << 30)
        metrics["Total"] = mem_info[1] / float(1 << 30)
    except:
        pass

    metrics["VRAM"] = torch.cuda.max_memory_allocated(device) / float(1 << 30)
    metrics["RAM"] = psutil.Process(os.getpid()).memory_info().rss / float(1 << 30)  # / 1024**3
    return metrics


def collect_system_metrics(cfg, metrics, kWh_counter, setup):
    # Finalize some compute metrics:
    metrics["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
    metrics["numGPUs"] = torch.cuda.device_count()
    metrics["hostname"] = socket.gethostname()
    metrics = collect_memory_usage(metrics, setup["device"])
    if torch.cuda.device_count() == 1:
        metrics["kWh"] = get_kWh(kWh_counter, setup)
    else:
        if torch.distributed.is_initialized():
            local_kWh = get_kWh(kWh_counter, setup)
            kWh_comm = torch.as_tensor(local_kWh).cuda() if torch.cuda.is_available() else kWh_comm.float()
            torch.distributed.all_reduce(kWh_comm, torch.distributed.ReduceOp.SUM, async_op=False)
            metrics["kWh"] = kWh_comm.item()
        else:
            metrics["kWh"] = float("NaN")
    return metrics


def get_kWh(kWh_counter, setup):
    miilijoule_final = pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(setup["device"].index))
    kWh_final = miilijoule_final * 1e-6 / 3600  # kilojoule per hour
    kWh = kWh_final - kWh_counter["initial_value"]
    return kWh


def pathfinder(cfg):
    with open_dict(cfg):
        cfg.original_cwd = hydra.utils.get_original_cwd()
        # ugliest way to get the absolute path to output subdir
        if not os.path.isabs(cfg.base_dir):
            base_dir_full_path = os.path.abspath(os.getcwd())
            while os.path.basename(base_dir_full_path) != cfg.base_dir:
                base_dir_full_path = os.path.dirname(base_dir_full_path)
                if base_dir_full_path == "/":
                    raise ValueError("Cannot find base directory.")
            cfg.base_dir = base_dir_full_path

        cfg.impl.path = os.path.expanduser(cfg.impl.path)
        if not os.path.isabs(cfg.impl.path):
            cfg.impl.path = os.path.join(cfg.base_dir, cfg.impl.path)
    return cfg


####### random helper functions
def get_cpus() -> int:
    # Number of threads
    # if 1:
    #     return 1
    try:
        return min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))  # covering both affinity and phys.
    except:
        pass
    try:
        return 1 #os.cpu_count()  # when running on mac
    except:
        return 1


def get_time_deltas(*times, set_format=True):
    now = time.time()
    output = []
    for i in range(len(times)):
        dt = now - times[i]
        if set_format:
            dt = format_time(dt, decimals=3)
        output.append(dt)
    return output


def format_time(seconds: int, decimals=-1) -> str:
    dt = str(datetime.timedelta(seconds=seconds))
    if decimals >= 0:
        splitted = dt.split(".")
        if len(splitted) > 1:
            after = splitted[1][:decimals]
            dt = ".".join([splitted[0], after])
    return dt

