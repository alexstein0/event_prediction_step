import logging
import os
import datetime
from typing import Dict, Tuple, Any, List

from .model_utils import get_model

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, f1_score, binary_f1_score, multiclass_f1_score

from event_prediction import tokenizer_utils, utils, get_data_processor, data_preparation

import time
import json

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


from transformers import DataCollatorForLanguageModeling, AutoTokenizer


log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTrainerInterface:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        data_loaders: Dict[str, DataLoader],
        train_eval: bool = True,
        prev_static_info: Dict = None,
        setup=None
    ):

        # data processing configs
        data_processor = get_data_processor(cfg.data)
        classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True)
        self.num_cols = classification_info["num_cols"]
        self.label_ids = classification_info["label_ids"]
        self.tokenizer = tokenizer
        self.train_eval = train_eval
        self.evaluate_all_positions = cfg.model.evaluate_all_positions

        _, self.col_to_loc, self.loc_to_col = data_preparation.get_col_to_id_dict(data_processor.get_data_cols(), dataset=None)
        consolidation_map = cfg.data.consolidate_columns
        self.consolidation_map = {}
        # if consolidation_map is not None and cfg.consolidate:
        #     assert len(consolidation_map) == 1, "only works for target now"
        #     for col_name, col_mapping in consolidation_map.items():
        #         assert len(col_mapping) == len(self.label_ids), "consolidation maps must have the same number of columns"
        #         for label_value, label_id in col_mapping.items():
        #             self.consolidation_map[self.label_ids[label_value]] = label_id

        if consolidation_map is not None and cfg.consolidate:
            for col_name, col_mapping in consolidation_map.items():
                col_label_map = self.label_ids[str(self.col_to_loc[col_name])]
                # todo only works if label_ids correspond to position in the table
                assert len(col_mapping) == len(col_label_map), "consolidation maps must have the same number of columns"
                temp_map = {}
                for label_value, label_id in col_mapping.items():
                    temp_map[col_label_map[label_value]] = label_id
                self.consolidation_map[self.col_to_loc[col_name]] = temp_map

        # DATA
        self.train_loader = data_loaders.get("train", None)
        if self.train_loader is None:
            log.info("No training data loaded!")

        self.valid_loader = data_loaders.get("test", None)
        if self.valid_loader is None:
            log.info("No test data loaded!")

        try:
            self.label_col_position = self.valid_loader.dataset.label_col_position
        except:
            log.info("No label column position provided!")
            self.label_col_position = self.num_cols - 1
        try:
            self.label_moved_to_position = self.valid_loader.dataset.label_moved_to_position
        except:
            log.info("No label column position provided!")
            self.label_moved_to_position = self.num_cols - 1

        self.epochs = cfg.model.epochs
        self.model_context_length = self.num_cols*cfg.model.seq_length
        self.randomize_order = cfg.model.randomize_order
        self.metric_calc_mode = cfg.model.metric_calc_mode

        self.suffix_options = ["", "_consolidated", "_last", "_last_consolidated"]
        if cfg.model.track_preds:
            # setting to none will track everything possible
            self.tracked_model_output_training = None
            self.tracked_model_output_eval = None
        else:
            self.tracked_model_output_training = ["rows", "loss", "loss_sequence", "accuracy"]
            self.tracked_model_output_eval = ["rows", "loss", "loss_sequence", "accuracy", "target_inds", "target_probs"]

        # todo is there a cleaner way of doing these?
        cfg.model.context_length = self.model_context_length
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if cfg.model_save_name is not None:
            self.model_save_name = cfg.model_save_name
        else:
            self.model_save_name = f"{cfg.name}-{cfg.seed}-{timestamp}"
        self.checkpoint_save_name = cfg.checkpoint_save_name
        log.info(f"Save name: {self.model_save_name}")
        which_checkpoints = cfg.impl.which_checkpoints
        self.save_best = which_checkpoints == "best" or which_checkpoints == "all" or which_checkpoints == "important"
        self.save_latest = which_checkpoints == "latest" or which_checkpoints == "all" or which_checkpoints == "important"
        self.save_all = which_checkpoints == "all"

        self.setup = {
            "device": device,
            "wandb_enabled": cfg.wandb.enabled,
        }
        if setup is not None:
            self.setup = setup

        # Load Model
        model = get_model(
            cfg.model, tokenizer
        )
        checkpoint = {}
        if cfg.model_save_name is not None:
            # load from .ckpt
            path = os.path.join(get_original_cwd(), cfg.model_dir, cfg.model_save_name, cfg.checkpoint_name)
            checkpoint = self.get_checkpoint(path)
            self.model = self.load_model_checkpoint(model, checkpoint)
            # self.model = checkpoint
            self.prev_static_info = prev_static_info
            log.info(f"Checkpoint loaded from: {path} {f'with static data' if prev_static_info is not None else ''}")
        else:
            # not loaded from checkpoint
            log.info(f"No checkpoint loaded")
            self.model = model
            self.prev_static_info = {}

        self.model.to(device)

        if train_eval:
            # Optimization params
            if cfg.model.optim == "AdamW":
                lr = cfg.model.lr
                self.optim = AdamW(self.model.parameters(), lr=lr)
            else:
                log.warning(f"Expected 'AdamW' but got {cfg.model.optim}, cannot train")
                self.optim = None
                raise NotImplementedError()

            try:
                self.optim.load_state_dict(checkpoint["optim_state"])
            except:
                log.info("Initialized Optimizer from scratch")

            # # TODO scheduler doesnt do anything yet
            # try:
            #     self.lr_scheduler = transformers.get_scheduler(
            #         name=cfg.model.lr_scheduler,
            #         optimizer=self.optim,
            #         num_warmup_steps=cfg.model.warmup_steps,
            #         num_training_steps=len(data_loaders.get("train")) * cfg.model.epochs,
            #     )
            #
            #     try:
            #         self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            #     except:
            #         log.info("Initialized Scheduler from scratch")
            #
            # except ValueError as e:
            #     self.lr_scheduler = None
            #     log.warning(f"Expected 'linear' or 'cosine' but got {cfg.model.lr_scheduler} so cannot train. {e}")
            #     raise NotImplementedError()
            #
            # except TypeError as e:
            #     self.lr_scheduler = None
            #     log.warning(f"Cannot init lr_scheduler {e}")
            #     raise NotImplementedError()

        self.steps = 0
        self.epoch = 0
        self.best_auc = 0
        # self.best_loss = float("inf")
        self.cfg = cfg
        self.static_info = {}
        self.static_info["tags"] = list(self.cfg.wandb.tags)
        self.static_info["name"] = self.cfg.name
        self.static_info["saved_name"] = self.model_save_name
        self.static_info["data_name"] = self.cfg.data.name
        self.static_info["lr"] = self.cfg.model.lr
        self.static_info["batch_size"] = self.cfg.model.batch_size
        self.static_info["seq_length"] = self.cfg.model.seq_length
        self.static_info["epochs"] = self.epochs
        self.static_info["num_cols"] = self.num_cols
        self.static_info["seed"] = self.cfg.seed
        self.static_info["label_column"] = self.loc_to_col.get(self.label_col_position)
        self.static_info["label_column_position"] = self.label_col_position
        self.static_info["label_moved_to_position"] = self.label_moved_to_position
        self.static_info["randomize_order"] = self.cfg.model.randomize_order
        self.static_info["mask_all_pct"] = self.cfg.model.percent_mask_all_labels_in_input
        self.static_info["mask_each_pct"] = self.cfg.model.percent_mask_labels_in_input
        self.static_info["consolidated"] = len(self.consolidation_map) > 0
        self.static_info["experiment_folder_name"] = self.cfg.experiment_folder_name
        self.prev_static_info = {f'train_{k}': v for k, v in self.prev_static_info.items() if k in self.static_info}

    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        log.info(f"Running training for {self.epochs} epochs")
        stats = dict()
        training_stats = dict()
        validation_stats = dict()
        start_time = time.time()
        last_print_time = start_time
        eval_time = 0
        last_log_step = -1

        for epoch in range(self.epochs):

            self.epoch = epoch
            self.train_loader.dataset.set_epoch = epoch
            self.valid_loader.dataset.set_epoch = epoch
            epoch_start_time = time.time()
            epoch_start_step = self.steps
            for data_idx, batch in enumerate(self.train_loader):
                self.optim.zero_grad()

                model_output = self.train_loop(data_idx, batch)
                stats = self.add_model_output_stats(model_output, stats, self.tracked_model_output_training)
                if (self.steps + 1) % self.cfg.impl.print_loss_every_nth_step == 0:
                    training_stats = self.gather_stats(stats)
                    elapsed_times = utils.get_time_deltas(start_time, last_print_time, set_format=False)

                    training_stats["total_elapsed_time"] = elapsed_times[0]
                    training_stats["step_avg_time"] = (elapsed_times[1] - eval_time) / (self.steps - last_log_step)
                    # note that average time subtracts eval time but not saved time
                    self.log(f"Train Step: {self.steps + 1}", training_stats, is_training=True)

                    last_log_step = self.steps
                    last_print_time = time.time()
                    eval_time = 0

                if (self.steps + 1) % self.cfg.impl.run_eval_every_nth_step == 0:
                    validation_stats, eval_time = self.run_validation(start_time, eval_time)

                if (self.cfg.impl.save_every_nth_step > 0 and
                        (self.steps + 1) % self.cfg.impl.save_every_nth_step == 0 and
                        self.steps != 0 and self.cfg.impl.save_intermediate_checkpoints):

                    if len(validation_stats) == 0:
                        # havent run eval in this iteration yet
                        validation_stats, eval_time = self.run_validation(start_time, eval_time)

                    if self.save_all:
                        if self.checkpoint_save_name is not None:
                            name = self.checkpoint_save_name
                        else:
                            name = f"checkpoint_step_{self.steps + 1}"
                        model_path = self.save_model(name)
                        log.info(f"Saving to {model_path}")
                    elif self.save_latest:
                        model_path = self.save_model("latest")
                        log.info(f"Saving to {model_path}")
                    if self.save_best and validation_stats["auc"] > self.best_auc:
                        model_path = self.save_model("best")
                        log.info(f"Saving to {model_path}")
                        self.best_auc = validation_stats["auc"]

                self.steps += 1

                # todo
                # dont calculate all training stats
                # learning rate tests for training

            training_stats, stats = self.reset_intermediate_stats(stats)
            eval_start_time = time.time()
            validation_stats = self.validate()
            elapsed_times = utils.get_time_deltas(start_time, eval_start_time, epoch_start_time, set_format=False)
            total_time = elapsed_times[0]
            eval_time += elapsed_times[1]
            epoch_time = elapsed_times[2]
            training_stats["step_avg_time"] = (elapsed_times[2] - eval_time) / (self.steps - epoch_start_step)
            eval_time = 0

            training_stats["total_elapsed_time"] = total_time
            training_stats["epoch_time"] = epoch_time

            validation_stats["total_elapsed_time"] = total_time
            validation_stats["epoch_time"] = epoch_time

            self.log(f"Training for epoch: {epoch}", training_stats, is_training=True)
            self.log(f"Eval for epoch: {epoch}", validation_stats, is_training=False)
            if self.cfg.impl.save_intermediate_checkpoints:
                if self.save_all:
                    name = f"checkpoint_end_{epoch}"
                    model_path = self.save_model(name)
                    log.info(f"Saving to {model_path}")
                elif self.save_latest:
                    model_path = self.save_model("latest")
                    log.info(f"Saving to {model_path}")
                if self.save_best and validation_stats["auc"] > self.best_auc:
                    model_path = self.save_model("best")
                    log.info(f"Saving to {model_path}")
                    self.best_auc = validation_stats["auc"]

        # todo add eval every once in a while
        # log to wandb
        # checkpoint model
        # track number of steps and ensure that scheduling is working correctly
        return validation_stats, training_stats

    def train_loop(self, idx, batch):
        self.model.train()
        # todo loop over devices/minibatches

        if "mask" not in batch:
            batch["mask"] = (batch["input_ids"] != 0).int()

        device_batch = self.to_device(batch, keys=["input_ids", "mask", "labels"])
        # loss_vals = []
        # log_ppls = []
        # stream_depth = device_batch["input_ids"].shape[1]
        # model_outputs = self.forward(device_batch, **model_outputs)

        # input_ids = device_batch["input_ids"]
        # targets = device_batch["targets"]

        outputs = self.forward(device_batch)
        loss = outputs["loss"]
        self.backward(loss)
        self.optimizer_step()
        metrics = self.calc_metrics_for_batch(device_batch, outputs)
        return metrics

    def optimizer_step(self):
        self.optim.step()

    def test(self):
        stats = self.validate()  # todo, is this different than validation?
        self.log(f"EVALUATION", stats, is_training=False)
        return stats

    def run_validation(self, start_time, eval_time):
        # Todo change the param, change the data
        eval_start_time = time.time()
        validation_stats = self.validate()
        elapsed_times = utils.get_time_deltas(start_time, eval_start_time, set_format=False)
        validation_stats["total_elapsed_time"] = elapsed_times[0]
        eval_time += elapsed_times[1]  # time elapsed for eval (subtracted from training time)
        validation_stats["eval_time"] = eval_time
        self.log(f"Eval step: {self.steps + 1}", validation_stats, is_training=False)
        return validation_stats, eval_time

    def validate(self):
        self.model.eval()
        stats = {}
        for data_idx, batch in enumerate(self.valid_loader):
            model_output = self.validation_loop(data_idx, batch)
            stats = self.add_model_output_stats(model_output, stats, self.tracked_model_output_eval)
        calced_stats = self.gather_stats(stats)
        return calced_stats

    def validation_loop(self, idx, batch):
        device_batch = self.to_device(batch, keys=["input_ids", "mask", "labels"])
        outputs = self.forward_inference(device_batch)
        metrics = self.calc_metrics_for_batch(device_batch, outputs)
        return metrics

    def save_model(self, name: str):
        model_path = self.get_model_path()
        full_path = os.path.join(model_path, name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), full_path)

        static_info = self.get_static_info()
        train_ids = self.train_loader.dataset.user_ids
        val_ids = self.valid_loader.dataset.user_ids
        static_info["train_ids"] = train_ids
        static_info["test_ids"] = val_ids
        utils.save_static_info(static_info, model_path)

        return full_path

    def get_model_path(self):
        return os.path.join(get_original_cwd(), self.cfg.model_dir, self.model_save_name)

    def get_static_info(self):
        return self.static_info.copy()

    def get_checkpoint(self, ckpt_path):
        try:
            return torch.load(ckpt_path)
        except:
            log.warning(f"Cannot load checkpoint {ckpt_path}")
            return {}

    def add_model_output_stats(self, model_outputs: Dict[str, Any],
                               intermediate_stats: Dict[str, Any],
                               stats_list: List[str] = None,  # can pass in list of desired stats
                               ) -> Dict[str, Any]:
        intermediate_stats["batches"] = intermediate_stats.get("batches", 0) + 1
        if stats_list is None:
            stats_list = model_outputs.keys()
        for k, v in model_outputs.items():
            included = False
            for stat in stats_list:
                if stat in k:
                    included = True
                    break
            if not included:
                continue

            prev = intermediate_stats.get(k, [])
            if len(v.shape) == 0:
                prev.append(v.detach().item())
            else:
                prev.append(v)
            intermediate_stats[k] = prev
        return intermediate_stats

    def reset_intermediate_stats(self, stats: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
        output_stats = self.gather_stats(stats)
        stats = {}
        return output_stats, stats

    def gather_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        output_stats = {}
        count = stats.get("batches", 1)
        output_stats["batches"] = count

        rows = sum(stats.get("rows", []))
        assert rows > 0
        output_stats["rows"] = rows
        # todo accuracy over rows of different length?

        if "loss" in stats:
            output_stats["loss"] = sum(stats["loss"]) / count

        acc_overall = []
        for loc, col_nam in self.loc_to_col.items():
            if f"accuracy_{col_nam}" in stats:
                acc = sum([i * (j/rows) for i, j in zip(stats[f"accuracy_{col_nam}"], stats[f"rows"])])
                output_stats[f"accuracy_{col_nam}"] = acc
                acc_overall.append(acc)

        if len(acc_overall) > 0:
            output_stats['accuracy_overall'] = sum(acc_overall) / len(acc_overall)

        for suffix in self.suffix_options:
            output_stats = self.get_eval_metrics(stats, output_stats, suffix)

        output_stats = utils.collect_memory_usage(output_stats, device)
        return output_stats

    def get_eval_metrics(self, stats: Dict[str, Any], output_stats: Dict[str, Any], suffix: str = "") -> Dict[str, Any]:
        if f"accuracy{suffix}" in stats:
            # need weighted average
            total_rows = sum(stats["rows"])
            output_stats[f"accuracy{suffix}"] = sum([i * (j/total_rows) for i, j in zip(stats[f"accuracy{suffix}"], stats[f"rows"])])
        if f"target_inds{suffix}" in stats:
            target_inds = torch.cat(stats[f"target_inds{suffix}"])
        if f"target_probs{suffix}" in stats:
            target_probs = torch.cat(stats[f"target_probs{suffix}"], dim=1)

        # Calculate metrics
        try:
            auc = multiclass_auroc(target_probs.T, target_inds, num_classes=target_probs.shape[0], thresholds=None)
            output_stats[f"auc{suffix}"] = auc.item()
            output_stats[f"f1_micro{suffix}"] = f1_score(target_probs.T, target_inds, num_classes=target_probs.shape[0], task="multiclass", average='micro').item()
            output_stats[f"f1_macro{suffix}"] = f1_score(target_probs.T, target_inds, num_classes=target_probs.shape[0], task="multiclass", average='macro').item()
            output_stats[f"f1_weighted{suffix}"] = f1_score(target_probs.T, target_inds, num_classes=target_probs.shape[0], task="multiclass", average='weighted').item()
            if target_probs.shape[0] == 2:
                output_stats[f"f1_binary{suffix}"] = binary_f1_score(target_probs[1, :], target_inds).item()
        except:
            pass
        return output_stats

    def calc_metrics_for_batch(self, device_batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        if "loss" in model_outputs:
            loss = model_outputs["loss"]
            metrics["loss"] = loss.clone().detach().to("cpu")

        metrics["rows"] = (device_batch["mask"] == self.label_col_position).sum()   # might be wrong, goes based on only one 0 per row

        mask_col = device_batch["mask"][:, 1:] == self.label_col_position
        mask_flattened_shifted, mask_flattened_last_shifted, labels_flattened, logits_flattened = self.shape_mask_labels_logits(mask_col, device_batch["labels"], model_outputs["logits"])
        #
        # # Convert logits over all vocabulary to target probability for use in accuracy
        # label_mask = device_batch["mask"][:, 1:] == self.label_col_position  # mask of where the labels are
        # mask_flattened_shifted = label_mask.reshape(-1)
        #
        # # where should metrics be calculated (like auc only on row labels)
        # last_label_mask = torch.zeros_like(label_mask, device=label_mask.device)
        # last_label_mask[torch.arange(label_mask.size(0)), torch.argmax((label_mask == 1).long().cumsum(dim=1) * label_mask, dim=1)] = 1
        # mask_flattened_last_shifted = last_label_mask.reshape(-1)
        #
        # # todo bos eos
        # shifted_outputs = logits[..., :-1, :].contiguous()
        # logits_flattened = shifted_outputs.view(-1, shifted_outputs.shape[-1])
        # shifted_labels = device_batch["labels"].contiguous()
        # labels_flattened = shifted_labels.view(-1)

        # track AUC stuff
        for suffix in self.suffix_options:
            consolidate = "consolidate" in suffix
            if consolidate and len(self.consolidation_map.get(self.label_col_position, [])) < 2:
                continue
            mask = mask_flattened_last_shifted if "last" in suffix else mask_flattened_shifted

            target_probs, target_inds = self.get_logits_and_probs(logits_flattened, labels_flattened, mask, consolidate=consolidate)
            metrics[f"target_probs{suffix}"] = target_probs.to("cpu")
            metrics[f"target_inds{suffix}"] = target_inds.to("cpu")
            preds = torch.argmax(target_probs, dim=0)  # (b*n/tokens_per_trans)
            accuracy = (preds == target_inds).float().mean()
            metrics[f"accuracy{suffix}"] = accuracy.detach().to("cpu")

        if not self.train_eval and self.evaluate_all_positions:
            # this is a specific bit of functionality to check accuracy on entire sequence
            for col_num, mapping in self.label_ids.items():
                col_num = int(col_num)
                mask_col = device_batch["mask"][:, 1:] == col_num  # mask of where the labels are
                col_mask_flattened_shifted, _, col_labels_flattened, col_logits_flattened \
                    = self.shape_mask_labels_logits(mask_col, device_batch["labels"], model_outputs["logits"])

                consolidate = len(self.consolidation_map.get(col_num, {})) > 1
                col_target_probs, col_target_inds = self.get_logits_and_probs(col_logits_flattened, col_labels_flattened, col_mask_flattened_shifted, column_number=col_num, consolidate=consolidate)
                col_preds = torch.argmax(col_target_probs, dim=0)  # (b*n/tokens_per_trans)
                col_accuracy = (col_preds == col_target_inds).float().mean()
                # metrics["overall correct"] = metrics.get("overall correct", 0) + sum(preds == target_inds)
                # metrics["overall total"] = metrics.get("overall total", 0) + len(preds)
                # metrics["accuracy_overall"] = metrics.get("accuracy_overall", 0) + (accuracy / len(self.label_ids))
                col_name = self.loc_to_col[col_num]
                metrics[f"accuracy_{col_name}"] = metrics.get(f"accuracy_{col_name}", 0) + col_accuracy

        return metrics

    def shape_mask_labels_logits(self, label_mask, labels, logits):
        mask_flattened_shifted = label_mask.reshape(-1)

        # where should metrics be calculated (like auc only on row labels)
        last_label_mask = torch.zeros_like(label_mask, device=label_mask.device)
        last_label_mask[
            torch.arange(label_mask.size(0)), torch.argmax((label_mask == 1).long().cumsum(dim=1) * label_mask,
                                                           dim=1)] = 1
        mask_flattened_last_shifted = last_label_mask.reshape(-1)

        # todo bos eos
        shifted_outputs = logits[..., :-1, :].contiguous()
        logits_flattened = shifted_outputs.view(-1, shifted_outputs.shape[-1])
        shifted_labels = labels.contiguous()
        labels_flattened = shifted_labels.view(-1)
        return mask_flattened_shifted, mask_flattened_last_shifted, labels_flattened, logits_flattened

    def get_logits_and_probs(self, logits_flattened: torch.Tensor,
                             labels_flattened: torch.Tensor,
                             mask_flattened_shifted: torch.Tensor,
                             column_number: int = -1,
                             consolidate: bool = True
                             ) -> Tuple[torch.Tensor, torch.Tensor]:

        if column_number < 0:
            column_number = self.label_col_position

        selected_logits = logits_flattened[mask_flattened_shifted == 1, :]
        selected_targets = labels_flattened[mask_flattened_shifted == 1]

        target_logits = []
        target_inds = torch.zeros_like(selected_targets) - 1
        consolidation_map = []

        label_dict = self.label_ids[str(column_number)]
        for i, (word, word_id) in enumerate(label_dict.items()):
            target_logits.append(selected_logits[:, word_id])
            target_inds[selected_targets == word_id] = i
            consolidation_map.append(self.consolidation_map.get(column_number, {}).get(word_id, -1))
        target_logits = torch.stack(target_logits)
        consolidation_map = torch.tensor(consolidation_map).to(target_inds.device)
        target_probs = F.softmax(target_logits, dim=0)

        if len(self.consolidation_map.get(column_number, {})) > 1 and consolidate:
            target_probs, target_inds = self.consolidate_column_values(consolidation_map, target_probs, target_inds)

        # assert (target_inds >= 0).all(), "all possible logits found"
        total_prob_is_one = target_probs.sum(dim=0).isclose(torch.tensor(1.0), atol=1e-3)

        assert total_prob_is_one.all(), (f"all probs add to 1, "
                                         f"but instead add to {total_prob_is_one.sum()} / {len(total_prob_is_one)} "
                                         f"for col number: {column_number} ({self.loc_to_col[column_number]}) "
                                         f"and consolidate: {consolidate}")
        return target_probs, target_inds

    def consolidate_column_values(self, consolidation_map, logits, inds):
        classes = consolidation_map.unique()
        output_logits = []
        for c in classes:
            output_logits.append(logits[consolidation_map == c].sum(dim=0))

        output_logits = torch.stack(output_logits)
        inds = consolidation_map[inds]
        return output_logits, inds

    def load_model_checkpoint(self, model, checkpoint: str, **kwargs: Any):
        model.load_state_dict(checkpoint)
        return model

    # def step(self, batch: dict[str, torch.Tensor]):
    #     loss = self.forward(**batch)["loss"]
    #     self.backward(loss)
    #     self.optimizer_step()
    #     return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids"]):
        """Move batch of data into device memory."""
        device_batch = {}
        for k, v in batch.items():
            if k in keys:
                v = v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            device_batch[k] = v
        return device_batch

    def forward(self, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"].clone()
        mask = (batch["mask"].clone() == self.label_col_position).long()
        model_outputs = self.model(input_ids=input_ids, target_mask=mask, labels=labels)
        return model_outputs

    @torch.no_grad()
    @torch._dynamo.disable()
    def forward_inference(self, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        mask = (batch["mask"].clone() == self.label_col_position).long()
        model_outputs = self.model(input_ids=input_ids, target_mask=mask)
        return model_outputs

    def backward(self, loss):
        loss.backward()
        # todo microbatching
        # context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        # with context():
        #     return self.scaler.scale(loss / self.accumulation_steps_expected).backward()

    def log(self, message: str, metrics: Dict[str, Any], is_training: bool):
        kw_str = [f"{message:25s}"]
        kw_str.append(f'''{f"batches/rows: {metrics['batches']}/{metrics['rows']}":30s} ''')
        kw_str.append(f'''{f"Total dur: {utils.format_time(metrics['total_elapsed_time'], 2)}":20s} ''' if 'total_elapsed_time' in metrics else "")
        kw_str.append(f'''{f"Epoch dur: {utils.format_time(metrics['epoch_time'], 2)}":20s} ''' if 'epoch_time' in metrics else "")
        kw_str.append(f'''{f"avg/step: {utils.format_time(metrics['step_avg_time'], 2)}":20s} ''' if 'step_avg_time' in metrics else "")
        kw_str.append(f'''{f"loss: {metrics['loss']:7.4f}":12s} ''' if 'loss' in metrics else "")

        kw_str.append(f'''{f"acc_overall: {metrics[f'accuracy_overall']:4.2%}":12s} ''' if f'accuracy_overall' in metrics else "")
        for loc, col_nam in self.loc_to_col.items():
            kw_str.append(f'''{f"acc_{col_nam}: {metrics[f'accuracy_{col_nam}']:4.2%}":12s} ''' if f'accuracy_{col_nam}' in metrics else "")

        # eval metrics
        for suffix in self.suffix_options:
            kw_str.append(f'''{f"acc{suffix}: {metrics[f'accuracy{suffix}']:4.2%}":12s} ''' if f'accuracy{suffix}' in metrics else "")
            kw_str.append(f'''{f"auc{suffix}: {metrics[f'auc{suffix}']:7.4f}":12s} ''' if f'auc{suffix}' in metrics else "")
            # kw_str.append(f'''{f"f1{suffix}: {metrics[f'f1_{suffix}']:7.4f}":12s} ''' if 'f1{suffix}' in metrics else "")
            kw_str.append(f'''{f"f1_micro{suffix}: {metrics[f'f1_micro{suffix}']:7.4f}":12s} ''' if f'f1_micro{suffix}' in metrics else "")
            kw_str.append(f'''{f"f1_macro{suffix}: {metrics[f'f1_macro{suffix}']:7.4f}":12s} ''' if f'f1_macro{suffix}' in metrics else "")
            kw_str.append(f'''{f"f1_weighted{suffix}: {metrics[f'f1_weighted{suffix}']:7.4f}":12s} ''' if f'f1_weighted{suffix}' in metrics else "")
            kw_str.append(f'''{f"f1_binary{suffix}: {metrics[f'f1_binary{suffix}']:7.4f}":12s} ''' if f'f1_binary{suffix}' in metrics else "")

        # kw_str.append(f'''{f"acc: {metrics['accuracy']:4.2%}":12s} ''' if 'accuracy' in metrics else "")
        # kw_str.append(f'''{f"acc2: {metrics['accuracy2']:4.2%}":12s} ''' if 'accuracy2' in metrics else "")
        # kw_str.append(f'''{f"auc: {metrics['auc']:7.4f}":12s} ''' if 'auc' in metrics else "")
        # kw_str.append(f'''{f"f1: {metrics['f1']:7.4f}":12s} ''' if 'f1' in metrics else "")
        # kw_str.append(f'''{f"f1_micro: {metrics['f1_micro']:7.4f}":12s} ''' if 'f1_micro' in metrics else "")
        # kw_str.append(f'''{f"f1_macro: {metrics['f1_macro']:7.4f}":12s} ''' if 'f1_macro' in metrics else "")
        # kw_str.append(f'''{f"f1_weighted: {metrics['f1_weighted']:7.4f}":12s} ''' if 'f1_weighted' in metrics else "")
        # kw_str.append(f'''{f"f1_binary: {metrics['f1_binary']:7.4f}":12s} ''' if 'f1_binary' in metrics else "")
        #
        # kw_str.append(f'''{f"acc last: {metrics['accuracy_last']:4.2%}":12s} ''' if 'accuracy_last' in metrics else "")
        # kw_str.append(f'''{f"acc2 last: {metrics['accuracy_last2']:4.2%}":12s} ''' if 'accuracy_last2' in metrics else "")
        # kw_str.append(f'''{f"auc last: {metrics['auc_last']:7.4f}":12s} ''' if 'auc_last' in metrics else "")
        # kw_str.append(f'''{f"f1 micro last: {metrics['f1_last_micro']:7.4f}":12s} ''' if 'f1_last_micro' in metrics else "")
        # kw_str.append(f'''{f"f1 macro last: {metrics['f1_last_macro']:7.4f}":12s} ''' if 'f1_last_macro' in metrics else "")
        # kw_str.append(f'''{f"f1 weighted last: {metrics['f1_last_weighted']:7.4f}":12s} ''' if 'f1_last_weighted' in metrics else "")
        # kw_str.append(f'''{f"f1 binary last: {metrics['f1_last_binary']:7.4f}":12s} ''' if 'f1_last_binary' in metrics else "")
        #
        # kw_str.append(f'''{f"acc consol: {metrics['accuracy_consolidated']:4.2%}":12s} ''' if 'accuracy_consolidated' in metrics else "")
        # kw_str.append(f'''{f"acc2 consol: {metrics['accuracy_consolidated2']:4.2%}":12s} ''' if 'accuracy_consolidated2' in metrics else "")
        # kw_str.append(f'''{f"auc consol: {metrics['auc_consolidated']:7.4f}":12s} ''' if 'auc_consolidated' in metrics else "")
        # kw_str.append(f'''{f"f1 micro consol: {metrics['f1_consolidated_micro']:7.4f}":12s} ''' if 'f1_consolidated_micro' in metrics else "")
        # kw_str.append(f'''{f"f1 macro consol: {metrics['f1_consolidated_macro']:7.4f}":12s} ''' if 'f1_consolidated_macro' in metrics else "")
        # kw_str.append(f'''{f"f1 weighted consol: {metrics['f1_consolidated_weighted']:7.4f}":12s} ''' if 'f1_consolidated_weighted' in metrics else "")
        # kw_str.append(f'''{f"f1 binary consol: {metrics['f1_consolidated_binary']:7.4f}":12s} ''' if 'f1_consolidated_binary' in metrics else "")
        #
        # kw_str.append(f'''{f"acc consol last: {metrics['accuracy_last_consolidated']:4.2%}":12s} ''' if 'accuracy_last_consolidated' in metrics else "")
        # kw_str.append(f'''{f"acc2 consol last: {metrics['accuracy_last_consolidated2']:4.2%}":12s} ''' if 'accuracy_last_consolidated2' in metrics else "")
        # kw_str.append(f'''{f"auc consol last: {metrics['auc_last_consolidated']:7.4f}":12s} ''' if 'auc_last_consolidated' in metrics else "")
        # kw_str.append(f'''{f"f1 micro consol last: {metrics['f1_last_consolidated_micro']:7.4f}":12s} ''' if 'f1_last_consolidated_micro' in metrics else "")
        # kw_str.append(f'''{f"f1 macro consol last: {metrics['f1_last_consolidated_macro']:7.4f}":12s} ''' if 'f1_last_consolidated_macro' in metrics else "")
        # kw_str.append(f'''{f"f1 weighted consol last: {metrics['f1_last_consolidated_weighted']:7.4f}":12s} ''' if 'f1_last_consolidated_weighted' in metrics else "")
        # kw_str.append(f'''{f"f1 binary consol last: {metrics['f1_last_consolidated_binary']:7.4f}":12s} ''' if 'f1_last_consolidated_binary' in metrics else "")

        # memory stuff
        if metrics.get("VRAM", 0.0) > 0.0:
            kw_str.append(f'''{f"Mem (VRAM/RAM): {metrics['VRAM']:5.4f}GB/{metrics['RAM']:5.4f}":15s}GB ''')
            # kw_str.append(f'''{f"Mem (USAGE): {metrics['Usage']:5.4f}GB/{metrics['Total']:5.4f}":15s}GB ''')
        else:
            kw_str.append(f'''{f"Mem (RAM): {metrics['RAM']:4.2f}":15s} ''' if 'RAM' in metrics else "")

        char_per_line = 200
        cur_line = ""
        for i in kw_str:
            if len(cur_line) + len(i) > char_per_line:
                log.info(cur_line)
                cur_line = f"    {i}"
            else:
                cur_line += i
        if len(cur_line) > 0:
            log.info(cur_line)

        logged = ["batches", "rows", "epoch_time", "total_elapsed_time", "step_avg_time", "loss", "accuracy", "auc", "f1"]
        prefix = "train_" if is_training else "eval_"
        # wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items() if k in logged}
        wandb_metrics = {}
        for k, v in metrics.items():
            included = False
            for log_name in logged:
                if log_name in k:
                    included = True
                    break
            if included:
                # avoid repeats
                wandb_metrics[f"{prefix}{k}"] = v

        wandb_metrics["epoch"] = self.epoch
        wandb_metrics["step"] = self.steps
        if self.setup.get("wandb_enabled", False):
            utils.wandb_log(wandb_metrics)

        # save static data
        # todo dont need to repeat it each time
        # if not is_training:
        static_info = self.get_static_info()
        wandb_metrics.update(static_info)

        if self.prev_static_info is not None:
            wandb_metrics.update(self.prev_static_info)

        if self.cfg.experiment_folder_name is None:
            csv_path = self.get_model_path()
            table_name = f"{prefix}stats"
        else:
            csv_path = os.path.join(get_original_cwd(), self.cfg.base_dir, self.cfg.experiment_folder_name)
            if "train_saved_name" in wandb_metrics:  # model save name is there so no need to redo
                table_name = f"{prefix}stats"
            else:
                table_name = f"{self.model_save_name}-{prefix}stats"
        utils.save_to_table(csv_path, table_name, self.cfg.dryrun, **wandb_metrics)

        # save static data to single json
        # todo this current will override for each run in the experiment
        # static_info = self.get_static_info()
        # train_ids = self.train_loader.dataset.user_ids
        # val_ids = self.valid_loader.dataset.user_ids
        # static_info["train_ids"] = train_ids
        # static_info["test_ids"] = val_ids
        # utils.save_static_info(static_info, csv_path)

    def _init_distributed(self, model):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.setup["device"]] if self.setup["device"].type == "cuda" else None,
            output_device=self.setup["device"] if self.setup["device"].type == "cuda" else None,
            broadcast_buffers=self.cfg_impl.broadcast_buffers,
            bucket_cap_mb=self.cfg_impl.bucket_cap_mb,
            gradient_as_bucket_view=self.cfg_impl.gradient_as_bucket_view,
            static_graph=self.cfg_impl.static_graph,
        )
        return model
