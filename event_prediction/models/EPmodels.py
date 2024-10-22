import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, AutoConfig, GPT2Model, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import logging
log = logging.getLogger(__name__)


# ideas adapted from fata-trans
class Decoder(GPT2LMHeadModel):
    """Decoder model (can be used with GPT pretrained)"""
    def __init__(self, cfg, tokenizer):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer.vocab),
            n_ctx=cfg.seq_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        super().__init__(config)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_token_id = tokenizer.mask_token_id
        # if tokenizer.mask_token_id is None:
        #     log.warning(f"mask token id {tokenizer.mask_token_id} was not provided")
        #     self.mask_token_id = 0
        self.percent_mask_all_labels_in_input = cfg.percent_mask_all_labels_in_input
        self.percent_mask_labels_in_input = cfg.percent_mask_labels_in_input
        self.sequence_label_type = cfg.sequence_label_type  # 'last', 'all', 'any' or none/labels
        self.epoch_to_switch = cfg.epoch_to_switch
        self.loss_calc_mode = cfg.loss_calc_mode  # all (all tokens) last (last label) or labels (all labels)
        # self.metric_calc_mode = cfg.metric_calc_mode  # last (last label) or labels (all labels)

        self.sequence_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # todo second head for sequence label
        assert (self.bos_token_id is None and self.eos_token_id is None) or (self.bos_token_id is not None and self.eos_token_id is not None), f"Either both none or both not none:{self.bos_token_id}, {self.eos_token_id}"
        if cfg.loss_fn == "CrossEntropyLoss":
            self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        # elif cfg.loss_fn == "RMSE":
        #     def RMSELoss(guess, target):
        #         return torch.sqrt(torch.mean((guess - target) ** 2))
        #     self.loss_fn = RMSELoss
        else:
            log.warning(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}, cannot train")
            raise NotImplementedError()

    def forward(self,
                input_ids=None,
                # past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                use_cache=True,
                target_mask=None
                ):
        mask_mask = target_mask.clone()  # masked candidates
        row_rand = torch.rand(mask_mask.size(0), device=input_ids.device)

        # mask_mask = torch.rand(*input_ids.size(), device=input_ids.device) < self.percent_mask_labels_in_input
        # input_ids[(mask_mask & target_mask) == 1] = self.mask_token_id
        # input_ids[(target_mask.T*mask_mask).T == 1] = self.mask_token_id
        # todo can probably do much faster
        for i in range(input_ids.size(0)):
            # randomly mask all labels in an input row
            if row_rand[i] < self.percent_mask_all_labels_in_input:  # will be masked fully masked
                continue

            elif row_rand[i] < self.percent_mask_all_labels_in_input + self.percent_mask_labels_in_input:  # will mask randomly
                # randomly mask each label in an input row
                mask_candidates_count = mask_mask.sum(1)[i]
                if mask_candidates_count < 2:  # dont want to mask all or none
                    mask_mask[0, :] = 0
                    continue
                mask_count = torch.randint(low=1, high=mask_candidates_count, size=(1,))
                masked_labels = torch.randperm(mask_candidates_count)[:mask_count]
                lab_mask = torch.zeros(mask_candidates_count, device=input_ids.device).long()
                lab_mask[masked_labels] = 1
                mask_mask[i, mask_mask[i] == 1] = lab_mask

            else:  # will not be masked
                mask_mask[i, :] = 0

        input_ids[mask_mask == 1] = self.mask_token_id

        model_outputs = self.transformer(
            input_ids,
            # past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = model_outputs[0]
        logits = self.lm_head(hidden_states)
        # sequence_logits = self.sequence_head(hidden_states[:, -1, :])
        outputs = {
            "logits": logits,
            # "sequence_logits": sequence_logits
        }

        if labels is not None:
            # todo this code is not used but could be if we wanted to add a sequence level loss term
            # if self.sequence_label_type == 'last':
            #     # the last label in a sequence
            #     m = torch.argmax((target_mask == 1).long().cumsum(dim=1) * target_mask, dim=1)
            #     labels_for_sequence = input_ids[torch.arange(input_ids.size(0)), m]
            #     loss = self.loss_fn(sequence_logits.view(-1, sequence_logits.shape[-1]), labels_for_sequence.view(-1))
            #     outputs['sequence_loss'] = loss
            # elif self.sequence_label_type == 'all':
            #     # todo
            #     labels_for_sequence = input_ids * target_mask  # returns all the labels in the sequence
            # elif self.sequence_label_type == 'any':
            #     # todo
            #     pass
            # else:
            #     labels_for_sequence = [-1]  # no label for sequence

            shifted_logits = logits[..., 1:-2, :].contiguous()  # remove the start/end of sequence?
            shifted_labels = labels[..., 1:-1].contiguous()  # remove the start/end of sequence?

            # Which part of the sequence to calculate the loss on, by default it is every token in the sequence
            loss_mask = torch.ones_like(target_mask, device=shifted_labels.device)
            if 0 <= self.epoch_to_switch <= self.epoch:
                if self.loss_calc_mode is None or self.loss_calc_mode == "all":
                    pass
                elif self.loss_calc_mode == "labels":
                    loss_mask = target_mask
                elif self.loss_calc_mode == "last":
                    loss_mask = torch.argmax((target_mask == 1).long().cumsum(dim=1) * target_mask, dim=1)
                elif self.loss_calc_mode == "any":
                    # todo
                    pass
                else:
                    raise

            shifted_labels = torch.mul(shifted_labels, loss_mask[..., 2:-1]).long()
            shifted_labels[shifted_labels == 0] = -100

            loss = self.loss_fn(shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1))
            outputs["loss"] = loss
        return outputs


class RowEncoder(nn.Module):
    """This is my user defined encoder to move a row into embedding space before doing autoregressive step"""
    def __init__(self, n_cols: int, vocab_size: int, hidden_size: int, col_hidden_size: int,
                 nheads: int = 8, nlayers: int = 1):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, col_hidden_size)

        enc = nn.TransformerEncoderLayer(d_model=col_hidden_size, nhead=nheads, dim_feedforward=col_hidden_size)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)

        self.linear = nn.Linear(col_hidden_size * n_cols, hidden_size)
        # self.hidden_size = hidden_size
        # self.col_hidden_size = col_hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(input_ids)
        embeds_shape = list(embedded.size())

        embedded = embedded.view([-1] + embeds_shape[-2:])
        embedded = embedded.permute(1, 0, 2)
        embedded = self.encoder(embedded)
        embedded = embedded.permute(1, 0, 2)
        embedded = embedded.contiguous().view(embeds_shape[0:2] + [-1])

        embedded = self.linear(embedded)

        return embedded


class HierarchicalModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids: torch):
        return self.decoder(self.encoder(input_ids))
