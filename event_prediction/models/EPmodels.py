import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, AutoConfig, GPT2Model, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from einops import rearrange

import logging

from torchmetrics.classification import BinaryAUROC

metric = BinaryAUROC(thresholds=None, compute_on_cpu=True)

log = logging.getLogger(__name__)


# ideas adapted from fata-trans
class Decoder(nn.Module):
    """Decoder model (can be used with GPT pretrained)"""
    def __init__(self, cfg, tokenizer):
        super().__init__()

        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer.vocab),
            n_ctx=cfg.seq_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if cfg.loss_fn == "CrossEntropyLoss":
            self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        else:
            log.warning(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}, cannot train")
            raise NotImplementedError()

        self.model = GPT2LMHeadModel(config)
        # self.model = GPT2ForSequenceClassification(config)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        model_input_ids = input_ids[:, :-1].contiguous()
        shifted_labels = input_ids[:, 1:].contiguous()

        if kwargs.get("mask", None) is not None:
            # todo
            masks = kwargs["mask"]
            shifted_labels = torch.mul(shifted_labels, masks[..., 1:])

        shifted_labels[shifted_labels == 0] = -100
        # outputs = self.model(input_ids=input_ids, labels=input_ids)  # can do this where the model calculates the loss
        outputs = self.model(input_ids=input_ids)
        shifted_outputs = outputs["logits"][:, :-1, :].contiguous()

        # todo try calc loss only on label col
        loss = self.loss_fn(shifted_outputs.view(-1, shifted_outputs.shape[-1]), shifted_labels.view(-1))
        # loss = outputs["loss"]
        outputs = {k: v for k, v in outputs.items()}
        return outputs, loss


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
