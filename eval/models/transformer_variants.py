"""Transformer encoder-decoder variants for trajectory prediction.

Two configurations:
  Transformer-3L: 3 encoder + 3 decoder layers, d_model=128, nhead=4
  Transformer-6L: 6 encoder + 6 decoder layers, d_model=256, nhead=8

Training uses teacher forcing; inference uses non-autoregressive decoding
(last history position repeated as decoder query) to avoid exposure bias.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


HIST_DIM  = 6
FUT_STEPS = 30
OUT_DIM   = 2


class _SinePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _TransformerBase(nn.Module):
    def __init__(self, d_model, nhead, n_enc, n_dec, ffn_dim, dropout):
        super().__init__()
        self.src_embed  = nn.Linear(HIST_DIM, d_model)
        self.tgt_embed  = nn.Linear(OUT_DIM,  d_model)
        self.src_pe     = _SinePE(d_model)
        self.tgt_pe     = _SinePE(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, OUT_DIM)
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, hist, target=None, tf_ratio=0.5):
        src_e = self.src_pe(self.src_embed(hist))
        # Teacher-forcing: use shifted target as decoder input
        if target is not None:
            tgt_in = torch.cat([hist[:, -1:, :2], target[:, :-1, :]], dim=1)
        else:
            # Non-AR inference: last position repeated
            tgt_in = hist[:, -1:, :2].expand(-1, FUT_STEPS, -1).contiguous()
        tgt_e  = self.tgt_pe(self.tgt_embed(tgt_in))
        T = tgt_in.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=hist.device)
        out    = self.transformer(src_e, tgt_e, tgt_mask=causal, tgt_is_causal=True)
        return self.out_proj(out)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None)


class Transformer3L(_TransformerBase):
    name = "Transformer-3L"

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__(d_model, nhead, 3, 3, ffn_dim, dropout)


class Transformer6L(_TransformerBase):
    name = "Transformer-6L"

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__(d_model, nhead, 6, 6, ffn_dim, dropout)
