"""
Non-Autoregressive (NAR) Transformer for trajectory prediction.

Why NAR?
  v1 Transformers used auto-regressive decoding at inference but were trained
  with teacher forcing → exposure-bias failure (ADE 838–1414 m).

  NAR predicts all 30 future positions in a SINGLE forward pass using
  30 learnable temporal queries that cross-attend to the encoded history.
  Training = inference: no sequential dependencies, zero exposure bias.

Architecture:
  Encoder  : hist (B,30,6) → TransformerEncoder → memory (B,30,d)
  Decoder  : 30 learned queries (B,30,d)
             → TransformerDecoder (cross-attn to memory)
             → Linear → future (B,30,2)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

HIST_DIM  = 6
FUT_STEPS = 30
OUT_DIM   = 2


class _SinePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerNAR(nn.Module):
    """
    Non-autoregressive trajectory Transformer.

    Parameters match a ~6M-param model comparable to Transformer-6L.
    Training and inference are identical (no sequential decoding).
    """
    name = "Transformer-NAR"

    def __init__(
        self,
        d_model:    int   = 256,
        nhead:      int   = 8,
        enc_layers: int   = 4,
        dec_layers: int   = 4,
        ffn_dim:    int   = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.hist_proj = nn.Linear(HIST_DIM, d_model)
        self.enc_pe    = _SinePE(d_model, max_len=256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # ── Decoder ──────────────────────────────────────────────────────
        # 30 learnable temporal queries — one per future timestep
        self.future_queries = nn.Parameter(torch.randn(FUT_STEPS, d_model) * 0.02)
        self.dec_pe         = _SinePE(d_model, max_len=256)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.out_proj = nn.Linear(d_model, OUT_DIM)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, hist, target=None, tf_ratio=None) -> torch.Tensor:
        """
        hist   : (B, 30, 6)  normalised history
        target / tf_ratio are accepted for API compatibility but ignored
                (NAR has no sequential decoding to teacher-force).
        Returns: (B, 30, 2)  predicted future positions (normalised)
        """
        B = hist.size(0)

        # Encode history
        mem = self.encoder(
            self.enc_pe(self.hist_proj(hist))
        )                                              # (B, 30, d)

        # Expand learned future queries for the batch
        queries = self.dec_pe(
            self.future_queries.unsqueeze(0).expand(B, -1, -1)
        )                                              # (B, 30, d)

        # Decode: cross-attend to history memory
        out = self.decoder(queries, mem)               # (B, 30, d)

        return self.out_proj(out)                      # (B, 30, 2)

    def predict(self, hist) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(hist)
