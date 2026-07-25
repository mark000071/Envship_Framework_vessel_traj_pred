"""Temporal Convolutional Network (TCN) for trajectory prediction.

Architecture:
  - Causal dilated 1D convolutions with exponentially increasing dilation
  - Residual connections
  - Direct regression: encodes full history, predicts all 30 future steps at once
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


HIST_DIM  = 6
FUT_STEPS = 30
OUT_DIM   = 2


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        # Causal: pad left only
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(ch, ch, kernel_size, dilation),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(ch, ch, kernel_size, dilation),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(ch, ch, 1) if True else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        # LayerNorm needs (B, T, C)
        out = x
        for layer in self.net:
            if isinstance(layer, nn.LayerNorm):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
            else:
                out = layer(out)
        return out + x    # residual


class TCNModel(nn.Module):
    name = "TCN"

    def __init__(
        self,
        hidden_dim:  int   = 128,
        n_levels:    int   = 6,
        kernel_size: int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(HIST_DIM, hidden_dim)
        # Dilations: 1, 2, 4, 8, 16, 32 → receptive field covers 30 steps
        blocks = []
        for i in range(n_levels):
            dilation = 2 ** i
            blocks.append(_ResBlock(hidden_dim, kernel_size, dilation, dropout))
        self.blocks  = nn.ModuleList(blocks)
        # Predict all T_fut steps from the last temporal feature
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, FUT_STEPS * OUT_DIM),
        )

    def forward(self, hist, target=None, tf_ratio=0.5):
        # hist: (B, 30, 6)
        x = self.input_proj(hist)          # (B, 30, H)
        x = x.transpose(1, 2)             # (B, H, 30)
        for block in self.blocks:
            x = block(x)
        feat = x[:, :, -1]                # (B, H) — last timestep
        out  = self.head(feat)            # (B, 30*2)
        return out.view(-1, FUT_STEPS, OUT_DIM)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist)
