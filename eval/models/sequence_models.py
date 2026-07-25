"""RNN-based trajectory prediction models.

All models follow the same interface:
  forward(hist, target=None, tf_ratio=0.5) -> (N, T_fut, 2)
  predict(hist) -> (N, T_fut, 2)    # teacher-forcing-free

hist:   (B, 30, 6)  float32  normalised
target: (B, 30, 2)  float32  normalised, used during training
"""

from __future__ import annotations
import torch
import torch.nn as nn


HIST_DIM    = 6
FUT_STEPS   = 30
OUT_DIM     = 2


# ---------------------------------------------------------------------------
# Shared encoder/decoder pieces
# ---------------------------------------------------------------------------

class _RNNDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(out_dim, hidden_dim, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, h, c):
        out, (h, c) = self.rnn(x, (h, c))
        return self.proj(out), h, c


def _run_decoder(decoder, start, h, c, T, target, tf_ratio):
    token = start                  # (B, 1, 2)
    preds = []
    for t in range(T):
        out, h, c = decoder(token, h, c)
        preds.append(out)
        if target is not None and torch.rand(1).item() < tf_ratio:
            token = target[:, t:t+1, :]
        else:
            token = out
    return torch.cat(preds, dim=1)   # (B, T, 2)


# ---------------------------------------------------------------------------
# LSTM 2-layer (hidden=256)
# ---------------------------------------------------------------------------

class LSTM2L(nn.Module):
    name = "LSTM-2L"

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(HIST_DIM, hidden_dim, 2,
                                batch_first=True,
                                dropout=dropout)
        self.decoder = _RNNDecoder(hidden_dim, OUT_DIM, 2, dropout)

    def forward(self, hist, target=None, tf_ratio=0.5):
        _, (h, c) = self.encoder(hist)
        start = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h, c, FUT_STEPS, target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# LSTM 4-layer (hidden=256)
# ---------------------------------------------------------------------------

class LSTM4L(nn.Module):
    name = "LSTM-4L"

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(HIST_DIM, hidden_dim, 4,
                                batch_first=True,
                                dropout=dropout)
        self.decoder = _RNNDecoder(hidden_dim, OUT_DIM, 4, dropout)

    def forward(self, hist, target=None, tf_ratio=0.5):
        _, (h, c) = self.encoder(hist)
        start = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h, c, FUT_STEPS, target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# GRU 2-layer (hidden=256)
# ---------------------------------------------------------------------------

class _GRUDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn  = nn.GRU(out_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        return self.proj(out), h


class GRU2L(nn.Module):
    name = "GRU-2L"

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(HIST_DIM, hidden_dim, 2,
                               batch_first=True, dropout=dropout)
        self.decoder = _GRUDecoder(hidden_dim, OUT_DIM, 2, dropout)

    def forward(self, hist, target=None, tf_ratio=0.5):
        _, h = self.encoder(hist)
        token = hist[:, -1:, :2]
        preds = []
        for t in range(FUT_STEPS):
            out, h = self.decoder(token, h)
            preds.append(out)
            if target is not None and torch.rand(1).item() < tf_ratio:
                token = target[:, t:t+1, :]
            else:
                token = out
        return torch.cat(preds, dim=1)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# Bi-LSTM 2-layer encoder + unidirectional 2-layer decoder (hidden=256)
# ---------------------------------------------------------------------------

class BiLSTM2L(nn.Module):
    name = "BiLSTM-2L"

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(HIST_DIM, hidden_dim, 2,
                                batch_first=True, bidirectional=True,
                                dropout=dropout)
        # Project bidirectional hidden to unidirectional for decoder
        self.proj_h  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_c  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = _RNNDecoder(hidden_dim, OUT_DIM, 2, dropout)

    def forward(self, hist, target=None, tf_ratio=0.5):
        _, (h, c) = self.encoder(hist)
        # h: (4, B, 256) → concat forward+backward for each layer
        B = hist.size(0)
        # Take last forward and last backward layers, project
        h_fwd = h[-2]                    # (B, H)
        h_bwd = h[-1]
        h_dec = self.proj_h(torch.cat([h_fwd, h_bwd], dim=-1)).unsqueeze(0).expand(2, -1, -1).contiguous()
        c_fwd = c[-2]
        c_bwd = c[-1]
        c_dec = self.proj_c(torch.cat([c_fwd, c_bwd], dim=-1)).unsqueeze(0).expand(2, -1, -1).contiguous()
        start = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.0)
