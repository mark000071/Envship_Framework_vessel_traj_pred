"""
V2 RNN sequence models — velocity-augmented decoder.

Key improvements over v1:

1. Decoder input: 4-D [x, y, dx, dy] instead of 2-D [x, y].
   The dx/dy terms give the decoder an explicit velocity signal at every
   prediction step, mimicking what Dead Reckoning exploits analytically.
   v1 forced all kinematic information into the initial (h0, c0) which
   fades over 30 decoder steps; v2 re-conditions on velocity each step.

2. Teacher forcing decays to 0 at 40 % of training (not 70 % in v1),
   so the model spends the majority of training in free-run mode and
   learns to handle its own prediction errors.
   (LR schedule is set in train_dl.py — CosineAnnealingLR, no restarts.)

Why NOT displacement prediction (tried, worse):
  Predicting (dx, dy) and cumsum-ing creates a 30× gradient amplification
  for the first displacement.  Grad-clipping mutes early-step updates while
  late steps under-learn, preventing convergence.  4-D absolute decoder
  avoids this because the loss gradient flows directly to each step's (x, y)
  prediction without accumulation.
"""

from __future__ import annotations
import torch
import torch.nn as nn

HIST_DIM  = 6
FUT_STEPS = 30
OUT_DIM   = 2
DEC_IN    = 4      # [x, y, dx, dy]
HIDDEN    = 256


# ---------------------------------------------------------------------------
# Shared decoder
# ---------------------------------------------------------------------------

class _Vel4Decoder(nn.Module):
    """LSTM decoder with 4-D input: [x_prev, y_prev, dx_prev, dy_prev]."""

    def __init__(self, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.rnn  = nn.LSTM(DEC_IN, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.)
        self.proj = nn.Linear(hidden, OUT_DIM)

    def forward(self, x, h, c):
        out, (h, c) = self.rnn(x, (h, c))
        return self.proj(out), h, c          # (B,1,2), h, c


def _make_start_token(hist: torch.Tensor) -> torch.Tensor:
    """
    Build the initial 4-D decoder token from the last two history steps.
    token = [x_{-1}, y_{-1}, x_{-1}-x_{-2}, y_{-1}-y_{-2}]
    """
    pos  = hist[:, -1:, :2]                          # (B, 1, 2)
    vel  = hist[:, -1:, :2] - hist[:, -2:-1, :2]     # (B, 1, 2) last velocity
    return torch.cat([pos, vel], dim=-1)              # (B, 1, 4)


def _run_vel4_decoder(decoder, start_tok, h, c, T, target, tf_ratio):
    """
    Decoder loop with 4-D tokens (absolute position + velocity).

    start_tok : (B, 1, 4)
    target    : (B, T, 2) normalised absolute future positions, or None
    Returns   : (B, T, 2) predicted absolute positions
    """
    tok      = start_tok          # (B, 1, 4)
    prev_pos = tok[:, :, :2]      # (B, 1, 2)  position part of token
    preds    = []

    for t in range(T):
        pred_pos, h, c = decoder(tok, h, c)   # (B, 1, 2) absolute position
        preds.append(pred_pos)

        if target is not None and torch.rand(1).item() < tf_ratio:
            next_pos = target[:, t:t+1, :]    # teacher force: true position
        else:
            next_pos = pred_pos               # free run

        vel = next_pos - prev_pos             # (B, 1, 2) velocity estimate
        tok = torch.cat([next_pos, vel], dim=-1)   # (B, 1, 4)
        prev_pos = next_pos

    return torch.cat(preds, dim=1)            # (B, T, 2)


# ---------------------------------------------------------------------------
# LSTM-2L  v2
# ---------------------------------------------------------------------------

class LSTM2L_v2(nn.Module):
    name = "LSTM-2L-v2"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(HIST_DIM, hidden, 2, batch_first=True,
                               dropout=dropout)
        self.decoder = _Vel4Decoder(hidden, 2, dropout)

    def forward(self, hist, target=None, tf_ratio: float = 0.5):
        _, (h, c) = self.encoder(hist)
        start = _make_start_token(hist)
        return _run_vel4_decoder(self.decoder, start, h, c, FUT_STEPS,
                                  target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.)


# ---------------------------------------------------------------------------
# BiLSTM-2L  v2
# ---------------------------------------------------------------------------

class BiLSTM2L_v2(nn.Module):
    name = "BiLSTM-2L-v2"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(HIST_DIM, hidden, 2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        self.h_proj  = nn.Linear(hidden * 2, hidden)
        self.c_proj  = nn.Linear(hidden * 2, hidden)
        self.decoder = _Vel4Decoder(hidden, 2, dropout)

    def _merge_bidir(self, h, c, B):
        H  = h.size(-1)
        hm = h.view(2, 2, B, H).permute(0, 2, 1, 3).contiguous().view(2 * B, 2 * H)
        cm = c.view(2, 2, B, H).permute(0, 2, 1, 3).contiguous().view(2 * B, 2 * H)
        return (self.h_proj(hm).view(2, B, -1),
                self.c_proj(cm).view(2, B, -1))

    def forward(self, hist, target=None, tf_ratio: float = 0.5):
        B = hist.size(0)
        _, (h, c) = self.encoder(hist)
        h2, c2 = self._merge_bidir(h, c, B)
        start = _make_start_token(hist)
        return _run_vel4_decoder(self.decoder, start, h2, c2, FUT_STEPS,
                                  target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.)
