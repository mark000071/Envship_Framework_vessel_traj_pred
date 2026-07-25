"""TrAISformer ported to local-metre coordinate frame.

Paper: "TrAISformer: A Generative Transformer for AIS Trajectory Prediction"
       (Capobianco et al., 2021) — github.com/CIA-Oceanix/TrAISformer

Key ideas retained from TrAISformer (Capobianco et al., 2021):
  1. Discrete tokenisation of (x, y, SOG, COG) into bin indices
  2. Separate learnable embedding per feature vocabulary
  3. GPT-style causal (decoder-only) Transformer
  4. Next-token prediction training (cross-entropy over bins)
  5. Greedy decoding at inference → bin centres → metre positions

Adaptation for local-metre frame:
  - lat/lon bins → x_bin, y_bin over ±6000 m, 50 m resolution (240 bins each)
  - SOG bins: 0–30 knots, 0.5 knot resolution (60 bins)
  - COG bins: 0–360°, 5° resolution (72 bins)

Each timestep produces 4 tokens; history of 30 steps = 120 input tokens.
Predict next 30 steps = 120 output tokens.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vocabulary sizes
# ---------------------------------------------------------------------------

X_BINS  = 240   # −6000 to +6000 m, 50 m per bin
Y_BINS  = 240
SOG_BINS = 60   # 0–30 kn, 0.5 kn per bin
COG_BINS = 72   # 0–360°, 5° per bin
TOKENS_PER_STEP = 4
TOTAL_BINS = X_BINS + Y_BINS + SOG_BINS + COG_BINS

X_RANGE  = (-6000.0,  6000.0)
Y_RANGE  = (-6000.0,  6000.0)
SOG_RANGE = (0.0,     30.0)
COG_RANGE = (0.0,     360.0)


# ---------------------------------------------------------------------------
# Tokeniser / de-tokeniser
# ---------------------------------------------------------------------------

def _quantise(values: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    clipped = np.clip(values, lo, hi - 1e-6)
    bins = np.floor((clipped - lo) / (hi - lo) * n_bins).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)   # guarantee [0, n_bins-1]


def _centres(bins: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    return lo + (bins.astype(np.float32) + 0.5) * (hi - lo) / n_bins


def tokenise_hist(hist_norm: np.ndarray, normalizer) -> np.ndarray:
    """
    hist_norm: (N, 30, 6) normalised
    Returns: (N, 30, 4) int64 token indices [x_bin, y_bin, sog_bin, cog_bin]
    """
    # Denormalise x, y
    xy  = hist_norm[:, :, :2] * normalizer.std[:2] + normalizer.mean[:2]
    x_b = _quantise(xy[:, :, 0], *X_RANGE,  X_BINS)
    y_b = _quantise(xy[:, :, 1], *Y_RANGE,  Y_BINS)
    # SOG (channel 4) — denormalise
    sog = hist_norm[:, :, 4] * normalizer.std[4] + normalizer.mean[4]
    sog = np.clip(sog, 0, 30)
    sog_b = _quantise(sog, *SOG_RANGE, SOG_BINS)
    # COG from vx, vy (channels 2, 3) — compute angle
    vx  = hist_norm[:, :, 2] * normalizer.std[2] + normalizer.mean[2]
    vy  = hist_norm[:, :, 3] * normalizer.std[3] + normalizer.mean[3]
    cog_deg = (np.degrees(np.arctan2(vx, vy)) + 360) % 360
    cog_b   = _quantise(cog_deg, *COG_RANGE, COG_BINS)
    return np.stack([x_b, y_b, sog_b, cog_b], axis=-1)


def tokenise_future(future_norm: np.ndarray, normalizer, hist_norm: np.ndarray) -> np.ndarray:
    """future_norm: (N, 30, 2) → (N, 30, 4) tokens.
    SOG/COG are approximated from step differences; all bins clamped.
    """
    xy  = future_norm * normalizer.std[:2] + normalizer.mean[:2]
    # Clamp positions to ±6000m (some fast vessels may exceed this; cap at bin boundary)
    xy  = np.clip(xy, X_RANGE[0] + 1e-3, X_RANGE[1] - 1e-3)
    x_b = _quantise(xy[:, :, 0], *X_RANGE, X_BINS)
    y_b = _quantise(xy[:, :, 1], *Y_RANGE, Y_BINS)
    # Approximate SOG from position step differences
    dx  = np.diff(xy[:, :, 0], axis=1, prepend=xy[:, :1, 0])
    dy  = np.diff(xy[:, :, 1], axis=1, prepend=xy[:, :1, 1])
    sog_mps  = np.sqrt(dx**2 + dy**2) / 20.0         # metres per 20s → m/s
    sog_knots = np.clip(sog_mps / 0.514444, 0, 30.0)
    cog_deg   = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    sog_b = _quantise(sog_knots, *SOG_RANGE, SOG_BINS)
    cog_b = _quantise(cog_deg,   *COG_RANGE, COG_BINS)
    return np.stack([x_b, y_b, sog_b, cog_b], axis=-1)


def detokenise_xy(x_bins: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    """bins (N, T) → metres (N, T, 2)"""
    x = _centres(x_bins, *X_RANGE, X_BINS)
    y = _centres(y_bins, *Y_RANGE, Y_BINS)
    return np.stack([x, y], axis=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _SinePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrAISformerLocal(nn.Module):
    """GPT-style decoder-only Transformer with discrete token inputs."""
    name = "TrAISformer (local)"

    def __init__(
        self,
        d_model:   int   = 128,
        nhead:     int   = 4,
        n_layers:  int   = 4,
        ffn_dim:   int   = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # Separate embedding per feature + shared PE
        self.embed_x   = nn.Embedding(X_BINS + 1,   d_model // 4)
        self.embed_y   = nn.Embedding(Y_BINS + 1,   d_model // 4)
        self.embed_sog = nn.Embedding(SOG_BINS + 1, d_model // 4)
        self.embed_cog = nn.Embedding(COG_BINS + 1, d_model // 4)
        self.step_proj = nn.Linear(d_model, d_model)
        self.pe        = _SinePE(d_model, max_len=256)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Output heads per feature
        self.head_x   = nn.Linear(d_model, X_BINS)
        self.head_y   = nn.Linear(d_model, Y_BINS)
        self.head_sog = nn.Linear(d_model, SOG_BINS)
        self.head_cog = nn.Linear(d_model, COG_BINS)
        self._init()

    def _init(self):
        for emb in [self.embed_x, self.embed_y, self.embed_sog, self.embed_cog]:
            nn.init.normal_(emb.weight, std=0.01)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, 4) int64 → (B, T, d_model)"""
        x_e   = self.embed_x(tokens[:, :, 0])
        y_e   = self.embed_y(tokens[:, :, 1])
        sog_e = self.embed_sog(tokens[:, :, 2])
        cog_e = self.embed_cog(tokens[:, :, 3])
        return self.step_proj(torch.cat([x_e, y_e, sog_e, cog_e], dim=-1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        tokens: (B, T, 4) — all timesteps concatenated (hist + future shifted)
        Returns tuple of logits: (logit_x, logit_y, logit_sog, logit_cog)
                each (B, T, vocab_size)
        """
        B, T, _ = tokens.shape
        emb  = self.pe(self._embed_tokens(tokens))   # (B, T, d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        out  = self.transformer(emb, mask=mask, is_causal=True)
        return (
            self.head_x(out),   self.head_y(out),
            self.head_sog(out), self.head_cog(out),
        )

    def predict_future(self, hist_tokens: torch.Tensor) -> torch.Tensor:
        """
        hist_tokens: (B, 30, 4)
        Returns: (B, 30, 2) predicted x, y in bin-centre metres
        """
        B = hist_tokens.size(0)
        device = hist_tokens.device
        tokens = hist_tokens.clone()                  # (B, 30, 4)
        future_x = torch.zeros(B, 30, dtype=torch.long, device=device)
        future_y = torch.zeros(B, 30, dtype=torch.long, device=device)

        for t in range(30):
            lx, ly, ls, lc = self.forward(tokens)    # each (B, T, vocab)
            # Greedy decode last position
            nx = lx[:, -1, :].argmax(dim=-1)          # (B,)
            ny = ly[:, -1, :].argmax(dim=-1)
            ns = ls[:, -1, :].argmax(dim=-1)
            nc = lc[:, -1, :].argmax(dim=-1)
            future_x[:, t] = nx
            future_y[:, t] = ny
            # Append predicted token
            new_tok = torch.stack([nx, ny, ns, nc], dim=-1).unsqueeze(1)  # (B, 1, 4)
            tokens  = torch.cat([tokens, new_tok], dim=1)

        # Convert to metres
        x_m = _centres(future_x.cpu().numpy(), *X_RANGE, X_BINS)
        y_m = _centres(future_y.cpu().numpy(), *Y_RANGE, Y_BINS)
        return torch.from_numpy(np.stack([x_m, y_m], axis=-1)).float().to(device)

    # Allow predict(hist_norm) interface used by eval framework
    # (hist_norm is already normalised; tokens must be pre-computed)
    def predict(self, hist_norm_dummy):
        raise NotImplementedError(
            "TrAISformerLocal requires tokenised input; use predict_future(hist_tokens) "
            "or the specialised train_traisformer.py wrapper."
        )
