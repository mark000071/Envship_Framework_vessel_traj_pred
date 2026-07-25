"""Context-aware trajectory prediction models.

All built on top of LSTM-2L encoder-decoder backbone.
Context is injected after the encoder (into the decoder initial state or as
additional input per step).

Models:
  LSTMSocialPool   — Neighbor-Pooling / Social-LSTM style
  LSTMSocialAttn   — Social Attention LSTM
  LSTMEnvDesc      — Environment descriptor conditioned LSTM
  LSTMEnvRaster    — Environment raster (CNN) conditioned LSTM
  LSTMSocialEnv    — Full context: social + env descriptor + raster
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequence_models import _RNNDecoder, _run_decoder

HIST_DIM    = 6
FUT_STEPS   = 30
OUT_DIM     = 2
HIDDEN      = 256
N_NEIGHBORS = 10
SOCIAL_FEAT = 5      # [rel_x, rel_y, rel_vx, rel_vy, dist]
ENV_DESC_DIM = 14


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

class _LSTMEncoder(nn.Module):
    def __init__(self, in_dim: int = HIST_DIM, hidden: int = HIDDEN,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, h, c


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Social-LSTM (Neighbor Pooling)
# ---------------------------------------------------------------------------

class LSTMSocialPool(nn.Module):
    """Social LSTM with max-pooling over neighbor embeddings.

    For each sample, encodes each neighbor's (rel_x, rel_y, rel_vx, rel_vy, dist)
    into a 64-dim vector, max-pools over neighbors, then injects into the
    decoder initial state alongside the LSTM encoder output.

    Reference: Alahi et al. "Social Force LSTM" (adapted here as pool variant).
    """
    name = "LSTM+Social-Pool"

    def __init__(self, hidden: int = HIDDEN, social_enc_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder  = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        # Neighbor encoder: (SOCIAL_FEAT) → social_enc_dim
        self.neigh_enc = _MLP(SOCIAL_FEAT, social_enc_dim, social_enc_dim * 2)
        # Fuse encoder hidden + pooled social context
        self.fuse_h   = _MLP(hidden + social_enc_dim, hidden, hidden)
        self.fuse_c   = _MLP(hidden + social_enc_dim, hidden, hidden)
        self.decoder  = _RNNDecoder(hidden, OUT_DIM, 2, dropout)

    def forward(self, hist, social_feat, social_mask, target=None, tf_ratio=0.5):
        """
        hist:        (B, 30, 6)
        social_feat: (B, 10, 5) — neighbor features, zero-padded
        social_mask: (B, 10)    — 1 = valid, 0 = padding
        """
        _, h, c = self.encoder(hist)       # each (2, B, H)
        # Encode each neighbor
        B, N, _ = social_feat.shape
        neigh_e = self.neigh_enc(social_feat.view(B * N, -1)).view(B, N, -1)  # (B, N, 64)
        # Mask and max-pool. Padded slots are filled with -inf so they can
        # never win the max (a zero-fill would clamp every pooled dim at >=0);
        # all-padded rows (no neighbours) fall back to a zero context vector.
        mask_3d = social_mask.unsqueeze(-1).bool()            # (B, N, 1)
        neigh_e = neigh_e.masked_fill(~mask_3d, float("-inf"))
        pooled  = neigh_e.max(dim=1).values                   # (B, 64)
        has_nb  = mask_3d.any(dim=1)                          # (B, 1)
        pooled  = torch.where(has_nb, pooled, torch.zeros_like(pooled))
        # Fuse into decoder state (only last layer of h, c)
        h_last  = h[-1]                       # (B, H)
        c_last  = c[-1]
        h_new   = self.fuse_h(torch.cat([h_last, pooled], dim=-1))
        c_new   = self.fuse_c(torch.cat([c_last, pooled], dim=-1))
        # Rebuild 2-layer hidden states
        h_dec   = torch.stack([h[0], h_new], dim=0)
        c_dec   = torch.stack([c[0], c_new], dim=0)
        start   = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, social_feat=None, social_mask=None, **kw):
        if social_feat is None:
            B = hist.shape[0]
            social_feat = torch.zeros(B, N_NEIGHBORS, SOCIAL_FEAT, device=hist.device)
            social_mask = torch.zeros(B, N_NEIGHBORS, device=hist.device)
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# Social Attention LSTM
# ---------------------------------------------------------------------------

class LSTMSocialAttn(nn.Module):
    """Social LSTM with attention-based aggregation over neighbors.

    Attention weight for each neighbor = softmax(score(h_enc, neigh_enc)).
    Inspired by the Attention-based Social Force model line.
    """
    name = "LSTM+Social-Attn"

    def __init__(self, hidden: int = HIDDEN, social_enc_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder   = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.neigh_enc = _MLP(SOCIAL_FEAT, social_enc_dim, social_enc_dim * 2)
        # Attention: query = encoder hidden, key = neighbor encoding
        self.attn_q    = nn.Linear(hidden,         social_enc_dim)
        self.attn_k    = nn.Linear(social_enc_dim, social_enc_dim)
        self.fuse_h    = _MLP(hidden + social_enc_dim, hidden, hidden)
        self.fuse_c    = _MLP(hidden + social_enc_dim, hidden, hidden)
        self.decoder   = _RNNDecoder(hidden, OUT_DIM, 2, dropout)
        self.scale     = social_enc_dim ** 0.5

    def forward(self, hist, social_feat, social_mask, target=None, tf_ratio=0.5):
        _, h, c = self.encoder(hist)
        B, N, _ = social_feat.shape
        neigh_e = self.neigh_enc(social_feat.view(B * N, -1)).view(B, N, -1)

        # Attention
        query = self.attn_q(h[-1]).unsqueeze(1)          # (B, 1, 64)
        keys  = self.attn_k(neigh_e)                     # (B, N, 64)
        scores = (query * keys).sum(-1) / self.scale     # (B, N)
        # Mask invalid neighbors
        scores = scores.masked_fill(social_mask == 0, float("-inf"))
        # If all neighbors are invalid → uniform zero context
        all_invalid = (social_mask.sum(dim=1) == 0)
        weights = torch.zeros_like(scores)
        if not all_invalid.all():
            weights[~all_invalid] = torch.softmax(scores[~all_invalid], dim=-1)
        context = (weights.unsqueeze(-1) * neigh_e).sum(dim=1)   # (B, 64)

        h_new = self.fuse_h(torch.cat([h[-1], context], dim=-1))
        c_new = self.fuse_c(torch.cat([c[-1], context], dim=-1))
        h_dec = torch.stack([h[0], h_new], dim=0)
        c_dec = torch.stack([c[0], c_new], dim=0)
        start = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, social_feat=None, social_mask=None, **kw):
        if social_feat is None:
            B = hist.shape[0]
            social_feat = torch.zeros(B, N_NEIGHBORS, SOCIAL_FEAT, device=hist.device)
            social_mask = torch.zeros(B, N_NEIGHBORS, device=hist.device)
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# Environment-Descriptor LSTM
# ---------------------------------------------------------------------------

class LSTMEnvDesc(nn.Module):
    """LSTM conditioned on scalar environment descriptors.

    14-dim env descriptor (scene one-hot + numeric) is encoded and injected
    into the decoder initial state.
    """
    name = "LSTM+Env-Desc"

    def __init__(self, hidden: int = HIDDEN, env_enc_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder  = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.env_enc  = _MLP(ENV_DESC_DIM, env_enc_dim, env_enc_dim * 2)
        self.fuse_h   = _MLP(hidden + env_enc_dim, hidden, hidden)
        self.fuse_c   = _MLP(hidden + env_enc_dim, hidden, hidden)
        self.decoder  = _RNNDecoder(hidden, OUT_DIM, 2, dropout)

    def forward(self, hist, env_desc, target=None, tf_ratio=0.5):
        """env_desc: (B, ENV_DESC_DIM)"""
        _, h, c = self.encoder(hist)
        env_e   = self.env_enc(env_desc)                 # (B, 32)
        h_new   = self.fuse_h(torch.cat([h[-1], env_e], dim=-1))
        c_new   = self.fuse_c(torch.cat([c[-1], env_e], dim=-1))
        h_dec   = torch.stack([h[0], h_new], dim=0)
        c_dec   = torch.stack([c[0], c_new], dim=0)
        start   = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, env_desc=None, **kw):
        if env_desc is None:
            env_desc = torch.zeros(hist.shape[0], ENV_DESC_DIM, device=hist.device)
        with torch.no_grad():
            return self.forward(hist, env_desc, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# Environment-Raster LSTM
# ---------------------------------------------------------------------------

class _RasterCNN(nn.Module):
    """Light CNN that encodes a 6×128×128 binary raster into a 64-dim vector."""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=4, padding=2), nn.ReLU(),   # → 16×32×32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # → 32×16×16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # → 64×8×8
            nn.AdaptiveAvgPool2d(1),                                # → 64×1×1
            nn.Flatten(),
            nn.Linear(64, out_dim), nn.GELU(),
        )

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        return self.net(raster.float() / 1.0)


class LSTMEnvRaster(nn.Module):
    """LSTM conditioned on the 128×128 environment raster (via CNN encoder)."""
    name = "LSTM+Env-Raster"

    def __init__(self, hidden: int = HIDDEN, cnn_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder  = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.cnn      = _RasterCNN(cnn_dim)
        self.fuse_h   = _MLP(hidden + cnn_dim, hidden, hidden)
        self.fuse_c   = _MLP(hidden + cnn_dim, hidden, hidden)
        self.decoder  = _RNNDecoder(hidden, OUT_DIM, 2, dropout)

    def forward(self, hist, env_raster, target=None, tf_ratio=0.5):
        """env_raster: (B, 6, 128, 128) uint8"""
        _, h, c  = self.encoder(hist)
        rast_e   = self.cnn(env_raster)                  # (B, 64)
        h_new    = self.fuse_h(torch.cat([h[-1], rast_e], dim=-1))
        c_new    = self.fuse_c(torch.cat([c[-1], rast_e], dim=-1))
        h_dec    = torch.stack([h[0], h_new], dim=0)
        c_dec    = torch.stack([c[0], c_new], dim=0)
        start    = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, env_raster=None, **kw):
        if env_raster is None:
            env_raster = torch.zeros(hist.shape[0], 6, 128, 128,
                                      device=hist.device, dtype=torch.uint8)
        with torch.no_grad():
            return self.forward(hist, env_raster, target=None, tf_ratio=0.0)


# ---------------------------------------------------------------------------
# Full context: Social + Env Descriptor + Env Raster
# ---------------------------------------------------------------------------

class LSTMSocialEnv(nn.Module):
    """LSTM conditioned on social context + env descriptor + env raster."""
    name = "LSTM+Social+Env"

    def __init__(self, hidden: int = HIDDEN, social_enc_dim: int = 64,
                 env_enc_dim: int = 32, cnn_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder   = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.neigh_enc = _MLP(SOCIAL_FEAT, social_enc_dim, social_enc_dim * 2)
        self.attn_q    = nn.Linear(hidden,         social_enc_dim)
        self.attn_k    = nn.Linear(social_enc_dim, social_enc_dim)
        self.scale     = social_enc_dim ** 0.5
        self.env_enc   = _MLP(ENV_DESC_DIM, env_enc_dim, env_enc_dim * 2)
        self.cnn       = _RasterCNN(cnn_dim)
        ctx_dim = social_enc_dim + env_enc_dim + cnn_dim
        self.fuse_h    = _MLP(hidden + ctx_dim, hidden, hidden)
        self.fuse_c    = _MLP(hidden + ctx_dim, hidden, hidden)
        self.decoder   = _RNNDecoder(hidden, OUT_DIM, 2, dropout)

    def forward(self, hist, social_feat, social_mask, env_desc, env_raster,
                target=None, tf_ratio=0.5):
        _, h, c = self.encoder(hist)
        B, N, _ = social_feat.shape

        # Social attention
        neigh_e = self.neigh_enc(social_feat.view(B * N, -1)).view(B, N, -1)
        query   = self.attn_q(h[-1]).unsqueeze(1)
        keys    = self.attn_k(neigh_e)
        scores  = (query * keys).sum(-1) / self.scale
        scores  = scores.masked_fill(social_mask == 0, float("-inf"))
        all_inv = (social_mask.sum(dim=1) == 0)
        weights = torch.zeros_like(scores)
        if not all_inv.all():
            weights[~all_inv] = torch.softmax(scores[~all_inv], dim=-1)
        soc_ctx = (weights.unsqueeze(-1) * neigh_e).sum(dim=1)

        # Env
        env_e   = self.env_enc(env_desc)
        rast_e  = self.cnn(env_raster)

        ctx     = torch.cat([soc_ctx, env_e, rast_e], dim=-1)
        h_new   = self.fuse_h(torch.cat([h[-1], ctx], dim=-1))
        c_new   = self.fuse_c(torch.cat([c[-1], ctx], dim=-1))
        h_dec   = torch.stack([h[0], h_new], dim=0)
        c_dec   = torch.stack([c[0], c_new], dim=0)
        start   = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, social_feat=None, social_mask=None,
                env_desc=None, env_raster=None, **kw):
        B = hist.shape[0]
        dev = hist.device
        if social_feat is None:
            social_feat = torch.zeros(B, N_NEIGHBORS, SOCIAL_FEAT, device=dev)
            social_mask = torch.zeros(B, N_NEIGHBORS, device=dev)
        if env_desc is None:
            env_desc    = torch.zeros(B, ENV_DESC_DIM, device=dev)
        if env_raster is None:
            env_raster  = torch.zeros(B, 6, 128, 128, device=dev, dtype=torch.uint8)
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask, env_desc, env_raster,
                                target=None, tf_ratio=0.0)
