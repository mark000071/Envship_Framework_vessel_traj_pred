"""V5 social-context model — addresses the failure of v1/v2 social baselines.

Diagnosed problems with the current social encoders (LSTM+Social-Pool, Attn):
  1. Feature poverty: only 5 per-neighbour features
       (rel_x, rel_y, rel_vx, rel_vy, dist).
       Closest-Point-of-Approach (CPA) and Time-to-CPA (TCPA) — already
       computed at build time and stored in the social CSV — are NOT exposed
       to the model.  These are the most signal-rich maritime-interaction
       features.
  2. Sparse signal swamping: ~82% of DMA train samples are isolated (0
     neighbours).  With one-shot pooling, the social pathway becomes mostly
     "predict zero", which the LSTM trivially learns to ignore.
  3. No graceful fall-back for isolated samples — the model has no explicit
     way to "switch off" the social pathway when there are zero neighbours.
  4. One-shot injection into h0 only — the social signal doesn't persist
     through 30 prediction steps.

LSTMSocialAttnV5 fixes all four:
  * Consumes an 8-d per-neighbour feature vector:
        [rel_x, rel_y, rel_vx, rel_vy, dist, cpa, tcpa, |rel_v|]
    (legacy 5-d vectors are auto-padded so old loaders still work).
  * Properly-masked softmax attention (set masked positions to -inf so they
    receive zero attention weight; never contaminate the context vector).
  * Per-step cross-attention: at every decoder step the hidden state queries
    the neighbour set, so the social signal persists for the full 10 min.
  * Learned gate from the number of valid neighbours — when count==0 the
    gate is near zero and the model falls back to the trajectory-only LSTM.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequence_models import _RNNDecoder
from .context_models   import _LSTMEncoder, HIST_DIM, FUT_STEPS, OUT_DIM, HIDDEN

MAX_NEIGHBORS = 10
SOCIAL_FEAT_V5 = 8   # rel_x, rel_y, rel_vx, rel_vy, dist, cpa, tcpa, |rel_v|


class LSTMSocialAttnV5(nn.Module):
    name = "LSTM+Social-AttnV5"

    def __init__(self, hidden: int = HIDDEN, neigh_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder    = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.neigh_enc  = nn.Sequential(
            nn.Linear(SOCIAL_FEAT_V5, neigh_dim), nn.GELU(),
            nn.Linear(neigh_dim, neigh_dim),
        )
        self.q_proj = nn.Linear(hidden,    neigh_dim)
        self.k_proj = nn.Linear(neigh_dim, neigh_dim)
        self.v_proj = nn.Linear(neigh_dim, neigh_dim)
        self.scale  = neigh_dim ** 0.5

        # Learned gate: scalar sigmoid that down-weights the social signal when
        # there are 0 neighbours; uses the neighbour count as input.
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        dec_in = OUT_DIM + neigh_dim                  # prev_xy + ctx
        self.dec_cell = nn.LSTMCell(dec_in, hidden)
        self.dec_proj = nn.Linear(hidden + neigh_dim, OUT_DIM)
        self.dropout  = nn.Dropout(dropout)

    # ── Feature-shape adapter ──
    @staticmethod
    def _to_v5(social_feat: torch.Tensor) -> torch.Tensor:
        """Pad a 5-d (legacy) input up to 8-d, leaving CPA / TCPA / |rel_v| = 0."""
        if social_feat.shape[-1] >= SOCIAL_FEAT_V5:
            return social_feat[..., :SOCIAL_FEAT_V5]
        B, N, _ = social_feat.shape
        pad = torch.zeros(B, N, SOCIAL_FEAT_V5 - social_feat.shape[-1],
                           device=social_feat.device, dtype=social_feat.dtype)
        return torch.cat([social_feat, pad], dim=-1)

    def forward(self, hist, social_feat, social_mask, target=None, tf_ratio=0.5):
        """
        hist        : (B, 30, 6)
        social_feat : (B, 10, 8) or (B, 10, 5) — auto-padded
        social_mask : (B, 10)
        """
        sf = self._to_v5(social_feat)
        B = hist.size(0)

        _, h_enc, c_enc = self.encoder(hist)
        h, c = h_enc[-1], c_enc[-1]               # (B, H)

        # Encode neighbours
        nE = self.neigh_enc(sf)                   # (B, N, neigh_dim)
        K  = self.k_proj(nE)
        V  = self.v_proj(nE)

        # Boolean mask: True = valid; will be inverted to -inf for softmax
        mask = social_mask.bool()                 # (B, N)
        any_valid = mask.any(dim=-1, keepdim=True)  # (B, 1)
        n_valid = mask.sum(dim=-1, keepdim=True).float()  # (B, 1)

        # Gate: f(n_valid) ∈ (0, 1)  — driven only by raw count
        g = self.gate(n_valid)                    # (B, 1)

        # Initial context = masked mean of V (or zero if all masked)
        m1 = mask.unsqueeze(-1).float()
        denom = m1.sum(dim=1).clamp(min=1.0)
        ctx = (V * m1).sum(dim=1) / denom        # (B, neigh_dim)
        ctx = ctx * g                             # gated

        prev_xy = hist[:, -1, :2]
        preds = []
        for t in range(FUT_STEPS):
            x_in = torch.cat([prev_xy, ctx], dim=-1)
            h, c = self.dec_cell(x_in, (h, c))
            h = self.dropout(h)

            # Per-step masked attention
            q = self.q_proj(h)                    # (B, neigh_dim)
            scores = torch.einsum("bf,bnf->bn", q, K) / self.scale
            scores = scores.masked_fill(~mask, float("-inf"))
            # All-masked rows: softmax→nan; replace with zero attention
            attn   = torch.where(any_valid,
                                  F.softmax(scores, dim=-1),
                                  torch.zeros_like(scores))
            ctx    = torch.einsum("bn,bnf->bf", attn, V) * g

            out = self.dec_proj(torch.cat([h, ctx], dim=-1))
            preds.append(out.unsqueeze(1))
            if target is not None and torch.rand(1).item() < tf_ratio:
                prev_xy = target[:, t, :]
            else:
                prev_xy = out

        return torch.cat(preds, dim=1)            # (B, 30, 2)

    def predict(self, hist, social_feat=None, social_mask=None, **kw):
        B, dev = hist.shape[0], hist.device
        if social_feat is None:
            social_feat = torch.zeros(B, MAX_NEIGHBORS, SOCIAL_FEAT_V5, device=dev)
            social_mask = torch.zeros(B, MAX_NEIGHBORS, device=dev)
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask,
                                target=None, tf_ratio=0.0)
