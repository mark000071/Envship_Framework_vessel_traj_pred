"""V4 context-aware models — exploiting the signed-distance field (SDF)
environment representation rather than the binary OSM raster.

Two models:

  LSTMEnvSDF
    - Drop-in replacement for LSTMEnvRaster.
    - Input: (B, 2, 128, 128) float16 SDF (shore + nav distance in metres)
      instead of (B, 6, 128, 128) uint8 binary masks.
    - Same global-pool CNN + h0/c0 fusion architecture.
    - Hypothesis: continuous distance per pixel is far more informative
      per parameter than 6-channel binary occupancy.

  LSTMEnvSpatialAttn
    - Keeps the CNN's spatial feature map (no global pool).
    - Decoder queries the (16x16) feature map at every step via single-head
      cross-attention.
    - Hypothesis: discarding spatial structure (LSTMEnvRaster's
      AdaptiveAvgPool2d(1)) is the dominant failure mode of the raster
      baseline. Per-step cross-attention lets the model ask
      "what's around me now?" at every prediction step.

Crosses map representation (binary raster vs. SDF) with aggregation
(global pooling vs. spatial attention) in a controlled 2x2 design.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequence_models import _RNNDecoder, _run_decoder
from .context_models import _LSTMEncoder, _MLP, HIST_DIM, FUT_STEPS, OUT_DIM, HIDDEN

# Patch radius (used to normalize SDF values to roughly [-1, 1])
PATCH_RADIUS_M = 5000.0


# ===========================================================================
# Plan A: LSTMEnvSDF — same architecture as LSTMEnvRaster, SDF input
# ===========================================================================

class _SDFCNN(nn.Module):
    """Lightweight CNN encoding a 2x128x128 SDF map into a 64-dim vector.

    Identical conv stack as _RasterCNN, but in_channels=2 to consume the
    SDF directly rather than the 6-channel binary mask.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=4, padding=2),   nn.ReLU(),  # → 16×32×32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.ReLU(),  # → 32×16×16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.ReLU(),  # → 64×8×8
            nn.AdaptiveAvgPool2d(1),                                # → 64×1×1
            nn.Flatten(),
            nn.Linear(64, out_dim), nn.GELU(),
        )

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        # Normalize SDF: values in metres, patch radius 5000m
        x = (sdf.float() / PATCH_RADIUS_M).clamp(-1.0, 1.0)
        return self.net(x)


class LSTMEnvSDF(nn.Module):
    """LSTM conditioned on a 128×128 signed-distance field environment.

    Architectural twin of LSTMEnvRaster — same encoder, fuser, decoder.
    Differs only in: (a) input channels (2 instead of 6),
    (b) input is continuous SDF in metres instead of binary occupancy.
    """
    name = "LSTM+Env-SDF"

    def __init__(self, hidden: int = HIDDEN, cnn_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder  = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.cnn      = _SDFCNN(cnn_dim)
        self.fuse_h   = _MLP(hidden + cnn_dim, hidden, hidden)
        self.fuse_c   = _MLP(hidden + cnn_dim, hidden, hidden)
        self.decoder  = _RNNDecoder(hidden, OUT_DIM, 2, dropout)

    def forward(self, hist, env_sdf, target=None, tf_ratio=0.5):
        """env_sdf: (B, 2, 128, 128) float16 or float32"""
        _, h, c  = self.encoder(hist)
        sdf_e    = self.cnn(env_sdf)
        h_new    = self.fuse_h(torch.cat([h[-1], sdf_e], dim=-1))
        c_new    = self.fuse_c(torch.cat([c[-1], sdf_e], dim=-1))
        h_dec    = torch.stack([h[0], h_new], dim=0)
        c_dec    = torch.stack([c[0], c_new], dim=0)
        start    = hist[:, -1:, :2]
        return _run_decoder(self.decoder, start, h_dec, c_dec, FUT_STEPS, target, tf_ratio)

    def predict(self, hist, env_sdf=None, **kw):
        if env_sdf is None:
            env_sdf = torch.zeros(hist.shape[0], 2, 128, 128,
                                   device=hist.device, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(hist, env_sdf, target=None, tf_ratio=0.0)


# ===========================================================================
# Plan B: LSTMEnvSpatialAttn — preserve spatial feature map + cross-attn
# ===========================================================================

class _SDFSpatialEncoder(nn.Module):
    """CNN that keeps the spatial dimension (16x16 feature map)."""
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2),   nn.GELU(),  # 128 → 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.GELU(),  # 64 → 32
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1), nn.GELU(),  # 32 → 16
        )
        self.feat_dim = feat_dim
        # Learned 2D positional embedding for the 16x16 grid
        self.pos_emb = nn.Parameter(torch.randn(1, 16 * 16, feat_dim) * 0.02)

    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        """Returns (B, 256, feat_dim) — 256 spatial tokens with positional info."""
        x = (sdf.float() / PATCH_RADIUS_M).clamp(-1.0, 1.0)
        feat = self.net(x)                       # (B, feat_dim, 16, 16)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) # (B, 256, feat_dim)
        return tokens + self.pos_emb


class LSTMEnvSpatialAttn(nn.Module):
    """LSTM with per-step cross-attention over the SDF feature map.

    At each of 30 prediction steps, the decoder hidden state queries the
    CNN feature map (16x16) via single-head cross-attention. This avoids
    the information loss of global pooling in LSTMEnvRaster.
    """
    name = "LSTM+Env-SpatialAttn"

    def __init__(self, hidden: int = HIDDEN, feat_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder    = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.spatial    = _SDFSpatialEncoder(feat_dim)

        self.q_proj     = nn.Linear(hidden, feat_dim)
        self.k_proj     = nn.Linear(feat_dim, feat_dim)
        self.v_proj     = nn.Linear(feat_dim, feat_dim)
        self.scale      = feat_dim ** 0.5

        # Decoder: 1-layer LSTMCell, input = [prev_xy(2) + ctx(feat_dim)]
        self.dec_in_dim = OUT_DIM + feat_dim
        self.dec_cell   = nn.LSTMCell(self.dec_in_dim, hidden)
        self.dec_proj   = nn.Linear(hidden + feat_dim, OUT_DIM)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, hist, env_sdf, target=None, tf_ratio=0.5):
        B = hist.size(0)
        _, h_enc, c_enc = self.encoder(hist)              # h_enc: (2, B, H)

        # Initial decoder state from the encoder's last layer
        h = h_enc[-1]                                      # (B, H)
        c = c_enc[-1]                                      # (B, H)

        # Encode SDF spatial map
        spatial_feat = self.spatial(env_sdf)               # (B, 256, F)
        K = self.k_proj(spatial_feat)                      # (B, 256, F)
        V = self.v_proj(spatial_feat)                      # (B, 256, F)

        # Initial context (mean of feature map)
        ctx = V.mean(dim=1)                                # (B, F)
        prev_xy = hist[:, -1, :2]                          # (B, 2)

        preds = []
        for t in range(FUT_STEPS):
            # Step the decoder
            x_in = torch.cat([prev_xy, ctx], dim=-1)       # (B, 2+F)
            h, c = self.dec_cell(x_in, (h, c))
            h = self.dropout(h)

            # Cross-attention: query feature map
            q = self.q_proj(h)                             # (B, F)
            scores = torch.einsum('bf,bnf->bn', q, K) / self.scale  # (B, 256)
            attn   = F.softmax(scores, dim=-1)             # (B, 256)
            ctx    = torch.einsum('bn,bnf->bf', attn, V)   # (B, F)

            # Predict next position
            out = self.dec_proj(torch.cat([h, ctx], dim=-1))   # (B, 2)
            preds.append(out.unsqueeze(1))

            # Teacher forcing or autoregressive
            if target is not None and torch.rand(1).item() < tf_ratio:
                prev_xy = target[:, t, :]
            else:
                prev_xy = out

        return torch.cat(preds, dim=1)                     # (B, 30, 2)

    def predict(self, hist, env_sdf=None, **kw):
        if env_sdf is None:
            env_sdf = torch.zeros(hist.shape[0], 2, 128, 128,
                                   device=hist.device, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(hist, env_sdf, target=None, tf_ratio=0.0)


# ===========================================================================
# Disentanglement control: Binary input + Spatial Attention
# (Tells us whether the v4 win is from SDF input, spatial attention, or both.)
# ===========================================================================

class _BinarySpatialEncoder(nn.Module):
    """Spatial-preserving CNN over the 6-channel BINARY raster."""
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2),   nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.GELU(),
            nn.Conv2d(32, feat_dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.feat_dim = feat_dim
        self.pos_emb  = nn.Parameter(torch.randn(1, 16 * 16, feat_dim) * 0.02)

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        feat = self.net(raster.float())              # (B, F, 16, 16)
        tokens = feat.flatten(2).transpose(1, 2)     # (B, 256, F)
        return tokens + self.pos_emb


class LSTMEnvBinarySpatialAttn(nn.Module):
    """Spatial-attention variant of LSTMEnvRaster (binary 6-ch mask).

    Same per-step cross-attention decoder as LSTMEnvSpatialAttn but the
    input is the original 6-channel binary OSM raster instead of the SDF.
    This completes the 2x2 ablation: {Binary, SDF} x {GlobalPool, SpatialAttn}.
    """
    name = "LSTM+EnvBinary-SpatialAttn"

    def __init__(self, hidden: int = HIDDEN, feat_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder    = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.spatial    = _BinarySpatialEncoder(feat_dim)
        self.q_proj     = nn.Linear(hidden, feat_dim)
        self.k_proj     = nn.Linear(feat_dim, feat_dim)
        self.v_proj     = nn.Linear(feat_dim, feat_dim)
        self.scale      = feat_dim ** 0.5
        self.dec_in_dim = OUT_DIM + feat_dim
        self.dec_cell   = nn.LSTMCell(self.dec_in_dim, hidden)
        self.dec_proj   = nn.Linear(hidden + feat_dim, OUT_DIM)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, hist, env_raster, target=None, tf_ratio=0.5):
        B = hist.size(0)
        _, h_enc, c_enc = self.encoder(hist)
        h = h_enc[-1]; c = c_enc[-1]

        spatial_feat = self.spatial(env_raster)             # (B, 256, F)
        K = self.k_proj(spatial_feat)
        V = self.v_proj(spatial_feat)
        ctx = V.mean(dim=1)
        prev_xy = hist[:, -1, :2]

        preds = []
        for t in range(FUT_STEPS):
            x_in = torch.cat([prev_xy, ctx], dim=-1)
            h, c = self.dec_cell(x_in, (h, c))
            h = self.dropout(h)
            q = self.q_proj(h)
            scores = torch.einsum('bf,bnf->bn', q, K) / self.scale
            attn   = F.softmax(scores, dim=-1)
            ctx    = torch.einsum('bn,bnf->bf', attn, V)
            out = self.dec_proj(torch.cat([h, ctx], dim=-1))
            preds.append(out.unsqueeze(1))
            if target is not None and torch.rand(1).item() < tf_ratio:
                prev_xy = target[:, t, :]
            else:
                prev_xy = out
        return torch.cat(preds, dim=1)

    def predict(self, hist, env_raster=None, **kw):
        if env_raster is None:
            env_raster = torch.zeros(hist.shape[0], 6, 128, 128,
                                      device=hist.device, dtype=torch.uint8)
        with torch.no_grad():
            return self.forward(hist, env_raster, target=None, tf_ratio=0.0)


# ===========================================================================
# Full context: Social attention + SDF environment + Spatial attention
# ===========================================================================

class LSTMSocialEnvSDF(nn.Module):
    """Combines social attention with SDF spatial-attn environmental context.

    Architecture:
      - Trajectory encoder: 2-layer LSTM (HIDDEN=256)
      - Social encoder: per-neighbor MLP + cross-attention over neighbours
      - Environment encoder: SDF spatial CNN -> 16x16 feature map
      - Per-step decoder: cross-attend over env spatial tokens AND over social
        neighbour embeddings; fuse [hidden, env_ctx, soc_ctx] -> displacement.
    """
    name = "LSTM+Social+Env-SDF"

    SOC_ENC_DIM = 64
    MAX_NB      = 10
    SOC_FEAT    = 5

    def __init__(self, hidden: int = HIDDEN, feat_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = _LSTMEncoder(HIST_DIM, hidden, 2, dropout)
        self.spatial = _SDFSpatialEncoder(feat_dim)
        self.neigh_enc = nn.Sequential(
            nn.Linear(self.SOC_FEAT, self.SOC_ENC_DIM), nn.GELU(),
            nn.Linear(self.SOC_ENC_DIM, self.SOC_ENC_DIM),
        )

        self.env_q     = nn.Linear(hidden, feat_dim)
        self.env_k     = nn.Linear(feat_dim, feat_dim)
        self.env_v     = nn.Linear(feat_dim, feat_dim)
        self.env_scale = feat_dim ** 0.5

        self.soc_q     = nn.Linear(hidden, self.SOC_ENC_DIM)
        self.soc_k     = nn.Linear(self.SOC_ENC_DIM, self.SOC_ENC_DIM)
        self.soc_v     = nn.Linear(self.SOC_ENC_DIM, self.SOC_ENC_DIM)
        self.soc_scale = self.SOC_ENC_DIM ** 0.5

        dec_in_dim = OUT_DIM + feat_dim + self.SOC_ENC_DIM
        self.dec_cell  = nn.LSTMCell(dec_in_dim, hidden)
        self.dec_proj  = nn.Linear(hidden + feat_dim + self.SOC_ENC_DIM, OUT_DIM)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, hist, env_sdf, social_feat, social_mask,
                target=None, tf_ratio=0.5):
        B = hist.size(0)
        _, h_enc, c_enc = self.encoder(hist)
        h = h_enc[-1]; c = c_enc[-1]

        spatial_feat = self.spatial(env_sdf)
        Ke = self.env_k(spatial_feat); Ve = self.env_v(spatial_feat)
        env_ctx = Ve.mean(dim=1)

        soc_e = self.neigh_enc(social_feat)                  # (B, MAX_NB, SOC_ENC_DIM)
        Ks    = self.soc_k(soc_e); Vs = self.soc_v(soc_e)
        mask  = social_mask.bool()                           # (B, MAX_NB)
        # Initial social context: masked mean
        m1 = mask.unsqueeze(-1).float()
        soc_ctx = (Vs * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1.0)

        prev_xy = hist[:, -1, :2]
        preds = []
        for t in range(FUT_STEPS):
            x_in = torch.cat([prev_xy, env_ctx, soc_ctx], dim=-1)
            h, c = self.dec_cell(x_in, (h, c))
            h = self.dropout(h)

            qe = self.env_q(h)
            es = torch.einsum('bf,bnf->bn', qe, Ke) / self.env_scale
            ea = F.softmax(es, dim=-1)
            env_ctx = torch.einsum('bn,bnf->bf', ea, Ve)

            qs = self.soc_q(h)
            ss = torch.einsum('bf,bnf->bn', qs, Ks) / self.soc_scale
            ss = ss.masked_fill(~mask, float('-inf'))
            # If all-masked (no neighbours), softmax→nan; fallback to uniform-zero
            valid = mask.any(dim=-1, keepdim=True)
            sa = torch.where(valid, F.softmax(ss, dim=-1), torch.zeros_like(ss))
            soc_ctx = torch.einsum('bn,bnf->bf', sa, Vs)

            out = self.dec_proj(torch.cat([h, env_ctx, soc_ctx], dim=-1))
            preds.append(out.unsqueeze(1))
            if target is not None and torch.rand(1).item() < tf_ratio:
                prev_xy = target[:, t, :]
            else:
                prev_xy = out
        return torch.cat(preds, dim=1)

    def predict(self, hist, env_sdf=None, social_feat=None, social_mask=None, **kw):
        B, dev = hist.shape[0], hist.device
        if env_sdf is None:
            env_sdf = torch.zeros(B, 2, 128, 128, device=dev, dtype=torch.float32)
        if social_feat is None:
            social_feat = torch.zeros(B, self.MAX_NB, self.SOC_FEAT, device=dev)
        if social_mask is None:
            social_mask = torch.zeros(B, self.MAX_NB, device=dev)
        with torch.no_grad():
            return self.forward(hist, env_sdf, social_feat, social_mask,
                                target=None, tf_ratio=0.0)
