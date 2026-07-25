"""
V2 context-aware trajectory prediction models.

Two key architectural upgrades over v1:

1. Per-step social cross-attention
   v1: social features injected once into decoder h0.
   v2: at every decoder step, the decoder hidden state queries neighbor
       embeddings via multi-head attention.  Context is fresh every step.

2. Per-step environment FiLM (Feature-wise Linear Modulation)
   v1: env descriptor injected once into h0.
   v2: env descriptor produces (γ, β) that linearly modulate the LSTM
       hidden output at every step.  Scene context persists through all
       30 prediction steps.

Both upgrades use the V2 displacement decoder from sequence_models_v2.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequence_models_v2 import HIDDEN, HIST_DIM, FUT_STEPS, OUT_DIM


class _DispDecoder(nn.Module):
    """Variable-input-dim LSTM decoder used by context models.
    Input dim = displacement(2) + optional social_ctx(64).
    Uses .step() + .project() so FiLM can be inserted between them.
    """
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.rnn  = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.)
        self.proj = nn.Linear(hidden, OUT_DIM)

    def step(self, x, h, c):
        raw, (h, c) = self.rnn(x, (h, c))
        return raw, h, c          # raw: (B, 1, H)

    def project(self, raw):
        return self.proj(raw)     # (B, 1, 2)


def _compute_target_disp(hist_n: torch.Tensor,
                          future_n: torch.Tensor) -> torch.Tensor:
    """Step-wise displacement targets for teacher forcing.
    Returns (B, T, 2): future_n[t] - future_n[t-1]  (first step uses last hist pos).
    """
    prev = torch.cat([hist_n[:, -1:, :2], future_n[:, :-1, :2]], dim=1)
    return future_n[:, :, :2] - prev

MAX_NEIGHBORS = 10
SOCIAL_FEAT   = 5       # [rel_x, rel_y, rel_vx, rel_vy, dist]
ENV_DESC_DIM  = 14
SOC_ENC_DIM   = 64      # dimension for neighbour embeddings / attention
ENV_ENC_DIM   = 64


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None):
        super().__init__()
        h = hidden or out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _LSTMEncoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(HIST_DIM, hidden, 2, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


# ---------------------------------------------------------------------------
# Per-step social cross-attention
# ---------------------------------------------------------------------------

class _SocialCrossAttn(nn.Module):
    """
    At each decoder step:
      query  = linear(h_last)              shape (B, 1, d_k)
      key, value = neigh_enc(social_feat)  shape (B, N, d_k)
      output = MHA(query, K, V)            shape (B, d_k)

    Masked for zero-padded neighbours (social_mask == 0).
    """
    def __init__(self, hidden_dim: int = HIDDEN,
                 soc_enc_dim: int = SOC_ENC_DIM, n_heads: int = 4):
        super().__init__()
        self.neigh_enc = _MLP(SOCIAL_FEAT, soc_enc_dim, soc_enc_dim * 2)
        self.q_proj    = nn.Linear(hidden_dim, soc_enc_dim)
        self.mha       = nn.MultiheadAttention(soc_enc_dim, n_heads,
                                                dropout=0.1, batch_first=True)
        self.norm      = nn.LayerNorm(soc_enc_dim)

    def encode_kv(self, social_feat: torch.Tensor) -> torch.Tensor:
        """Pre-compute key/value encodings (done once per sample).
        social_feat: (B, N, 5) -> (B, N, soc_enc_dim)
        """
        B, N, _ = social_feat.shape
        return self.neigh_enc(social_feat.view(B * N, -1)).view(B, N, -1)

    def attend(self, h_last: torch.Tensor,
               kv: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        """
        h_last : (B, hidden_dim)    — last-layer decoder hidden state
        kv     : (B, N, soc_enc_dim)
        mask   : (B, N) float 0/1  — 1 = valid neighbour
        Returns: (B, soc_enc_dim)

        Handles the all-masked case robustly: PyTorch MHA produces NaN
        when key_padding_mask is all-True for a sample (empty attention).
        We unmask the first slot for those samples and zero the output.
        """
        q   = self.q_proj(h_last).unsqueeze(1)   # (B, 1, d_k)
        kpm = (mask < 0.5)                        # (B, N) True = ignore

        # Per-sample guard: if a sample has no valid neighbour, temporarily
        # unmask slot 0 so softmax doesn't see all -inf → NaN.
        all_masked = kpm.all(dim=1)               # (B,)
        if all_masked.any():
            kpm_safe = kpm.clone()
            kpm_safe[all_masked, 0] = False       # unmask one slot
        else:
            kpm_safe = kpm

        ctx, _ = self.mha(q, kv, kv, key_padding_mask=kpm_safe)
        ctx = ctx.squeeze(1)                      # (B, d_k)

        # Zero out samples that had no real neighbours (their "context" is noise)
        if all_masked.any():
            ctx[all_masked] = 0.0

        return self.norm(ctx)                     # (B, soc_enc_dim)


# ---------------------------------------------------------------------------
# Per-step environment FiLM
# ---------------------------------------------------------------------------

class _EnvFiLM(nn.Module):
    """
    Produces (γ, β) from env_desc, applies h_new = (1 + γ) * h + β.
    Applied to the LSTM raw output (before projection to displacement).
    """
    def __init__(self, env_dim: int = ENV_DESC_DIM, hidden_dim: int = HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, raw: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """
        raw : (B, 1, hidden_dim) — LSTM output before proj
        env : (B, env_dim)
        """
        gb    = self.mlp(env)                        # (B, 2*H)
        gamma, beta = gb.chunk(2, dim=-1)            # each (B, H)
        return (1 + gamma.unsqueeze(1)) * raw + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Context decoder loop
# ---------------------------------------------------------------------------

def _run_ctx_decoder(decoder: _DispDecoder,
                      start_disp: torch.Tensor,
                      h: torch.Tensor, c: torch.Tensor,
                      T: int,
                      tgt_disp,
                      tf_ratio: float,
                      last_pos: torch.Tensor,
                      social_attn: _SocialCrossAttn | None = None,
                      social_kv: torch.Tensor | None = None,
                      social_mask: torch.Tensor | None = None,
                      film: _EnvFiLM | None = None,
                      env_emb: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-step context decoder loop.

    start_disp : (B, 1, 2)  — pure displacement token (NEVER pre-augmented).
    The loop concatenates social context at every step (including step 0).
    FiLM is applied to LSTM output if env context is present.
    """
    tok  = start_disp  # (B, 1, 2) — always pure 2-D displacement
    cur  = last_pos    # (B, 1, 2) — running absolute position
    preds: list[torch.Tensor] = []

    for t in range(T):
        # ── Build augmented input ─────────────────────────────────────────
        inp_parts = [tok]                              # (B, 1, 2)
        if social_attn is not None and social_kv is not None:
            soc_ctx = social_attn.attend(
                h[-1], social_kv, social_mask)         # (B, soc_enc_dim)
            inp_parts.append(soc_ctx.unsqueeze(1))     # (B, 1, 64)
        inp = torch.cat(inp_parts, dim=-1)             # (B, 1, 2 or 66)

        # ── LSTM step ─────────────────────────────────────────────────────
        raw, h, c = decoder.step(inp, h, c)            # raw: (B, 1, H)

        # ── FiLM env conditioning ─────────────────────────────────────────
        if film is not None and env_emb is not None:
            raw = film(raw, env_emb)

        disp = decoder.project(raw)                    # (B, 1, 2)
        cur  = cur + disp
        preds.append(cur)

        # ── Teacher forcing — next token is always 2-D displacement ───────
        if tgt_disp is not None and torch.rand(1).item() < tf_ratio:
            tok = tgt_disp[:, t:t+1, :]               # true disp (2-D)
        else:
            tok = disp                                 # predicted disp (2-D)

    return torch.cat(preds, dim=1)   # (B, T, 2)


# ---------------------------------------------------------------------------
# LSTM + Social Attention  v2
# ---------------------------------------------------------------------------

class LSTMSocialAttnV2(nn.Module):
    """Per-step social cross-attention + displacement decoder."""
    name = "LSTM+Social-Attn-v2"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.encoder      = _LSTMEncoder(hidden, dropout)
        self.social_attn  = _SocialCrossAttn(hidden, SOC_ENC_DIM)
        # Decoder input = displacement(2) + social_ctx(64)
        self.decoder      = _DispDecoder(OUT_DIM + SOC_ENC_DIM, hidden, 2, dropout)

    def forward(self, hist, social_feat, social_mask,
                target=None, tf_ratio: float = 0.5):
        h, c  = self.encoder(hist)
        kv    = self.social_attn.encode_kv(social_feat)   # (B, N, 64)
        start = hist[:, -1:, :2] - hist[:, -2:-1, :2]    # (B, 1, 2) pure disp
        lpos  = hist[:, -1:, :2]
        td    = _compute_target_disp(hist, target) if target is not None else None

        return _run_ctx_decoder(
            self.decoder, start, h, c, FUT_STEPS, td, tf_ratio, lpos,
            social_attn=self.social_attn, social_kv=kv,
            social_mask=social_mask)

    def predict(self, hist, social_feat, social_mask):
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask,
                                 target=None, tf_ratio=0.)


# ---------------------------------------------------------------------------
# LSTM + Env Descriptor  v2  (FiLM per step)
# ---------------------------------------------------------------------------

class LSTMEnvDescV2(nn.Module):
    """Per-step FiLM env conditioning + displacement decoder."""
    name = "LSTM+Env-Desc-v2"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.encoder  = _LSTMEncoder(hidden, dropout)
        self.env_proj = _MLP(ENV_DESC_DIM, ENV_ENC_DIM)
        self.film     = _EnvFiLM(ENV_ENC_DIM, hidden)
        # Decoder input = displacement only (FiLM modulates hidden, not input)
        self.decoder  = _DispDecoder(OUT_DIM, hidden, 2, dropout)

    def forward(self, hist, env_desc, target=None, tf_ratio: float = 0.5):
        h, c     = self.encoder(hist)
        env_emb  = self.env_proj(env_desc)               # (B, 64)
        start    = hist[:, -1:, :2] - hist[:, -2:-1, :2]
        lpos     = hist[:, -1:, :2]
        td       = _compute_target_disp(hist, target) if target is not None else None
        return _run_ctx_decoder(
            self.decoder, start, h, c, FUT_STEPS, td, tf_ratio, lpos,
            film=self.film, env_emb=env_emb)

    def predict(self, hist, env_desc):
        with torch.no_grad():
            return self.forward(hist, env_desc, target=None, tf_ratio=0.)


# ---------------------------------------------------------------------------
# LSTM + Social + Env  v2
# ---------------------------------------------------------------------------

class LSTMSocialEnvV2(nn.Module):
    """Per-step social attention + per-step env FiLM + displacement decoder."""
    name = "LSTM+Social+Env-v2"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.encoder     = _LSTMEncoder(hidden, dropout)
        self.social_attn = _SocialCrossAttn(hidden, SOC_ENC_DIM)
        self.env_proj    = _MLP(ENV_DESC_DIM, ENV_ENC_DIM)
        self.film        = _EnvFiLM(ENV_ENC_DIM, hidden)
        self.decoder     = _DispDecoder(OUT_DIM + SOC_ENC_DIM, hidden, 2, dropout)

    def forward(self, hist, social_feat, social_mask, env_desc,
                target=None, tf_ratio: float = 0.5):
        h, c    = self.encoder(hist)
        kv      = self.social_attn.encode_kv(social_feat)
        env_emb = self.env_proj(env_desc)
        start   = hist[:, -1:, :2] - hist[:, -2:-1, :2]
        lpos    = hist[:, -1:, :2]
        td      = _compute_target_disp(hist, target) if target is not None else None

        return _run_ctx_decoder(
            self.decoder, start, h, c, FUT_STEPS, td, tf_ratio, lpos,
            social_attn=self.social_attn, social_kv=kv,
            social_mask=social_mask,
            film=self.film, env_emb=env_emb)

    def predict(self, hist, social_feat, social_mask, env_desc):
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask, env_desc,
                                 target=None, tf_ratio=0.)
