"""
V3 context-aware trajectory prediction models.

Three improvements over v2:

1. Encoder cross-attention (from sequence_models_v3)
   Decoder queries ALL BiLSTM encoder states each step → full history access.

2. Velocity-augmented decoder (absolute position, no gradient amplification)
   All context models switch from _DispDecoder to _Vel4CrossDecoder.

3. Social context as gated FiLM instead of LSTM-input concatenation
   v2: zero-padding 64 social dims pollutes LSTM input for 80% no-neighbor
   v3: social FiLM with hard gate (zero ctx when no neighbors) → FiLM ≈
       identity for no-neighbor samples. Zero-init last layer ensures
       clean start.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .sequence_models_v3 import (
    _EncoderCrossAttn, _Vel4CrossDecoder, _make_start_token, _merge_bidir,
    HIDDEN, HIST_DIM, FUT_STEPS, OUT_DIM, CTX_DIM, ENC_DIM,
)

MAX_NEIGHBORS = 10
SOCIAL_FEAT   = 5
ENV_DESC_DIM  = 14
SOC_ENC_DIM   = 64
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


class _BiLSTMEncoder(nn.Module):
    """Shared BiLSTM encoder for all v3 context models."""
    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.lstm   = nn.LSTM(HIST_DIM, hidden, 2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        self.h_proj = nn.Linear(hidden * 2, hidden)
        self.c_proj = nn.Linear(hidden * 2, hidden)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        H = self.lstm.hidden_size
        enc_out, (h, c) = self.lstm(x)             # (B,30,2H)
        h2, c2 = _merge_bidir(h, c, B, H, self.h_proj, self.c_proj)
        return enc_out, h2, c2                     # (B,30,2H), (2,B,H), (2,B,H)


# ---------------------------------------------------------------------------
# Env FiLM (same logic as v2)
# ---------------------------------------------------------------------------

class _EnvFiLM(nn.Module):
    """h_out = (1+γ) * h + β,  [γ, β] = MLP(env_emb)."""
    def __init__(self, env_dim: int = ENV_ENC_DIM, hidden_dim: int = HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(env_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, raw: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        gb           = self.mlp(env)               # (B, 2H)
        gamma, beta  = gb.chunk(2, dim=-1)
        return (1 + gamma.unsqueeze(1)) * raw + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Social cross-attention (same guard as v2)
# ---------------------------------------------------------------------------

class _SocialCrossAttn(nn.Module):
    """Per-step social cross-attention with all-masked NaN guard."""
    def __init__(self, hidden_dim: int = HIDDEN,
                 soc_enc_dim: int = SOC_ENC_DIM, n_heads: int = 4):
        super().__init__()
        self.neigh_enc = _MLP(SOCIAL_FEAT, soc_enc_dim, soc_enc_dim * 2)
        self.q_proj    = nn.Linear(hidden_dim, soc_enc_dim)
        self.mha       = nn.MultiheadAttention(soc_enc_dim, n_heads,
                                                dropout=0.1, batch_first=True)
        self.norm      = nn.LayerNorm(soc_enc_dim)

    def encode_kv(self, social_feat: torch.Tensor) -> torch.Tensor:
        B, N, _ = social_feat.shape
        return self.neigh_enc(social_feat.view(B * N, -1)).view(B, N, -1)

    def attend(self, h_last: torch.Tensor,
               kv: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        q   = self.q_proj(h_last).unsqueeze(1)
        kpm = (mask < 0.5)
        all_masked = kpm.all(dim=1)
        if all_masked.any():
            kpm_safe = kpm.clone()
            kpm_safe[all_masked, 0] = False
        else:
            kpm_safe = kpm
        ctx, _ = self.mha(q, kv, kv, key_padding_mask=kpm_safe)
        ctx    = ctx.squeeze(1)
        if all_masked.any():
            ctx[all_masked] = 0.0
        return self.norm(ctx)   # (B, soc_enc_dim)


# ---------------------------------------------------------------------------
# Gated Social FiLM
# ---------------------------------------------------------------------------

class _SocialFiLM(nn.Module):
    """
    Social context as FiLM modulation with hard gate.

    When social_ctx = 0 (no neighbors, hard-gated):
      - zero-init last layer → γ=0, β=0 → FiLM = identity at start
      - gradient from no-neighbor samples does not update this module

    h_out = (1+γ) * h + β,  [γ, β] = MLP(social_ctx * neighbor_gate)
    """
    def __init__(self, soc_dim: int = SOC_ENC_DIM, hidden_dim: int = HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(soc_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # Zero-init last layer: identity FiLM when soc_ctx=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, raw: torch.Tensor, soc_ctx: torch.Tensor,
                social_mask: torch.Tensor) -> torch.Tensor:
        """
        raw       : (B, 1, H)       raw LSTM output
        soc_ctx   : (B, soc_enc_dim)
        social_mask: (B, N)  1=valid — used for hard gate
        """
        # Hard gate: zero social influence when no real neighbors
        has_nb   = (social_mask.sum(dim=1) > 0).float()   # (B,)
        gated    = soc_ctx * has_nb.unsqueeze(1)           # (B, soc_dim)
        gb       = self.mlp(gated)                         # (B, 2H)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1 + gamma.unsqueeze(1)) * raw + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# V3 context decoder loop
# ---------------------------------------------------------------------------

def _run_v3_ctx_decoder(decoder: _Vel4CrossDecoder,
                         enc_cross_attn: _EncoderCrossAttn,
                         start_tok: torch.Tensor,
                         h, c,
                         enc_memory: torch.Tensor,
                         T: int,
                         target, tf_ratio: float,
                         env_film: _EnvFiLM | None = None,
                         env_emb: torch.Tensor | None = None,
                         soc_attn: _SocialCrossAttn | None = None,
                         soc_kv: torch.Tensor | None = None,
                         soc_mask: torch.Tensor | None = None,
                         soc_film: _SocialFiLM | None = None,
                         ) -> torch.Tensor:
    """
    Velocity-augmented decoder loop with encoder cross-attn + optional FiLMs.

    Ordering of per-step operations:
      1. enc_cross_attn: query encoder memory → enc_ctx
      2. LSTM step: input = [x, y, dx, dy, enc_ctx]  → raw hidden
      3. env FiLM (if present): modulates raw with env descriptor
      4. social FiLM (if present): modulates raw with gated social ctx
      5. project raw → absolute (x, y)
    """
    tok      = start_tok         # (B,1,4)
    prev_pos = tok[:, :, :2]
    preds: list[torch.Tensor] = []

    for t in range(T):
        # 1. Encoder cross-attention
        enc_ctx = enc_cross_attn(h[-1], enc_memory)    # (B, CTX_DIM)

        # 2. LSTM step
        raw, h, c = decoder.step_raw(tok, enc_ctx, h, c)  # (B,1,H)

        # 3. Env FiLM
        if env_film is not None and env_emb is not None:
            raw = env_film(raw, env_emb)

        # 4. Social FiLM (gated)
        if (soc_attn is not None and soc_kv is not None
                and soc_mask is not None and soc_film is not None):
            soc_ctx = soc_attn.attend(h[-1], soc_kv, soc_mask)
            raw     = soc_film(raw, soc_ctx, soc_mask)

        # 5. Project
        pred_pos = decoder.project(raw)                # (B,1,2)
        preds.append(pred_pos)

        # Teacher forcing / free run
        if target is not None and torch.rand(1).item() < tf_ratio:
            next_pos = target[:, t:t+1, :]
        else:
            next_pos = pred_pos

        vel      = next_pos - prev_pos
        tok      = torch.cat([next_pos, vel], dim=-1)  # (B,1,4)
        prev_pos = next_pos

    return torch.cat(preds, dim=1)   # (B,T,2)


# ---------------------------------------------------------------------------
# LSTM + Env-Desc v3  (BiLSTM + enc_cross_attn + per-step env FiLM)
# ---------------------------------------------------------------------------

class LSTMEnvDescV3(nn.Module):
    """BiLSTM encoder-cross-attn + per-step env FiLM + vel4 decoder."""
    name = "LSTM+Env-Desc-v3"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1,
                 ctx_dim: int = CTX_DIM):
        super().__init__()
        self.encoder    = _BiLSTMEncoder(hidden, dropout)
        self.cross_attn = _EncoderCrossAttn(hidden, hidden * 2, ctx_dim)
        self.env_proj   = _MLP(ENV_DESC_DIM, ENV_ENC_DIM)
        self.env_film   = _EnvFiLM(ENV_ENC_DIM, hidden)
        self.decoder    = _Vel4CrossDecoder(hidden, 2, dropout, ctx_dim)

    def forward(self, hist, env_desc, target=None, tf_ratio: float = 0.5):
        enc_out, h, c = self.encoder(hist)
        env_emb = self.env_proj(env_desc)
        start   = _make_start_token(hist)
        return _run_v3_ctx_decoder(
            self.decoder, self.cross_attn, start, h, c, enc_out,
            FUT_STEPS, target, tf_ratio,
            env_film=self.env_film, env_emb=env_emb)

    def predict(self, hist, env_desc):
        with torch.no_grad():
            return self.forward(hist, env_desc, target=None, tf_ratio=0.)


# ---------------------------------------------------------------------------
# LSTM + Social + Env v3  (full context: enc_cross_attn + env FiLM + soc FiLM)
# ---------------------------------------------------------------------------

class LSTMSocialEnvV3(nn.Module):
    """BiLSTM enc-cross-attn + per-step env FiLM + gated social FiLM."""
    name = "LSTM+Social+Env-v3"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1,
                 ctx_dim: int = CTX_DIM):
        super().__init__()
        self.encoder     = _BiLSTMEncoder(hidden, dropout)
        self.cross_attn  = _EncoderCrossAttn(hidden, hidden * 2, ctx_dim)
        self.env_proj    = _MLP(ENV_DESC_DIM, ENV_ENC_DIM)
        self.env_film    = _EnvFiLM(ENV_ENC_DIM, hidden)
        self.social_attn = _SocialCrossAttn(hidden, SOC_ENC_DIM)
        self.soc_film    = _SocialFiLM(SOC_ENC_DIM, hidden)
        self.decoder     = _Vel4CrossDecoder(hidden, 2, dropout, ctx_dim)

    def forward(self, hist, social_feat, social_mask, env_desc,
                target=None, tf_ratio: float = 0.5):
        enc_out, h, c = self.encoder(hist)
        env_emb  = self.env_proj(env_desc)
        kv       = self.social_attn.encode_kv(social_feat)
        start    = _make_start_token(hist)
        return _run_v3_ctx_decoder(
            self.decoder, self.cross_attn, start, h, c, enc_out,
            FUT_STEPS, target, tf_ratio,
            env_film=self.env_film,  env_emb=env_emb,
            soc_attn=self.social_attn, soc_kv=kv, soc_mask=social_mask,
            soc_film=self.soc_film)

    def predict(self, hist, social_feat, social_mask, env_desc):
        with torch.no_grad():
            return self.forward(hist, social_feat, social_mask, env_desc,
                                 target=None, tf_ratio=0.)
