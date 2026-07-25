"""
V3 sequence models — BiLSTM encoder with per-step cross-attention.

Why encoder cross-attention beats plain (h_n, c_n):
  TCN achieves ADE=87.9m because dilated convolutions process all 30 history
  steps at every scale simultaneously. LSTM v2 only carries history through
  (h_n, c_n) which degrades over 30 decoder steps.

  Here, at every decoder step, h[-1] cross-attends to all BiLSTM encoder
  output states — equivalent to "looking back at the full history at each
  prediction step". This closes the gap with TCN.

Architecture:
  Encoder : BiLSTM-2L → enc_out (B, 30, 2H) + projected (h, c)
  Decoder : LSTM-2L with input [x, y, dx, dy, enc_ctx]  (4 + CTX_DIM)
  enc_ctx : per-step multi-head cross-attention over enc_out
"""

from __future__ import annotations
import torch
import torch.nn as nn

HIST_DIM  = 6
FUT_STEPS = 30
OUT_DIM   = 2
DEC_IN    = 4       # [x, y, dx, dy]
HIDDEN    = 256
CTX_DIM   = 64      # encoder cross-attention output dim
ENC_DIM   = HIDDEN * 2   # BiLSTM enc_out dim


# ---------------------------------------------------------------------------
# Encoder cross-attention
# ---------------------------------------------------------------------------

class _EncoderCrossAttn(nn.Module):
    """
    Lightweight cross-attention: decoder h[-1] queries BiLSTM encoder outputs.

    query : linear(h_last)       (B, 1, CTX_DIM)
    key/v : linear(enc_memory)   (B, T, CTX_DIM)
    output: (B, CTX_DIM)
    """
    def __init__(self, hidden: int = HIDDEN, enc_dim: int = ENC_DIM,
                 ctx_dim: int = CTX_DIM, n_heads: int = 4):
        super().__init__()
        self.q_proj  = nn.Linear(hidden,  ctx_dim)
        self.kv_proj = nn.Linear(enc_dim, ctx_dim)
        self.mha     = nn.MultiheadAttention(ctx_dim, n_heads, dropout=0.1,
                                              batch_first=True)
        self.norm    = nn.LayerNorm(ctx_dim)

    def forward(self, h_last: torch.Tensor,
                enc_memory: torch.Tensor) -> torch.Tensor:
        """
        h_last    : (B, hidden)
        enc_memory: (B, T, enc_dim)
        Returns   : (B, ctx_dim)
        """
        q      = self.q_proj(h_last).unsqueeze(1)   # (B, 1, ctx_dim)
        kv     = self.kv_proj(enc_memory)            # (B, T, ctx_dim)
        ctx, _ = self.mha(q, kv, kv)
        return self.norm(ctx.squeeze(1))             # (B, ctx_dim)


# ---------------------------------------------------------------------------
# Velocity-augmented cross-attention decoder
# ---------------------------------------------------------------------------

class _Vel4CrossDecoder(nn.Module):
    """
    LSTM decoder: input = [x, y, dx, dy, enc_ctx].
    Exposes step_raw() + project() so context models can insert FiLM between.
    """
    def __init__(self, hidden: int, n_layers: int, dropout: float,
                 ctx_dim: int = CTX_DIM):
        super().__init__()
        self.rnn  = nn.LSTM(DEC_IN + ctx_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.)
        self.proj = nn.Linear(hidden, OUT_DIM)

    def forward(self, tok: torch.Tensor, ctx: torch.Tensor, h, c):
        """Simple path: no FiLM insertion."""
        inp = torch.cat([tok, ctx.unsqueeze(1)], dim=-1)  # (B,1,4+ctx)
        out, (h, c) = self.rnn(inp, (h, c))
        return self.proj(out), h, c                        # (B,1,2), h, c

    def step_raw(self, tok: torch.Tensor, ctx: torch.Tensor, h, c):
        """For FiLM insertion between LSTM and projection."""
        inp = torch.cat([tok, ctx.unsqueeze(1)], dim=-1)
        raw, (h, c) = self.rnn(inp, (h, c))
        return raw, h, c   # (B,1,H), h, c

    def project(self, raw: torch.Tensor) -> torch.Tensor:
        return self.proj(raw)  # (B,1,2)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _make_start_token(hist: torch.Tensor) -> torch.Tensor:
    """(B,1,4) = [last_pos, last_vel] from history."""
    pos = hist[:, -1:, :2]
    vel = hist[:, -1:, :2] - hist[:, -2:-1, :2]
    return torch.cat([pos, vel], dim=-1)


def _merge_bidir(h, c, B: int, H: int,
                 h_proj: nn.Linear, c_proj: nn.Linear):
    """Merge 2-layer bidirectional (h,c) to 2-layer unidirectional."""
    hm = h.view(2, 2, B, H).permute(0, 2, 1, 3).contiguous().view(2 * B, 2 * H)
    cm = c.view(2, 2, B, H).permute(0, 2, 1, 3).contiguous().view(2 * B, 2 * H)
    return (h_proj(hm).view(2, B, -1),
            c_proj(cm).view(2, B, -1))


def _run_vel4_cross_decoder(decoder: _Vel4CrossDecoder,
                             cross_attn: _EncoderCrossAttn,
                             start_tok: torch.Tensor,
                             h, c,
                             enc_memory: torch.Tensor,
                             T: int,
                             target, tf_ratio: float) -> torch.Tensor:
    """
    Velocity-augmented decoder loop with encoder cross-attention.
    Uses decoder.forward() — no FiLM insertion.
    """
    tok      = start_tok          # (B,1,4)
    prev_pos = tok[:, :, :2]
    preds    = []

    for t in range(T):
        ctx            = cross_attn(h[-1], enc_memory)   # (B, ctx_dim)
        pred_pos, h, c = decoder(tok, ctx, h, c)         # (B,1,2)
        preds.append(pred_pos)

        if target is not None and torch.rand(1).item() < tf_ratio:
            next_pos = target[:, t:t+1, :]
        else:
            next_pos = pred_pos

        vel      = next_pos - prev_pos
        tok      = torch.cat([next_pos, vel], dim=-1)    # (B,1,4)
        prev_pos = next_pos

    return torch.cat(preds, dim=1)  # (B,T,2)


# ---------------------------------------------------------------------------
# BiLSTM-2L v3
# ---------------------------------------------------------------------------

class BiLSTM2L_v3(nn.Module):
    """BiLSTM encoder + per-step encoder cross-attention + vel4 decoder."""
    name = "BiLSTM-2L-v3"

    def __init__(self, hidden: int = HIDDEN, dropout: float = 0.1,
                 ctx_dim: int = CTX_DIM):
        super().__init__()
        self.encoder    = nn.LSTM(HIST_DIM, hidden, 2, batch_first=True,
                                   bidirectional=True, dropout=dropout)
        self.h_proj     = nn.Linear(hidden * 2, hidden)
        self.c_proj     = nn.Linear(hidden * 2, hidden)
        self.cross_attn = _EncoderCrossAttn(hidden, hidden * 2, ctx_dim)
        self.decoder    = _Vel4CrossDecoder(hidden, 2, dropout, ctx_dim)

    def forward(self, hist, target=None, tf_ratio: float = 0.5):
        B = hist.size(0)
        H = self.encoder.hidden_size
        enc_out, (h, c) = self.encoder(hist)              # (B,30,2H)
        h2, c2 = _merge_bidir(h, c, B, H, self.h_proj, self.c_proj)
        start  = _make_start_token(hist)
        return _run_vel4_cross_decoder(
            self.decoder, self.cross_attn, start, h2, c2, enc_out,
            FUT_STEPS, target, tf_ratio)

    def predict(self, hist):
        with torch.no_grad():
            return self.forward(hist, target=None, tf_ratio=0.)
