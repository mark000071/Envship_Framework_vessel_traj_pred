"""Transformer Seq2Seq baseline for ship trajectory prediction.

A Transformer encoder-decoder with positional encoding.
Input: 30-step history [x, y, vx, vy, sog, tod_sin]
Output: 30-step future [x, y]
"""

from __future__ import annotations

import math, time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrajectoryTransformer(nn.Module):
    """Encoder-decoder Transformer for trajectory prediction."""

    def __init__(
        self,
        input_dim:    int   = 6,
        d_model:      int   = 128,
        nhead:        int   = 4,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        dim_feedforward: int = 512,
        dropout:      float = 0.1,
        future_steps: int   = 30,
        output_dim:   int   = 2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.d_model = d_model

        self.src_embed = nn.Linear(input_dim, d_model)
        self.tgt_embed = nn.Linear(output_dim, d_model)
        self.src_pe    = SinusoidalPE(d_model)
        self.tgt_pe    = SinusoidalPE(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,          # (B, T_hist, input_dim)
        tgt_input: torch.Tensor,    # (B, T_fut, 2)
    ) -> torch.Tensor:
        B, T_fut, _ = tgt_input.shape
        src_e = self.src_pe(self.src_embed(src))        # (B, T_hist, d_model)
        tgt_e = self.tgt_pe(self.tgt_embed(tgt_input))  # (B, T_fut,  d_model)

        # Causal mask for decoder
        causal = nn.Transformer.generate_square_subsequent_mask(T_fut, device=src.device)
        out = self.transformer(src_e, tgt_e, tgt_mask=causal)
        return self.output_proj(out)   # (B, T_fut, 2)

    @torch.no_grad()
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """Inference: use last history position as all decoder inputs (non-autoregressive).

        This avoids the exposure-bias / error-accumulation issue of autoregressive
        decoding for models trained primarily with teacher forcing.
        The encoder captures global history context; the decoder attends to it
        conditioned on the last known position.
        """
        last = src[:, -1:, :2].expand(-1, self.future_steps, -1).clone()  # (B, T, 2)
        return self.forward(src, last)   # (B, T_fut, 2)


# ---------------------------------------------------------------------------
# Training + inference
# ---------------------------------------------------------------------------

def _normalise(hist, future):
    disp = np.abs(hist[:, :, :2]).max(axis=(1, 2), keepdims=True).clip(min=1.0)
    hist_n = hist.copy().astype(np.float32)
    hist_n[:,:,:2] /= disp
    hist_n[:,:,2]  /= (disp[:,:,0] / 20.0)
    hist_n[:,:,3]  /= (disp[:,:,0] / 20.0)
    return hist_n, (future / disp).astype(np.float32), disp[:, 0, 0]


def train(
    track_root:  str | Path,
    save_path:   str | Path,
    d_model:     int   = 128,
    nhead:       int   = 4,
    num_layers:  int   = 3,
    epochs:      int   = 20,
    batch_size:  int   = 512,
    lr:          float = 1e-3,
    warmup_epochs: int = 2,
    device:      str   = "auto",
) -> TrajectoryTransformer:
    assert _TORCH_AVAILABLE, "PyTorch required"
    from ..dataset import load_split

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Transformer] training on {device}")

    print("[Transformer] loading train split...")
    data = load_split(track_root, "train")
    hist_n, fut_n, _ = _normalise(data["hist"], data["future"])

    ds = TensorDataset(torch.from_numpy(hist_n), torch.from_numpy(fut_n))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))

    model = TrajectoryTransformer(d_model=d_model, nhead=nhead,
                                   num_enc_layers=num_layers, num_dec_layers=num_layers).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < warmup_epochs: return ep / max(1, warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss(delta=1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        t0 = time.time()
        for hist_b, fut_b in dl:
            hist_b, fut_b = hist_b.to(device), fut_b.to(device)
            opt.zero_grad()
            # Teacher forcing: shift future by one step
            tgt_in = torch.cat([hist_b[:, -1:, :2], fut_b[:, :-1, :]], dim=1)
            pred = model(hist_b, tgt_in)
            loss = loss_fn(pred, fut_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(dl)
        print(f"[Transformer] epoch {epoch:02d}/{epochs}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}  t={time.time()-t0:.0f}s")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers}, save_path)
    print(f"[Transformer] saved to {save_path}")
    return model


class TransformerBaseline:
    name = "Transformer Seq2Seq"

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        assert _TORCH_AVAILABLE, "PyTorch required"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = TrajectoryTransformer(
            d_model=ckpt["d_model"], nhead=ckpt["nhead"],
            num_enc_layers=ckpt["num_layers"], num_dec_layers=ckpt["num_layers"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def __call__(self, data: dict) -> np.ndarray:
        hist = data["hist"].astype(np.float32)
        fut  = data["future"].astype(np.float32)
        hist_n, _, scales = _normalise(hist, fut)
        with torch.no_grad():
            pred_n = self.model.predict(
                torch.from_numpy(hist_n).to(self.device)
            ).cpu().numpy()
        return pred_n * scales[:, None, None]
