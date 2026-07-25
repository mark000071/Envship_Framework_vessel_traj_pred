"""LSTM Seq2Seq baseline for ship trajectory prediction.

A simple encoder-decoder LSTM trained on the Standard Track train split.
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
# Model definition
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        return self.out(out), h, c


class LSTMSeq2Seq(nn.Module):
    """Encoder-decoder LSTM for trajectory prediction."""

    def __init__(
        self,
        input_dim:  int   = 6,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        output_dim: int   = 2,
        future_steps: int = 30,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_dim, output_dim, num_layers, dropout)

    def forward(self, hist: torch.Tensor, teacher_forcing_ratio: float = 0.5,
                target: torch.Tensor | None = None) -> torch.Tensor:
        """
        hist:   (B, T_hist, input_dim)
        target: (B, T_fut, 2)   — used for teacher forcing during training
        Returns: (B, T_fut, 2)
        """
        B = hist.size(0)
        h, c = self.encoder(hist)

        # Start token = last history position (should be ~0,0 in local frame)
        dec_input = hist[:, -1:, :2]     # (B, 1, 2)
        outputs   = []
        for t in range(self.future_steps):
            out, h, c = self.decoder(dec_input, h, c)
            outputs.append(out)
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = target[:, t:t+1, :]
            else:
                dec_input = out
        return torch.cat(outputs, dim=1)   # (B, T_fut, 2)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _normalise(hist: np.ndarray, future: np.ndarray):
    """Normalise by per-sample max absolute history displacement."""
    # hist: (N, 30, 6), future: (N, 30, 2)
    # Scale by max displacement in history
    disp = np.abs(hist[:, :, :2]).max(axis=(1, 2), keepdims=True)  # (N, 1, 1)
    disp = np.where(disp < 1.0, 1.0, disp)                         # avoid div by 0
    hist_n         = hist.copy()
    hist_n[:,:,:2] = hist[:,:,:2] / disp
    hist_n[:,:,2]  = hist[:,:,2]  / (disp[:, :, 0] / 20.0)   # vx scaled
    hist_n[:,:,3]  = hist[:,:,3]  / (disp[:, :, 0] / 20.0)   # vy scaled
    future_n = future / disp
    return hist_n.astype(np.float32), future_n.astype(np.float32), disp[:, 0, 0]


def train(
    track_root: str | Path,
    save_path:  str | Path,
    hidden_dim: int   = 256,
    num_layers: int   = 2,
    epochs:     int   = 15,
    batch_size: int   = 512,
    lr:         float = 3e-4,
    device:     str   = "auto",
) -> LSTMSeq2Seq:
    assert _TORCH_AVAILABLE, "PyTorch required for LSTM training"
    from ..dataset import load_split

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LSTM] training on {device}")

    print("[LSTM] loading train split...")
    data = load_split(track_root, "train")
    hist_n, fut_n, _ = _normalise(data["hist"], data["future"])

    ds = TensorDataset(torch.from_numpy(hist_n), torch.from_numpy(fut_n))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))

    model = LSTMSeq2Seq(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for hist_b, fut_b in dl:
            hist_b = hist_b.to(device)
            fut_b  = fut_b.to(device)
            opt.zero_grad()
            tf_ratio = max(0.0, 0.5 * (1.0 - epoch / epochs))
            pred = model(hist_b, teacher_forcing_ratio=tf_ratio, target=fut_b)
            loss = loss_fn(pred, fut_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg = total_loss / len(dl)
        print(f"[LSTM] epoch {epoch:02d}/{epochs}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}  t={time.time()-t0:.0f}s")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "hidden_dim": hidden_dim, "num_layers": num_layers}, save_path)
    print(f"[LSTM] saved to {save_path}")
    return model


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class LSTMBaseline:
    name = "LSTM Seq2Seq"

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        assert _TORCH_AVAILABLE, "PyTorch required"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = LSTMSeq2Seq(
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def __call__(self, data: dict) -> np.ndarray:
        hist = data["hist"].astype(np.float32)   # (N, 30, 6)
        fut  = data["future"].astype(np.float32) # (N, 30, 2)
        hist_n, _, scales = _normalise(hist, fut)
        with torch.no_grad():
            pred_n = self.model(
                torch.from_numpy(hist_n).to(self.device),
                teacher_forcing_ratio=0.0
            ).cpu().numpy()                       # (N, 30, 2)
        # Denormalise
        return pred_n * scales[:, None, None]
