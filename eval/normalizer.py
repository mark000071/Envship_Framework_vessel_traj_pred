"""Global mean-std normalizer for trajectory data.

Fitted on the training split; applied to val/test.
Normalizes each of the 6 input channels independently.
Future positions (x, y) use the same stats as channels 0 and 1 of hist.
"""

from __future__ import annotations
import json, numpy as np
from pathlib import Path


class GlobalNormalizer:
    """Per-channel mean/std normalizer fitted on training data."""

    CHANNELS = ["x_m", "y_m", "vx_mps", "vy_mps", "sog_kn", "tod_sin"]

    def __init__(self):
        self.mean: np.ndarray | None = None   # (6,)
        self.std:  np.ndarray | None = None   # (6,)

    def fit(self, hist: np.ndarray) -> "GlobalNormalizer":
        """hist: (N, T, 6) float32"""
        flat = hist.reshape(-1, hist.shape[-1])   # (N*T, 6)
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std  = flat.std(axis=0).astype(np.float32)
        self.std  = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform_hist(self, hist: np.ndarray) -> np.ndarray:
        return ((hist - self.mean) / self.std).astype(np.float32)

    def transform_future(self, future: np.ndarray) -> np.ndarray:
        """future: (N, T, 2) — only x and y channels."""
        return ((future - self.mean[:2]) / self.std[:2]).astype(np.float32)

    def inverse_future(self, pred_norm: np.ndarray) -> np.ndarray:
        """Undo normalization on predicted future (N, T, 2)."""
        return (pred_norm * self.std[:2] + self.mean[:2]).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "GlobalNormalizer":
        d = json.loads(Path(path).read_text())
        n = cls()
        n.mean = np.array(d["mean"], dtype=np.float32)
        n.std  = np.array(d["std"],  dtype=np.float32)
        return n
