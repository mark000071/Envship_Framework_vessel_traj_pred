"""Constant Velocity (CV) baseline.

Estimates velocity from the last few history steps and extrapolates linearly.
No training required.
"""

import numpy as np


def predict(hist_xy: np.ndarray, future_steps: int = 30, window: int = 5) -> np.ndarray:
    """
    Args:
        hist_xy:      (N, T_hist, 2) float32 — history positions in local metres
        future_steps: number of steps to predict
        window:       number of trailing history steps used for velocity estimate

    Returns:
        pred: (N, future_steps, 2) float32
    """
    N = hist_xy.shape[0]
    dt = 20.0  # seconds per step

    # Mean velocity over the trailing window
    # v[i] ≈ (x[T-1] - x[T-1-window]) / (window * dt)
    w = min(window, hist_xy.shape[1] - 1)
    vx = (hist_xy[:, -1, 0] - hist_xy[:, -1 - w, 0]) / (w * dt)  # (N,)
    vy = (hist_xy[:, -1, 1] - hist_xy[:, -1 - w, 1]) / (w * dt)

    # Extrapolate: x0 = hist_xy[:, -1, :] ≈ (0, 0) in the local frame
    t = np.arange(1, future_steps + 1, dtype=np.float32) * dt   # (T,)
    pred_x = hist_xy[:, -1:, 0] + vx[:, None] * t[None, :]      # (N, T)
    pred_y = hist_xy[:, -1:, 1] + vy[:, None] * t[None, :]      # (N, T)

    return np.stack([pred_x, pred_y], axis=-1)                   # (N, T, 2)


class ConstantVelocity:
    name = "Constant Velocity"

    def __init__(self, window: int = 5):
        self.window = window

    def __call__(self, data: dict) -> np.ndarray:
        return predict(data["hist"][:, :, :2], window=self.window)
