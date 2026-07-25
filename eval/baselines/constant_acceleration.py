"""Constant Acceleration (CA) baseline.

Fits a quadratic (constant acceleration) model on the trailing history steps.
No training required.
"""

import numpy as np


def predict(hist_xy: np.ndarray, future_steps: int = 30, window: int = 8) -> np.ndarray:
    """
    Args:
        hist_xy:      (N, T_hist, 2) — history positions in local metres
        future_steps: number of steps to predict
        window:       trailing steps used to fit position / velocity / acceleration

    Returns:
        pred: (N, future_steps, 2)
    """
    dt = 20.0
    w = min(window, hist_xy.shape[1])
    recent = hist_xy[:, -w:, :]          # (N, w, 2)
    N = hist_xy.shape[0]

    # Fit 2nd-order polynomial per axis using least squares
    t = np.arange(w, dtype=np.float64) * dt    # relative time within window
    # Design matrix: [1, t, t^2]
    A = np.stack([np.ones(w), t, 0.5 * t * t], axis=1)  # (w, 3)
    # Solve via pseudo-inverse (same A for all samples)
    A_pinv = np.linalg.pinv(A)                            # (3, w)

    # Coefficients: [x0, vx, ax] for x-axis; same for y
    cx = A_pinv @ recent[:, :, 0].T   # (3, N)
    cy = A_pinv @ recent[:, :, 1].T   # (3, N)

    # t_hist[-1] = (w-1)*dt corresponds to hist[-1] in the polynomial
    t_hist_end = (w - 1) * dt
    # Future times: one step beyond the last history point
    t_fut = np.arange(1, future_steps + 1, dtype=np.float64) * dt + t_hist_end  # (T,)
    A_fut     = np.stack([np.ones(future_steps), t_fut, 0.5 * t_fut * t_fut], axis=1)  # (T, 3)
    A_hist_end = np.array([1.0, t_hist_end, 0.5 * t_hist_end * t_hist_end])             # (3,)

    # Raw predictions from polynomial
    pred_x = (A_fut @ cx).T.astype(np.float32)   # (N, T)
    pred_y = (A_fut @ cy).T.astype(np.float32)

    # Correct so the prediction at t_hist_end matches hist[-1] exactly
    # (removes any constant offset from least-squares fit)
    poly_at_end_x = (A_hist_end @ cx).astype(np.float32)  # (N,)
    poly_at_end_y = (A_hist_end @ cy).astype(np.float32)
    offset_x = hist_xy[:, -1, 0].astype(np.float32) - poly_at_end_x  # (N,)
    offset_y = hist_xy[:, -1, 1].astype(np.float32) - poly_at_end_y
    pred_x = pred_x + offset_x[:, None]
    pred_y = pred_y + offset_y[:, None]

    return np.stack([pred_x, pred_y], axis=-1).astype(np.float32)


class ConstantAcceleration:
    name = "Constant Acceleration"

    def __init__(self, window: int = 8):
        self.window = window

    def __call__(self, data: dict) -> np.ndarray:
        return predict(data["hist"][:, :, :2], window=self.window)
