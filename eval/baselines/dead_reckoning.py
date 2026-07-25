"""Dead Reckoning (DR) baseline.

Uses the last reported SOG and COG (from AIS) to project forward.
This is the domain-specific maritime baseline — it uses the exact
reported velocity from the AIS message at the end of the history window.
No training required.
"""

import math
import numpy as np


KNOTS_TO_MPS = 0.514444


def predict(
    hist_xy:  np.ndarray,   # (N, 30, 2) local metres
    hist_sog: np.ndarray,   # (N, 30) knots
    hist_cog_sin: np.ndarray,  # (N, 30)
    hist_cog_cos: np.ndarray,  # (N, 30)
    future_steps: int = 30,
    smooth_window: int = 3,
) -> np.ndarray:
    """
    Returns:
        pred: (N, future_steps, 2) in local metres
    """
    dt = 20.0
    N = hist_xy.shape[0]
    w = min(smooth_window, hist_sog.shape[1])

    # Smooth SOG and COG over trailing window to reduce noise
    sog_kn  = hist_sog[:, -w:].mean(axis=1)           # (N,) knots
    cog_sin = hist_cog_sin[:, -w:].mean(axis=1)        # (N,)
    cog_cos = hist_cog_cos[:, -w:].mean(axis=1)        # (N,)

    # Speed in m/s
    speed_mps = sog_kn * KNOTS_TO_MPS                  # (N,)

    # Normalize COG vector to unit direction
    cog_norm = np.sqrt(cog_sin ** 2 + cog_cos ** 2).clip(min=1e-6)
    cog_sin_n = cog_sin / cog_norm
    cog_cos_n = cog_cos / cog_norm

    # COG is measured clockwise from North; convert to local XY bearing
    # In local metres: x = East, y = North
    # COG angle θ: x-component = sin(θ), y-component = cos(θ)
    vx = speed_mps * cog_sin_n    # (N,)  East component
    vy = speed_mps * cog_cos_n    # (N,)  North component

    # Extrapolate from last history position
    t    = np.arange(1, future_steps + 1, dtype=np.float32) * dt   # (T,)
    x0   = hist_xy[:, -1, 0:1]   # (N, 1)
    y0   = hist_xy[:, -1, 1:2]   # (N, 1)
    pred_x = x0 + vx[:, None] * t[None, :]   # (N, T)
    pred_y = y0 + vy[:, None] * t[None, :]   # (N, T)

    return np.stack([pred_x, pred_y], axis=-1)   # (N, T, 2)


class DeadReckoning:
    name = "Dead Reckoning"

    def __init__(self, smooth_window: int = 3):
        self.smooth_window = smooth_window

    def __call__(self, data: dict) -> np.ndarray:
        hist = data["hist"]   # (N, 30, 6): [x, y, vx, vy, sog, tod_sin]
        # Reconstruct COG sin/cos from velocity
        vx = hist[:, :, 2]   # (N, 30)
        vy = hist[:, :, 3]
        speed = np.sqrt(vx**2 + vy**2).clip(min=1e-6)
        cog_sin = vx / speed
        cog_cos = vy / speed
        sog = hist[:, :, 4]   # knots

        return predict(
            hist[:, :, :2], sog, cog_sin, cog_cos,
            smooth_window=self.smooth_window,
        )
