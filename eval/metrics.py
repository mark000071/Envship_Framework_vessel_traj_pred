"""Standard trajectory prediction metrics for EnvShip-Bench.

All positions in local metres (same coordinate frame as the benchmark).
Shapes: pred / gt are (N, T, 2) where T=30, last dim is (x, y).
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Sequence


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def displacement_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-sample per-step L2 displacement. Returns (N, T)."""
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1))


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error over all steps and all samples (metres)."""
    return float(displacement_error(pred, gt).mean())


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error at the last prediction step (metres)."""
    return float(displacement_error(pred, gt)[:, -1].mean())


def ade_at_step(pred: np.ndarray, gt: np.ndarray, step: int) -> float:
    """ADE evaluated only up to `step` (1-indexed, so step=9 means 3 min)."""
    return float(displacement_error(pred, gt)[:, :step].mean())


def min_ade(preds: np.ndarray, gt: np.ndarray) -> float:
    """minADE_K for probabilistic models. preds: (N, K, T, 2), gt: (N, T, 2)."""
    gt_exp  = gt[:, None, :, :]                          # (N, 1, T, 2)
    errors  = displacement_error(preds, gt_exp)           # (N, K, T)
    sample_ade = errors.mean(axis=-1)                     # (N, K)
    return float(sample_ade.min(axis=-1).mean())          # scalar


def min_fde(preds: np.ndarray, gt: np.ndarray) -> float:
    """minFDE_K. preds: (N, K, T, 2), gt: (N, T, 2)."""
    gt_exp  = gt[:, None, :, :]
    errors  = displacement_error(preds, gt_exp)           # (N, K, T)
    final   = errors[:, :, -1]                            # (N, K)
    return float(final.min(axis=-1).mean())


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    pred:     np.ndarray,          # (N, T, 2)
    gt:       np.ndarray,          # (N, T, 2)
    metadata: dict[str, Sequence], # optional per-sample metadata arrays
) -> dict[str, object]:
    """Compute ADE / FDE and per-stratum breakdowns."""
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    N, T, _ = pred.shape

    err = displacement_error(pred, gt)  # (N, T)

    results: dict[str, object] = {
        "n_samples": N,
        "T":         T,
        "ADE":       round(float(err.mean()),       3),
        "FDE":       round(float(err[:, -1].mean()),3),
        "ADE_3min":  round(float(err[:, :9].mean()),  3),   # 3 min = steps 1-9
        "ADE_6min":  round(float(err[:, :18].mean()),  3),  # 6 min = steps 1-18
        "ADE_10min": round(float(err[:, :30].mean()),  3),  # 10 min = all 30 steps
    }

    # Per-difficulty breakdown
    if "difficulty_tier" in metadata:
        tiers = np.asarray(metadata["difficulty_tier"])
        for tier in ("easy", "medium", "hard"):
            mask = tiers == tier
            if mask.sum() > 0:
                results[f"ADE_{tier}"] = round(float(err[mask].mean()), 3)
                results[f"FDE_{tier}"] = round(float(err[mask, -1].mean()), 3)
                results[f"n_{tier}"]   = int(mask.sum())

    # Per-vessel-type breakdown
    if "ship_group" in metadata:
        groups = np.asarray(metadata["ship_group"])
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 0:
                key = str(g).replace("/", "_")
                results[f"ADE_{key}"]  = round(float(err[mask].mean()), 3)
                results[f"n_{key}"]    = int(mask.sum())

    # Per-scene breakdown
    if "scene_type" in metadata:
        scenes = np.asarray(metadata["scene_type"])
        for sc in np.unique(scenes):
            mask = scenes == sc
            if mask.sum() > 0:
                results[f"ADE_{sc}"]  = round(float(err[mask].mean()), 3)
                results[f"n_{sc}"]    = int(mask.sum())

    # Per-source breakdown (for cross-domain pooled evaluation)
    if "source" in metadata:
        sources = np.asarray(metadata["source"])
        for src in np.unique(sources):
            mask = sources == src
            if mask.sum() > 0:
                results[f"ADE_src_{src}"] = round(float(err[mask].mean()), 3)
                results[f"FDE_src_{src}"] = round(float(err[mask, -1].mean()), 3)
                results[f"n_src_{src}"]   = int(mask.sum())

    return results


def format_results_table(results_by_model: dict[str, dict]) -> str:
    """Format a Markdown comparison table of model results."""
    models = list(results_by_model.keys())
    key_cols = ["ADE", "FDE", "ADE_3min", "ADE_6min",
                "ADE_easy", "ADE_medium", "ADE_hard",
                "ADE_cargo_tanker", "ADE_fishing", "ADE_sailing_leisure",
                "ADE_open_water", "ADE_nearshore", "ADE_harbor"]

    # Build table
    present_cols = [c for c in key_cols
                    if any(c in results_by_model[m] for m in models)]

    header  = "| Model | " + " | ".join(present_cols) + " |"
    sep     = "|" + "|".join([":---:"] * (len(present_cols) + 1)) + "|"
    rows    = [header, sep]
    for name, res in results_by_model.items():
        vals = [str(res.get(c, "-")) for c in present_cols]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(rows)
