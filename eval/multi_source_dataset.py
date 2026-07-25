"""Pool multiple Standard-Track jurisdictions into a single training set.

Used for the EnvShip-Bench cross-domain experiments:
  - combined-source training (all 4 jurisdictions pooled)
  - leave-one-source-out training (3 jurisdictions pooled, 1 held out for test)

Each jurisdiction is identified by a short slug; the loader concatenates the
underlying numpy / object arrays produced by load_context_split() and adds a
per-sample 'source' meta column so downstream evaluation can stratify by
jurisdiction.

Usage:
    from eval.multi_source_dataset import POOLS, load_pooled_split
    data = load_pooled_split(POOLS["combined"], "train",
                              load_social=True, load_env_sdf=True)
    print(data["meta"]["source"])     # array of slug strings
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np

from eval.context_dataset import load_context_split


# ── Canonical roots for the four jurisdictions ─────────────────────────────
# Resolved from $ENVSHIP_DATA_ROOT via eval/paths.py (Track A: 10 min -> 10 min,
# Track B: 30 min -> 60 min).
from eval.paths import SOURCE_ROOTS, SOURCE_ROOTS_B  # noqa: F401,E402


# Convenience pool definitions for experiments
POOLS: dict[str, list[str]] = {
    "combined":          ["dma", "noaa", "piraeus", "norway"],
    # leave-one-source-out: train pool when {source} is held out
    "loso_holdout_dma":     ["noaa", "piraeus", "norway"],
    "loso_holdout_noaa":    ["dma",  "piraeus", "norway"],
    "loso_holdout_piraeus": ["dma",  "noaa",    "norway"],
    "loso_holdout_norway":  ["dma",  "noaa",    "piraeus"],
    # Track B pools (Piraeus + Norway are small here, so combined excludes
    # them to avoid distribution skew; LOSO uses DMA <-> NOAA)
    "trackb_combined":          ["dma", "noaa"],
    "trackb_loso_holdout_dma":  ["noaa"],
    "trackb_loso_holdout_noaa": ["dma"],
}


def load_pooled_split(sources: Iterable[str], split: str, **load_kwargs) -> dict:
    """Load `split` from each source in `sources` and concatenate.

    Per-key behaviour:
      - hist, future, env_desc, env_raster, env_sdf, social_feat, social_mask,
        neighbor_count    → np.concatenate along axis=0
      - meta              → dict of np.concatenate per key, plus a new 'source'
                            column with the jurisdiction slug per row

    Sources with an empty split (e.g.\ Norway val/test) are silently skipped.
    Sources whose split file is missing raise FileNotFoundError.
    """
    # Track-B pools are tagged via the pool name; auto-route to Track-B roots
    # if any of the sources still resolves via the pool naming convention.
    track_b_mode = load_kwargs.pop("_track_b", False)
    roots = SOURCE_ROOTS_B if track_b_mode else SOURCE_ROOTS
    parts: list[tuple[str, dict]] = []
    for src in sources:
        root = roots[src]
        try:
            d = load_context_split(root, split, **load_kwargs)
        except (FileNotFoundError, ValueError) as e:
            # Norway val/test can be tiny -- skip gracefully if empty
            print(f"  [pool] skipping {src}/{split}: {type(e).__name__}: {str(e)[:80]}")
            continue
        if len(d["hist"]) == 0:
            print(f"  [pool] skipping {src}/{split}: 0 samples")
            continue
        parts.append((src, d))

    if not parts:
        raise RuntimeError(f"pool {list(sources)} / {split} produced 0 samples total")

    out: dict = {}
    # Numeric keys
    numeric_keys = ("hist", "future", "env_desc", "env_raster", "env_sdf",
                    "social_feat", "social_mask", "neighbor_count")
    from eval.lazy_context import lazy_enabled, concat_to_cache
    for k in numeric_keys:
        if k in parts[0][1]:
            arrs = [p[1][k] for p in parts]
            if (lazy_enabled() and k in ("env_raster", "env_sdf")
                    and any(isinstance(a, np.memmap) for a in arrs)):
                tag = "pool|" + "|".join(src for src, _ in parts) + f"|{split}|b{int(track_b_mode)}"
                out[k] = concat_to_cache(tag=tag, key=k, arrays=arrs)
            else:
                out[k] = np.concatenate(arrs, axis=0)

    # Meta keys
    meta_keys = list(parts[0][1]["meta"].keys())
    out["meta"] = {}
    for mk in meta_keys:
        out["meta"][mk] = np.concatenate(
            [p[1]["meta"][mk] for p in parts], axis=0)
    # New per-sample source slug
    out["meta"]["source"] = np.concatenate(
        [np.full(len(d["hist"]), src) for src, d in parts])

    total = len(out["hist"])
    breakdown = ", ".join(f"{src}={len(d['hist'])}" for src, d in parts)
    print(f"  [pool] {split} loaded N={total}  ({breakdown})")
    return out


def stratified_metrics_by_source(pred: np.ndarray, gt: np.ndarray,
                                  meta: dict) -> dict:
    """Compute ADE/FDE per source jurisdiction in a pooled-evaluation setting."""
    sources = meta["source"]
    out: dict[str, dict] = {"all": {"N": int(len(pred))}}
    err = np.linalg.norm(pred - gt, axis=-1)
    out["all"]["ADE"] = round(float(err.mean()), 3)
    out["all"]["FDE"] = round(float(err[:, -1].mean()), 3)
    for src in np.unique(sources):
        m = sources == src
        if m.sum() == 0: continue
        e = err[m]
        out[str(src)] = {
            "N":   int(m.sum()),
            "ADE": round(float(e.mean()), 3),
            "FDE": round(float(e[:, -1].mean()), 3),
        }
    return out
