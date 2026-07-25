#!/usr/bin/env python3
"""Traditional ML baselines: Random Forest and XGBoost.

Both use multi-output regression to predict all 30×2 future positions
from the flattened 30×6 history features.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.dataset import load_split
from eval.metrics import compute_metrics

from eval.paths import DEFAULT_TRACK_ROOT, RES_DIR
TRACK_ROOT = DEFAULT_TRACK_ROOT   # resolves from $ENVSHIP_DATA_ROOT (see eval/paths.py)


def flatten_features(hist: np.ndarray) -> np.ndarray:
    """(N, 30, 6) → (N, 180)  flattened trajectory features."""
    return hist.reshape(hist.shape[0], -1)


def flatten_targets(future: np.ndarray) -> np.ndarray:
    """(N, 30, 2) → (N, 60)"""
    return future.reshape(future.shape[0], -1)


def unflatten_targets(flat: np.ndarray, fut_steps: int = 30) -> np.ndarray:
    """(N, fut_steps*2) → (N, fut_steps, 2).  Track A uses 30; Track B uses 180."""
    return flat.reshape(flat.shape[0], fut_steps, 2)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_rf(n_estimators: int = 100, max_depth: int = 20, n_jobs: int = -1,
             track_root: Path | None = None) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    root = Path(track_root) if track_root else TRACK_ROOT
    print(f"[RF] loading data from {root} ...")
    # use_augmented=True for the test split so scene_type is available;
    # without it the augmented columns aren't loaded and scene-stratified
    # ADE collapses to "all-open-water".
    data_tr = load_split(root, "train", use_augmented=True)
    data_te = load_split(root, "test",  use_augmented=True)

    X_tr = flatten_features(data_tr["hist"])
    y_tr = flatten_targets(data_tr["future"])
    X_te = flatten_features(data_te["hist"])

    print(f"[RF] fitting n_estimators={n_estimators}  train_shape={X_tr.shape} ...")
    t0   = time.time()
    # RF natively supports multi-output, no need for MultiOutputRegressor
    rf   = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=5, n_jobs=n_jobs, random_state=42)
    rf.fit(X_tr, y_tr)
    print(f"[RF] fit done in {time.time()-t0:.0f}s")

    pred = unflatten_targets(rf.predict(X_te), fut_steps=data_te["future"].shape[1])
    res  = compute_metrics(pred, data_te["future"], data_te["meta"])
    res["model"] = "Random Forest"
    print(f"[RF] ADE={res['ADE']}  FDE={res['FDE']}")
    return res


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgb(n_estimators: int = 300, max_depth: int = 6,
              learning_rate: float = 0.1, n_jobs: int = -1,
              track_root: Path | None = None) -> dict:
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor

    root = Path(track_root) if track_root else TRACK_ROOT
    print(f"[XGB] loading data from {root} ...")
    # use_augmented=True for the test split so scene_type is available;
    # without it the augmented columns aren't loaded and scene-stratified
    # ADE collapses to "all-open-water".
    data_tr = load_split(root, "train", use_augmented=True)
    data_te = load_split(root, "test",  use_augmented=True)

    X_tr = flatten_features(data_tr["hist"])
    y_tr = flatten_targets(data_tr["future"])
    X_te = flatten_features(data_te["hist"])

    print(f"[XGB] fitting n_estimators={n_estimators}  train_shape={X_tr.shape} ...")
    t0 = time.time()
    # XGBoost natively supports multi-output in newer versions
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=n_jobs, random_state=42, tree_method="hist",
        multi_strategy="multi_output_tree",   # native multi-output support
        verbosity=1,
    )
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_tr[:5000], y_tr[:5000])],
                  verbose=50)
    print(f"[XGB] fit done in {time.time()-t0:.0f}s")

    pred = unflatten_targets(xgb_model.predict(X_te), fut_steps=data_te["future"].shape[1])
    res  = compute_metrics(pred, data_te["future"], data_te["meta"])
    res["model"] = "XGBoost"
    print(f"[XGB] ADE={res['ADE']}  FDE={res['FDE']}")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rf", "xgb", "all"], default="all")
    p.add_argument("--track-root", type=Path, default=None,
                   help="Standard-Track-v1 root (default: DMA).  Pass NOAA / Singapore "
                        "/ Piraeus to retrain on another jurisdiction.")
    p.add_argument("--res-dir",    type=Path, default=None,
                   help="Where to write *.json (default: eval/results).")
    args = p.parse_args()

    res_root = Path(args.res_dir).resolve() if args.res_dir else RES_DIR
    res_root.mkdir(parents=True, exist_ok=True)

    results = []
    if args.model in ("rf", "all"):
        r = train_rf(track_root=args.track_root)
        results.append(r)
        (res_root / "random_forest.json").write_text(json.dumps(r, indent=2))

    if args.model in ("xgb", "all"):
        r = train_xgb(track_root=args.track_root)
        results.append(r)
        (res_root / "xgboost.json").write_text(json.dumps(r, indent=2))

    for r in results:
        print(f"  {r['model']}: ADE={r['ADE']}  FDE={r['FDE']}")
