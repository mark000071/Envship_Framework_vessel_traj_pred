#!/usr/bin/env python3
"""
Social-dense subset evaluation for EnvShip-Bench.

Standard Track test set is ~80 % no-neighbor samples → social context
models cannot be fairly evaluated on it.  This script filters to
samples with neighbor_count >= MIN_NEIGHBORS and re-evaluates all
social models, yielding a fairer comparison.

Usage:
    python eval/social_dense_eval.py            # uses all trained checkpoints
    python eval/social_dense_eval.py --min-nb 3 # stricter threshold
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.normalizer      import GlobalNormalizer
from eval.context_dataset import load_context_split
from eval.metrics         import compute_metrics
from eval.models          import MODEL_REGISTRY

TRACK_ROOT = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets"
                  "/multi_type_mini_bench_build/standard_track_v1")
CKPT_DIR   = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/checkpoints")
RES_DIR    = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results")

# Social models to compare (v1 and v2)
SOCIAL_MODELS = [
    "lstm_social_pool",
    "lstm_social_attn",
    "lstm_social_env",
    "lstm_social_attn_v2",
    "lstm_social_env_v2",
]
# Non-social baselines for reference
BASELINE_MODELS = ["lstm_2l", "lstm_2l_v2", "tcn"]


def _load_model(name: str, device: torch.device):
    ckpt_path = CKPT_DIR / f"{name}.pt"
    if not ckpt_path.exists():
        return None
    model = MODEL_REGISTRY[name]().to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state"])
    except RuntimeError as e:
        print(f"  {name:<28}  [checkpoint mismatch — skip: {e}]")
        return None
    model.eval()
    return model


def evaluate_on_subset(model, norm, data, idx, device, batch_size=256):
    """Run inference + ADE/FDE on filtered sample indices."""
    hist_n = torch.from_numpy(norm.transform_hist(data["hist"][idx]))

    # Build extra context tensors if needed
    has_social = "social_feat" in data
    has_env    = "env_desc"    in data

    preds = []
    B = batch_size
    with torch.no_grad():
        for i in range(0, len(idx), B):
            sl = slice(i, i + B)
            hb = hist_n[sl].to(device)
            extra: dict[str, torch.Tensor] = {}

            if has_social and hasattr(model, "social_attn"):
                extra["social_feat"] = torch.from_numpy(
                    data["social_feat"][idx[sl]]).to(device)
                extra["social_mask"] = torch.from_numpy(
                    data["social_mask"][idx[sl]]).to(device)

            if has_env and (hasattr(model, "env_proj") or hasattr(model, "film")):
                extra["env_desc"] = torch.from_numpy(
                    data["env_desc"][idx[sl]]).to(device)

            if extra:
                pred_n = model.predict(hb, **extra).cpu().numpy()
            else:
                pred_n = model.predict(hb).cpu().numpy()

            preds.append(norm.inverse_future(pred_n))

    pred = np.concatenate(preds)
    gt   = data["future"][idx]
    meta_sub = {k: np.array(v)[idx] for k, v in data["meta"].items()}
    return compute_metrics(pred, gt, meta_sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-nb", type=int, default=2,
                    help="Minimum neighbour count for social-dense subset (default 2)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load normalizer
    norm_path = CKPT_DIR / "lstm_2l_v2_norm.json"
    if not norm_path.exists():
        norm_path = CKPT_DIR / "lstm_2l_norm.json"
    norm = GlobalNormalizer.load(norm_path)

    # Load context test data (social + env desc, no raster)
    print(f"Loading context test data (min_neighbors >= {args.min_nb}) ...")
    data = load_context_split(TRACK_ROOT, "test",
                               load_social=True,
                               load_env_desc=True,
                               load_env_raster=False)

    # Build dense subset mask
    nb_counts = data["neighbor_count"]
    idx_all   = np.arange(len(nb_counts))
    idx_dense = np.where(nb_counts >= args.min_nb)[0]

    print(f"Total test samples  : {len(idx_all):,}")
    print(f"Social-dense (>={args.min_nb} nb): {len(idx_dense):,} "
          f"({len(idx_dense)/len(idx_all)*100:.1f}%)")
    print()

    all_models = SOCIAL_MODELS + BASELINE_MODELS
    results = {}

    for name in all_models:
        model = _load_model(name, device)
        if model is None:
            print(f"  {name:<28}  [checkpoint not found — skip]")
            continue

        # Evaluate on full test
        r_full  = evaluate_on_subset(model, norm, data, idx_all,  device)
        # Evaluate on dense subset
        r_dense = evaluate_on_subset(model, norm, data, idx_dense, device)

        results[name] = {"full": r_full, "dense": r_dense}
        tag = "social" if name in SOCIAL_MODELS else "base  "
        print(f"  [{tag}] {name:<28} "
              f"full ADE={r_full['ADE']:.1f}  "
              f"dense ADE={r_dense['ADE']:.1f}  "
              f"dense FDE={r_dense['FDE']:.1f}")

    # Save
    out_path = RES_DIR / "social_dense_eval.json"
    out_path.write_text(json.dumps({"min_neighbors": args.min_nb,
                                     "n_dense": int(len(idx_dense)),
                                     "n_total": int(len(idx_all)),
                                     "results": results}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
