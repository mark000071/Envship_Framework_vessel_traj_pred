"""Stratified evaluation for social-context models.

EnvShip-Bench DMA test split is 79.5% isolated (zero neighbours), so the
aggregate ADE is dominated by the isolated subset.  This script re-evaluates
a trained model and breaks ADE down by neighbour-count buckets:
    n=0   |  n≥1   |  n≥3   |  n≥5
so we can see whether a social-aware model genuinely helps where it should.

Usage:
    python eval/eval_social_stratified.py \
        --model lstm_social_attn_v5 \
        --ckpt  eval/checkpoints/lstm_social_attn_v5.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.context_dataset import load_context_split
from eval.normalizer      import GlobalNormalizer
from eval.metrics         import compute_metrics
from eval.models          import MODEL_REGISTRY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--ckpt",  type=Path, required=True)
    p.add_argument("--norm",  type=Path, default=None,
                   help="GlobalNormalizer JSON. Default: <ckpt-dir>/<model>_norm.json")
    p.add_argument("--track-root", type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets"
                                "/multi_type_mini_bench_build/standard_track_v1"))
    p.add_argument("--social-feat-dim", type=int, default=8)
    p.add_argument("--out",   type=Path, default=None,
                   help="Optional JSON output path.")
    return p.parse_args()


def main():
    args = parse_args()
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_path = args.norm or args.ckpt.parent / f"{args.model}_norm.json"
    norm = GlobalNormalizer.load(norm_path)

    print(f"[strat-eval] loading test split with {args.social_feat_dim}-d social feat ...")
    data = load_context_split(
        args.track_root, "test",
        load_social=True, load_env_desc=False,
        load_env_raster=False, load_env_sdf=False,
        social_feat_dim=args.social_feat_dim,
    )

    model = MODEL_REGISTRY[args.model]().to(dev)
    sd    = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd.get("state", sd))
    model.eval()

    hist = norm.transform_hist(data["hist"])
    bs = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(hist), bs):
            hb = torch.from_numpy(hist[i:i+bs]).to(dev)
            sf = torch.from_numpy(data["social_feat"][i:i+bs]).to(dev)
            sm = torch.from_numpy(data["social_mask"][i:i+bs]).to(dev)
            p  = model.predict(hb, social_feat=sf, social_mask=sm).cpu().numpy()
            preds.append(norm.inverse_future(p))
    pred = np.concatenate(preds)
    gt   = data["future"]
    nc   = data["neighbor_count"]

    def ade(idx: np.ndarray) -> dict:
        if idx.sum() == 0:
            return {"N": 0, "ADE": None, "FDE": None}
        err = np.linalg.norm(pred[idx] - gt[idx], axis=-1)
        return {
            "N":   int(idx.sum()),
            "ADE": round(float(err.mean()), 2),
            "FDE": round(float(err[:, -1].mean()), 2),
        }

    report = {
        "model": args.model,
        "all":         ade(np.ones_like(nc, dtype=bool)),
        "isolated":    ade(nc == 0),
        "n_geq_1":     ade(nc >= 1),
        "n_geq_3":     ade(nc >= 3),
        "n_geq_5":     ade(nc >= 5),
    }
    print("\n=== stratified ADE ===")
    for k, v in report.items():
        if k == "model":
            continue
        print(f"  {k:<10}  N={v['N']:>5}  ADE={v['ADE']}  FDE={v['FDE']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
