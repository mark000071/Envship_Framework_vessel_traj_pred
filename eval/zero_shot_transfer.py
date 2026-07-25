"""Evaluate existing DMA-trained checkpoints on each jurisdiction's test set.

For each (jurisdiction, model) we load the DMA-trained .pt + norm, run
inference over the jurisdiction's test split (in the original metres
coordinate frame), and compute compute_metrics() on the result.

Usage:
    python eval/zero_shot_transfer.py \
        --ckpt-dir eval/checkpoints \
        --out-dir  eval/results/zero_shot
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.multi_source_dataset import SOURCE_ROOTS
from eval.context_dataset      import load_context_split
from eval.dataset              import load_split
from eval.normalizer           import GlobalNormalizer
from eval.metrics              import compute_metrics
from eval.models               import MODEL_REGISTRY, CONTEXT_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_LOAD_CFG = {
    "tcn":                  {"context": False},
    "lstm_env_sdf":         {"load_social": False, "load_env_desc": False,
                              "load_env_raster": False, "load_env_sdf": True},
    "lstm_social_env_sdf":  {"load_social": True,  "load_env_desc": False,
                              "load_env_raster": False, "load_env_sdf": True,
                              "social_feat_dim": 5},
}


def evaluate(model_name: str, ckpt_dir: Path, target_root: Path,
             target_slug: str) -> dict:
    cfg = MODEL_LOAD_CFG[model_name]
    is_context = (model_name in CONTEXT_MODELS)
    norm = GlobalNormalizer.load(ckpt_dir / f"{model_name}_norm.json")
    if is_context:
        cfg_ctx = {k: v for k, v in cfg.items() if k != "context"}
        d = load_context_split(target_root, "test", **cfg_ctx)
    else:
        d = load_split(target_root, "test", use_augmented=True)

    if len(d["hist"]) == 0:
        return {"model": model_name, "target": target_slug, "n": 0,
                "ADE": None, "FDE": None, "note": "empty test split"}

    model = MODEL_REGISTRY[model_name]().to(DEVICE)
    sd = torch.load(ckpt_dir / f"{model_name}.pt", map_location=DEVICE)
    model.load_state_dict(sd.get("state", sd))
    model.eval()

    hist = norm.transform_hist(d["hist"])
    bs = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(hist), bs):
            hb = torch.from_numpy(hist[i:i+bs]).to(DEVICE)
            kw = {}
            if "env_sdf" in d:
                kw["env_sdf"] = torch.from_numpy(d["env_sdf"][i:i+bs]).to(DEVICE)
            if "social_feat" in d:
                kw["social_feat"] = torch.from_numpy(d["social_feat"][i:i+bs]).to(DEVICE)
                kw["social_mask"] = torch.from_numpy(d["social_mask"][i:i+bs]).to(DEVICE)
            if "env_desc" in d:
                kw["env_desc"] = torch.from_numpy(d["env_desc"][i:i+bs]).to(DEVICE)
            if "env_raster" in d:
                kw["env_raster"] = torch.from_numpy(d["env_raster"][i:i+bs]).float().to(DEVICE)
            try:
                p = model.predict(hb, **kw).cpu().numpy()
            except TypeError:
                p = model.predict(hb).cpu().numpy()
            preds.append(norm.inverse_future(p))
    pred = np.concatenate(preds)
    gt = d["future"]
    res = compute_metrics(pred, gt, d["meta"])
    res["model"]  = model_name
    res["target"] = target_slug
    res["n"]      = len(pred)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/checkpoints"))
    ap.add_argument("--out-dir",  type=Path,
                    default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results/zero_shot"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model in MODEL_LOAD_CFG:
        for target_slug, target_root in SOURCE_ROOTS.items():
            print(f"\n── {model:<22} → {target_slug:<8} ──")
            try:
                r = evaluate(model, args.ckpt_dir, target_root, target_slug)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                r = {"model": model, "target": target_slug, "error": str(e)}
            all_results.append(r)
            print(f"  ADE={r.get('ADE')}  FDE={r.get('FDE')}  n={r.get('n')}")
            (args.out_dir / f"{model}__{target_slug}.json").write_text(json.dumps(r, indent=2))

    (args.out_dir / "all.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n→ saved {len(all_results)} entries to {args.out_dir}")


if __name__ == "__main__":
    main()
