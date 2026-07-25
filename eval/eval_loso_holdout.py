"""Re-evaluate LOSO checkpoints on the genuinely held-out test split.

Background: train_dl.py with --pool=loso_holdout_<src> trains on three
jurisdictions but its built-in test pass evaluates on those same three
jurisdictions' test splits (not the held-out one).  Proper LOSO transfer
requires evaluating on the test split of the source that was NOT in the
training pool.

This script loads each LOSO checkpoint and runs inference on the held-out
source's test split, then overwrites the corresponding JSON with the
LOSO-correct metrics.

Usage:
    python eval/eval_loso_holdout.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.multi_source_dataset import SOURCE_ROOTS
from eval.context_dataset      import load_context_split
from eval.normalizer           import GlobalNormalizer
from eval.metrics              import compute_metrics
from eval.models               import MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO   = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets")

MODELS = ["tcn", "lstm_env_sdf", "lstm_social_env_sdf"]
SOURCES = ["dma", "noaa", "piraeus", "norway"]

LOAD_CFG = {
    "tcn": {"context": False},
    "lstm_env_sdf": {"load_social": False, "load_env_desc": False,
                      "load_env_raster": False, "load_env_sdf": True},
    "lstm_social_env_sdf": {"load_social": True,  "load_env_desc": False,
                             "load_env_raster": False, "load_env_sdf": True,
                             "social_feat_dim": 5},
}


def evaluate(model_name: str, ckpt_dir: Path, holdout_src: str) -> dict:
    cfg_in = LOAD_CFG[model_name]
    is_context = (cfg_in.get("context", True) is not False)

    ckpt_pt = ckpt_dir / f"{model_name}.pt"
    norm_p  = ckpt_dir / f"{model_name}_norm.json"
    if not ckpt_pt.exists():
        return {"error": f"missing checkpoint {ckpt_pt}"}

    norm = GlobalNormalizer.load(norm_p)
    target_root = SOURCE_ROOTS[holdout_src]
    cfg_ctx = {k: v for k, v in cfg_in.items() if k != "context"}
    if is_context:
        d = load_context_split(target_root, "test", **cfg_ctx)
    else:
        from eval.dataset import load_split
        d = load_split(target_root, "test", use_augmented=True)
    if len(d["hist"]) == 0:
        return {"error": "empty holdout test split"}

    model = MODEL_REGISTRY[model_name]().to(DEVICE)
    sd = torch.load(ckpt_pt, map_location=DEVICE)
    model.load_state_dict(sd.get("state", sd))
    model.eval()

    hist = norm.transform_hist(d["hist"])
    bs = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(hist), bs):
            hb = torch.from_numpy(hist[i:i+bs]).to(DEVICE)
            kw = {}
            for k in ("env_sdf", "env_desc"):
                if k in d:
                    kw[k] = torch.from_numpy(d[k][i:i+bs]).to(DEVICE)
            if "env_raster" in d:
                kw["env_raster"] = torch.from_numpy(d["env_raster"][i:i+bs]).float().to(DEVICE)
            if "social_feat" in d:
                kw["social_feat"] = torch.from_numpy(d["social_feat"][i:i+bs]).to(DEVICE)
                kw["social_mask"] = torch.from_numpy(d["social_mask"][i:i+bs]).to(DEVICE)
            try:
                p = model.predict(hb, **kw).cpu().numpy()
            except TypeError:
                p = model.predict(hb).cpu().numpy()
            preds.append(norm.inverse_future(p))
    pred = np.concatenate(preds)
    gt = d["future"]
    res = compute_metrics(pred, gt, d["meta"])
    res["model"]   = model_name
    res["holdout"] = holdout_src
    res["n_samples"] = len(pred)
    return res


def main():
    for holdout in SOURCES:
        ckpt_dir = REPO / "eval" / "checkpoints" / f"loso_{holdout}"
        res_dir  = REPO / "eval" / "results"     / f"loso_{holdout}"
        if not ckpt_dir.exists():
            print(f"── loso_{holdout}: ckpt dir missing, skipping ──")
            continue
        print(f"\n── loso_{holdout} (test on {holdout}) ──")
        for m in MODELS:
            ckpt_pt = ckpt_dir / f"{m}.pt"
            if not ckpt_pt.exists():
                print(f"  {m}: NO CKPT (not yet trained)")
                continue
            try:
                r = evaluate(m, ckpt_dir, holdout)
                if "error" in r:
                    print(f"  {m}: ERROR {r['error']}")
                    continue
                print(f"  {m}: ADE={r['ADE']:.2f}  FDE={r['FDE']:.2f}  N={r['n_samples']}")
                # Save with explicit LOSO-correct suffix; keep original test
                # result also.
                (res_dir / f"{m}_holdout_eval.json").write_text(json.dumps(r, indent=2))
            except Exception as e:
                print(f"  {m}: EXCEPTION {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
