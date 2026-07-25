"""Deep-ensemble K=3 evaluator for Track B.

For each (jurisdiction, model) with 3 single-modal seeds already trained,
load all 3 checkpoints, run inference on the test set, and compute
minADE@3 / minFDE@3 by taking the minimum across the K=3 per-sample
predictions.

This is the "no architectural change" K=3 control baseline reported
alongside the learned K=3 head.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets")
sys.path.insert(0, str(REPO))

from eval.train_dl import (
    MODEL_REGISTRY, CONTEXT_MODELS, _MODEL_LOAD_CFG, _feat_keys_for_model,
    _make_loader, GlobalNormalizer, _unpack_batch
)
from eval.dataset import load_split
from eval.context_dataset import load_context_split


def get_trackb_root(domain):
    if domain == "DMA":
        return "/mnt/nfs/kun/DeepJSCC/Agent_paper_exp_ALL_folder/Cross-domain-datasets/track_b/DMA/multi_type_mini_bench_build/standard_track_v1"
    elif domain == "NOAA":
        return "/mnt/nfs/kun/DeepJSCC/Agent_paper_exp_ALL_folder/Cross-domain-datasets/track_b/NOAA/multi_type_mini_bench_build/standard_track_v1"
    raise ValueError(domain)


def _set_fut_steps(n):
    from eval.models import (sequence_models, sequence_models_v2,
                              sequence_models_v3, context_models,
                              context_models_v2, context_models_v3,
                              context_models_v4, context_models_v5,
                              tcn as _tcn, transformer_nar as _tnar,
                              transformer_variants as _tv)
    for mod in (sequence_models, sequence_models_v2, sequence_models_v3,
                context_models, context_models_v2, context_models_v3,
                context_models_v4, context_models_v5,
                _tcn, _tnar, _tv):
        mod.FUT_STEPS = n


def eval_ensemble(domain: str, model_slug: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track_root = Path(get_trackb_root(domain)).resolve()
    ckpt_dir = REPO / "eval" / "checkpoints" / "v4" / "track_b" / domain
    seeds_present = sorted([
        int(p.stem.split("seed")[-1])
        for p in ckpt_dir.glob(f"{model_slug}_seed*.pt")
    ])
    if len(seeds_present) < 2:
        return {"model": model_slug, "domain": domain, "n_seeds": len(seeds_present),
                "minADE_3": None, "minFDE_3": None, "error": "insufficient seeds"}

    is_context = model_slug in CONTEXT_MODELS
    if is_context:
        cfg = _MODEL_LOAD_CFG[model_slug]
        data_te = load_context_split(track_root, "test", **cfg)
    else:
        data_te = load_split(track_root, "test")
    fut_steps = int(data_te["future"].shape[1])
    _set_fut_steps(fut_steps)

    # Use seed=1 normalizer
    norm = GlobalNormalizer.load(ckpt_dir / f"{model_slug}_seed1_norm.json")

    test_loader = _make_loader(data_te, norm, 256, False, model_slug, str(device))

    # Aggregate per-sample test predictions from each seed
    per_seed_preds = []
    target = None
    for s in seeds_present[:3]:  # use 3 seeds for K=3
        ckpt_path = ckpt_dir / f"{model_slug}_seed{s}.pt"
        model = MODEL_REGISTRY[model_slug]().to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        model.eval()
        preds_seed = []
        targets_seed = []
        with torch.no_grad():
            for batch in test_loader:
                b = _unpack_batch(batch, model_slug, device)
                out = model(**b["inputs"])
                # Un-normalize
                out_unnorm = norm.inverse_targets(out)
                preds_seed.append(out_unnorm.cpu().numpy())
                if target is None:
                    targets_seed.append(b["target_unnorm"].cpu().numpy())
        preds_seed = np.concatenate(preds_seed, axis=0)
        per_seed_preds.append(preds_seed)
        if target is None:
            target = np.concatenate(targets_seed, axis=0)
        del model
        torch.cuda.empty_cache()

    if len(per_seed_preds) < 2:
        return {"model": model_slug, "domain": domain, "n_seeds": len(seeds_present),
                "minADE_3": None, "minFDE_3": None, "error": "ensemble too small"}

    # Stack to (B, K, T, 2)
    preds_stack = np.stack(per_seed_preds, axis=1)
    B, K, T, _ = preds_stack.shape

    diff = preds_stack - target[:, None, :, :]
    ade_per_k = np.linalg.norm(diff, axis=-1).mean(axis=-1)   # (B, K)
    min_ade = ade_per_k.min(axis=-1).mean()
    fde_per_k = np.linalg.norm(diff[..., -1, :], axis=-1)     # (B, K)
    min_fde = fde_per_k.min(axis=-1).mean()

    return {
        "model": model_slug, "domain": domain, "n_seeds": K,
        "minADE_3": float(min_ade), "minFDE_3": float(min_fde),
        "n_test": int(B), "T": int(T),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["DMA", "NOAA"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = eval_ensemble(args.domain, args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
