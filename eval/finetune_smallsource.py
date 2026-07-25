"""Fine-tune the combined-source checkpoint on a small jurisdiction.

Method-fix proposed for the Piraeus / Norway tail-error problem identified in
the cross-domain analysis: take the combined-source checkpoint (which has
seen all four jurisdictions including 168K DMA + NOAA samples) and continue
training for a small number of epochs on the target jurisdiction's training
split with a reduced learning rate.  This should preferentially close the
gap on the hard tail of the target distribution without forgetting the
DMA/NOAA backbone.

Usage:
    python eval/finetune_smallsource.py \
        --model lstm_env_sdf --target piraeus \
        --lr 5e-5 --epochs 30 --patience 8
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.multi_source_dataset import SOURCE_ROOTS
from eval.context_dataset      import load_context_split
from eval.dataset              import load_split
from eval.normalizer           import GlobalNormalizer
from eval.metrics              import compute_metrics
from eval.models               import MODEL_REGISTRY, CONTEXT_MODELS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets")

LOAD_CFG = {
    "tcn":                 {"context": False},
    "lstm_env_sdf":        {"load_social": False, "load_env_desc": False,
                              "load_env_raster": False, "load_env_sdf": True},
    "lstm_social_env_sdf": {"load_social": True,  "load_env_desc": False,
                              "load_env_raster": False, "load_env_sdf": True,
                              "social_feat_dim": 5},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, choices=list(LOAD_CFG))
    p.add_argument("--target",  required=True, choices=list(SOURCE_ROOTS))
    p.add_argument("--source-ckpt", type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/checkpoints/combined"),
                   help="Where to start from.  Default: combined-source ckpt.")
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--patience",    type=int,   default=8)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--out-dir",     type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results/finetune"))
    return p.parse_args()


def load_set(target_root: Path, split: str, model_name: str):
    cfg = LOAD_CFG[model_name]
    if cfg.get("context", True) is False:
        return load_split(target_root, split, use_augmented=True)
    cfg_ctx = {k: v for k, v in cfg.items() if k != "context"}
    return load_context_split(target_root, split, **cfg_ctx)


def to_tensor_batches(d, norm, batch_size: int, shuffle=False):
    hist_n = norm.transform_hist(d["hist"])
    fut_n  = norm.transform_future(d["future"])
    tensors = [torch.from_numpy(hist_n), torch.from_numpy(fut_n)]
    extras = {}
    for k in ("env_sdf", "env_desc", "social_feat", "social_mask"):
        if k in d:
            tensors.append(torch.from_numpy(d[k]))
            extras[k] = len(tensors) - 1
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=DEV.type == "cuda"), extras


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + normaliser from source checkpoint ───────────────────
    norm = GlobalNormalizer.load(args.source_ckpt / f"{args.model}_norm.json")
    model = MODEL_REGISTRY[args.model]().to(DEV)
    sd = torch.load(args.source_ckpt / f"{args.model}.pt", map_location=DEV)
    model.load_state_dict(sd.get("state", sd))
    print(f"[{args.model}] loaded source ckpt from {args.source_ckpt}")

    # ── Load target train/val/test ───────────────────────────────────────
    target_root = SOURCE_ROOTS[args.target]
    d_tr = load_set(target_root, "train", args.model)
    d_va = load_set(target_root, "val",   args.model)
    d_te = load_set(target_root, "test",  args.model)
    print(f"[{args.model}] target={args.target}  N_train={len(d_tr['hist'])} "
          f"val={len(d_va['hist'])} test={len(d_te['hist'])}")

    if len(d_tr["hist"]) < 5:
        print("ERROR: target train set too small to fine-tune"); return

    tr_dl, ext = to_tensor_batches(d_tr, norm, args.batch_size, shuffle=True)
    if len(d_va["hist"]) >= 1:
        va_dl, _ = to_tensor_batches(d_va, norm, args.batch_size, shuffle=False)
    else:
        va_dl = None

    # ── Pre-finetune test eval (sanity / baseline) ───────────────────────
    def evaluate(d):
        model.eval()
        hist = norm.transform_hist(d["hist"])
        preds = []
        with torch.no_grad():
            for i in range(0, len(hist), 256):
                hb = torch.from_numpy(hist[i:i+256]).to(DEV)
                kw = {}
                for k in ("env_sdf","env_desc","social_feat","social_mask"):
                    if k in d:
                        kw[k] = torch.from_numpy(d[k][i:i+256]).to(DEV)
                try:    p = model.predict(hb, **kw).cpu().numpy()
                except TypeError: p = model.predict(hb).cpu().numpy()
                preds.append(norm.inverse_future(p))
        pred = np.concatenate(preds)
        return compute_metrics(pred, d["future"], d["meta"])

    res_pre = evaluate(d_te)
    print(f"[{args.model}] pre-finetune Piraeus  ADE={res_pre['ADE']:.2f}  FDE={res_pre['FDE']:.2f}")

    # ── Fine-tune ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_val = float("inf"); pat_left = args.patience; best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for batch in tr_dl:
            hist_b, fut_b = batch[0].to(DEV), batch[1].to(DEV)
            kw = {}
            for k, idx in ext.items():
                kw[k] = batch[idx].to(DEV)
            try:
                pred = model(hist_b, **kw, target=fut_b, tf_ratio=0.0)
            except TypeError:
                pred = model(hist_b, target=fut_b, tf_ratio=0.0)
            loss = loss_fn(pred, fut_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tr_loss = tot / max(len(tr_dl), 1)
        sched.step()

        if va_dl is not None:
            model.eval(); v = 0.0
            with torch.no_grad():
                for batch in va_dl:
                    hist_b, fut_b = batch[0].to(DEV), batch[1].to(DEV)
                    kw = {k: batch[idx].to(DEV) for k, idx in ext.items()}
                    try: pred = model(hist_b, **kw, target=None, tf_ratio=0.0)
                    except TypeError: pred = model(hist_b, target=None, tf_ratio=0.0)
                    v += loss_fn(pred, fut_b).item()
            val_loss = v / max(len(va_dl), 1)
        else:
            val_loss = tr_loss

        print(f"  ep {ep:02d}/{args.epochs}  tr={tr_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss; pat_left = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_left -= 1
            if pat_left <= 0:
                print("  early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    res_post = evaluate(d_te)
    print(f"[{args.model}] post-finetune target ADE={res_post['ADE']:.2f}  FDE={res_post['FDE']:.2f}")
    print(f"  delta: ADE {res_pre['ADE']:.1f} → {res_post['ADE']:.1f}   "
          f"FDE {res_pre['FDE']:.1f} → {res_post['FDE']:.1f}")
    out = {"model": args.model, "target": args.target,
           "pre":  res_pre,  "post": res_post}
    (args.out_dir / f"{args.model}__{args.target}__finetune.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
