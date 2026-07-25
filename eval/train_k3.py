#!/usr/bin/env python3
"""K=3 multi-modal trajectory training for EnvShip-Bench Track B.

Wraps a base model (TCN / LSTM+Env-SDF / LSTM+Social+Env-SDF) with K=3
independent branches that each produce a full (T, 2) trajectory.  A small
classifier head over the encoded history features picks the most likely
branch.  Training uses best-of-K (by anchor) Huber loss on the full
trajectory plus a 0.1-weight CE on the classifier targeting argmin-anchor.

Evaluation metrics on the test split:
  - minADE@3 / minFDE@3   : best-of-K average / final-step displacement
  - ADE@1   / FDE@1       : metrics on the argmax-classifier branch

Usage:
    python eval/train_k3.py --model tcn \
        --track-root /path/to/standard_track_v1 \
        --ckpt-dir   /path/to/ckpts \
        --res-dir    /path/to/results \
        --save-json  /path/to/res.json \
        --seed 1 --epochs 80 --batch-size 256 --lr 5e-4 --patience 20 --k 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.context_dataset import load_context_split
from eval.dataset import load_split
from eval.models import CONTEXT_MODELS, MODEL_REGISTRY
from eval.models.multimodal_k3 import compute_anchors_kmeans
from eval.normalizer import GlobalNormalizer
from eval.train_dl import (
    _MODEL_LOAD_CFG,
    _feat_keys_for_model,
    _make_loader,
    _unpack_batch,
)


# Supported base models for the K=3 wrapper (per spec).
SUPPORTED_MODELS = ("tcn", "lstm_env_sdf", "lstm_social_env_sdf")


# ---------------------------------------------------------------------------
# FUT_STEPS patching (mirror train_dl._set_fut_steps)
# ---------------------------------------------------------------------------

def _set_fut_steps(n: int) -> None:
    """Override module-level FUT_STEPS so every base-model forward uses `n`."""
    from eval.models import (
        context_models,
        context_models_v2,
        context_models_v3,
        context_models_v4,
        context_models_v5,
        sequence_models,
        sequence_models_v2,
        sequence_models_v3,
        tcn as _tcn,
        transformer_nar as _tnar,
        transformer_variants as _tv,
    )

    for mod in (
        sequence_models,
        sequence_models_v2,
        sequence_models_v3,
        context_models,
        context_models_v2,
        context_models_v3,
        context_models_v4,
        context_models_v5,
        _tcn,
        _tnar,
        _tv,
    ):
        mod.FUT_STEPS = n


# ---------------------------------------------------------------------------
# K3 Wrapper
# ---------------------------------------------------------------------------

class K3Wrapper(nn.Module):
    """K=3 multi-modal wrapper around a base trajectory predictor.

    Holds K independent instances of `model_cls`.  Each branch produces a
    (B, T, 2) trajectory in normalized space; the wrapper stacks them into
    (B, K, T, 2).  A tiny `nn.Linear` classifier on the mean of the
    raw history tensor produces logits (B, K).
    """

    def __init__(self, model_cls, k: int, hist_feat_dim: int, anchors: torch.Tensor):
        super().__init__()
        self.k = k
        self.branches = nn.ModuleList([model_cls() for _ in range(k)])
        self.classifier = nn.Linear(hist_feat_dim, k)
        # Anchors are K endpoints in normalized space — used to pick k*
        self.register_buffer("anchors", anchors.clone().float())

    # ------------------------------------------------------------------
    # Branch dispatching: each branch's forward signature differs based
    # on whether it consumes social / env-sdf extra inputs.  We rely on
    # **extra kwargs unpacking that the caller provides.
    # ------------------------------------------------------------------
    def _branch_forward(self, branch: nn.Module, hist, extra, target, tf_ratio):
        if extra:
            return branch(hist, **extra, target=target, tf_ratio=tf_ratio)
        return branch(hist, target=target, tf_ratio=tf_ratio)

    def _branch_predict(self, branch: nn.Module, hist, extra):
        if extra:
            return branch.predict(hist, **extra)
        return branch.predict(hist)

    def forward(self, hist, extra=None, target=None, tf_ratio=0.5):
        """Run all K branches on the same input.

        Returns:
            preds: (B, K, T, 2)   stacked trajectory predictions
            logits: (B, K)         classifier logits
        """
        extra = extra or {}
        outs = []
        for branch in self.branches:
            outs.append(self._branch_forward(branch, hist, extra, target, tf_ratio))
        preds = torch.stack(outs, dim=1)  # (B, K, T, 2)
        # Classifier on mean over time of the raw history features
        h_mean = hist.mean(dim=1)         # (B, hist_feat_dim)
        logits = self.classifier(h_mean)  # (B, K)
        return preds, logits

    @torch.no_grad()
    def predict(self, hist, extra=None):
        extra = extra or {}
        outs = []
        for branch in self.branches:
            outs.append(self._branch_predict(branch, hist, extra))
        preds = torch.stack(outs, dim=1)  # (B, K, T, 2)
        h_mean = hist.mean(dim=1)
        logits = self.classifier(h_mean)
        return preds, logits


# ---------------------------------------------------------------------------
# Loss: best-of-K-by-anchor Huber + classifier CE
# ---------------------------------------------------------------------------

def k3_loss(preds: torch.Tensor,        # (B, K, T, 2) normalized
            logits: torch.Tensor,        # (B, K)
            target: torch.Tensor,        # (B, T, 2)   normalized
            anchors: torch.Tensor,       # (K, 2)     normalized endpoints
            huber_delta: float = 1.0,
            cls_weight: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (total_loss, k_star)."""
    # Find which anchor's endpoint is closest to the GT endpoint
    gt_end = target[:, -1, :]                                  # (B, 2)
    anchor_dists = torch.cdist(gt_end.unsqueeze(1), anchors.unsqueeze(0))  # (B, 1, K)
    k_star = anchor_dists.squeeze(1).argmin(dim=-1)            # (B,)
    # Gather the chosen branch's full trajectory
    idx = k_star.view(-1, 1, 1, 1).expand(-1, 1, preds.size(2), preds.size(3))
    sel = preds.gather(1, idx).squeeze(1)                      # (B, T, 2)
    huber = F.huber_loss(sel, target, delta=huber_delta, reduction="mean")
    ce = F.cross_entropy(logits, k_star)
    return huber + cls_weight * ce, k_star


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def min_ade_fde_k(preds: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """preds: (N, K, T, 2)  unnormalized; gt: (N, T, 2) unnormalized."""
    diffs = preds - gt[:, None, :, :]                          # (N, K, T, 2)
    norms = np.linalg.norm(diffs, axis=-1)                     # (N, K, T)
    ade_per_k = norms.mean(axis=-1)                            # (N, K)
    fde_per_k = norms[..., -1]                                 # (N, K)
    return float(ade_per_k.min(axis=-1).mean()), float(fde_per_k.min(axis=-1).mean())


def ade_fde_branch(preds_branch: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """preds_branch: (N, T, 2)  unnormalized; gt: (N, T, 2) unnormalized."""
    diffs = preds_branch - gt                                  # (N, T, 2)
    norms = np.linalg.norm(diffs, axis=-1)                     # (N, T)
    return float(norms.mean(axis=-1).mean()), float(norms[:, -1].mean())


# ---------------------------------------------------------------------------
# Anchor computation (over training set future endpoints in normalized space)
# ---------------------------------------------------------------------------

def _compute_anchors_norm(data_tr: dict, norm: GlobalNormalizer, k: int,
                          seed: int) -> torch.Tensor:
    """Run k-means on the training-set normalized future endpoints."""
    fut_n = norm.transform_future(data_tr["future"])           # (N, T, 2)
    endpoints = torch.from_numpy(fut_n[:, -1, :]).float()      # (N, 2)
    # k-means is sensitive to init; seed RNG state for determinism.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed if seed else 0)
    # The function uses torch.randperm globally; just use it directly.
    return compute_anchors_kmeans(endpoints, k=k)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    seed = int(getattr(args, "seed", 0) or 0)
    if seed:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"[{args.model}_k3] device={device_str}  epochs={args.epochs}  "
          f"lr={args.lr}  seed={seed}  k={args.k}")

    if args.model not in SUPPORTED_MODELS:
        raise SystemExit(
            f"--model {args.model} not in supported list {SUPPORTED_MODELS}")

    track_root = Path(args.track_root).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    res_dir = Path(args.res_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.model}_k3] track_root={track_root}")

    # ── Load data ──────────────────────────────────────────────────────────
    is_context = args.model in CONTEXT_MODELS
    if is_context:
        cfg = _MODEL_LOAD_CFG[args.model]
        print(f"[{args.model}_k3] loading context data  cfg={cfg}")
        data_tr = load_context_split(track_root, "train", **cfg)
        data_va = load_context_split(track_root, "val",   **cfg)
        data_te = load_context_split(track_root, "test",  **cfg)
    else:
        print(f"[{args.model}_k3] loading trajectory-only data ...")
        data_tr = load_split(track_root, "train")
        data_va = load_split(track_root, "val")
        data_te = load_split(track_root, "test")

    fut_steps_actual = int(data_tr["future"].shape[1])
    hist_steps_actual = int(data_tr["hist"].shape[1])
    _set_fut_steps(fut_steps_actual)
    print(f"[{args.model}_k3] fut_steps={fut_steps_actual}  "
          f"hist_steps={hist_steps_actual}  "
          f"n_train={data_tr['hist'].shape[0]}  "
          f"n_val={data_va['hist'].shape[0]}  "
          f"n_test={data_te['hist'].shape[0]}")

    # ── Normalizer ─────────────────────────────────────────────────────────
    norm_path = ckpt_dir / f"{args.model}_k3_seed{seed}_norm.json"
    norm = GlobalNormalizer().fit(data_tr["hist"])
    norm.save(norm_path)

    # ── Anchors (k-means over training future endpoints in norm space) ─────
    anchors = _compute_anchors_norm(data_tr, norm, args.k, seed)
    print(f"[{args.model}_k3] anchors (normalized):\n{anchors.numpy()}")

    # ── Build K3 wrapper ───────────────────────────────────────────────────
    model_cls = MODEL_REGISTRY[args.model]
    hist_feat_dim = int(data_tr["hist"].shape[-1])
    wrapper = K3Wrapper(model_cls, args.k, hist_feat_dim, anchors).to(device)
    n_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    base_name = wrapper.branches[0].name
    print(f"[{args.model}_k3] base={base_name}  total_params={n_params:,}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    feat_keys = _feat_keys_for_model(args.model, data_tr)
    tr_dl = _make_loader(data_tr, norm, args.batch_size, True, args.model, device_str)
    va_dl = _make_loader(data_va, norm, args.batch_size, False, args.model, device_str)

    # ── Optim ──────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(wrapper.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    ckpt_path = ckpt_dir / f"{args.model}_k3_seed{seed}.pt"
    best_val = float("inf")
    patience_left = args.patience
    anchors_dev = anchors.to(device)

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        total_tr = 0.0
        t0 = time.time()
        for batch in tr_dl:
            hist_b, fut_b, extra = _unpack_batch(batch, feat_keys, device)
            tf_ratio = max(0.0, 0.5 * (1.0 - epoch / (args.epochs * 0.4)))
            opt.zero_grad()
            preds, logits = wrapper(hist_b, extra=extra, target=fut_b,
                                     tf_ratio=tf_ratio)
            loss, _ = k3_loss(preds, logits, fut_b, anchors_dev,
                              huber_delta=1.0, cls_weight=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            opt.step()
            total_tr += loss.item()
        sched.step()

        # ── Validation: Huber on selected-anchor branch ───────────────────
        wrapper.eval()
        total_va = 0.0
        with torch.no_grad():
            for batch in va_dl:
                hist_b, fut_b, extra = _unpack_batch(batch, feat_keys, device)
                preds, logits = wrapper(hist_b, extra=extra, target=None,
                                         tf_ratio=0.0)
                # Huber on the anchor-chosen branch (matches training objective)
                gt_end = fut_b[:, -1, :]
                ad = torch.cdist(gt_end.unsqueeze(1), anchors_dev.unsqueeze(0))
                k_star = ad.squeeze(1).argmin(dim=-1)
                idx = k_star.view(-1, 1, 1, 1).expand(-1, 1, preds.size(2),
                                                       preds.size(3))
                sel = preds.gather(1, idx).squeeze(1)
                huber = F.huber_loss(sel, fut_b, delta=1.0, reduction="mean")
                total_va += huber.item()

        val_loss = total_va / max(1, len(va_dl))
        tr_loss = total_tr / max(1, len(tr_dl))
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch <= 5 or epoch == args.epochs:
            print(f"[{args.model}_k3] ep {epoch:03d}/{args.epochs}  "
                  f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  t={elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = args.patience
            torch.save({"state": wrapper.state_dict(),
                        "anchors": anchors.cpu()}, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{args.model}_k3] early stop at epoch {epoch}  "
                      f"best_val={best_val:.4f}")
                break

    # ── Reload best checkpoint and evaluate on test ────────────────────────
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(sd["state"])
    wrapper.eval()
    print(f"[{args.model}_k3] training done  best_val={best_val:.4f}  "
          f"evaluating on test ({data_te['hist'].shape[0]} samples) ...")

    hist_te_n = torch.from_numpy(norm.transform_hist(data_te["hist"]))
    feat_keys_te = _feat_keys_for_model(args.model, data_te)
    bs = args.batch_size * 2

    all_preds_k_norm = []     # list of (b, K, T, 2)
    all_logits = []           # list of (b, K)
    with torch.no_grad():
        for i in range(0, len(hist_te_n), bs):
            hist_b = hist_te_n[i:i + bs].to(device)
            extra = {key: torch.from_numpy(data_te[key][i:i + bs]).to(device)
                     for key in feat_keys_te}
            preds, logits = wrapper.predict(hist_b, extra=extra)
            all_preds_k_norm.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    preds_k_norm = np.concatenate(all_preds_k_norm, axis=0)  # (N, K, T, 2)
    logits_all = np.concatenate(all_logits, axis=0)          # (N, K)

    # Inverse-normalize to physical metres for all K branches
    N, K, T, _ = preds_k_norm.shape
    preds_k = norm.inverse_future(preds_k_norm.reshape(N * K, T, 2)).reshape(N, K, T, 2)
    gt = data_te["future"].astype(np.float32)

    minADE3, minFDE3 = min_ade_fde_k(preds_k, gt)
    # ADE@1: argmax classifier branch
    argmax_k = logits_all.argmax(axis=-1)                    # (N,)
    sel_pred = preds_k[np.arange(N), argmax_k]               # (N, T, 2)
    ADE1, FDE1 = ade_fde_branch(sel_pred, gt)

    print(f"[{args.model}_k3] minADE_3={minADE3:.2f}  minFDE_3={minFDE3:.2f}  "
          f"ADE_1={ADE1:.2f}  FDE_1={FDE1:.2f}")

    return {
        "model": f"{base_name}_k3",
        "k": int(args.k),
        "ADE_1": float(ADE1),
        "FDE_1": float(FDE1),
        "minADE_3": float(minADE3),
        "minFDE_3": float(minFDE3),
        "val_huber": float(best_val),
        "n_test": int(N),
        "seed": int(seed),
        "anchors": anchors.cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    p.add_argument("--track-root", type=Path, required=True,
                   help="Path to a standard_track_v1 directory (Track B).")
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--res-dir", type=Path, required=True)
    p.add_argument("--save-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--k", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = train(args)
    out = args.save_json or (args.res_dir / f"{args.model}_k3_seed{args.seed}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"[done] saved to {out}")
