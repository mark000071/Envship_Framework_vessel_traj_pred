#!/usr/bin/env python3
"""Unified DL training framework for EnvShip-Bench.

Usage:
    python eval/train_dl.py --model lstm_2l
    python eval/train_dl.py --model lstm_social_pool --epochs 150
    python eval/train_dl.py --model traisformer
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.normalizer     import GlobalNormalizer
from eval.dataset        import load_split
from eval.context_dataset import load_context_split, MAX_NEIGHBORS, SOCIAL_FEAT, N_ENV_DESC
from eval.metrics        import compute_metrics
from eval.models         import MODEL_REGISTRY, CONTEXT_MODELS, RASTER_MODELS

from eval.paths import DEFAULT_TRACK_ROOT, CKPT_DIR, RES_DIR

# Data root resolves from $ENVSHIP_DATA_ROOT (see eval/paths.py); override a
# single run with --track-root. Defaults point at DMA Track A.
TRACK_ROOT = DEFAULT_TRACK_ROOT


# ---------------------------------------------------------------------------
# TrAISformer-specific training
# ---------------------------------------------------------------------------

def _train_traisformer(model, args, device, norm):
    """Token-level cross-entropy training for TrAISformer."""
    from eval.models.traisformer_local import tokenise_hist, tokenise_future, detokenise_xy, X_BINS, Y_BINS, SOG_BINS, COG_BINS

    print("[traisformer] loading data & tokenising ...")
    raw_tr = load_split(TRACK_ROOT, "train")
    raw_va = load_split(TRACK_ROOT, "val")

    def tokenise_dataset(raw):
        hist_n = norm.transform_hist(raw["hist"])
        fut_n  = norm.transform_future(raw["future"])
        ht = tokenise_hist(hist_n, norm)    # (N, 30, 4)
        ft = tokenise_future(fut_n, norm, hist_n)  # (N, 30, 4)
        return ht, ft

    tr_h, tr_f = tokenise_dataset(raw_tr)
    va_h, va_f = tokenise_dataset(raw_va)

    tr_ds = TensorDataset(torch.from_numpy(tr_h), torch.from_numpy(tr_f))
    va_ds = TensorDataset(torch.from_numpy(va_h), torch.from_numpy(va_f))
    # num_workers=0 / no pin_memory: the datasets are small and the host has
    # only ~2G headroom under the harness; worker subprocess copies + pinned
    # buffers were OOM-killing transformer_nar on Track B.
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=False)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")
    patience_left = args.patience
    ckpt_path = CKPT_DIR / f"{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_tr = 0.0
        for ht_b, ft_b in tr_dl:
            ht_b, ft_b = ht_b.to(device), ft_b.to(device)
            # Input: hist tokens + shifted future (all but last)
            seq_in = torch.cat([ht_b, ft_b[:, :-1, :]], dim=1)    # (B, 59, 4)
            lx, ly, ls, lc = model(seq_in)
            # Targets: predict next token starting from hist[-1] to fut[-1]
            t_x = torch.cat([ht_b[:, 1:, 0],  ft_b[:, :, 0]], dim=1)   # (B, 59)
            t_y = torch.cat([ht_b[:, 1:, 1],  ft_b[:, :, 1]], dim=1)
            t_s = torch.cat([ht_b[:, 1:, 2],  ft_b[:, :, 2]], dim=1)
            t_c = torch.cat([ht_b[:, 1:, 3],  ft_b[:, :, 3]], dim=1)
            loss = (nn.functional.cross_entropy(lx.view(-1, X_BINS),   t_x.view(-1))
                  + nn.functional.cross_entropy(ly.view(-1, Y_BINS),   t_y.view(-1))
                  + nn.functional.cross_entropy(ls.view(-1, SOG_BINS), t_s.view(-1))
                  + nn.functional.cross_entropy(lc.view(-1, COG_BINS), t_c.view(-1))) / 4
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total_tr += loss.item()
        sched.step()

        model.eval()
        total_va = 0.0
        with torch.no_grad():
            for ht_b, ft_b in va_dl:
                ht_b, ft_b = ht_b.to(device), ft_b.to(device)
                seq_in = torch.cat([ht_b, ft_b[:, :-1, :]], dim=1)
                lx, ly, ls, lc = model(seq_in)
                t_x = torch.cat([ht_b[:, 1:, 0], ft_b[:, :, 0]], dim=1)
                t_y = torch.cat([ht_b[:, 1:, 1], ft_b[:, :, 1]], dim=1)
                t_s = torch.cat([ht_b[:, 1:, 2], ft_b[:, :, 2]], dim=1)
                t_c = torch.cat([ht_b[:, 1:, 3], ft_b[:, :, 3]], dim=1)
                loss = (nn.functional.cross_entropy(lx.view(-1, X_BINS), t_x.view(-1))
                      + nn.functional.cross_entropy(ly.view(-1, Y_BINS), t_y.view(-1))
                      + nn.functional.cross_entropy(ls.view(-1, SOG_BINS), t_s.view(-1))
                      + nn.functional.cross_entropy(lc.view(-1, COG_BINS), t_c.view(-1))) / 4
                total_va += loss.item()
        val_loss = total_va / len(va_dl)
        tr_loss  = total_tr / len(tr_dl)
        lr_now   = sched.get_last_lr()[0]
        print(f"[{args.model}] epoch {epoch:03d}  tr={tr_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss; patience_left = args.patience
            torch.save({"state": model.state_dict(), "norm_path": str(ckpt_path.parent / f"{args.model}_norm.json")}, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{args.model}] early stop at epoch {epoch}")
                break

    # Evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state"])
    model.eval()
    raw_te = load_split(TRACK_ROOT, "test")
    te_h, _ = tokenise_dataset(raw_te)
    te_ht = torch.from_numpy(te_h)
    preds = []
    bs    = args.batch_size
    with torch.no_grad():
        for i in range(0, len(te_ht), bs):
            ht_b = te_ht[i:i+bs].to(device)
            p = model.predict_future(ht_b).cpu().numpy()
            preds.append(p)
    pred = np.concatenate(preds)
    gt   = raw_te["future"]
    res  = compute_metrics(pred, gt, raw_te["meta"])
    res["model"] = args.model
    return res


# ---------------------------------------------------------------------------
# Build DataLoader for a split
# ---------------------------------------------------------------------------

# Explicit feature map per context model (avoids fragile string matching)
_MODEL_FEAT_SPEC: dict[str, list[str]] = {
    # v1
    "lstm_social_pool":      ["social_feat", "social_mask"],
    "lstm_social_attn":      ["social_feat", "social_mask"],
    "lstm_env_desc":         ["env_desc"],
    "lstm_env_raster":       ["env_raster"],
    "lstm_social_env":       ["social_feat", "social_mask", "env_desc", "env_raster"],
    # v2
    "lstm_social_attn_v2":   ["social_feat", "social_mask"],
    "lstm_env_desc_v2":      ["env_desc"],
    "lstm_social_env_v2":    ["social_feat", "social_mask", "env_desc"],
    # v3
    "lstm_env_desc_v3":      ["env_desc"],
    "lstm_social_env_v3":    ["social_feat", "social_mask", "env_desc"],
    # v4
    "lstm_env_sdf":               ["env_sdf"],
    "lstm_env_spatial_attn":      ["env_sdf"],
    "lstm_env_binary_spatial_attn": ["env_raster"],
    "lstm_social_env_sdf":        ["env_sdf", "social_feat", "social_mask"],
    # v5
    "lstm_social_attn_v5":        ["social_feat", "social_mask"],
}

# What context data to load per model (keys match load_context_split kwargs)
_MODEL_LOAD_CFG: dict[str, dict[str, bool]] = {
    # v1
    "lstm_social_pool":    {"load_social": True,  "load_env_desc": False, "load_env_raster": False},
    "lstm_social_attn":    {"load_social": True,  "load_env_desc": False, "load_env_raster": False},
    "lstm_env_desc":       {"load_social": False, "load_env_desc": True,  "load_env_raster": False},
    "lstm_env_raster":     {"load_social": False, "load_env_desc": False, "load_env_raster": True},
    "lstm_social_env":     {"load_social": True,  "load_env_desc": True,  "load_env_raster": True},
    # v2
    "lstm_social_attn_v2": {"load_social": True,  "load_env_desc": False, "load_env_raster": False},
    "lstm_env_desc_v2":    {"load_social": False, "load_env_desc": True,  "load_env_raster": False},
    "lstm_social_env_v2":  {"load_social": True,  "load_env_desc": True,  "load_env_raster": False},
    # v3
    "lstm_env_desc_v3":    {"load_social": False, "load_env_desc": True,  "load_env_raster": False},
    "lstm_social_env_v3":  {"load_social": True,  "load_env_desc": True,  "load_env_raster": False},
    # v4
    "lstm_env_sdf":               {"load_social": False, "load_env_desc": False, "load_env_raster": False, "load_env_sdf": True},
    "lstm_env_spatial_attn":      {"load_social": False, "load_env_desc": False, "load_env_raster": False, "load_env_sdf": True},
    "lstm_env_binary_spatial_attn": {"load_social": False, "load_env_desc": False, "load_env_raster": True,  "load_env_sdf": False},
    "lstm_social_env_sdf":        {"load_social": True,  "load_env_desc": False, "load_env_raster": False, "load_env_sdf": True},
    # v5 — uses 8-d social features (CPA + TCPA + |rel_v|)
    "lstm_social_attn_v5":        {"load_social": True,  "load_env_desc": False, "load_env_raster": False, "load_env_sdf": False, "social_feat_dim": 8},
}


def _feat_keys_for_model(model_name: str, data: dict) -> list[str]:
    """Return ordered list of extra feature keys present in data for this model."""
    if model_name not in _MODEL_FEAT_SPEC:
        return []
    return [k for k in _MODEL_FEAT_SPEC[model_name] if k in data]


def _make_loader(data: dict, norm: GlobalNormalizer, batch_size: int,
                 shuffle: bool, model_name: str, device_str: str) -> DataLoader:
    hist_n = torch.from_numpy(norm.transform_hist(data["hist"]))
    fut_n  = torch.from_numpy(norm.transform_future(data["future"]))
    tensors = [hist_n, fut_n]

    for key in _feat_keys_for_model(model_name, data):
        tensors.append(torch.from_numpy(data[key]))

    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def _unpack_batch(batch: tuple, feat_keys: list[str], device: torch.device):
    """Return (hist, fut, extra_kwargs) using ordered feat_keys."""
    hist_b = batch[0].to(device)
    fut_b  = batch[1].to(device)
    extra  = {key: batch[2 + i].to(device) for i, key in enumerate(feat_keys)}
    return hist_b, fut_b, extra


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    seed = int(getattr(args, "seed", 0) or 0)
    if seed:
        import random
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    print(f"[{args.model}] device={device_str}  epochs={args.epochs}  lr={args.lr}  seed={seed}")

    # Allow CLI override of track / ckpt / results dirs so the same script
    # trains on DMA, NOAA, Singapore, Piraeus, ... by just swapping --track-root.
    global TRACK_ROOT, CKPT_DIR, RES_DIR
    if getattr(args, "track_root", None):
        TRACK_ROOT = Path(args.track_root).resolve()
        print(f"[{args.model}] track_root override → {TRACK_ROOT}")
    if getattr(args, "ckpt_dir", None):
        CKPT_DIR = Path(args.ckpt_dir).resolve()
    if getattr(args, "res_dir", None):
        RES_DIR = Path(args.res_dir).resolve()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Track B support: override FUT_STEPS dynamically ───────────────────
    # All models use `FUT_STEPS` (module-level constant) at runtime, not at
    # __init__.  By overriding the constant in each module BEFORE building
    # the model, every forward() loop reads the new value.  This is the
    # least-invasive way to support both Track A (30 future steps) and Track
    # B (180 future steps) without changing 11 model files.
    def _set_fut_steps(n: int):
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

    # ── Load data ──────────────────────────────────────────────────────────
    pool = getattr(args, "pool", None) or None
    is_context = args.model in CONTEXT_MODELS
    if pool:
        # Multi-source pooled training (cross-domain experiments).
        from eval.multi_source_dataset import POOLS, load_pooled_split
        if pool not in POOLS:
            raise SystemExit(f"unknown --pool {pool}; options: {list(POOLS)}")
        sources = POOLS[pool]
        cfg = _MODEL_LOAD_CFG.get(args.model, {}) if is_context else {}
        # Pools beginning with "trackb_" use the Track-B (30min/60min) source roots
        is_track_b = pool.startswith("trackb_")
        cfg["_track_b"] = is_track_b
        print(f"[{args.model}] pool={pool}  sources={sources}  track_b={is_track_b}  cfg={cfg}")
        data_tr = load_pooled_split(sources, "train", **cfg)
        data_va = load_pooled_split(sources, "val",   **cfg)
        data_te = load_pooled_split(sources, "test",  **cfg)
        cfg.pop("_track_b", None)
    elif is_context:
        cfg = _MODEL_LOAD_CFG[args.model]
        print(f"[{args.model}] loading context data  cfg={cfg}")
        data_tr = load_context_split(TRACK_ROOT, "train", **cfg)
        data_va = load_context_split(TRACK_ROOT, "val",   **cfg)
        data_te = load_context_split(TRACK_ROOT, "test",  **cfg)
    else:
        print(f"[{args.model}] loading trajectory-only data ...")
        data_tr = load_split(TRACK_ROOT, "train")
        data_va = load_split(TRACK_ROOT, "val")
        data_te = load_split(TRACK_ROOT, "test")

    # Phase-E support: truncate history and/or future to specified lengths.
    # hist_truncate=K keeps only the LAST K steps of the history; future_truncate=N
    # keeps only the FIRST N steps of the future.
    hist_trunc   = int(getattr(args, "hist_truncate",   0) or 0)
    future_trunc = int(getattr(args, "future_truncate", 0) or 0)
    if hist_trunc > 0:
        for d in (data_tr, data_va, data_te):
            d["hist"] = d["hist"][:, -hist_trunc:, :]
        print(f"[{args.model}] hist truncated to last {hist_trunc} steps")
    if future_trunc > 0:
        for d in (data_tr, data_va, data_te):
            d["future"] = d["future"][:, :future_trunc, :]
        print(f"[{args.model}] future truncated to first {future_trunc} steps")

    # Auto-derive prediction horizon from data; override module-level
    # FUT_STEPS in all model modules so the runtime forward() loop uses the
    # correct number of steps for Track A (30) or Track B (180).
    fut_steps_actual = int(data_tr["future"].shape[1])
    _set_fut_steps(fut_steps_actual)
    print(f"[{args.model}] fut_steps={fut_steps_actual}  "
          f"hist_steps={data_tr['hist'].shape[1]}")

    # ── Normalizer ─────────────────────────────────────────────────────────
    norm_path = CKPT_DIR / f"{args.model}_norm.json"
    if args.load_ckpt and norm_path.exists():
        norm = GlobalNormalizer.load(norm_path)
    else:
        norm = GlobalNormalizer().fit(data_tr["hist"])
        norm.save(norm_path)

    # ── Model ──────────────────────────────────────────────────────────────
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model}] {model.name}  params={n_params:,}")

    ckpt_path = CKPT_DIR / f"{args.model}.pt"

    # TrAISformer has its own training loop
    if args.model == "traisformer":
        return _train_traisformer(model, args, device, norm)

    if args.load_ckpt and ckpt_path.exists():
        print(f"[{args.model}] loading from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state"])
    else:
        # ── DataLoaders ────────────────────────────────────────────────────
        feat_keys = _feat_keys_for_model(args.model, data_tr)
        tr_dl = _make_loader(data_tr, norm, args.batch_size, True,  args.model, device_str)
        va_dl = _make_loader(data_va, norm, args.batch_size, False, args.model, device_str)

        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Single cosine decay — no warm restarts that disrupt learned weights
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val = float("inf")
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_tr = 0.0
            t0 = time.time()
            for batch in tr_dl:
                hist_b, fut_b, extra = _unpack_batch(batch, feat_keys, device)
                # Decay TF ratio to 0 by 40 % of training (not 70 %)
                # so the model spends the majority of training doing free-run inference,
                # eliminating the train/inference distribution mismatch.
                tf_ratio = max(0.0, 0.5 * (1.0 - epoch / (args.epochs * 0.4)))
                opt.zero_grad()
                if extra:
                    pred = model(hist_b, **extra, target=fut_b, tf_ratio=tf_ratio)
                else:
                    pred = model(hist_b, target=fut_b, tf_ratio=tf_ratio)
                loss = loss_fn(pred, fut_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); total_tr += loss.item()
            sched.step()

            # Validation
            model.eval()
            total_va = 0.0
            with torch.no_grad():
                for batch in va_dl:
                    hist_b, fut_b, extra = _unpack_batch(batch, feat_keys, device)
                    if extra:
                        pred = model(hist_b, **extra, target=None, tf_ratio=0.0)
                    else:
                        pred = model(hist_b, target=None, tf_ratio=0.0)
                    total_va += loss_fn(pred, fut_b).item()
            val_loss = total_va / len(va_dl)
            tr_loss  = total_tr / len(tr_dl)
            elapsed  = time.time() - t0

            if epoch % 10 == 0 or epoch <= 5:
                print(f"[{args.model}] ep {epoch:03d}/{args.epochs}  "
                      f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  t={elapsed:.0f}s")

            if val_loss < best_val:
                best_val = val_loss; patience_left = args.patience
                torch.save({"state": model.state_dict()}, ckpt_path)
            else:
                # Early stopping is suspended until --min-epochs (e.g. the end
                # of the teacher-forcing anneal window): stopping inside the
                # scheduled-sampling transition kills runs that would still
                # converge (observed as bimodal seed behaviour).
                min_ep = int(getattr(args, "min_epochs", 0) or 0)
                if epoch <= min_ep:
                    continue
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[{args.model}] early stop at epoch {epoch}  best_val={best_val:.4f}")
                    break

        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state"])
        print(f"[{args.model}] training done  best_val={best_val:.4f}")

    # ── Evaluate on test ───────────────────────────────────────────────────
    model.eval()
    hist_te_n = torch.from_numpy(norm.transform_hist(data_te["hist"]))
    feat_keys_te = _feat_keys_for_model(args.model, data_te)
    bs    = args.batch_size          # was *4; the 1024-wide eval batch spiked host RAM
    preds = []
    with torch.no_grad():
        for i in range(0, len(hist_te_n), bs):
            hist_b = hist_te_n[i:i+bs].to(device)
            extra  = {key: torch.from_numpy(data_te[key][i:i+bs]).to(device)
                      for key in feat_keys_te}
            if extra:
                pred_n = model.predict(hist_b, **extra).cpu().numpy()
            else:
                pred_n = model.predict(hist_b).cpu().numpy()
            preds.append(norm.inverse_future(pred_n))

    pred = np.concatenate(preds)
    gt   = data_te["future"]
    res  = compute_metrics(pred, gt, data_te["meta"])
    res["model"]   = f"{model.name}"
    res["n_params"] = n_params
    print(f"[{args.model}] test ADE={res['ADE']}  FDE={res['FDE']}")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--track-root", type=Path, default=None,
                   help="Path to a Standard-Track-v1 directory. Defaults to the "
                        "DMA track for backward compatibility; pass NOAA / Singapore / "
                        "Piraeus root to retrain on another jurisdiction.")
    p.add_argument("--ckpt-dir",   type=Path, default=None,
                   help="Override checkpoint directory (default: eval/checkpoints).")
    p.add_argument("--res-dir",    type=Path, default=None,
                   help="Override results directory (default: eval/results).")
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--batch-size",type=int,   default=512)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--patience",  type=int,   default=25)
    p.add_argument("--min-epochs", type=int,  default=0,
                   help="No early stopping before this epoch (0 = legacy "
                        "behaviour). Set to the teacher-forcing anneal end, "
                        "e.g. 32 for 80-epoch runs with a 40%% anneal.")
    p.add_argument("--seed",      type=int,   default=0,
                   help="Seed for torch / numpy / random.  Default 0 reproduces "
                        "the legacy non-seeded behaviour; pass 1..N for multi-seed runs.")
    p.add_argument("--pool",      type=str,   default=None,
                   help="Cross-domain pool name (e.g. combined, loso_holdout_dma); "
                        "see eval/multi_source_dataset.py:POOLS.  Mutually exclusive "
                        "with --track-root semantically; pool takes precedence.")
    p.add_argument("--hist-truncate", type=int, default=0,
                   help="Use only the LAST K steps of the history tensor "
                        "(0 = no truncation).  Used by Phase E (horizon sweep).")
    p.add_argument("--future-truncate", type=int, default=0,
                   help="Use only the FIRST N steps of the future tensor "
                        "(0 = no truncation).  Used by Phase E to fix 10-min "
                        "prediction across all observation horizons.")
    p.add_argument("--load-ckpt", action="store_true")
    p.add_argument("--save-json", type=Path,  default=None)
    return p.parse_args()


if __name__ == "__main__":
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    res  = train(args)
    out  = args.save_json or (RES_DIR / f"{args.model}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"[done] saved to {out}")
