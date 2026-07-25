#!/usr/bin/env python3
"""Run all baselines on the Standard Track test split and produce a results table.

Usage:
    # Run analytic baselines only (no GPU required, instant):
    python eval/run_baselines.py --track-root ./multi_type_mini_bench_build/standard_track_v1

    # Train and evaluate LSTM + Transformer:
    python eval/run_baselines.py --track-root ./multi_type_mini_bench_build/standard_track_v1 \
        --train-learned --epochs 15

    # Evaluate from saved checkpoints:
    python eval/run_baselines.py --track-root ./multi_type_mini_bench_build/standard_track_v1 \
        --lstm-ckpt eval/checkpoints/lstm.pt \
        --transformer-ckpt eval/checkpoints/transformer.pt
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.dataset import load_split
from eval.metrics import compute_metrics, format_results_table
from eval.baselines.constant_velocity    import ConstantVelocity
from eval.baselines.constant_acceleration import ConstantAcceleration
from eval.baselines.dead_reckoning       import DeadReckoning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_analytic(model, data: dict) -> np.ndarray:
    return model(data)


def evaluate_model(model_fn, data: dict, name: str) -> dict:
    t0   = time.time()
    pred = model_fn(data)          # (N, 30, 2)
    gt   = data["future"]          # (N, 30, 2)
    results = compute_metrics(pred, gt, data["meta"])
    elapsed = time.time() - t0
    results["model"]   = name
    results["elapsed"] = round(elapsed, 2)
    return results


def print_summary(results: dict) -> None:
    print(f"\n  {'Model':<28} ADE(m)  FDE(m) ADE_3m ADE_6m  |  easy  medium  hard  |  cargo fishing sailing")
    print("  " + "-" * 100)
    for res in results:
        nm   = res.get("model", "?")[:27]
        ade  = res.get("ADE", "-")
        fde  = res.get("FDE", "-")
        a3   = res.get("ADE_3min", "-")
        a6   = res.get("ADE_6min", "-")
        ea   = res.get("ADE_easy",   "-")
        me   = res.get("ADE_medium", "-")
        ha   = res.get("ADE_hard",   "-")
        ca   = res.get("ADE_cargo_tanker",    "-")
        fi   = res.get("ADE_fishing",          "-")
        sa   = res.get("ADE_sailing_leisure",  "-")
        print(f"  {nm:<28} {ade:>6}  {fde:>6} {a3:>6} {a6:>6}  |  {ea:>5}  {me:>6}  {ha:>5}  |  {ca:>6}  {fi:>7}  {sa:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track-root",   type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/multi_type_mini_bench_build/standard_track_v1"))
    p.add_argument("--split",        default="test", choices=["val", "test"])
    p.add_argument("--train-learned",action="store_true", help="Train LSTM + Transformer from scratch")
    p.add_argument("--epochs",       type=int, default=15)
    p.add_argument("--lstm-ckpt",    type=Path, default=None)
    p.add_argument("--transformer-ckpt", type=Path, default=None)
    p.add_argument("--ckpt-dir",     type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/checkpoints"))
    p.add_argument("--results-json", type=Path,
                   default=Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/baseline_results.json"))
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*65}")
    print("EnvShip-Bench Baseline Evaluation")
    print(f"{'='*65}")
    print(f"  track_root : {args.track_root}")
    print(f"  split      : {args.split}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[data] loading {args.split} split ...")
    data = load_split(args.track_root, args.split, use_augmented=True)
    print(f"[data] N={len(data['hist']):,}  hist={data['hist'].shape}  future={data['future'].shape}")
    print()

    all_results = []

    # ── Analytic baselines ────────────────────────────────────────────────
    print("[baselines] running analytic models ...")
    for model in [
        ConstantVelocity(window=5),
        ConstantVelocity(window=3),
        ConstantAcceleration(window=8),
        DeadReckoning(smooth_window=3),
    ]:
        res = evaluate_model(lambda d, m=model: m(d), data, model.name)
        all_results.append(res)
        print(f"  {model.name:<28} ADE={res['ADE']}m  FDE={res['FDE']}m  ({res['elapsed']:.1f}s)")

    # ── Learned baselines ─────────────────────────────────────────────────
    try:
        import torch
        from eval.baselines.lstm_seq2seq       import train as train_lstm, LSTMBaseline
        from eval.baselines.transformer_baseline import train as train_transformer, TransformerBaseline

        lstm_ckpt = args.lstm_ckpt or (args.ckpt_dir / "lstm_seq2seq.pt")
        tf_ckpt   = args.transformer_ckpt or (args.ckpt_dir / "transformer.pt")

        if args.train_learned or not lstm_ckpt.exists():
            print("\n[LSTM] training ...")
            train_lstm(args.track_root, lstm_ckpt, epochs=args.epochs, device=args.device)
        if lstm_ckpt.exists():
            baseline = LSTMBaseline(lstm_ckpt, device=args.device)
            res = evaluate_model(baseline, data, baseline.name)
            all_results.append(res)
            print(f"  {'LSTM Seq2Seq':<28} ADE={res['ADE']}m  FDE={res['FDE']}m  ({res['elapsed']:.1f}s)")

        if args.train_learned or not tf_ckpt.exists():
            print("\n[Transformer] training ...")
            train_transformer(args.track_root, tf_ckpt, epochs=args.epochs, device=args.device)
        if tf_ckpt.exists():
            baseline = TransformerBaseline(tf_ckpt, device=args.device)
            res = evaluate_model(baseline, data, baseline.name)
            all_results.append(res)
            print(f"  {'Transformer Seq2Seq':<28} ADE={res['ADE']}m  FDE={res['FDE']}m  ({res['elapsed']:.1f}s)")

    except ImportError:
        print("[skip] PyTorch not available — skipping learned baselines")

    # ── Print full table ──────────────────────────────────────────────────
    print()
    print_summary(all_results)

    # ── Markdown table ───────────────────────────────────────────────────
    md_table = format_results_table({r["model"]: r for r in all_results})
    print("\n" + "="*65)
    print("Markdown table (copy to paper):")
    print("="*65)
    print(md_table)

    # ── Save JSON ────────────────────────────────────────────────────────
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[done] results saved to {args.results_json}")


if __name__ == "__main__":
    main()
