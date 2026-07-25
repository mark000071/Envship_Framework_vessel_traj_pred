"""Statistical analysis for the DMA multi-seed grid.

Aggregates per-seed JSONs produced by eval/multi_seed_dma.sh and emits:
  - eval/results/multi_seed/summary.json     -- mean / 95% CI / std per model
  - eval/results/multi_seed/significance.json -- paired Wilcoxon p-values
  - figures/multi_seed_ci.pdf                -- ADE error-bar (mean +/- 95% CI)
  - paper/sections/_multi_seed_table_body.tex -- LaTeX block ready to \input

Usage:
    python eval/analyze_multi_seed.py \
        --results-dir /mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results/multi_seed \
        --fig-dir     /mnt/nfs/kun/DeepJSCC/Agent_paper_exp_ALL_folder/ICDE_conferece_dataset_paper/figures \
        --tex-dir     /mnt/nfs/kun/DeepJSCC/Agent_paper_exp_ALL_folder/ICDE_conferece_dataset_paper/paper/sections
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np


MODELS = [
    ("tcn",                 "TCN"),
    ("lstm_env_desc_v2",    "LSTM+Env-Desc-v2"),
    ("lstm_env_sdf",        "LSTM+Env-SDF"),
    ("lstm_social_env_sdf", "LSTM+Social+Env-SDF"),
]

# Optional reference comparators (single-run, deterministic).
DR_REF = {"name": "Dead Reckoning", "ADE": 91.6, "FDE": 231.9}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--fig-dir",     type=Path, required=True)
    p.add_argument("--tex-dir",     type=Path, required=True)
    return p.parse_args()


def t_ci_95(values: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% CI for the mean using the Student-t with n-1 df."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    se = values.std(ddof=1) / math.sqrt(n)
    # n<=5 -> t critical at 95% two-sided is ~2.776 (df=4).  We hard-code
    # the small-n table to avoid scipy dependency at analysis time.
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n-1, 1.96)
    half = t_crit * se
    return values.mean() - half, values.mean() + half


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test, two-sided.

    Implementation note: with only 5 paired samples per model the exact null
    distribution has 2^5 = 32 rearrangements, so we enumerate all of them
    to obtain a real (non-asymptotic) two-sided p-value.  This is more
    honest than the normal-approximation usually employed by scipy.
    """
    diff = a - b
    diff = diff[diff != 0.0]
    n = len(diff)
    if n == 0:
        return {"n": 0, "W": 0.0, "p": 1.0, "mean_diff": 0.0}
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1.0     # 1..n
    sign = np.sign(diff)
    W_pos = float(((sign > 0) * ranks).sum())
    W_neg = float(((sign < 0) * ranks).sum())
    W_obs = min(W_pos, W_neg)
    # Exhaustive enumeration over all 2^n sign assignments
    total = 1 << n
    extreme = 0
    for mask in range(total):
        signs = np.array([+1 if (mask >> i) & 1 else -1 for i in range(n)])
        wp = float(((signs > 0) * ranks).sum())
        wn = float(((signs < 0) * ranks).sum())
        if min(wp, wn) <= W_obs:
            extreme += 1
    return {"n": int(n),
            "W": W_obs,
            "p": extreme / total,
            "mean_diff": float(diff.mean())}


def main():
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.tex_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-seed results ────────────────────────────────────────────
    by_model: dict[str, dict] = {}
    for key, disp in MODELS:
        seed_files = sorted(args.results_dir.glob(f"{key}_seed*.json"))
        ades, fdes = [], []
        for sf in seed_files:
            r = json.loads(sf.read_text())
            ades.append(float(r["ADE"]))
            fdes.append(float(r["FDE"]))
        if not ades:
            print(f"  WARN: no seeds found for {key}")
            continue
        ades = np.asarray(ades, dtype=float)
        fdes = np.asarray(fdes, dtype=float)
        by_model[key] = {
            "name": disp, "n_seeds": len(ades),
            "ADE":  ades.tolist(), "FDE": fdes.tolist(),
            "ADE_mean": float(ades.mean()), "ADE_std": float(ades.std(ddof=1)) if len(ades) > 1 else 0.0,
            "FDE_mean": float(fdes.mean()), "FDE_std": float(fdes.std(ddof=1)) if len(fdes) > 1 else 0.0,
        }
        lo, hi = t_ci_95(ades)
        by_model[key]["ADE_ci95"] = [lo, hi]
        lo, hi = t_ci_95(fdes)
        by_model[key]["FDE_ci95"] = [lo, hi]

    (args.results_dir / "summary.json").write_text(json.dumps(by_model, indent=2))
    print(f"summary -> {args.results_dir/'summary.json'}")

    # ── Paired Wilcoxon: each model vs TCN baseline (paired by seed index) ─
    significance: dict[str, dict] = {}
    if "tcn" in by_model:
        base = np.asarray(by_model["tcn"]["ADE"])
        for key, disp in MODELS:
            if key == "tcn" or key not in by_model:
                continue
            cmp = np.asarray(by_model[key]["ADE"])
            n = min(len(base), len(cmp))
            if n < 2:
                continue
            res = wilcoxon_signed_rank(base[:n], cmp[:n])
            significance[f"tcn_vs_{key}"] = {
                "baseline": "TCN", "challenger": disp,
                "n_seeds": n,
                **res,
                "delta_mean_ADE": float(cmp[:n].mean() - base[:n].mean()),
            }
    (args.results_dir / "significance.json").write_text(json.dumps(significance, indent=2))
    print(f"significance -> {args.results_dir/'significance.json'}")

    # ── ASCII summary table ──────────────────────────────────────────────
    print("\n=== Per-model ADE (mean ± half-CI95) ===")
    for key, disp in MODELS:
        m = by_model.get(key)
        if m is None: continue
        lo, hi = m["ADE_ci95"]
        half = (hi - lo) / 2 if hi != lo else 0.0
        print(f"  {disp:<24}  n={m['n_seeds']:>2}  ADE={m['ADE_mean']:.2f} ± {half:.2f}  (std={m['ADE_std']:.2f})")
    print(f"\n  Dead Reckoning (deterministic):  ADE={DR_REF['ADE']}")

    if significance:
        print("\n=== Paired Wilcoxon vs TCN (two-sided, exact) ===")
        for k, v in significance.items():
            star = " *" if v["p"] < 0.05 else ""
            print(f"  {v['challenger']:<24}  n={v['n_seeds']}  W={v['W']:.1f}  p={v['p']:.4f}  Δ={v['delta_mean_ADE']:+.2f}m{star}")

    # ── LaTeX table block (wrapped in \resizebox for column-fit) ─────────
    lines = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Model & $n$ & ADE $\pm$ 95\% CI (m) & $p$ vs.\ TCN \\",
        r"\midrule",
    ]
    # DR row first (no CI, deterministic)
    lines.append(f"Dead Reckoning            & 1 & ${DR_REF['ADE']:.1f}$ & --- \\\\")
    # Identify best (lowest mean) for bolding
    best_mean = min((m["ADE_mean"] for m in by_model.values()), default=None)
    for key, disp in MODELS:
        m = by_model.get(key)
        if m is None:
            lines.append(f"{disp} & 0 & --- & --- \\\\");  continue
        lo, hi = m["ADE_ci95"]
        half = (hi - lo) / 2
        p_str = "ref" if key == "tcn" else (
            f"${significance.get(f'tcn_vs_{key}', {}).get('p', '---'):.3f}$"
            if isinstance(significance.get(f"tcn_vs_{key}", {}).get("p"), float)
            else "---"
        )
        if m["ADE_mean"] == best_mean:
            lines.append(f"\\textbf{{{disp}}} & {m['n_seeds']} & "
                          f"$\\mathbf{{{m['ADE_mean']:.2f} \\pm {half:.2f}}}$ & {p_str} \\\\")
        else:
            lines.append(f"{disp} & {m['n_seeds']} & "
                          f"${m['ADE_mean']:.2f} \\pm {half:.2f}$ & {p_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    tex_path = args.tex_dir / "_multi_seed_table_body.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\ntex -> {tex_path}")

    # ── CI figure (matplotlib) ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        names, means, halfs = [], [], []
        for key, disp in MODELS:
            m = by_model.get(key)
            if m is None: continue
            names.append(disp.replace("LSTM+", "L+"))
            means.append(m["ADE_mean"])
            lo, hi = m["ADE_ci95"]
            halfs.append((hi - lo) / 2)
        # Add DR reference horizontal line
        ax.axhline(DR_REF["ADE"], color="#7f7f7f", lw=1.2, ls="--",
                   label=f"Dead Reckoning ({DR_REF['ADE']:.1f})")
        x = np.arange(len(names))
        ax.errorbar(x, means, yerr=halfs, fmt="o", color="#2A6FDB",
                    capsize=4, markersize=7, lw=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("ADE (m)", fontsize=10)
        ax.set_title("DMA multi-seed ADE (mean $\\pm$ 95\\% CI, 5 seeds)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_fig = args.fig_dir / "multi_seed_ci.pdf"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"figure -> {out_fig}")
    except Exception as e:
        print(f"  WARN: figure generation skipped ({e})")


if __name__ == "__main__":
    main()
