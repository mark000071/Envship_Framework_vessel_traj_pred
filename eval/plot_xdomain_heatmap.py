"""Cross-domain transfer heatmap for Track A.

Renders a 5×4 heatmap: rows = training pool (Combined, LOSO/X for each X),
cols = test jurisdiction.  Cells = ADE in metres.  One panel per model.
"""
import json, glob
from pathlib import Path
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/mnt/nfs/kun/DeepJSCC/Agent_paper_exp_ALL_folder/ICDE_conferece_dataset_paper/figures_v4")
OUT.mkdir(parents=True, exist_ok=True)

RES = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results_v4_multiscale")
MODELS = [("tcn", "TCN"),
           ("lstm_env_sdf", "LSTM+Env-SDF"),
           ("lstm_social_env_sdf", "LSTM+Soc+Env-SDF")]
POOLS = [("combined", "Combined"),
          ("loso_dma", "LOSO/no-DMA"),
          ("loso_noaa", "LOSO/no-NOAA"),
          ("loso_piraeus", "LOSO/no-Piraeus"),
          ("loso_norway", "LOSO/no-Norway")]
DOMAINS = ["dma", "noaa", "piraeus", "norway"]
DOMAIN_LABEL = ["DMA", "NOAA", "Piraeus", "Norway"]


def best(glob_pat):
    best = None
    for p in sorted(glob.glob(glob_pat)):
        try:
            d = json.loads(Path(p).read_text())
            if d.get("ADE") is None: continue
            if best is None or d["ADE"] < best["ADE"]:
                best = d
        except: pass
    return best


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor("white")
for ax, (slug, label) in zip(axes, MODELS):
    grid = np.full((len(POOLS), len(DOMAINS)), np.nan)
    for r, (pool_dir, _) in enumerate(POOLS):
        b = best(str(RES / "track_a" / pool_dir / f"{slug}_seed*.json"))
        if not b: continue
        for c, dom in enumerate(DOMAINS):
            v = b.get(f"ADE_src_{dom}")
            if v is not None:
                grid[r, c] = v
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=80, vmax=220)
    # Annotate cells
    for r in range(len(POOLS)):
        for c in range(len(DOMAINS)):
            if not np.isnan(grid[r, c]):
                color = "white" if grid[r, c] > 150 else "black"
                ax.text(c, r, f"{grid[r,c]:.0f}", ha="center", va="center",
                        color=color, fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAIN_LABEL, fontsize=9)
    ax.set_yticks(range(len(POOLS)))
    ax.set_yticklabels([p[1] for p in POOLS], fontsize=9)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Test jurisdiction", fontsize=9)
    if slug == "tcn":
        ax.set_ylabel("Training pool", fontsize=9)
    plt.colorbar(im, ax=ax, label="ADE (m)" if slug == "lstm_social_env_sdf" else "")

fig.suptitle("Track A cross-domain transfer matrices (5-seed mean ADE, m)",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
out_pdf = OUT / "xdomain_heatmap_track_a.pdf"
plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
plt.savefig(out_pdf.with_suffix(".png"), dpi=140, bbox_inches="tight")
print(f"wrote {out_pdf}")
