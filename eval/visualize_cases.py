#!/usr/bin/env python3
"""
EnvShip-Bench Dataset Case Visualizations.

Produces 4 figures saved to eval/visualizations/:
  1. scene_overview.png       — 4 scene types × 5 cases each (20 panels)
  2. difficulty_overview.png  — easy/medium/hard × 5 cases each (15 panels)
  3. social_overview.png      — ≥1 / ≥2 / ≥3 neighbor cases (12 panels)
  4. env_detail.png           — 8 selected cases, each with raster map +
                                signed-distance field + env descriptor chart

Each raster panel shows:
  · Land (tan) / Water (light blue) / Navigable (sky blue)
  · Natural boundary (coastline, green overlay)
  · Manmade boundary (pier/quay, red overlay)
  · Signed-distance contours (in detail view)
  · History track (blue) → GT future (green dashed)
  · Neighbor ships (orange circle + velocity arrow)
"""

from __future__ import annotations
import sys, gzip, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.context_dataset import load_context_split

# ── Paths ────────────────────────────────────────────────────────────────────
TRACK_ROOT = Path("multi_type_mini_bench_build/standard_track_v1")
RAST_DIR   = TRACK_ROOT / "context_v1/environment/rasters/test"
OUT_DIR    = Path("eval/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_RADIUS_M = 5000          # 5 km half-width
GRID_SIZE      = 128
M_PER_PIX      = 2 * PATCH_RADIUS_M / GRID_SIZE   # 78.125 m/pixel
EXTENT         = [-PATCH_RADIUS_M, PATCH_RADIUS_M,
                  -PATCH_RADIUS_M, PATCH_RADIUS_M]

# ── Colour palette ───────────────────────────────────────────────────────────
C_OCEAN    = np.array([28,  107, 162], dtype=np.uint8)   # deep water
C_WATER    = np.array([100, 180, 230], dtype=np.uint8)   # coastal water
C_NAV      = np.array([160, 218, 245], dtype=np.uint8)   # navigable
C_LAND     = np.array([210, 180, 140], dtype=np.uint8)   # tan land
C_NAT_BDRY = np.array([ 34, 139,  34], dtype=np.uint8)  # green coastline
C_MAN_BDRY = np.array([180,  30,  30], dtype=np.uint8)  # red pier/quay

TRAJ_HIST  = "#1f77b4"   # blue
TRAJ_GT    = "#2ca02c"   # green
TRAJ_NB    = "#d62728"   # red for neighbours
MARKER_S   = 60


# ── Raster helpers ───────────────────────────────────────────────────────────

def render_raster(rast: np.ndarray) -> np.ndarray:
    """Convert 6-channel binary raster (6,128,128) → RGB uint8 (128,128,3)."""
    land  = rast[0]; water = rast[1]; nav  = rast[2]
    nat_b = rast[3]; man_b = rast[4]

    H, W = land.shape
    rgb  = np.full((H, W, 3), C_OCEAN, dtype=np.uint8)
    rgb[water == 1] = C_WATER
    rgb[nav   == 1] = C_NAV
    rgb[land  == 1] = C_LAND
    rgb[nat_b == 1] = C_NAT_BDRY
    rgb[man_b == 1] = C_MAN_BDRY
    return rgb


def load_rasters():
    """Return (rasters, id_to_idx) with mmap for memory efficiency."""
    rasters = np.load(RAST_DIR / "masks.npy", mmap_mode="r")
    ids     = np.load(RAST_DIR / "sample_ids.npy", allow_pickle=True)
    id_map  = {str(s): i for i, s in enumerate(ids.tolist())}
    return rasters, id_map


def load_sdist():
    """Return (sds_shore, sds_nav) memory-mapped signed-distance arrays."""
    sds = np.load(RAST_DIR / "signed_dist_shore.npy", mmap_mode="r")
    sdn = np.load(RAST_DIR / "signed_dist_nav.npy",   mmap_mode="r")
    return sds, sdn


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_trajectory(ax, hist, fut,
                     soc_feat=None, soc_mask=None,
                     show_legend: bool = False,
                     alpha_hist: float = 1.0):
    """Overlay history + GT future + neighbours on an already-set-up axis."""
    hx, hy = hist[:, 0], hist[:, 1]
    fx, fy = fut[:, 0],  fut[:, 1]

    # History track
    ax.plot(hx, hy, "-", color=TRAJ_HIST, lw=2.0, alpha=alpha_hist, zorder=4)
    ax.scatter(hx[0],  hy[0],  c=TRAJ_HIST, s=MARKER_S, marker="o",
               zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter(hx[-1], hy[-1], c=TRAJ_HIST, s=MARKER_S*1.5, marker="^",
               zorder=5, edgecolors="white", linewidths=0.8)

    # GT future
    ax.plot(fx, fy, "--", color=TRAJ_GT, lw=2.0, zorder=4)
    ax.scatter(fx[-1], fy[-1], c=TRAJ_GT, s=MARKER_S*1.5, marker="^",
               zorder=5, edgecolors="white", linewidths=0.8)

    # Neighbours
    if soc_feat is not None and soc_mask is not None:
        for i in range(len(soc_mask)):
            if soc_mask[i] < 0.5:
                continue
            # De-normalise: feat = [rel_x/3000, rel_y/3000, rel_vx/5, rel_vy/5, dist/3000]
            nb_rx = float(soc_feat[i, 0]) * 3000.0
            nb_ry = float(soc_feat[i, 1]) * 3000.0
            nb_vx = float(soc_feat[i, 2]) * 5.0      # m/s
            nb_vy = float(soc_feat[i, 3]) * 5.0
            nb_x  = hx[-1] + nb_rx
            nb_y  = hy[-1] + nb_ry
            # 60 s extrapolation for arrow
            ax.scatter(nb_x, nb_y, c=TRAJ_NB, s=MARKER_S*1.2, marker="D",
                       zorder=6, edgecolors="white", linewidths=0.8)
            if abs(nb_vx) + abs(nb_vy) > 0.3:
                ax.annotate("", xy=(nb_x + nb_vx*60, nb_y + nb_vy*60),
                            xytext=(nb_x, nb_y),
                            arrowprops=dict(arrowstyle="-|>", color=TRAJ_NB,
                                            lw=1.5, mutation_scale=12),
                            zorder=7)

    if show_legend:
        legend_elements = [
            Line2D([0],[0], color=TRAJ_HIST, lw=2, label="History"),
            Line2D([0],[0], color=TRAJ_GT, lw=2, ls="--", label="GT Future"),
            Line2D([0],[0], color=TRAJ_NB,  marker="D", lw=0,
                   markersize=7, label="Neighbour"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right",
                  framealpha=0.85)


def _setup_raster_ax(ax, rast_rgb, title: str = ""):
    """Show RGB raster and configure axis."""
    ax.imshow(rast_rgb, origin="upper", extent=EXTENT, interpolation="nearest")
    # Cross-hair at anchor (origin)
    ax.axhline(0, color="white", lw=0.4, alpha=0.4, zorder=1)
    ax.axvline(0, color="white", lw=0.4, alpha=0.4, zorder=1)
    ax.set_xlim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
    ax.set_ylim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
    ax.set_aspect("equal")
    ax.set_xticks([-4000, -2000, 0, 2000, 4000])
    ax.set_xticklabels(["-4km", "-2km", "0", "+2km", "+4km"], fontsize=6)
    ax.set_yticks([-4000, -2000, 0, 2000, 4000])
    ax.set_yticklabels(["-4km", "-2km", "0", "+2km", "+4km"], fontsize=6)
    if title:
        ax.set_title(title, fontsize=8, pad=3)


def _add_scale_bar(ax, length_m=2000):
    """Add a simple scale bar."""
    x0 = PATCH_RADIUS_M * 0.55
    y0 = -PATCH_RADIUS_M * 0.90
    ax.plot([x0, x0 + length_m], [y0, y0], "w-", lw=2.5, solid_capstyle="butt", zorder=8)
    ax.text(x0 + length_m/2, y0 + 250, f"{length_m//1000} km",
            ha="center", va="bottom", fontsize=6, color="white", fontweight="bold", zorder=8)


def _env_desc_panel(ax, env_desc: np.ndarray, meta: dict):
    """Draw env descriptor summary as horizontal bar chart."""
    ax.set_facecolor("#f9f9f9")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    labels = [
        "Water ratio",
        "Navigable ratio",
        "Barrier density × 10",
        "Nat. boundary × 10",
        "Man. boundary × 10",
        "Shore dist (norm)",
        "Manmade dist (norm)",
        "Has natural bdry",
        "Has manmade bdry",
        "Env quality",
    ]
    # env_desc layout: [open_water, nearshore, harbor, constrained,
    #                   nearest_shore/5000, nearest_manmade/5000,
    #                   water_ratio, navigable_ratio, barrier_density*10,
    #                   nat_bdry_density*10, manmade_bdry_density*10,
    #                   has_natural, has_manmade, env_quality]
    vals = [
        float(env_desc[6]),               # water_ratio
        float(env_desc[7]),               # navigable_ratio
        float(env_desc[8]) * 10,          # barrier_density
        float(env_desc[9]) * 10,          # nat boundary density
        float(env_desc[10]) * 10,         # manmade boundary density
        float(env_desc[4]),               # shore dist (0-1)
        float(env_desc[5]),               # manmade dist (0-1)
        float(env_desc[11]),              # has_natural_boundary
        float(env_desc[12]),              # has_manmade_boundary
        float(env_desc[13]),              # env_quality
    ]
    colours = [
        "#4C9BE8", "#7EC8E3", "#E07B54", "#4CAF50", "#E57373",
        "#90A4AE", "#78909C", "#66BB6A", "#EF5350", "#FFA726",
    ]
    for j, (lab, val, col) in enumerate(zip(labels, vals, colours)):
        y = 9 - j
        ax.barh(y, min(val, 1.0), height=0.6, color=col, alpha=0.8, zorder=2)
        ax.text(-0.03, y, lab, ha="right", va="center", fontsize=6.5, color="#333")
        ax.text(min(val, 1.0) + 0.02, y, f"{val:.2f}", ha="left", va="center",
                fontsize=6.5, color="#333")
    ax.axvline(0, color="#aaa", lw=0.5)

    # Scene type and metadata annotation
    sc   = meta.get("scene_type", "?")
    diff = meta.get("difficulty_tier", "?")
    ship = meta.get("ship_group", "?")
    ax.text(0.5, -0.4, f"Scene: {sc}  |  Difficulty: {diff}  |  Ship: {ship}",
            ha="center", va="center", fontsize=7, color="#444",
            transform=ax.transAxes)

    # Scene one-hot bar (top)
    scene_names = ["open_water", "nearshore", "harbor", "constrained"]
    scene_cols  = ["#4C9BE8", "#4CAF50", "#E57373", "#9C27B0"]
    for k, (sn, scol) in enumerate(zip(scene_names, scene_cols)):
        v = float(env_desc[k])
        ax.barh(10.2, v, left=k * 0.26, height=0.4, color=scol, alpha=0.85)
        ax.text(k * 0.26 + 0.13, 10.2, sn, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold")
    ax.set_ylim(-0.8, 10.8)


# ── Case selection ────────────────────────────────────────────────────────────

def pick_cases(data: dict, scene_type: str | None = None,
               difficulty: str | None = None,
               min_neighbors: int = 0,
               n: int = 5,
               seed: int = 42) -> list[int]:
    """Return `n` sample indices matching filters, sorted by trajectory curvature."""
    meta    = data["meta"]
    scenes  = np.array(meta["scene_type"])
    diffs   = np.array(meta["difficulty_tier"])
    nb_cnt  = data.get("neighbor_count", np.zeros(len(scenes), dtype=int))

    mask = np.ones(len(scenes), dtype=bool)
    if scene_type is not None:
        mask &= (scenes == scene_type)
    if difficulty is not None:
        mask &= (diffs == difficulty)
    if min_neighbors > 0:
        mask &= (nb_cnt >= min_neighbors)

    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []

    # Score by trajectory curvature (interesting = more curved)
    curvatures = []
    for i in idxs:
        fut = data["future"][i]
        dx = np.diff(fut[:, 0]); dy = np.diff(fut[:, 1])
        ang = np.arctan2(dy, dx)
        ang_change = np.abs(np.diff(np.unwrap(ang))).sum()
        curvatures.append(ang_change)
    curvatures = np.array(curvatures)

    # Mix: half most-curved, half random (for variety)
    rng     = np.random.default_rng(seed)
    n_curve = min(n // 2 + 1, len(idxs))
    top_idx = np.argsort(curvatures)[-n_curve:][::-1]
    rem_idx = np.setdiff1d(np.arange(len(idxs)), top_idx)
    if len(rem_idx) >= n - n_curve:
        rand_idx = rng.choice(rem_idx, size=n - n_curve, replace=False)
    else:
        rand_idx = rem_idx
    chosen = idxs[np.concatenate([top_idx, rand_idx])][:n]
    return chosen.tolist()


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_scene_overview(data, rasters, id_map):
    """4 scene types × 5 cases each = 20 panels."""
    print("  Building scene_overview.png ...")
    scene_types = ["open_water", "nearshore", "constrained", "harbor"]
    scene_label = ["Open Water", "Nearshore", "Constrained", "Harbor"]
    scene_cols  = ["#4C9BE8", "#4CAF50", "#9C27B0", "#E57373"]
    N_PER_ROW   = 5

    fig, axes = plt.subplots(4, N_PER_ROW, figsize=(N_PER_ROW * 4, 4 * 4.2))
    fig.patch.set_facecolor("#1a1a2e")

    for row, (sc, slab, scol) in enumerate(zip(scene_types, scene_label, scene_cols)):
        idxs = pick_cases(data, scene_type=sc, n=N_PER_ROW, seed=row * 17)
        # Pad with random if not enough
        if len(idxs) < N_PER_ROW:
            idxs = idxs + pick_cases(data, scene_type=sc, n=N_PER_ROW, seed=99)
            idxs = list(dict.fromkeys(idxs))[:N_PER_ROW]

        for col, idx in enumerate(idxs[:N_PER_ROW]):
            ax = axes[row][col]
            sid = str(data["meta"]["sample_id"][idx])
            ridx = id_map.get(sid, -1)
            rast_rgb = render_raster(rasters[ridx]) if ridx >= 0 else \
                       np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)

            nb_count = int(data["neighbor_count"][idx])
            diff     = str(data["meta"]["difficulty_tier"][idx])
            speed    = float(data["meta"]["avg_speed_knots"][idx])

            title = (f"{diff}  |  {speed:.1f}kn"
                     + (f"  |  {nb_count}nb" if nb_count > 0 else ""))

            _setup_raster_ax(ax, rast_rgb, title)
            _add_scale_bar(ax)

            sf = data["social_feat"][idx]  if "social_feat" in data else None
            sm = data["social_mask"][idx]  if "social_mask" in data else None
            _plot_trajectory(ax, data["hist"][idx], data["future"][idx],
                             sf, sm, show_legend=(col == N_PER_ROW - 1))

            if col == 0:
                ax.set_ylabel(slab, fontsize=10, color=scol, fontweight="bold", labelpad=6)

    fig.suptitle("EnvShip-Bench — Scene Type Overview", fontsize=14, color="white",
                 fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.6)
    out = OUT_DIR / "scene_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_difficulty_overview(data, rasters, id_map):
    """3 difficulty tiers × 5 cases each = 15 panels."""
    print("  Building difficulty_overview.png ...")
    diffs  = ["easy", "medium", "hard"]
    dcols  = ["#4CAF50", "#FF9800", "#F44336"]
    dlabs  = ["Easy (straight-line, predictable)", "Medium (moderate course change)",
              "Hard (complex manoeuvre)"]
    N_PER_ROW = 5

    fig, axes = plt.subplots(3, N_PER_ROW, figsize=(N_PER_ROW * 4, 3 * 4.2))
    fig.patch.set_facecolor("#1a1a2e")

    for row, (diff, dcol, dlab) in enumerate(zip(diffs, dcols, dlabs)):
        idxs = pick_cases(data, difficulty=diff, n=N_PER_ROW, seed=row * 31)
        for col, idx in enumerate(idxs[:N_PER_ROW]):
            ax = axes[row][col]
            sid  = str(data["meta"]["sample_id"][idx])
            ridx = id_map.get(sid, -1)
            rast_rgb = render_raster(rasters[ridx]) if ridx >= 0 else \
                       np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)

            sc    = str(data["meta"]["scene_type"][idx])
            speed = float(data["meta"]["avg_speed_knots"][idx])
            nb    = int(data["neighbor_count"][idx])
            title = (f"{sc}  |  {speed:.1f}kn"
                     + (f"  |  {nb}nb" if nb > 0 else ""))

            _setup_raster_ax(ax, rast_rgb, title)
            _add_scale_bar(ax)

            sf = data["social_feat"][idx] if "social_feat" in data else None
            sm = data["social_mask"][idx] if "social_mask" in data else None
            _plot_trajectory(ax, data["hist"][idx], data["future"][idx],
                             sf, sm, show_legend=(col == N_PER_ROW - 1))

            if col == 0:
                ax.set_ylabel(dlab, fontsize=9, color=dcol, fontweight="bold",
                              labelpad=6, wrap=True)

    fig.suptitle("EnvShip-Bench — Trajectory Difficulty Overview", fontsize=14,
                 color="white", fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.6)
    out = OUT_DIR / "difficulty_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_social_overview(data, rasters, id_map):
    """Social interaction cases: 3 rows (≥1, ≥2, ≥3 neighbours) × 4 cases."""
    print("  Building social_overview.png ...")
    N_PER_ROW = 4
    rows_cfg  = [
        (1, "1 neighbour",   "#FF9800"),
        (2, "2 neighbours",  "#F44336"),
        (3, "≥3 neighbours", "#9C27B0"),
    ]

    fig, axes = plt.subplots(3, N_PER_ROW, figsize=(N_PER_ROW * 4, 3 * 4.2))
    fig.patch.set_facecolor("#1a1a2e")

    for row, (nb_min, rlabel, rcol) in enumerate(rows_cfg):
        nb_cnt = data["neighbor_count"]
        if nb_min == 1:
            mask_exact = (nb_cnt == 1)
        elif nb_min == 2:
            mask_exact = (nb_cnt == 2)
        else:
            mask_exact = (nb_cnt >= 3)

        idxs_all = np.where(mask_exact)[0]
        rng = np.random.default_rng(42 + row)
        if len(idxs_all) >= N_PER_ROW:
            idxs = rng.choice(idxs_all, size=N_PER_ROW, replace=False).tolist()
        else:
            idxs = idxs_all.tolist()

        for col, idx in enumerate(idxs[:N_PER_ROW]):
            ax = axes[row][col]
            sid  = str(data["meta"]["sample_id"][idx])
            ridx = id_map.get(sid, -1)
            rast_rgb = render_raster(rasters[ridx]) if ridx >= 0 else \
                       np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)

            sc    = str(data["meta"]["scene_type"][idx])
            speed = float(data["meta"]["avg_speed_knots"][idx])
            diff  = str(data["meta"]["difficulty_tier"][idx])
            nb    = int(nb_cnt[idx])
            title = f"{sc}  |  {diff}  |  {speed:.1f}kn  |  {nb}nb"

            _setup_raster_ax(ax, rast_rgb, title)
            _add_scale_bar(ax)
            _plot_trajectory(ax, data["hist"][idx], data["future"][idx],
                             data["social_feat"][idx], data["social_mask"][idx],
                             show_legend=(col == N_PER_ROW - 1))

            if col == 0:
                ax.set_ylabel(rlabel, fontsize=10, color=rcol, fontweight="bold",
                              labelpad=6)

    fig.suptitle("EnvShip-Bench — Social Interaction Cases", fontsize=14,
                 color="white", fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.6)
    out = OUT_DIR / "social_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_env_detail(data, rasters, id_map, sds_shore, sds_nav):
    """
    Detailed 3-panel view for 8 representative cases:
      Col 0: Raster map (6-channel RGB) + trajectory
      Col 1: Signed-distance field (shore) + trajectory
      Col 2: Environment descriptor bar chart
    """
    print("  Building env_detail.png ...")
    # Select 8 interesting cases: 2 each from nearshore/constrained/harbor + 2 hard open_water
    selected = []
    for sc, nd, nn in [("harbor",     None, 0),
                        ("constrained", None, 0),
                        ("nearshore",  "hard",   0),
                        ("nearshore",  "medium", 0),
                        ("open_water", "hard",   2),   # hard with neighbours
                        ("open_water", "hard",   0),
                        ("open_water", "medium", 1),
                        ("open_water", "medium", 0),
                        ]:
        idxs = pick_cases(data, scene_type=sc, difficulty=nd,
                          min_neighbors=nn, n=2, seed=7)
        selected.extend(idxs[:1])
        if len(selected) >= 8:
            break
    # Fill remaining with harbour + hard
    all_harbor = np.where(np.array(data["meta"]["scene_type"]) == "harbor")[0]
    for i in all_harbor:
        if i not in selected:
            selected.append(int(i))
        if len(selected) >= 8:
            break
    selected = selected[:8]

    N = len(selected)
    fig = plt.figure(figsize=(18, N * 3.6))
    fig.patch.set_facecolor("#1a1a2e")

    # Custom colormap for signed-distance field
    cmap_sdf = LinearSegmentedColormap.from_list(
        "sdf", [(0,"#5B2333"), (0.45,"#d4a373"), (0.5,"white"), (0.55,"#90e0ef"), (1,"#03045e")],
    )

    for row, idx in enumerate(selected):
        sid  = str(data["meta"]["sample_id"][idx])
        ridx = id_map.get(sid, -1)
        rast_rgb  = render_raster(rasters[ridx]) if ridx >= 0 else \
                    np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)
        sdf_shore = sds_shore[ridx].astype(float) if ridx >= 0 else \
                    np.zeros((128, 128))
        # ── Panel 0: raster ──────────────────────────────────────────
        ax0 = fig.add_subplot(N, 3, row * 3 + 1)
        _setup_raster_ax(ax0, rast_rgb)
        _add_scale_bar(ax0)
        sf = data["social_feat"][idx] if "social_feat" in data else None
        sm = data["social_mask"][idx] if "social_mask" in data else None
        _plot_trajectory(ax0, data["hist"][idx], data["future"][idx], sf, sm,
                         show_legend=(row == 0))
        # Row label
        sc   = str(data["meta"]["scene_type"][idx])
        diff = str(data["meta"]["difficulty_tier"][idx])
        nb   = int(data["neighbor_count"][idx])
        ship = str(data["meta"]["ship_group"][idx])
        row_label = f"[{row+1}] {sc} · {diff} · {ship} · {nb}nb"
        ax0.set_ylabel(row_label, fontsize=7.5, color="white", labelpad=4)
        if row == 0:
            ax0.set_title("Raster Map (6-ch OSM)", fontsize=9, color="white", pad=4)

        # ── Panel 1: signed-distance field ───────────────────────────
        ax1 = fig.add_subplot(N, 3, row * 3 + 2)
        vmax = max(abs(sdf_shore).max(), 1)
        im = ax1.imshow(sdf_shore, origin="upper", extent=EXTENT,
                        cmap=cmap_sdf, vmin=-vmax, vmax=vmax,
                        interpolation="bilinear")
        # Shoreline contour at 0
        ax1.contour(sdf_shore, levels=[0], colors=["#f0f0f0"], linewidths=[1.2],
                    extent=EXTENT, origin="upper")
        ax1.set_xlim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
        ax1.set_ylim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
        ax1.set_aspect("equal")
        ax1.tick_params(labelsize=6)
        ax1.axhline(0, color="white", lw=0.3, alpha=0.4)
        ax1.axvline(0, color="white", lw=0.3, alpha=0.4)
        _plot_trajectory(ax1, data["hist"][idx], data["future"][idx], sf, sm)
        if row == 0:
            ax1.set_title("Signed-Distance Field (shore)", fontsize=9, color="white", pad=4)
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5, colors="white")
        cbar.set_label("dist to shore (m)", fontsize=5.5, color="white")
        cbar.outline.set_edgecolor("white")

        # ── Panel 2: env descriptor chart ────────────────────────────
        ax2 = fig.add_subplot(N, 3, row * 3 + 3)
        meta_row = {k: str(v[idx]) for k, v in data["meta"].items()}
        _env_desc_panel(ax2, data["env_desc"][idx], meta_row)
        if row == 0:
            ax2.set_title("Environment Descriptor", fontsize=9, color="#333", pad=4)

    fig.suptitle("EnvShip-Bench — Detailed Case Analysis (Raster · SDF · Env Descriptor)",
                 fontsize=13, color="white", fontweight="bold", y=1.005)
    plt.tight_layout(pad=0.8, h_pad=0.5, w_pad=0.8)
    out = OUT_DIR / "env_detail.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_harbor_all(data, rasters, id_map, sds_shore):
    """Show all harbor cases (rare: only 15 in test set) in a single figure."""
    print("  Building harbor_all.png ...")
    harbor_idx = np.where(np.array(data["meta"]["scene_type"]) == "harbor")[0]
    N = len(harbor_idx)
    ncols = 5
    nrows = (N + ncols - 1) // ncols

    cmap_sdf = LinearSegmentedColormap.from_list(
        "sdf", [(0,"#5B2333"),(0.45,"#d4a373"),(0.5,"white"),(0.55,"#90e0ef"),(1,"#03045e")]
    )

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 8, nrows * 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    for k, idx in enumerate(harbor_idx):
        row_k = k // ncols
        col_k = (k % ncols) * 2

        sid  = str(data["meta"]["sample_id"][idx])
        ridx = id_map.get(sid, -1)
        rast_rgb  = render_raster(rasters[ridx]) if ridx >= 0 else \
                    np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)
        sdf_shore = sds_shore[ridx].astype(float) if ridx >= 0 else np.zeros((128, 128))

        ax0 = axes[row_k][col_k]
        ax1 = axes[row_k][col_k + 1]

        nb   = int(data["neighbor_count"][idx])
        diff = str(data["meta"]["difficulty_tier"][idx])
        spd  = float(data["meta"]["avg_speed_knots"][idx])
        ship = str(data["meta"]["ship_group"][idx])

        sf = data["social_feat"][idx] if "social_feat" in data else None
        sm = data["social_mask"][idx] if "social_mask" in data else None

        _setup_raster_ax(ax0, rast_rgb,
                         f"#{k+1} {diff} | {ship} | {spd:.1f}kn | {nb}nb")
        _add_scale_bar(ax0)
        _plot_trajectory(ax0, data["hist"][idx], data["future"][idx], sf, sm,
                         show_legend=(k == 0))

        vmax = max(abs(sdf_shore).max(), 1)
        ax1.imshow(sdf_shore, origin="upper", extent=EXTENT,
                   cmap=cmap_sdf, vmin=-vmax, vmax=vmax, interpolation="bilinear")
        ax1.contour(sdf_shore, levels=[0], colors=["#f0f0f0"], linewidths=[1.5],
                    extent=EXTENT, origin="upper")
        ax1.set_xlim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
        ax1.set_ylim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
        ax1.set_aspect("equal")
        ax1.tick_params(labelsize=5)
        _plot_trajectory(ax1, data["hist"][idx], data["future"][idx], sf, sm)

    # Hide unused axes
    for k in range(N, nrows * ncols):
        row_k = k // ncols; col_k = (k % ncols) * 2
        axes[row_k][col_k].axis("off")
        axes[row_k][col_k + 1].axis("off")

    fig.suptitle(f"EnvShip-Bench — All {N} Harbor Cases (test split)",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.4)
    out = OUT_DIR / "harbor_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_constrained_nearshore(data, rasters, id_map, sds_shore):
    """8 constrained + 8 nearshore cases, raster + SDF side by side."""
    print("  Building constrained_nearshore.png ...")
    configs = [
        ("constrained", 8, "Constrained Channel"),
        ("nearshore",   8, "Nearshore"),
    ]
    N_PER  = 8
    N_COLS = 4    # 4 cases per row, 2 rows per scene type

    cmap_sdf = LinearSegmentedColormap.from_list(
        "sdf", [(0,"#5B2333"),(0.45,"#d4a373"),(0.5,"white"),(0.55,"#90e0ef"),(1,"#03045e")]
    )

    # 2 scene types × 2 rows × 4 cols × 2 panels = 32 subplots
    fig = plt.figure(figsize=(N_COLS * 8, 4 * 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(4, N_COLS * 2, figure=fig, hspace=0.55, wspace=0.25)

    global_row = 0
    for sc, n_req, sc_label in configs:
        idxs = pick_cases(data, scene_type=sc, n=n_req, seed=13)
        for k, idx in enumerate(idxs[:n_req]):
            row_k = global_row + k // N_COLS
            col_k = (k % N_COLS) * 2

            sid  = str(data["meta"]["sample_id"][idx])
            ridx = id_map.get(sid, -1)
            rast_rgb  = render_raster(rasters[ridx]) if ridx >= 0 else \
                        np.full((128, 128, 3), C_OCEAN, dtype=np.uint8)
            sdf_shore = sds_shore[ridx].astype(float) if ridx >= 0 else np.zeros((128, 128))

            ax0 = fig.add_subplot(gs[row_k, col_k])
            ax1 = fig.add_subplot(gs[row_k, col_k + 1])

            nb   = int(data["neighbor_count"][idx])
            diff = str(data["meta"]["difficulty_tier"][idx])
            spd  = float(data["meta"]["avg_speed_knots"][idx])
            sf = data["social_feat"][idx] if "social_feat" in data else None
            sm = data["social_mask"][idx] if "social_mask" in data else None

            title = f"{diff} | {spd:.1f}kn" + (f" | {nb}nb" if nb > 0 else "")
            _setup_raster_ax(ax0, rast_rgb, title)
            _add_scale_bar(ax0)
            _plot_trajectory(ax0, data["hist"][idx], data["future"][idx], sf, sm)

            vmax = max(abs(sdf_shore).max(), 1)
            ax1.imshow(sdf_shore, origin="upper", extent=EXTENT,
                       cmap=cmap_sdf, vmin=-vmax, vmax=vmax, interpolation="bilinear")
            ax1.contour(sdf_shore, levels=[0], colors=["#f0f0f0"], linewidths=[1.5],
                        extent=EXTENT, origin="upper")
            ax1.set_xlim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
            ax1.set_ylim(-PATCH_RADIUS_M, PATCH_RADIUS_M)
            ax1.set_aspect("equal")
            ax1.tick_params(labelsize=5)
            _plot_trajectory(ax1, data["hist"][idx], data["future"][idx], sf, sm)

            if col_k == 0:
                ax0.set_ylabel(f"{sc_label}\n#{k+1}", fontsize=8, color="white",
                               fontweight="bold", labelpad=4)

        global_row += 2

    # Add section labels
    fig.text(0.01, 0.98, "■ Constrained Channel Cases", color="#9C27B0",
             fontsize=11, fontweight="bold", va="top")
    fig.text(0.01, 0.50, "■ Nearshore Cases", color="#4CAF50",
             fontsize=11, fontweight="bold", va="top")

    fig.suptitle("EnvShip-Bench — Constrained & Nearshore Cases (Raster | SDF)",
                 fontsize=13, color="white", fontweight="bold", y=1.02)
    out = OUT_DIR / "constrained_nearshore.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading test data with full context ...")
    data = load_context_split(TRACK_ROOT, "test",
                               load_social=True,
                               load_env_desc=True,
                               load_env_raster=False)

    print("Loading rasters (mmap) ...")
    rasters, id_map = load_rasters()

    print("Loading signed-distance fields (mmap) ...")
    sds_shore, sds_nav = load_sdist()

    print("Generating figures ...")
    fig_scene_overview(data, rasters, id_map)
    fig_difficulty_overview(data, rasters, id_map)
    fig_social_overview(data, rasters, id_map)
    fig_env_detail(data, rasters, id_map, sds_shore, sds_nav)
    fig_harbor_all(data, rasters, id_map, sds_shore)
    fig_constrained_nearshore(data, rasters, id_map, sds_shore)

    print(f"\nAll figures saved to {OUT_DIR.resolve()}/")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
