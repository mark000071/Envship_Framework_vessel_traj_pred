"""Extended dataset loader with social and environment context.

Extends the base load_split() with:
  - Social: neighbor relative positions, velocities (max 10 neighbors, zero-padded)
  - Env descriptor: scene_type one-hot + numeric scalars (14-dim)
  - Env raster: 6×128×128 binary masks (optional, large)
"""

from __future__ import annotations
import csv, gzip, json, math
from pathlib import Path

import numpy as np


MAX_NEIGHBORS = 10
SOCIAL_FEAT   = 5     # [rel_x, rel_y, rel_vx, rel_vy, dist] per neighbor
N_ENV_DESC    = 14    # scene one-hot(4) + 10 numeric scalars


# ---------------------------------------------------------------------------
# Social feature parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> list:
    try:
        return json.loads(text) if text and text != "[]" else []
    except Exception:
        return []


def _safe_f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _encode_neighbor_row(row: dict, social_feat_dim: int = 5) -> np.ndarray:
    """Per-sample neighbour-features tensor (zero-padded to MAX_NEIGHBORS slots).

    Args:
        row: a single row of social_features.csv
        social_feat_dim: 5 (legacy) or 8 (V5).
            5-d → [rel_x, rel_y, rel_vx, rel_vy, dist]
            8-d → [rel_x, rel_y, rel_vx, rel_vy, dist, cpa, tcpa, |rel_v|]
    """
    feat  = np.zeros((MAX_NEIGHBORS, social_feat_dim), dtype=np.float32)
    mask  = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

    rx   = _parse_json(row.get("neighbor_rel_x_m_json",    "[]"))
    ry   = _parse_json(row.get("neighbor_rel_y_m_json",    "[]"))
    rvx  = _parse_json(row.get("neighbor_rel_vx_mps_json", "[]"))
    rvy  = _parse_json(row.get("neighbor_rel_vy_mps_json", "[]"))
    dist = _parse_json(row.get("neighbor_distance_m_json", "[]"))
    cpa  = _parse_json(row.get("neighbor_cpa_m_json",      "[]"))
    tcpa = _parse_json(row.get("neighbor_tcpa_s_json",     "[]"))

    n = min(MAX_NEIGHBORS, len(rx))
    for i in range(n):
        feat[i, 0] = _safe_f(rx[i])   / 3000.0   # normalize to ~[-1,1]
        feat[i, 1] = _safe_f(ry[i])   / 3000.0
        feat[i, 2] = _safe_f(rvx[i] if i < len(rvx) else 0) / 5.0
        feat[i, 3] = _safe_f(rvy[i] if i < len(rvy) else 0) / 5.0
        feat[i, 4] = _safe_f(dist[i] if i < len(dist) else 0) / 3000.0
        if social_feat_dim >= 8:
            # CPA in [0, 3000], scaled to ~[0, 1]
            cpa_v  = _safe_f(cpa[i]  if i < len(cpa)  else 0.0, default=0.0)
            tcpa_v = _safe_f(tcpa[i] if i < len(tcpa) else 0.0, default=0.0)
            vx_v   = _safe_f(rvx[i]  if i < len(rvx) else 0.0)
            vy_v   = _safe_f(rvy[i]  if i < len(rvy) else 0.0)
            feat[i, 5] = (cpa_v  if cpa_v  is not None else 0.0) / 3000.0
            feat[i, 6] = (tcpa_v if tcpa_v is not None else 0.0) / 600.0   # ±10 min scale
            feat[i, 7] = math.sqrt(vx_v*vx_v + vy_v*vy_v) / 10.0
        mask[i]    = 1.0
    return feat, mask


def _encode_env_desc(row: dict) -> np.ndarray:
    """Return (N_ENV_DESC,) float32 environment descriptor."""
    sc = row.get("scene_type", "open_water")
    sc_enc = [
        float(sc == "open_water"),
        float(sc == "nearshore"),
        float(sc == "harbor"),
        float(sc == "constrained"),
    ]
    numeric = [
        _safe_f(row.get("nearest_shore_dist_m", 5000)) / 5000.0,
        _safe_f(row.get("nearest_manmade_dist_m", 5000)) / 5000.0,
        _safe_f(row.get("water_ratio",     1.0)),
        _safe_f(row.get("navigable_ratio", 1.0)),
        _safe_f(row.get("barrier_density",  0.0)),
        _safe_f(row.get("natural_boundary_density", 0.0)),
        _safe_f(row.get("manmade_boundary_density", 0.0)),
        float(row.get("has_natural_boundary", 0) or 0),
        float(row.get("has_manmade_boundary", 0) or 0),
        _safe_f(row.get("env_quality_score", 1.0)),
    ]
    return np.array(sc_enc + numeric, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_context_split(
    track_root: str | Path,
    split: str,
    load_social: bool = True,
    load_env_desc: bool = True,
    load_env_raster: bool = False,
    load_env_sdf: bool = False,
    social_feat_dim: int = 5,
    max_samples: int = 0,
) -> dict:
    """
    Returns dict with keys:
      hist            : (N, 30, 6)
      future          : (N, 30, 2)
      social_feat     : (N, MAX_NEIGHBORS, 5)   if load_social
      social_mask     : (N, MAX_NEIGHBORS)       if load_social
      neighbor_count  : (N,) int                 if load_social
      env_desc        : (N, N_ENV_DESC)           if load_env_desc
      env_raster      : (N, 6, 128, 128) uint8    if load_env_raster
      env_sdf         : (N, 2, 128, 128) float16  if load_env_sdf
                        (channel 0: signed_dist_shore, channel 1: signed_dist_nav, in metres)
      meta            : dict of string arrays
    """
    track_root = Path(track_root)
    ctx_root   = track_root / "context_v1"
    aug_path   = ctx_root / "augmented" / split / "part-000.csv.gz"

    # Build social lookup by sample_id
    social_lookup: dict[str, dict] = {}
    if load_social:
        soc_path = ctx_root / "social" / "features" / split / "social_features.csv"
        if soc_path.exists():
            with open(soc_path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    social_lookup[row["sample_id"]] = row

    # Load rasters (memory-mapped)
    rasters = None
    raster_ids: dict[str, int] = {}
    sdf_shore = None
    sdf_nav   = None
    sdf_ids: dict[str, int] = {}
    if load_env_raster:
        rast_dir = ctx_root / "environment" / "rasters" / split
        import numpy as _np
        rasters   = _np.load(rast_dir / "masks.npy", mmap_mode="r")
        raster_ids_arr = _np.load(rast_dir / "sample_ids.npy", allow_pickle=True)
        raster_ids = {str(sid): i for i, sid in enumerate(raster_ids_arr.tolist())}
    if load_env_sdf:
        rast_dir = ctx_root / "environment" / "rasters" / split
        import numpy as _np
        sdf_shore = _np.load(rast_dir / "signed_dist_shore.npy", mmap_mode="r")
        sdf_nav   = _np.load(rast_dir / "signed_dist_nav.npy",   mmap_mode="r")
        sdf_ids_arr = _np.load(rast_dir / "sample_ids.npy", allow_pickle=True)
        sdf_ids   = {str(sid): i for i, sid in enumerate(sdf_ids_arr.tolist())}

    # Load metadata for difficulty_tier
    diff_tier_map: dict[str, str] = {}
    meta_path = track_root / "reports" / f"{split}_selected_metadata.csv"
    if meta_path.exists():
        with open(meta_path, "r", newline="", encoding="utf-8") as mf:
            for row in csv.DictReader(mf):
                if "difficulty_tier" in row:
                    diff_tier_map[row["sample_id"]] = row["difficulty_tier"]

    # Stream the augmented CSV
    from eval.dataset import _parse_f32, _classify_ship
    from eval.lazy_context import lazy_enabled, gather_to_cache
    _lazy_ctx = lazy_enabled()
    hists, futures = [], []
    soc_feats, soc_masks, n_counts = [], [], []
    env_descs = []
    rast_list = []
    sdf_list  = []
    rast_idx_list: list[int] = []
    sdf_idx_list: list[int] = []
    meta: dict[str, list] = {
        "sample_id": [], "mmsi": [], "ship_group": [],
        "difficulty_tier": [], "scene_type": [],
        "avg_speed_knots": [], "interp_ratio_total": [],
    }

    n = 0
    with gzip.open(aug_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hx  = _parse_f32(row["hist_x_json"])
            hy  = _parse_f32(row["hist_y_json"])
            fx  = _parse_f32(row["fut_x_json"])
            fy  = _parse_f32(row["fut_y_json"])
            sog = _parse_f32(row["hist_sog_json"])
            tod = _parse_f32(row["hist_time_of_day_sin_json"])
            dt  = 20.0
            vx  = np.gradient(hx, dt)
            vy  = np.gradient(hy, dt)

            hists.append(np.stack([hx, hy, vx, vy, sog, tod], axis=-1).astype(np.float32))
            futures.append(np.stack([fx, fy], axis=-1).astype(np.float32))

            sid = row["sample_id"]

            if load_social:
                srow = social_lookup.get(sid, {})
                sf, sm = _encode_neighbor_row(srow, social_feat_dim=social_feat_dim)
                nc = int(_safe_f(srow.get("neighbor_count_used", 0)))
                soc_feats.append(sf); soc_masks.append(sm); n_counts.append(nc)

            if load_env_desc:
                env_descs.append(_encode_env_desc(row))

            if load_env_raster and rasters is not None:
                idx = raster_ids.get(sid, -1)
                if _lazy_ctx:
                    rast_idx_list.append(idx)
                elif idx >= 0:
                    rast_list.append(rasters[idx].copy())
                else:
                    rast_list.append(np.zeros((6, 128, 128), dtype=np.uint8))

            if load_env_sdf and sdf_shore is not None:
                idx = sdf_ids.get(sid, -1)
                if _lazy_ctx:
                    sdf_idx_list.append(idx)
                elif idx >= 0:
                    sdf_list.append(np.stack([sdf_shore[idx], sdf_nav[idx]], axis=0))
                else:
                    sdf_list.append(np.zeros((2, 128, 128), dtype=np.float16))

            sg = _classify_ship(row.get("ship_type",""), row.get("ship_class",""))
            dt_field = diff_tier_map.get(sid, row.get("difficulty_tier",""))
            meta["sample_id"].append(sid)
            meta["mmsi"].append(int(row.get("mmsi", 0)))
            meta["ship_group"].append(sg)
            meta["difficulty_tier"].append(dt_field)
            meta["scene_type"].append(row.get("scene_type","open_water"))
            meta["avg_speed_knots"].append(float(row.get("avg_speed_knots","") or 0))
            meta["interp_ratio_total"].append(float(row.get("interp_ratio_total","") or 0))

            n += 1
            if max_samples > 0 and n >= max_samples:
                break

    result = {
        "hist":   np.stack(hists),
        "future": np.stack(futures),
        "meta":   {k: np.asarray(v) for k, v in meta.items()},
    }
    if load_social:
        result["social_feat"]    = np.stack(soc_feats)    # (N, 10, 5)
        result["social_mask"]    = np.stack(soc_masks)    # (N, 10)
        result["neighbor_count"] = np.array(n_counts)
    if load_env_desc:
        result["env_desc"]       = np.stack(env_descs)    # (N, 14)
    if load_env_raster and rast_list:
        result["env_raster"]     = np.stack(rast_list)    # (N, 6, 128, 128)
    if load_env_sdf and sdf_list:
        result["env_sdf"]        = np.stack(sdf_list)     # (N, 2, 128, 128) float16
    # Lazy path: stream the gather into a local memmap cache instead of RAM.
    if _lazy_ctx and load_env_raster and rasters is not None and rast_idx_list:
        result["env_raster"] = gather_to_cache(
            tag=f"{track_root}|{split}", key="env_raster", idx=rast_idx_list,
            sources=rasters, shape=(6, 128, 128), dtype=np.uint8)
    if _lazy_ctx and load_env_sdf and sdf_shore is not None and sdf_idx_list:
        result["env_sdf"] = gather_to_cache(
            tag=f"{track_root}|{split}", key="env_sdf", idx=sdf_idx_list,
            sources=(sdf_shore, sdf_nav), shape=(2, 128, 128), dtype=np.float16)
    return result
