#!/usr/bin/env python3
"""Build Standard Track v1 for EnvShip-Bench.

Standard Track design goals:
  - All major vessel types (cargo/tanker, passenger/ferry, fishing, tug/service, sailing/leisure)
  - Three difficulty tiers: easy / medium / hard (35 / 40 / 25 %)
  - Five ship-type groups with explicit target fractions
  - Relaxed quality thresholds vs the clean subset to allow curves and maneuvers
  - Scale: 120 K train / 15 K val / 15 K test

Stratum key: (ship_group, speed_bin, difficulty_tier)
Selection within hard tier is ranked by descending difficulty (hardest first);
easy and medium tiers are ranked by descending quality score.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOTS_PER_MPS = 1.9438444924406
STEP_SECONDS = 20.0

# Ship group target fractions (must sum to 1.0 after excluding other_unknown).
# Calibrated to DMA Danish-waters source data availability:
#   cargo_tanker eligible ~794K (80%), fishing ~105K (11%), passenger_ferry ~50K (5%),
#   sailing_leisure ~28K (3%), tug_service ~14K (1%).
# Fractions below intentionally over-represent minority types relative to raw availability
# while remaining reachable given per-MMSI diversity limits.
GROUP_TARGET_FRACS: dict[str, float] = {
    "cargo_tanker":    0.62,
    "fishing":         0.13,
    "passenger_ferry": 0.09,
    "sailing_leisure": 0.09,
    "tug_service":     0.07,
}

# Difficulty tier target fractions within each group
DIFFICULTY_TARGET_FRACS: dict[str, float] = {
    "easy":   0.35,
    "medium": 0.40,
    "hard":   0.25,
}

# difficulty_score boundaries
DIFFICULTY_EASY_MAX   = 5.0
DIFFICULTY_MEDIUM_MAX = 12.0
# hard: >= 12.0

# Speed weights inside each (group, difficulty) bucket
SPEED_WEIGHTS: dict[str, float] = {
    "0_5":   0.55,
    "5_10":  0.85,
    "10_15": 1.00,
    "15_20": 0.90,
    "gt20":  0.70,
}

DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/benchmark/core"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/multi_type_mini_bench_build"
    "/standard_track_v1"
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    sample_id: str
    hash_value: int
    split: str
    mmsi: int
    segment_id: str
    sample_step: int
    ship_group: str
    speed_bin: str
    difficulty_tier: str          # "easy" / "medium" / "hard"
    quality_score: float
    difficulty_score: float
    avg_speed_knots: float
    hist_median_speed_knots: float
    hist_speed_cv: float
    hist_efficiency: float
    fut_efficiency: float
    future_linearity: float
    hist_nonzero_ratio: float
    fut_nonzero_ratio: float
    hist_pause_ratio: float
    fut_pause_ratio: float
    bridge_turn_deg: float
    hist_turn_mean_abs_deg: float
    fut_turn_mean_abs_deg: float
    hist_turn_max_abs_deg: float
    fut_turn_max_abs_deg: float
    hist_step_outlier_ratio: float
    fut_step_outlier_ratio: float
    interp_ratio_hist: float
    interp_ratio_fut: float
    interp_ratio_total: float
    hist_interp_true_ratio: float
    fut_interp_true_ratio: float
    hist_cog_motion_mae_deg: float
    hist_heading_cog_mae_deg: float
    hist_displacement_m: float
    fut_displacement_m: float
    future_path_length_m: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--train-target", type=int, default=120_000)
    p.add_argument("--val-target",   type=int, default=15_000)
    p.add_argument("--test-target",  type=int, default=15_000)
    # vessel groups
    p.add_argument(
        "--allow-ship-groups",
        default="cargo_tanker,passenger_ferry,fishing,tug_service,sailing_leisure",
    )
    p.add_argument("--exclude-ship-groups", default="other_unknown")
    # redundancy limits — train uses tighter diversity controls;
    # eval (val/test) uses looser limits because there are far fewer unique MMSI
    # (MMSI-hash split assigns ~10 % of vessels to each eval split)
    p.add_argument("--max-per-mmsi-train",        type=int,   default=55)
    p.add_argument("--max-per-mmsi-eval",         type=int,   default=70)
    p.add_argument("--max-per-segment-train",     type=int,   default=12)
    p.add_argument("--max-per-segment-eval",      type=int,   default=20)
    # step_gap=2 means consecutive selected windows from the same segment are at least
    # 2 window-indices apart = 2 * 5-point-step * 20s = 200s (3.3 min) of temporal separation.
    p.add_argument("--min-segment-step-gap-train",type=int,   default=2)
    p.add_argument("--min-segment-step-gap-eval", type=int,   default=3)
    # quota smooth exponent: 0 = purely proportional to target fracs (recommended)
    p.add_argument("--quota-sqrt-smooth",      type=float, default=0.1,
                   help="Exponent on count inside each stratum quota (0=proportional, 1=fully count-driven)")
    # -----------------------------------------------------------------------
    # Quality / filter thresholds  — significantly relaxed vs clean subset
    # -----------------------------------------------------------------------
    p.add_argument("--min-avg-speed-knots",          type=float, default=1.5)
    p.add_argument("--max-avg-speed-knots",          type=float, default=30.0)
    p.add_argument("--min-hist-median-speed-knots",  type=float, default=1.5)
    p.add_argument("--max-hist-speed-cv",            type=float, default=0.65)
    p.add_argument("--min-hist-efficiency",          type=float, default=0.70)
    p.add_argument("--min-fut-efficiency",           type=float, default=0.65)
    p.add_argument("--min-future-linearity",         type=float, default=0.75)
    p.add_argument("--min-hist-nonzero-ratio",       type=float, default=0.60)
    p.add_argument("--min-fut-nonzero-ratio",        type=float, default=0.65)
    p.add_argument("--max-hist-pause-ratio",         type=float, default=0.45)
    p.add_argument("--max-fut-pause-ratio",          type=float, default=0.40)
    p.add_argument("--max-bridge-turn-deg",          type=float, default=50.0)
    p.add_argument("--max-hist-turn-mean-deg",       type=float, default=20.0)
    p.add_argument("--max-fut-turn-mean-deg",        type=float, default=15.0)
    p.add_argument("--max-hist-turn-max-deg",        type=float, default=70.0)
    p.add_argument("--max-fut-turn-max-deg",         type=float, default=60.0)
    p.add_argument("--max-step-outlier-ratio",       type=float, default=5.0)
    p.add_argument("--max-interp-ratio-total",       type=float, default=0.20)
    p.add_argument("--max-interp-ratio-side",        type=float, default=0.20)
    p.add_argument("--max-interp-true-ratio",        type=float, default=0.20)
    p.add_argument("--max-hist-cog-motion-mae-deg",  type=float, default=35.0)
    p.add_argument("--max-hist-heading-cog-mae-deg", type=float, default=45.0)
    p.add_argument("--min-quality-score",            type=float, default=30.0)
    # misc
    p.add_argument("--log-every", type=int, default=100_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in ("", None) else default
    except Exception:
        return default


def safe_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    t = normalize_text(value)
    if t in {"true", "1", "yes", "y"}:
        return True
    if t in {"false", "0", "no", "n"}:
        return False
    return default


def parse_json_array(text: str) -> list[object]:
    t = str(text or "").strip()
    if not t:
        return []
    try:
        data = json.loads(t)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parse_float_array(text: str) -> list[float]:
    result = []
    for item in parse_json_array(text):
        try:
            result.append(float(item))
        except Exception:
            pass
    return result


def parse_bool_array(text: str) -> list[bool]:
    return [bool(item) for item in parse_json_array(text)]


def stable_hash(sample_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), byteorder="big"
    )


def parse_sample_step(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Ship-group classification
# ---------------------------------------------------------------------------

def ship_group_from_type(ship_type: object, ship_class: object | None = None) -> str:
    text  = normalize_text(ship_type)
    text2 = normalize_text(ship_class)
    full  = f"{text} {text2}".strip()
    if any(k in full for k in ["cargo", "tanker", "bulk", "container"]):
        return "cargo_tanker"
    if any(k in full for k in ["passenger", "ferry"]):
        return "passenger_ferry"
    if "fishing" in full:
        return "fishing"
    if any(k in full for k in ["tug", "service", "pilot", "tow"]):
        return "tug_service"
    if any(k in full for k in ["sailing", "pleasure", "yacht"]):
        return "sailing_leisure"
    return "other_unknown"


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

def speed_bin_from_knots(speed: float) -> str:
    if speed < 5:   return "0_5"
    if speed < 10:  return "5_10"
    if speed < 15:  return "10_15"
    if speed < 20:  return "15_20"
    return "gt20"


def difficulty_tier_from_score(score: float) -> str:
    if score < DIFFICULTY_EASY_MAX:   return "easy"
    if score < DIFFICULTY_MEDIUM_MAX: return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Geometry / motion metrics
# ---------------------------------------------------------------------------

def wrap_angle_diff_deg(a: float, b: float) -> float:
    diff = (b - a + 180.0) % 360.0 - 180.0
    return abs(diff)


def mean_angle_deg(angles: list[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(math.radians(v)) for v in angles)
    c = sum(math.cos(math.radians(v)) for v in angles)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return float(angles[-1])
    return math.degrees(math.atan2(s, c))


def _fmean(values: list[float], default: float = 0.0) -> float:
    return (sum(values) / len(values)) if values else default


def _fmedian(values: list[float], default: float = 0.0) -> float:
    return float(statistics.median(values)) if values else default


def _pstdev(values: list[float], default: float = 0.0) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else default


def angle_from_sin_cos(sins: list[float], coss: list[float]) -> list[float]:
    n = min(len(sins), len(coss))
    return [math.degrees(math.atan2(sins[i], coss[i])) for i in range(n)]


def compute_path_metrics(
    points: list[tuple[float, float]],
    prepend_origin: bool,
) -> dict[str, float]:
    seq = ([(0.0, 0.0)] + points) if prepend_origin else points
    if len(seq) < 2:
        return {
            "path_length_m": 0.0, "displacement_m": 0.0, "efficiency": 0.0,
            "median_speed_kn": 0.0, "speed_cv": 0.0,
            "nonzero_ratio": 0.0, "pause_ratio": 1.0,
            "mean_abs_turn_deg": 180.0, "max_abs_turn_deg": 180.0,
            "step_outlier_ratio": 999.0, "recent_bearing_deg": 0.0,
        }

    steps, bearings = [], []
    for i in range(1, len(seq)):
        dx = seq[i][0] - seq[i-1][0]
        dy = seq[i][1] - seq[i-1][1]
        steps.append(math.hypot(dx, dy))
        bearings.append(math.degrees(math.atan2(dy, dx)))

    if prepend_origin:
        disp = math.hypot(points[-1][0], points[-1][1]) if points else 0.0
    else:
        disp = math.hypot(points[-1][0] - points[0][0],
                          points[-1][1] - points[0][1]) if len(points) >= 2 else 0.0

    path = sum(steps)
    eff  = disp / path if path > 1e-6 else 0.0
    spd  = [s / STEP_SECONDS * KNOTS_PER_MPS for s in steps]
    med_spd  = _fmedian(spd)
    mean_spd = _fmean(spd)
    cv       = _pstdev(spd) / mean_spd if mean_spd > 1e-6 else 0.0
    nonzero  = sum(1 for s in steps if s >= 5.0) / len(steps)
    pause    = sum(1 for s in steps if s <= 10.0) / len(steps)

    turns = [wrap_angle_diff_deg(bearings[i-1], bearings[i]) for i in range(1, len(bearings))]
    mean_turn = _fmean(turns)
    max_turn  = max(turns) if turns else 0.0

    med_step = max(_fmedian(steps, default=0.0), 1.0)
    outlier  = (max(steps) / med_step) if steps else 999.0
    recent   = mean_angle_deg(bearings[-5:]) if bearings else 0.0

    return {
        "path_length_m": path, "displacement_m": disp, "efficiency": eff,
        "median_speed_kn": med_spd, "speed_cv": cv,
        "nonzero_ratio": nonzero, "pause_ratio": pause,
        "mean_abs_turn_deg": mean_turn, "max_abs_turn_deg": max_turn,
        "step_outlier_ratio": outlier, "recent_bearing_deg": recent,
    }


def mean_abs_angle_error(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return _fmean([wrap_angle_diff_deg(a[i], b[i]) for i in range(n)])


def recent_motion_bearings(points: list[tuple[float, float]], n: int = 6) -> list[float]:
    if len(points) < 2:
        return []
    start = max(1, len(points) - n)
    bearings = []
    for i in range(start, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        if math.hypot(dx, dy) < 1e-6:
            continue
        bearings.append(math.degrees(math.atan2(dy, dx)))
    return bearings


# ---------------------------------------------------------------------------
# Quality / difficulty scoring  (ship-type bonus removed for neutrality)
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_scores(metrics: dict[str, float]) -> tuple[float, float]:
    """Return (quality_score, difficulty_score).

    quality_score measures how clean and benchmark-friendly a sample is.
    difficulty_score measures forecast complexity.
    Ship-type bonus is removed so all groups are scored on equal footing.
    """
    quality_score = (
        14.0 * metrics["fut_efficiency"]
        + 10.0 * metrics["hist_efficiency"]
        +  9.0 * metrics["future_linearity"]
        +  8.0 * _clamp01(1.0 - metrics["hist_speed_cv"]       / 0.65)
        +  8.0 * _clamp01(1.0 - metrics["fut_turn_mean_abs_deg"]  / 15.0)
        +  6.0 * _clamp01(1.0 - metrics["hist_turn_mean_abs_deg"] / 20.0)
        +  6.0 * _clamp01(1.0 - metrics["bridge_turn_deg"]       / 50.0)
        +  5.0 * _clamp01(1.0 - metrics["hist_pause_ratio"]      / 0.45)
        +  5.0 * _clamp01(1.0 - metrics["fut_pause_ratio"]       / 0.40)
        +  5.0 * _clamp01(1.0 - metrics["interp_ratio_total"]    / 0.20)
        +  4.0 * _clamp01(1.0 - metrics["hist_step_outlier_ratio"] / 5.0)
        +  4.0 * _clamp01(1.0 - metrics["fut_step_outlier_ratio"]  / 5.0)
        +  4.0 * _clamp01(1.0 - metrics["hist_cog_motion_mae_deg"]   / 35.0)
        +  4.0 * _clamp01(1.0 - metrics["hist_heading_cog_mae_deg"]  / 45.0)
        + 8.0  # fixed base (replaces ship-type bonus)
    )

    difficulty_score = (
        30.0 * (1.0 - metrics["fut_efficiency"])
        + 20.0 * _clamp01(metrics["fut_turn_mean_abs_deg"] / 15.0)
        + 15.0 * _clamp01(metrics["bridge_turn_deg"]       / 45.0)
        + 15.0 * metrics["fut_pause_ratio"]
        + 10.0 * _clamp01(metrics["fut_step_outlier_ratio"] / 5.0)
        + 10.0 * _clamp01(metrics["interp_ratio_total"]    / 0.20)
    )
    return quality_score, difficulty_score


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def check_filters(metrics: dict[str, float], args: argparse.Namespace) -> str | None:
    checks = [
        ("avg_speed_out_of_range",
         args.min_avg_speed_knots <= metrics["avg_speed_knots"] <= args.max_avg_speed_knots),
        ("low_hist_median_speed",
         metrics["hist_median_speed_knots"] >= args.min_hist_median_speed_knots),
        ("hist_speed_variability",
         metrics["hist_speed_cv"] <= args.max_hist_speed_cv),
        ("hist_efficiency",
         metrics["hist_efficiency"] >= args.min_hist_efficiency),
        ("fut_efficiency",
         metrics["fut_efficiency"] >= args.min_fut_efficiency),
        ("future_linearity",
         metrics["future_linearity"] >= args.min_future_linearity),
        ("hist_nonzero_ratio",
         metrics["hist_nonzero_ratio"] >= args.min_hist_nonzero_ratio),
        ("fut_nonzero_ratio",
         metrics["fut_nonzero_ratio"] >= args.min_fut_nonzero_ratio),
        ("hist_pause_ratio",
         metrics["hist_pause_ratio"] <= args.max_hist_pause_ratio),
        ("fut_pause_ratio",
         metrics["fut_pause_ratio"] <= args.max_fut_pause_ratio),
        ("bridge_turn",
         metrics["bridge_turn_deg"] <= args.max_bridge_turn_deg),
        ("hist_turn_mean",
         metrics["hist_turn_mean_abs_deg"] <= args.max_hist_turn_mean_deg),
        ("fut_turn_mean",
         metrics["fut_turn_mean_abs_deg"] <= args.max_fut_turn_mean_deg),
        ("hist_turn_max",
         metrics["hist_turn_max_abs_deg"] <= args.max_hist_turn_max_deg),
        ("fut_turn_max",
         metrics["fut_turn_max_abs_deg"] <= args.max_fut_turn_max_deg),
        ("step_outlier_ratio",
         metrics["hist_step_outlier_ratio"] <= args.max_step_outlier_ratio
         and metrics["fut_step_outlier_ratio"] <= args.max_step_outlier_ratio),
        ("interp_ratio_total",
         metrics["interp_ratio_total"] <= args.max_interp_ratio_total),
        ("interp_ratio_side",
         metrics["interp_ratio_hist"] <= args.max_interp_ratio_side
         and metrics["interp_ratio_fut"] <= args.max_interp_ratio_side),
        ("interp_true_ratio",
         metrics["hist_interp_true_ratio"] <= args.max_interp_true_ratio
         and metrics["fut_interp_true_ratio"] <= args.max_interp_true_ratio),
        ("hist_cog_motion_mae",
         metrics["hist_cog_motion_mae_deg"] <= args.max_hist_cog_motion_mae_deg),
        ("hist_heading_cog_mae",
         metrics["hist_heading_cog_mae_deg"] <= args.max_hist_heading_cog_mae_deg),
    ]
    for name, ok in checks:
        if not ok:
            return name
    return None


# ---------------------------------------------------------------------------
# Row evaluation
# ---------------------------------------------------------------------------

def evaluate_row(
    row: dict[str, str],
    split: str,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> tuple[Candidate | None, str | None]:
    ship_group = ship_group_from_type(row.get("ship_type"), row.get("ship_class"))
    if ship_group in excluded_groups or ship_group not in allowed_groups:
        return None, "ship_group_excluded"
    if "core_eligible" in row and not safe_bool(row.get("core_eligible"), default=True):
        return None, "not_core_eligible"
    if "quality_tier" in row and normalize_text(row.get("quality_tier")) not in {"", "core"}:
        return None, "quality_tier_not_core"
    if "hist_gap_ok" in row and not safe_bool(row.get("hist_gap_ok"), default=True):
        return None, "hist_gap_not_ok"
    if "fut_gap_ok" in row and not safe_bool(row.get("fut_gap_ok"), default=True):
        return None, "fut_gap_not_ok"

    hist_x = parse_float_array(row.get("hist_x_json", ""))
    hist_y = parse_float_array(row.get("hist_y_json", ""))
    fut_x  = parse_float_array(row.get("fut_x_json", ""))
    fut_y  = parse_float_array(row.get("fut_y_json", ""))

    hist_pts = [(hist_x[i], hist_y[i]) for i in range(min(len(hist_x), len(hist_y)))]
    fut_pts  = [(fut_x[i],  fut_y[i])  for i in range(min(len(fut_x),  len(fut_y)))]
    if len(hist_pts) < 12 or len(fut_pts) < 12:
        return None, "too_few_points"

    hist_sog     = parse_float_array(row.get("hist_sog_json", ""))
    hist_cog     = angle_from_sin_cos(
        parse_float_array(row.get("hist_cog_sin_json", "")),
        parse_float_array(row.get("hist_cog_cos_json", "")),
    )
    hist_heading = angle_from_sin_cos(
        parse_float_array(row.get("hist_heading_sin_json", "")),
        parse_float_array(row.get("hist_heading_cos_json", "")),
    )
    hist_interp  = parse_bool_array(row.get("hist_interp_json", ""))
    fut_interp   = parse_bool_array(row.get("fut_interp_json", ""))

    hm = compute_path_metrics(hist_pts, prepend_origin=False)
    fm = compute_path_metrics(fut_pts,  prepend_origin=True)

    hist_motion_b = recent_motion_bearings(hist_pts)
    recent_hist_b = mean_angle_deg(hist_motion_b[-5:]) if hist_motion_b else hm["recent_bearing_deg"]
    bridge_turn   = wrap_angle_diff_deg(recent_hist_b, fm["recent_bearing_deg"])

    hist_cog_motion_mae = (
        mean_abs_angle_error(hist_cog[-len(hist_motion_b):], hist_motion_b)
        if hist_motion_b and hist_cog else 0.0
    )
    hist_heading_cog_mae = mean_abs_angle_error(hist_heading, hist_cog) if hist_heading and hist_cog else 0.0

    hist_interp_true = _fmean([1.0 if f else 0.0 for f in hist_interp]) if hist_interp else 0.0
    fut_interp_true  = _fmean([1.0 if f else 0.0 for f in fut_interp])  if fut_interp  else 0.0

    hist_disp = safe_float(row.get("hist_displacement_m"), hm["displacement_m"])
    fut_disp  = safe_float(row.get("fut_displacement_m"),  fm["displacement_m"])
    irh       = safe_float(row.get("interp_ratio_hist"),   hist_interp_true)
    irf       = safe_float(row.get("interp_ratio_fut"),    fut_interp_true)
    irt       = safe_float(row.get("interp_ratio_total"),  0.5 * (irh + irf))

    hist_sog_med = _fmedian(hist_sog) if hist_sog else hm["median_speed_kn"]
    avg_spd      = 0.5 * (hist_disp / 600.0 + fut_disp / 600.0) * KNOTS_PER_MPS
    fut_lin      = fut_disp / max(fm["path_length_m"], 1.0)

    metrics = {
        "avg_speed_knots":          avg_spd,
        "hist_median_speed_knots":  hist_sog_med,
        "hist_speed_cv":            hm["speed_cv"],
        "hist_efficiency":          hm["efficiency"],
        "fut_efficiency":           fm["efficiency"],
        "future_linearity":         fut_lin,
        "hist_nonzero_ratio":       hm["nonzero_ratio"],
        "fut_nonzero_ratio":        fm["nonzero_ratio"],
        "hist_pause_ratio":         hm["pause_ratio"],
        "fut_pause_ratio":          fm["pause_ratio"],
        "bridge_turn_deg":          bridge_turn,
        "hist_turn_mean_abs_deg":   hm["mean_abs_turn_deg"],
        "fut_turn_mean_abs_deg":    fm["mean_abs_turn_deg"],
        "hist_turn_max_abs_deg":    hm["max_abs_turn_deg"],
        "fut_turn_max_abs_deg":     fm["max_abs_turn_deg"],
        "hist_step_outlier_ratio":  hm["step_outlier_ratio"],
        "fut_step_outlier_ratio":   fm["step_outlier_ratio"],
        "interp_ratio_hist":        irh,
        "interp_ratio_fut":         irf,
        "interp_ratio_total":       irt,
        "hist_interp_true_ratio":   hist_interp_true,
        "fut_interp_true_ratio":    fut_interp_true,
        "hist_cog_motion_mae_deg":  hist_cog_motion_mae,
        "hist_heading_cog_mae_deg": hist_heading_cog_mae,
    }

    reason = check_filters(metrics, args)
    if reason is not None:
        return None, reason

    quality_score, difficulty_score = compute_scores(metrics)
    if quality_score < args.min_quality_score:
        return None, "quality_score"

    sample_id = str(row["sample_id"])
    return Candidate(
        sample_id=sample_id,
        hash_value=stable_hash(sample_id),
        split=split,
        mmsi=safe_int(row.get("mmsi")),
        segment_id=str(row.get("segment_id", "")),
        sample_step=parse_sample_step(sample_id),
        ship_group=ship_group,
        speed_bin=speed_bin_from_knots(avg_spd),
        difficulty_tier=difficulty_tier_from_score(difficulty_score),
        quality_score=quality_score,
        difficulty_score=difficulty_score,
        avg_speed_knots=avg_spd,
        hist_median_speed_knots=hist_sog_med,
        hist_speed_cv=hm["speed_cv"],
        hist_efficiency=hm["efficiency"],
        fut_efficiency=fm["efficiency"],
        future_linearity=fut_lin,
        hist_nonzero_ratio=hm["nonzero_ratio"],
        fut_nonzero_ratio=fm["nonzero_ratio"],
        hist_pause_ratio=hm["pause_ratio"],
        fut_pause_ratio=fm["pause_ratio"],
        bridge_turn_deg=bridge_turn,
        hist_turn_mean_abs_deg=hm["mean_abs_turn_deg"],
        fut_turn_mean_abs_deg=fm["mean_abs_turn_deg"],
        hist_turn_max_abs_deg=hm["max_abs_turn_deg"],
        fut_turn_max_abs_deg=fm["max_abs_turn_deg"],
        hist_step_outlier_ratio=hm["step_outlier_ratio"],
        fut_step_outlier_ratio=fm["step_outlier_ratio"],
        interp_ratio_hist=irh,
        interp_ratio_fut=irf,
        interp_ratio_total=irt,
        hist_interp_true_ratio=hist_interp_true,
        fut_interp_true_ratio=fut_interp_true,
        hist_cog_motion_mae_deg=hist_cog_motion_mae,
        hist_heading_cog_mae_deg=hist_heading_cog_mae,
        hist_displacement_m=hist_disp,
        fut_displacement_m=fut_disp,
        future_path_length_m=fm["path_length_m"],
    ), None


# ---------------------------------------------------------------------------
# Hierarchical quota
# ---------------------------------------------------------------------------

def _strata_key(c: Candidate) -> tuple[str, str, str]:
    return (c.ship_group, c.speed_bin, c.difficulty_tier)


def hierarchical_quota(
    by_key: dict[tuple[str, str, str], list[Candidate]],
    target: int,
    group_fracs: dict[str, float],
    diff_fracs: dict[str, float],
    speed_w: dict[str, float],
    smooth_exp: float,
) -> dict[tuple[str, str, str], int]:
    """Three-level quota: group → difficulty → speed bin."""

    counts = {k: len(v) for k, v in by_key.items()}

    # Raw importance weight for each stratum
    raw_w: dict[tuple[str, str, str], float] = {}
    for (g, s, d), n in counts.items():
        raw_w[(g, s, d)] = (
            group_fracs.get(g, 0.0)
            * diff_fracs.get(d, 0.0)
            * speed_w.get(s, 0.5)
            * (n ** smooth_exp)
        )

    total_w = sum(raw_w.values())
    if total_w <= 0:
        return {k: 0 for k in counts}

    # First pass: floor allocation
    quota: dict[tuple[str, str, str], int] = {}
    fractional: list[tuple[float, int, tuple[str, str, str]]] = []
    for key, w in raw_w.items():
        exact = target * w / total_w
        avail = counts[key]
        base  = min(avail, int(math.floor(exact)))
        quota[key] = base
        fractional.append((exact - math.floor(exact), avail - base, key))

    # Second pass: fill remainder with largest fractional parts
    remaining = target - sum(quota.values())
    for _, extra_avail, key in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        if extra_avail > 0:
            quota[key] += 1
            remaining  -= 1

    return quota


# ---------------------------------------------------------------------------
# Candidate selection with diversity constraints
# ---------------------------------------------------------------------------

def sort_key_for_tier(c: Candidate) -> tuple[float, float, int]:
    """Hard tier: prefer higher difficulty; easy/medium: prefer higher quality."""
    if c.difficulty_tier == "hard":
        return (-c.difficulty_score, -c.quality_score, c.hash_value)
    return (-c.quality_score, c.difficulty_score, c.hash_value)


def can_take(
    c: Candidate,
    used_mmsi: Counter[int],
    used_seg: Counter[str],
    used_steps: defaultdict[str, list[int]],
    max_per_mmsi: int,
    max_per_segment: int,
    min_segment_step_gap: int,
) -> bool:
    if used_mmsi[c.mmsi]      >= max_per_mmsi:     return False
    if used_seg[c.segment_id] >= max_per_segment:  return False
    if c.sample_step >= 0:
        for prev in used_steps[c.segment_id]:
            if abs(prev - c.sample_step) < min_segment_step_gap:
                return False
    return True


def select_candidates(
    split: str,
    candidates: list[Candidate],
    target: int,
    args: argparse.Namespace,
    max_per_mmsi: int,
    max_per_segment: int,
    min_segment_step_gap: int,
) -> tuple[list[Candidate], dict[str, object]]:
    by_key: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_key[_strata_key(c)].append(c)
    for pool in by_key.values():
        pool.sort(key=sort_key_for_tier)

    quota = hierarchical_quota(
        by_key,
        target=target,
        group_fracs=GROUP_TARGET_FRACS,
        diff_fracs=DIFFICULTY_TARGET_FRACS,
        speed_w=SPEED_WEIGHTS,
        smooth_exp=args.quota_sqrt_smooth,
    )

    selected: list[Candidate] = []
    leftovers: list[Candidate] = []
    used_mmsi:  Counter[int]                   = Counter()
    used_seg:   Counter[str]                   = Counter()
    used_steps: defaultdict[str, list[int]]    = defaultdict(list)
    strata_report: dict[str, dict[str, object]] = {}

    for key, pool in sorted(by_key.items()):
        taken  = 0
        wanted = quota.get(key, 0)
        for c in pool:
            if taken < wanted and can_take(
                c, used_mmsi, used_seg, used_steps,
                max_per_mmsi, max_per_segment, min_segment_step_gap,
            ):
                selected.append(c)
                used_mmsi[c.mmsi]      += 1
                used_seg[c.segment_id] += 1
                if c.sample_step >= 0:
                    used_steps[c.segment_id].append(c.sample_step)
                taken += 1
            else:
                leftovers.append(c)
        strata_report["|".join(key)] = {
            "eligible": len(pool),
            "quota":    wanted,
            "selected": taken,
            "mean_quality_score":    round(_fmean([x.quality_score    for x in pool]), 4),
            "mean_difficulty_score": round(_fmean([x.difficulty_score for x in pool]), 4),
        }

    # Fill remaining slots from leftovers (sorted by quality desc)
    leftovers.sort(key=lambda c: (-c.quality_score, c.difficulty_score, c.hash_value))
    fill_added = 0
    for c in leftovers:
        if len(selected) >= target:
            break
        if can_take(
            c, used_mmsi, used_seg, used_steps,
            max_per_mmsi, max_per_segment, min_segment_step_gap,
        ):
            selected.append(c)
            used_mmsi[c.mmsi]      += 1
            used_seg[c.segment_id] += 1
            if c.sample_step >= 0:
                used_steps[c.segment_id].append(c.sample_step)
            fill_added += 1

    selected = selected[:target]
    report = {
        "split":               split,
        "target":              target,
        "eligible_candidates": len(candidates),
        "selected":            len(selected),
        "fill_added":          fill_added,
        "strata_report":       strata_report,
    }
    return selected, report


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_selected_rows_v2(
    split_dir: Path,
    selected_ids: set[str],
    output_path: Path,
) -> int:
    """Single-pass export across all part shards into one .csv.gz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept   = 0
    writer: csv.DictWriter | None = None
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as dst:
        for file_path in sorted(split_dir.glob("*.csv.gz")):
            with gzip.open(file_path, "rt", encoding="utf-8", newline="") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    if row.get("sample_id") in selected_ids:
                        writer.writerow(row)
                        kept += 1
    return kept


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def summarize_numeric(items: list[Candidate], attr: str) -> dict[str, float]:
    vals = sorted(float(getattr(c, attr)) for c in items)
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    p90i = min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))
    return {
        "mean":   round(_fmean(vals),    4),
        "median": round(_fmedian(vals),  4),
        "p90":    round(vals[p90i],      4),
        "min":    round(vals[0],         4),
        "max":    round(vals[-1],        4),
    }


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_split(
    split: str,
    split_dir: Path,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> tuple[list[Candidate], dict[str, object]]:
    candidates: list[Candidate] = []
    filter_counts: Counter[str] = Counter()
    input_rows = 0
    t0 = time.time()

    part_files = sorted(split_dir.glob("*.csv.gz"))
    for fi, fpath in enumerate(part_files, start=1):
        print(f"[scan:{split}] file {fi}/{len(part_files)} {fpath.name}", flush=True)
        with gzip.open(fpath, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                input_rows += 1
                cand, reason = evaluate_row(row, split, args, allowed_groups, excluded_groups)
                if cand is None:
                    filter_counts[reason or "unknown"] += 1
                else:
                    candidates.append(cand)
                if input_rows % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[scan:{split}] rows={input_rows:,}  eligible={len(candidates):,}"
                        f"  filtered={input_rows - len(candidates):,}"
                        f"  elapsed={elapsed:.0f}s",
                        flush=True,
                    )

    print(
        f"[scan:{split}] DONE rows={input_rows:,}  eligible={len(candidates):,}"
        f"  elapsed={time.time()-t0:.0f}s",
        flush=True,
    )
    return candidates, {
        "input_rows":          input_rows,
        "eligible_candidates": len(candidates),
        "filter_counts":       dict(filter_counts),
    }


# ---------------------------------------------------------------------------
# Build one split
# ---------------------------------------------------------------------------

def _split_diversity_params(
    split: str, args: argparse.Namespace
) -> tuple[int, int, int]:
    """Return (max_per_mmsi, max_per_segment, min_segment_step_gap) for split."""
    if split == "train":
        return args.max_per_mmsi_train, args.max_per_segment_train, args.min_segment_step_gap_train
    return args.max_per_mmsi_eval, args.max_per_segment_eval, args.min_segment_step_gap_eval


def build_split(
    split: str,
    target: int,
    benchmark_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> dict[str, object]:
    split_dir = benchmark_root / split
    max_per_mmsi, max_per_segment, min_step_gap = _split_diversity_params(split, args)
    print(
        f"[build:{split}] diversity_params: max_per_mmsi={max_per_mmsi}"
        f"  max_per_segment={max_per_segment}  min_step_gap={min_step_gap}",
        flush=True,
    )
    candidates, scan_stats = scan_split(split, split_dir, args, allowed_groups, excluded_groups)
    selected, sel_report   = select_candidates(
        split, candidates, target, args, max_per_mmsi, max_per_segment, min_step_gap
    )
    selected_ids           = {c.sample_id for c in selected}

    # Output dirs
    (output_root / split).mkdir(parents=True, exist_ok=True)
    (output_root / "sample_ids").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    # Write CSV shard
    out_csv = output_root / split / "part-000.csv.gz"
    exported = export_selected_rows_v2(split_dir, selected_ids, out_csv)

    # Write sample ID list
    sid_path = output_root / "sample_ids" / f"{split}_sample_ids.txt"
    sid_path.write_text("\n".join(sorted(selected_ids)) + "\n", encoding="utf-8")

    # Write metadata CSV
    meta_path = output_root / "reports" / f"{split}_selected_metadata.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        fields = list(asdict(selected[0]).keys())
        with open(meta_path, "w", encoding="utf-8", newline="") as mh:
            w = csv.DictWriter(mh, fieldnames=fields)
            w.writeheader()
            for c in selected:
                w.writerow(asdict(c))

    # Write per-split report
    report: dict[str, object] = {
        "split":             split,
        "target":            target,
        "selected":          len(selected),
        "exported_rows":     exported,
        "unique_mmsi":       len({c.mmsi        for c in selected}),
        "unique_segments":   len({c.segment_id  for c in selected}),
        "ship_group_counts": dict(Counter(c.ship_group      for c in selected)),
        "speed_bin_counts":  dict(Counter(c.speed_bin       for c in selected)),
        "difficulty_tier_counts": dict(Counter(c.difficulty_tier for c in selected)),
        "filter_counts":     scan_stats["filter_counts"],
        "input_rows":        scan_stats["input_rows"],
        "eligible_candidates": scan_stats["eligible_candidates"],
        "selection_report":  sel_report,
        "quality_stats": {
            "quality_score":        summarize_numeric(selected, "quality_score"),
            "difficulty_score":     summarize_numeric(selected, "difficulty_score"),
            "avg_speed_knots":      summarize_numeric(selected, "avg_speed_knots"),
            "hist_speed_cv":        summarize_numeric(selected, "hist_speed_cv"),
            "hist_efficiency":      summarize_numeric(selected, "hist_efficiency"),
            "fut_efficiency":       summarize_numeric(selected, "fut_efficiency"),
            "future_linearity":     summarize_numeric(selected, "future_linearity"),
            "bridge_turn_deg":      summarize_numeric(selected, "bridge_turn_deg"),
            "hist_turn_mean_abs_deg": summarize_numeric(selected, "hist_turn_mean_abs_deg"),
            "fut_turn_mean_abs_deg":  summarize_numeric(selected, "fut_turn_mean_abs_deg"),
            "interp_ratio_total":   summarize_numeric(selected, "interp_ratio_total"),
        },
    }
    rep_path = output_root / "reports" / f"{split}_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[build:{split}] selected={len(selected):,}  exported={exported:,}"
        f"  unique_mmsi={report['unique_mmsi']}  unique_segments={report['unique_segments']}",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    allowed  = {g.strip() for g in args.allow_ship_groups.split(",")  if g.strip()}
    excluded = {g.strip() for g in args.exclude_ship_groups.split(",") if g.strip()}

    print(f"[build] benchmark_root : {args.benchmark_root}", flush=True)
    print(f"[build] output_root    : {args.output_root}",    flush=True)
    print(f"[build] targets        : train={args.train_target:,}  val={args.val_target:,}  test={args.test_target:,}", flush=True)
    print(f"[build] allowed_groups : {sorted(allowed)}",  flush=True)
    print(f"[build] excluded_groups: {sorted(excluded)}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "config": {
            "version":         "standard_track_v1",
            "benchmark_root":  str(args.benchmark_root),
            "output_root":     str(args.output_root),
            "train_target":    args.train_target,
            "val_target":      args.val_target,
            "test_target":     args.test_target,
            "allow_ship_groups":   sorted(allowed),
            "exclude_ship_groups": sorted(excluded),
            "group_target_fracs":      GROUP_TARGET_FRACS,
            "difficulty_target_fracs": DIFFICULTY_TARGET_FRACS,
            "difficulty_tier_boundaries": {
                "easy":   [0.0,                 DIFFICULTY_EASY_MAX],
                "medium": [DIFFICULTY_EASY_MAX, DIFFICULTY_MEDIUM_MAX],
                "hard":   [DIFFICULTY_MEDIUM_MAX, "inf"],
            },
            "max_per_mmsi_train":          args.max_per_mmsi_train,
            "max_per_mmsi_eval":           args.max_per_mmsi_eval,
            "max_per_segment_train":       args.max_per_segment_train,
            "max_per_segment_eval":        args.max_per_segment_eval,
            "min_segment_step_gap_train":  args.min_segment_step_gap_train,
            "min_segment_step_gap_eval":   args.min_segment_step_gap_eval,
            "min_future_linearity": args.min_future_linearity,
            "max_hist_cog_motion_mae_deg": args.max_hist_cog_motion_mae_deg,
            "max_bridge_turn_deg":  args.max_bridge_turn_deg,
        }
    }

    t_total = time.time()
    for split, target in [
        ("train", args.train_target),
        ("val",   args.val_target),
        ("test",  args.test_target),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"[build] === {split.upper()}  target={target:,} ===", flush=True)
        print(f"{'='*60}", flush=True)
        summary[split] = build_split(
            split=split,
            target=target,
            benchmark_root=args.benchmark_root,
            output_root=args.output_root,
            args=args,
            allowed_groups=allowed,
            excluded_groups=excluded,
        )

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[build] ALL DONE  total_elapsed={time.time()-t_total:.0f}s", flush=True)
    print(f"[build] summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
