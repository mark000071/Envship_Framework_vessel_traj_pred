#!/usr/bin/env python3
"""Stage 17 — OSM temporal-consistency check for standard_track samples.

The OSM rasters used to build the env context come from a single OSM
snapshot (typically the current 2026 OSM). When the AIS data is older
than the OSM snapshot (e.g. Piraeus 2019 vs OSM 2026), recently-built
ports/piers/breakwaters appear as land in the SDF while the original
AIS trajectory was on water. This stage flags such samples.

For each sample:
  1. Decode hist+fut XY arrays back to meters relative to anchor.
  2. Project each (x, y) onto the per-sample signed_dist_shore raster.
  3. signed_dist_shore < 0 means the point is on land per the OSM snapshot.
  4. Aggregate:
        max_inland_depth_m       — deepest inland penetration in meters
        n_inland_points          — number of trajectory points flagged inland
        max_consec_inland_run    — longest run of consecutive inland points
        n_uncheckable_points     — points outside the 5-km SDF patch (Track B)
  5. Flag:
        osm_temporal_consistent  = (max_inland_depth_m <= MAX_DEPTH_M)
                                   AND (max_consec_inland_run < MAX_RUN)

Defaults: MAX_DEPTH_M = 30, MAX_RUN = 3. Both are configurable.

Output: <track>/osm_temporal_consistency/<split>_flags.csv per dataset.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

GRID = 128
RADIUS_M = 5000.0
CELL_M = (2 * RADIUS_M) / GRID  # ≈ 78.125 m


def xy_to_cell(x: float, y: float, r: float = RADIUS_M, g: int = GRID):
    """Match the orientation used by build_standard_track_context_v1.py:
       gx = floor((x + r) / (2r) * g)
       gy = floor((r - y) / (2r) * g)   # flipped vertically
    """
    gx = int(math.floor((x + r) / (2 * r) * g))
    gy = int(math.floor((r - y) / (2 * r) * g))
    return gx, gy


def longest_consecutive_run(flags: Iterable[bool]) -> int:
    """Length of the longest run of consecutive True values."""
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def process_split(
    track_root: Path,
    context_root: Path,
    split: str,
    out_dir: Path,
    max_depth_m: float,
    max_run_threshold: int,
) -> dict:
    """Process a single split. Returns aggregate counters."""
    sdf_shore_path = context_root / "environment" / "rasters" / split / "signed_dist_shore.npy"
    sample_ids_path = context_root / "environment" / "rasters" / split / "sample_ids.npy"
    csv_path = track_root / split / "part-000.csv.gz"

    if not (sdf_shore_path.exists() and sample_ids_path.exists() and csv_path.exists()):
        return {"split": split, "skipped": True, "reason": "missing inputs"}

    sdf = np.load(sdf_shore_path)             # (N, g, g) float32 meters
    sample_ids = np.load(sample_ids_path, allow_pickle=True)
    id_to_row = {str(sid): idx for idx, sid in enumerate(sample_ids)}

    out_dir.mkdir(parents=True, exist_ok=True)
    flag_path = out_dir / f"{split}_flags.csv"
    fields = [
        "sample_id",
        "n_hist_points", "n_fut_points",
        "n_hist_inland", "n_fut_inland", "n_inland_points",
        "n_uncheckable_points",
        "max_inland_depth_m", "max_consec_inland_run",
        "any_anchor_inland", "anchor_inland_depth_m",
        "osm_temporal_consistent",
    ]
    total = 0
    n_consistent = 0
    n_uncheckable_only = 0
    n_no_sdf = 0

    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as fh, \
         flag_path.open("w", newline="", encoding="utf-8") as out_fh:
        reader = csv.DictReader(fh)
        writer = csv.DictWriter(out_fh, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            sid = row["sample_id"]
            row_idx = id_to_row.get(sid)
            if row_idx is None:
                n_no_sdf += 1
                writer.writerow({
                    "sample_id": sid,
                    "n_hist_points": 0, "n_fut_points": 0,
                    "n_hist_inland": 0, "n_fut_inland": 0, "n_inland_points": 0,
                    "n_uncheckable_points": 0,
                    "max_inland_depth_m": 0.0, "max_consec_inland_run": 0,
                    "any_anchor_inland": "",
                    "anchor_inland_depth_m": "",
                    "osm_temporal_consistent": "",   # uncheckable
                })
                total += 1
                continue
            try:
                hist_x = json.loads(row["hist_x_json"])
                hist_y = json.loads(row["hist_y_json"])
                fut_x  = json.loads(row["fut_x_json"])
                fut_y  = json.loads(row["fut_y_json"])
            except Exception:
                n_no_sdf += 1
                continue
            sdf_patch = sdf[row_idx]  # (g, g)
            ordered_flags = []
            hist_inland = 0
            fut_inland = 0
            max_depth = 0.0
            n_uncheckable = 0
            for i, (x, y) in enumerate(zip(hist_x, hist_y)):
                gx, gy = xy_to_cell(x, y)
                if 0 <= gx < GRID and 0 <= gy < GRID:
                    shore = float(sdf_patch[gy, gx])
                    inland = shore < 0
                    if inland:
                        depth = -shore
                        max_depth = max(max_depth, depth)
                        hist_inland += 1
                    ordered_flags.append(inland)
                else:
                    n_uncheckable += 1
                    ordered_flags.append(False)
            for i, (x, y) in enumerate(zip(fut_x, fut_y)):
                gx, gy = xy_to_cell(x, y)
                if 0 <= gx < GRID and 0 <= gy < GRID:
                    shore = float(sdf_patch[gy, gx])
                    inland = shore < 0
                    if inland:
                        depth = -shore
                        max_depth = max(max_depth, depth)
                        fut_inland += 1
                    ordered_flags.append(inland)
                else:
                    n_uncheckable += 1
                    ordered_flags.append(False)
            n_total_pts = len(ordered_flags)
            n_inland = hist_inland + fut_inland
            n_check = n_total_pts - n_uncheckable
            max_run_len = longest_consecutive_run(ordered_flags)
            # Anchor (centre of patch) — cell (g/2, g/2)
            anchor_shore = float(sdf_patch[GRID // 2, GRID // 2])
            anchor_inland = anchor_shore < 0
            consistent = (max_depth <= max_depth_m) and (max_run_len < max_run_threshold)
            # Edge case: if every point was outside patch (uncheckable), flag as None
            if n_check == 0:
                consistent_str = ""           # uncheckable
                n_uncheckable_only += 1
            else:
                consistent_str = "true" if consistent else "false"
                if consistent:
                    n_consistent += 1
            writer.writerow({
                "sample_id": sid,
                "n_hist_points": len(hist_x), "n_fut_points": len(fut_x),
                "n_hist_inland": hist_inland, "n_fut_inland": fut_inland,
                "n_inland_points": n_inland,
                "n_uncheckable_points": n_uncheckable,
                "max_inland_depth_m": f"{max_depth:.2f}",
                "max_consec_inland_run": max_run_len,
                "any_anchor_inland": "true" if anchor_inland else "false",
                "anchor_inland_depth_m": f"{(-anchor_shore):.2f}" if anchor_inland else "0.00",
                "osm_temporal_consistent": consistent_str,
            })
            total += 1

    return {
        "split": split,
        "total": total,
        "consistent": n_consistent,
        "uncheckable_only": n_uncheckable_only,
        "no_sdf": n_no_sdf,
        "flag_path": str(flag_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-root",   type=Path, required=True,
                   help="standard_track_v1 root (with train/val/test/part-000.csv.gz)")
    p.add_argument("--context-root", type=Path, required=True,
                   help="context_v1 root (containing environment/rasters/<split>/)")
    p.add_argument("--output-dir",   type=Path, required=True,
                   help="Where to write <split>_flags.csv + summary.json")
    p.add_argument("--max-depth-m", type=float, default=30.0,
                   help="Tolerance: max inland depth allowed [m]")
    p.add_argument("--max-run",     type=int,   default=3,
                   help="Tolerance: max consecutive inland points")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    print(f"[stage17] track={args.track_root}")
    print(f"[stage17] context={args.context_root}")
    print(f"[stage17] threshold: max_depth_m={args.max_depth_m}  max_run={args.max_run}")
    all_stats = {}
    for split in args.splits:
        print(f"[stage17] split={split} ...", flush=True)
        stats = process_split(args.track_root, args.context_root, split,
                              args.output_dir, args.max_depth_m, args.max_run)
        all_stats[split] = stats
        print(f"[stage17]   {stats}", flush=True)
    summary = {
        "track_root":   str(args.track_root),
        "context_root": str(args.context_root),
        "max_depth_m":  args.max_depth_m,
        "max_run":      args.max_run,
        "splits":       all_stats,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[stage17] DONE → {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
