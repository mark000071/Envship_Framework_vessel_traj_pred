#!/usr/bin/env python3
"""Finalize stage 03 from existing _tmp_partitions/ — recovery script.

Use when stage 03 crashed during the finalize loop after producing complete
temp partition files. Skips the expensive chunked re-write and just runs
the dedup/sort/write step from existing temps.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_dir, write_json, write_stage_csv

DEDUP_SCORE_COLUMNS = ["lat","lon","sog","cog","heading","nav_status","ship_type","length","width","draught","imo"]
CANONICAL = ["mmsi","timestamp_utc","lat","lon","sog","cog","heading","nav_status","ship_type","length","width","draught","imo","source_file"]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: 03b_finalize_from_tmp.py <stage-03-output-dir>")
        sys.exit(2)
    out_root = Path(sys.argv[1])
    temp_dir = out_root / "_tmp_partitions"
    out_dir = out_root / "partitions"
    ensure_dir(out_dir)
    if not temp_dir.exists():
        print(f"no temp dir at {temp_dir}", file=sys.stderr); sys.exit(2)
    files = sorted(temp_dir.glob("part-*.csv"))
    print(f"[finalize] {len(files)} temp partitions to finalize", file=sys.stderr)

    exact_dedup_rows = same_ts_rows = unique_mmsi = total_rows = 0
    for idx, partition_path in enumerate(files):
        try:
            df = pd.read_csv(partition_path, parse_dates=["timestamp_utc"])
        except ValueError as exc:
            if "Missing column provided to 'parse_dates'" not in str(exc):
                raise
            df = pd.read_csv(partition_path, header=None, names=CANONICAL,
                             parse_dates=["timestamp_utc"])
        except pd.errors.EmptyDataError:
            print(f"[finalize] {partition_path.name}: EMPTY, skip", file=sys.stderr)
            continue
        df["_ingest_order"] = range(len(df))
        before = len(df)
        df = df.drop_duplicates(keep="last")
        exact_dedup_rows += before - len(df)
        df["_missing_count"] = df[DEDUP_SCORE_COLUMNS].isna().sum(axis=1)
        df = df.sort_values(["mmsi","timestamp_utc","_missing_count","_ingest_order"])
        before_ts = len(df)
        df = df.groupby(["mmsi","timestamp_utc"], as_index=False, sort=False).tail(1)
        same_ts_rows += before_ts - len(df)
        df = df.sort_values(["mmsi","timestamp_utc","_ingest_order"]).drop(columns=["_ingest_order","_missing_count"])
        unique_mmsi += int(df["mmsi"].nunique())
        total_rows += len(df)
        write_stage_csv(df, out_dir / f"{partition_path.stem}.csv.gz")
        if (idx + 1) % 5 == 0 or idx + 1 == len(files):
            print(f"[finalize] {idx+1}/{len(files)} done", file=sys.stderr)
    write_json(out_root / "summary.json", {
        "stage": "03_sort_dedup",
        "rows_after": total_rows,
        "exact_duplicate_rows_removed": exact_dedup_rows,
        "same_mmsi_timestamp_rows_removed": same_ts_rows,
        "unique_mmsi_approx_sum_over_partitions": unique_mmsi,
        "partition_count": len(files),
    })
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"rows_after={total_rows}")


if __name__ == "__main__":
    main()
