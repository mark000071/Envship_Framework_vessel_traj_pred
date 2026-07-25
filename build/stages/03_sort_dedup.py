#!/usr/bin/env python3
"""Stage 03: sort by MMSI/timestamp and deduplicate."""

from __future__ import annotations

import shutil

import pandas as pd

from pipeline_utils import (
    ensure_dir,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    read_stage_chunks,
    write_json,
    write_partitioned_chunk,
    write_stage_csv,
)


DEDUP_SCORE_COLUMNS = [
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "nav_status",
    "ship_type",
    "length",
    "width",
    "draught",
    "imo",
]


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 03")
    config = load_config(args.config)
    ensure_dir(args.output)
    inputs = list_primary_stage_csvs(args.input)
    if not inputs:
        raise SystemExit("No filtered CSV files found.")
    chunksize = int(config["io"]["chunksize"])
    partition_count = int(config["io"]["partition_count"])
    temp_dir = args.output / "_tmp_partitions"
    shutil.rmtree(temp_dir, ignore_errors=True)
    ensure_dir(temp_dir)

    rows_before = 0
    for path in inputs:
        for chunk in read_stage_chunks(path, chunksize=chunksize, parse_dates=["timestamp_utc"]):
            rows_before += len(chunk)
            write_partitioned_chunk(chunk, temp_dir, partition_count)

    exact_dedup_rows = 0
    same_ts_rows = 0
    unique_mmsi = 0
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)

    for partition_path in sorted(temp_dir.glob("part-*.csv")):
        df = pd.read_csv(partition_path, parse_dates=["timestamp_utc"])
        df["_ingest_order"] = range(len(df))
        before = len(df)
        df = df.drop_duplicates(keep="last")
        exact_dedup_rows += before - len(df)
        df["_missing_count"] = df[DEDUP_SCORE_COLUMNS].isna().sum(axis=1)
        df = df.sort_values(["mmsi", "timestamp_utc", "_missing_count", "_ingest_order"])
        before_same_ts = len(df)
        df = df.groupby(["mmsi", "timestamp_utc"], as_index=False, sort=False).tail(1)
        same_ts_rows += before_same_ts - len(df)
        df = df.sort_values(["mmsi", "timestamp_utc", "_ingest_order"]).drop(
            columns=["_ingest_order", "_missing_count"]
        )
        unique_mmsi += int(df["mmsi"].nunique())
        write_stage_csv(df, out_dir / f"{partition_path.stem}.csv.gz")

    shutil.rmtree(temp_dir, ignore_errors=True)
    write_json(
        args.output / "summary.json",
        {
            "stage": "03_sort_dedup",
            "rows_before": rows_before,
            "rows_after": rows_before - exact_dedup_rows - same_ts_rows,
            "exact_duplicate_rows_removed": exact_dedup_rows,
            "same_mmsi_timestamp_rows_removed": same_ts_rows,
            "unique_mmsi_approx_sum_over_partitions": unique_mmsi,
            "partition_count": partition_count,
        },
    )
    print(f"rows_after={rows_before - exact_dedup_rows - same_ts_rows}")


if __name__ == "__main__":
    main()
