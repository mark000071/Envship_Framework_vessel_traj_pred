#!/usr/bin/env python3
"""Stage 05: split tracks by gap and implied speed."""

from __future__ import annotations

import shutil

import pandas as pd

from pipeline_utils import (
    KNOTS_PER_MPS,
    ensure_dir,
    haversine_m,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    read_stage_chunks,
    soft_speed_cap,
    write_partitioned_chunk,
    write_json,
    write_stage_csv,
)


def process_group(group: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, int]:
    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    segment_ids: list[str] = []
    break_reasons: list[str] = []
    dt_prev: list[float | None] = []
    imp_speed: list[float | None] = []
    dropped_rows = 0
    current_segment = 0
    valid_rows: list[int] = []
    gap_seconds = int(config["thresholds"]["segmentation_gap_seconds"])
    speed_factor = float(config["thresholds"]["implied_speed_factor"])
    hard_cap = float(config["thresholds"]["global_hard_speed_cap_knots"])

    for idx in range(len(group)):
        if idx == 0:
            valid_rows.append(idx)
            segment_ids.append(f"{int(group.loc[idx, 'mmsi'])}_{current_segment:06d}")
            break_reasons.append("start")
            dt_prev.append(np.nan)
            imp_speed.append(np.nan)
            continue

        prev = group.loc[valid_rows[-1]]
        curr = group.loc[idx]
        dt_seconds = (curr["timestamp_utc"] - prev["timestamp_utc"]).total_seconds()
        if dt_seconds <= 0:
            dropped_rows += 1
            continue

        distance_m = haversine_m(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        implied_knots = distance_m / dt_seconds * KNOTS_PER_MPS
        prev_sog = float(prev["sog"]) if pd.notna(prev["sog"]) else 0.0
        curr_sog = float(curr["sog"]) if pd.notna(curr["sog"]) else 0.0
        ship_cap = max(
            soft_speed_cap(prev["ship_type"], config) or hard_cap,
            soft_speed_cap(curr["ship_type"], config) or hard_cap,
        )
        threshold = max(speed_factor * max(prev_sog, curr_sog), ship_cap + 5.0)

        if dt_seconds > gap_seconds:
            current_segment += 1
            reason = "time_gap"
        elif implied_knots > threshold:
            current_segment += 1
            reason = "implied_speed"
        else:
            reason = "continue"

        valid_rows.append(idx)
        segment_ids.append(f"{int(curr['mmsi'])}_{current_segment:06d}")
        break_reasons.append(reason)
        dt_prev.append(dt_seconds)
        imp_speed.append(implied_knots)

    result = group.loc[valid_rows].copy().reset_index(drop=True)
    result["segment_id"] = segment_ids
    result["segment_break_before"] = break_reasons
    result["dt_to_prev_seconds"] = dt_prev
    result["implied_speed_from_prev_knots"] = imp_speed
    return result, dropped_rows


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 05")
    config = load_config(args.config)
    ensure_dir(args.output)
    input_paths = list_primary_stage_csvs(args.input)
    chunksize = int(config["io"]["chunksize"])
    partition_count = int(config["io"]["partition_count"])
    temp_dir = args.output / "_tmp_partitions"
    shutil.rmtree(temp_dir, ignore_errors=True)
    ensure_dir(temp_dir)
    for path in input_paths:
        for chunk in read_stage_chunks(path, chunksize=chunksize, parse_dates=["timestamp_utc"]):
            write_partitioned_chunk(chunk, temp_dir, partition_count)

    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    segment_summaries = []
    dropped_non_positive_dt = 0
    total_rows = 0
    total_segments = 0
    for partition_path in sorted(temp_dir.glob("part-*.csv")):
        df = pd.read_csv(partition_path, parse_dates=["timestamp_utc"]).sort_values(["mmsi", "timestamp_utc"])
        segmented_parts = []
        for _, group in df.groupby("mmsi", sort=False):
            result, dropped = process_group(group, config)
            segmented_parts.append(result)
            dropped_non_positive_dt += dropped
        segmented = pd.concat(segmented_parts, ignore_index=True) if segmented_parts else df.iloc[0:0].copy()
        total_rows += len(segmented)
        total_segments += int(segmented["segment_id"].nunique())
        segment_summary = (
            segmented.groupby("segment_id", as_index=False)
            .agg(
                mmsi=("mmsi", "first"),
                start_time=("timestamp_utc", "min"),
                end_time=("timestamp_utc", "max"),
                num_points=("segment_id", "size"),
            )
            .sort_values(["mmsi", "start_time"])
        )
        segment_summaries.append(segment_summary)
        write_stage_csv(segmented, out_dir / f"{partition_path.stem}.csv.gz")

    shutil.rmtree(temp_dir, ignore_errors=True)
    if segment_summaries:
        write_stage_csv(pd.concat(segment_summaries, ignore_index=True), args.output / "segment_summary.csv.gz")
    write_json(
        args.output / "summary.json",
        {
            "stage": "05_segment_tracks",
            "rows_after": total_rows,
            "segments": total_segments,
            "dropped_non_positive_dt_rows": dropped_non_positive_dt,
            "partition_count": partition_count,
        },
    )
    print(f"segments={total_segments}")


if __name__ == "__main__":
    import numpy as np

    main()
