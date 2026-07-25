#!/usr/bin/env python3
"""Stage 06: remove berthing and anchoring segments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_utils import (
    append_stage_csv,
    consecutive_flag_duration_seconds,
    ensure_dir,
    has_low_displacement_window,
    haversine_m,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    read_stage_csv,
    write_json,
    write_stage_csv,
)


def segment_flags(segment_id: str, group: pd.DataFrame, config: dict) -> dict:
    timestamps_ns = group["timestamp_utc"].astype("int64").to_numpy()
    low_speed_threshold = float(config["thresholds"]["underway_low_speed_knots"])
    low_speed_duration_seconds = int(config["thresholds"]["underway_low_speed_duration_seconds"])
    anchor_duration_seconds = int(config["thresholds"]["anchor_moored_duration_seconds"])
    low_disp_window_seconds = int(config["thresholds"]["low_displacement_window_seconds"])
    low_disp_window_meters = float(config["thresholds"]["low_displacement_window_meters"])

    nav_status = group["nav_status"].fillna("").str.lower()
    anchor_flags = nav_status.str.contains("anchor") | nav_status.str.contains("moored")
    low_speed_flags = group["sog"].fillna(9999.0) < low_speed_threshold

    longest_anchor_run = consecutive_flag_duration_seconds(anchor_flags.to_numpy(), timestamps_ns)
    longest_low_speed_run = consecutive_flag_duration_seconds(low_speed_flags.to_numpy(), timestamps_ns)
    end_to_end_m = haversine_m(
        float(group.iloc[0]["lat"]),
        float(group.iloc[0]["lon"]),
        float(group.iloc[-1]["lat"]),
        float(group.iloc[-1]["lon"]),
    )
    low_disp_window = has_low_displacement_window(
        group["lat"].to_numpy(),
        group["lon"].to_numpy(),
        timestamps_ns,
        low_disp_window_seconds,
        low_disp_window_meters,
    )

    exclude = (
        longest_anchor_run >= anchor_duration_seconds
        or longest_low_speed_run >= low_speed_duration_seconds
        or low_disp_window
    )
    return {
        "segment_id": segment_id,
        "mmsi": int(group["mmsi"].iloc[0]),
        "num_points": len(group),
        "start_time": group["timestamp_utc"].min(),
        "end_time": group["timestamp_utc"].max(),
        "duration_seconds": float((group["timestamp_utc"].max() - group["timestamp_utc"].min()).total_seconds()),
        "end_to_end_displacement_m": end_to_end_m,
        "longest_anchor_or_moored_seconds": longest_anchor_run,
        "longest_low_speed_seconds": longest_low_speed_run,
        "has_low_displacement_20min_window": bool(low_disp_window),
        "exclude_from_underway_only": bool(exclude),
    }


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 06")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    stats_parts = []
    all_motion_rows = 0
    underway_only_rows = 0
    input_paths = list_primary_stage_csvs(args.input)
    for input_path in input_paths:
        df = read_stage_csv(input_path).sort_values(["segment_id", "timestamp_utc"])
        all_motion_rows += len(df)
        append_stage_csv(df, args.output / "all_motion.csv.gz")
        stats = pd.DataFrame(
            [segment_flags(segment_id, group, config) for segment_id, group in df.groupby("segment_id", sort=False)]
        )
        excluded = set(stats.loc[stats["exclude_from_underway_only"], "segment_id"])
        underway_only = df.loc[~df["segment_id"].isin(excluded)].copy()
        underway_only_rows += len(underway_only)
        write_stage_csv(underway_only, out_dir / input_path.name)
        stats_parts.append(stats)

    stats = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    write_stage_csv(stats, args.output / "segment_labels.csv.gz")
    write_json(
        args.output / "summary.json",
        {
            "stage": "06_remove_anchorage",
            "all_motion_rows": all_motion_rows,
            "underway_only_rows": underway_only_rows,
            "excluded_segments": int(stats["exclude_from_underway_only"].sum()),
            "kept_segments": int((~stats["exclude_from_underway_only"]).sum()),
        },
    )
    print(f"underway_only_rows={underway_only_rows}")


if __name__ == "__main__":
    main()
