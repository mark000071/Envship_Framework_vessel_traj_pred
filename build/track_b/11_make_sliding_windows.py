#!/usr/bin/env python3
"""Stage 11: slice 10 min to 10 min sliding windows."""

from __future__ import annotations

import json

import pandas as pd

from pipeline_utils import ensure_dir, json_dumps_compact, list_primary_stage_csvs, load_config, parse_stage_args, read_stage_csv, write_json, write_stage_csv


def make_windows(group: pd.DataFrame, config: dict) -> list[dict]:
    hist = int(config["thresholds"]["sliding_window_hist_points"])
    fut = int(config["thresholds"]["sliding_window_fut_points"])
    step = int(config["thresholds"]["sliding_window_step_points"])
    total = hist + fut
    windows = []
    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    if len(group) < total:
        return windows
    gap_imputed_series = (
        group["is_gap_imputed_point"]
        if "is_gap_imputed_point" in group.columns
        else group["is_interpolated_point"].fillna(False)
    )
    grid_interp_series = (
        group["is_grid_interpolated_point"]
        if "is_grid_interpolated_point" in group.columns
        else pd.Series(False, index=group.index)
    )

    for start in range(0, len(group) - total + 1, step):
        window = group.iloc[start : start + total].copy()
        sample_id = f"{window['segment_id'].iloc[0]}_{start:05d}"
        windows.append(
            {
                "sample_id": sample_id,
                "mmsi": int(window["mmsi"].iloc[0]),
                "segment_id": window["segment_id"].iloc[0],
                "ship_type": window["ship_type"].iloc[-1] if "ship_type" in window else None,
                "ship_class": window["ship_class"].iloc[-1] if "ship_class" in window else None,
                "timestamps_json": json_dumps_compact([ts.isoformat() for ts in window["timestamp_utc"]]),
                "lat_json": json_dumps_compact(window["lat"].round(7).tolist()),
                "lon_json": json_dumps_compact(window["lon"].round(7).tolist()),
                "sog_json": json_dumps_compact([None if pd.isna(v) else round(float(v), 4) for v in window["sog"]]),
                "cog_json": json_dumps_compact([None if pd.isna(v) else round(float(v), 4) for v in window["cog"]]),
                "heading_json": json_dumps_compact([None if pd.isna(v) else round(float(v), 4) for v in window["heading"]]),
                "gap_imputed_json": json_dumps_compact([bool(v) for v in gap_imputed_series.iloc[start : start + total].fillna(False)]),
                "grid_interp_json": json_dumps_compact([bool(v) for v in grid_interp_series.iloc[start : start + total].fillna(False)]),
                "interp_json": json_dumps_compact([bool(v) for v in gap_imputed_series.iloc[start : start + total].fillna(False)]),
                "gap_class_json": json_dumps_compact(window["gap_class_before"].fillna("none").tolist()),
            }
        )
    return windows


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 11")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    sample_count = 0

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        if df.empty:
            continue
        rows = []
        for _, group in df.groupby("segment_id", sort=False):
            rows.extend(make_windows(group, config))
        result = pd.DataFrame(rows)
        if not result.empty:
            sample_count += len(result)
            write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    write_json(summary_path, {"stage": "11_make_sliding_windows", "samples_after": sample_count})
    print(f"samples_after={sample_count}")


if __name__ == "__main__":
    main()
