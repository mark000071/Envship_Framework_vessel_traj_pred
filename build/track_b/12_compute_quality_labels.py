#!/usr/bin/env python3
"""Stage 12: compute quality labels and interpolation ratios."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline_utils import (
    ensure_dir,
    json_dumps_compact,
    json_loads,
    latlon_to_local_xy,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    ship_type_id,
    read_stage_csv,
    write_json,
    write_stage_csv,
)


def cyclical(values: np.ndarray, period: float) -> tuple[list[float], list[float]]:
    angles = 2.0 * math.pi * values / period
    return np.sin(angles).round(6).tolist(), np.cos(angles).round(6).tolist()


def process_sample(row: pd.Series, config: dict) -> dict:
    hist = int(config["thresholds"]["sliding_window_hist_points"])
    fut = int(config["thresholds"]["sliding_window_fut_points"])
    timestamps = pd.to_datetime(json_loads(row["timestamps_json"]), utc=True, format="ISO8601")
    lat = np.array(json_loads(row["lat_json"]), dtype=float)
    lon = np.array(json_loads(row["lon_json"]), dtype=float)
    sog = np.array([np.nan if v is None else float(v) for v in json_loads(row["sog_json"])], dtype=float)
    cog = np.array([np.nan if v is None else float(v) for v in json_loads(row["cog_json"])], dtype=float)
    heading = np.array([np.nan if v is None else float(v) for v in json_loads(row["heading_json"])], dtype=float)
    gap_imputed = np.array(json_loads(row.get("gap_imputed_json", row["interp_json"])), dtype=bool)
    grid_interp = np.array(json_loads(row.get("grid_interp_json", "[]") or "[]"), dtype=bool)
    if grid_interp.size == 0:
        grid_interp = np.zeros_like(gap_imputed, dtype=bool)

    ref_idx = hist - 1
    x, y = latlon_to_local_xy(lat, lon, float(lat[ref_idx]), float(lon[ref_idx]))
    hist_x = x[:hist].round(3).tolist()
    hist_y = y[:hist].round(3).tolist()
    fut_x = x[hist:].round(3).tolist()
    fut_y = y[hist:].round(3).tolist()

    hist_disp = float(math.hypot(hist_x[-1] - hist_x[0], hist_y[-1] - hist_y[0]))
    fut_disp = float(math.hypot(fut_x[-1] - fut_x[0], fut_y[-1] - fut_y[0]))
    time_values = np.array([ts.hour * 3600 + ts.minute * 60 + ts.second for ts in timestamps[:hist]], dtype=float)
    dow_values = np.array([ts.dayofweek for ts in timestamps[:hist]], dtype=float)
    tod_sin, tod_cos = cyclical(time_values, 86400.0)
    dow_sin, dow_cos = cyclical(dow_values, 7.0)

    cog_valid = np.where(np.isnan(cog[:hist]), 0.0, cog[:hist])
    heading_valid = np.where(np.isnan(heading[:hist]), 0.0, heading[:hist])
    cog_sin, cog_cos = cyclical(cog_valid, 360.0)
    heading_sin, heading_cos = cyclical(heading_valid, 360.0)

    dt_seconds = np.diff(np.array([ts.value for ts in timestamps], dtype=np.int64)) / 1e9
    hist_gap_ok = bool(dt_seconds[: hist - 1].max(initial=0.0) <= float(config["thresholds"]["max_unprocessed_gap_seconds"]))
    fut_gap_ok = bool(dt_seconds[hist:].max(initial=0.0) <= float(config["thresholds"]["max_unprocessed_gap_seconds"]))

    interp_ratio_hist = float(gap_imputed[:hist].mean())
    interp_ratio_fut = float(gap_imputed[hist:].mean())
    interp_ratio_total = float(gap_imputed.mean())
    grid_interp_ratio_hist = float(grid_interp[:hist].mean())
    grid_interp_ratio_fut = float(grid_interp[hist:].mean())
    grid_interp_ratio_total = float(grid_interp.mean())

    full_eligible = (
        hist_gap_ok
        and fut_gap_ok
        and hist_disp >= float(config["thresholds"]["min_hist_displacement_meters"])
        and fut_disp >= float(config["thresholds"]["min_fut_displacement_meters"])
    )
    core_eligible = (
        full_eligible
        and interp_ratio_hist <= float(config["thresholds"]["core_interp_ratio_hist_max"])
        and interp_ratio_fut <= float(config["thresholds"]["core_interp_ratio_fut_max"])
        and interp_ratio_total <= float(config["thresholds"]["core_interp_ratio_total_max"])
    )

    return {
        "sample_id": row["sample_id"],
        "mmsi": int(row["mmsi"]),
        "segment_id": row["segment_id"],
        "ship_type": row["ship_type"],
        "ship_class": row["ship_class"],
        "ship_type_id": ship_type_id(row["ship_type"]),
        "hist_end_ts": timestamps[hist - 1].isoformat(),
        "pred_end_ts": timestamps[-1].isoformat(),
        "hist_x_json": json_dumps_compact(hist_x),
        "hist_y_json": json_dumps_compact(hist_y),
        "fut_x_json": json_dumps_compact(fut_x),
        "fut_y_json": json_dumps_compact(fut_y),
        "hist_sog_json": json_dumps_compact(np.nan_to_num(sog[:hist], nan=0.0).round(4).tolist()),
        "hist_cog_sin_json": json_dumps_compact(cog_sin),
        "hist_cog_cos_json": json_dumps_compact(cog_cos),
        "hist_heading_sin_json": json_dumps_compact(heading_sin),
        "hist_heading_cos_json": json_dumps_compact(heading_cos),
        "hist_time_of_day_sin_json": json_dumps_compact(tod_sin),
        "hist_time_of_day_cos_json": json_dumps_compact(tod_cos),
        "hist_day_of_week_sin_json": json_dumps_compact(dow_sin),
        "hist_day_of_week_cos_json": json_dumps_compact(dow_cos),
        "hist_interp_json": json_dumps_compact(gap_imputed[:hist].tolist()),
        "fut_interp_json": json_dumps_compact(gap_imputed[hist:].tolist()),
        "hist_grid_interp_json": json_dumps_compact(grid_interp[:hist].tolist()),
        "fut_grid_interp_json": json_dumps_compact(grid_interp[hist:].tolist()),
        "interp_ratio_hist": interp_ratio_hist,
        "interp_ratio_fut": interp_ratio_fut,
        "interp_ratio_total": interp_ratio_total,
        "grid_interp_ratio_hist": grid_interp_ratio_hist,
        "grid_interp_ratio_fut": grid_interp_ratio_fut,
        "grid_interp_ratio_total": grid_interp_ratio_total,
        "hist_gap_ok": hist_gap_ok,
        "fut_gap_ok": fut_gap_ok,
        "hist_displacement_m": hist_disp,
        "fut_displacement_m": fut_disp,
        "core_eligible": core_eligible,
        "full_eligible": full_eligible,
        "quality_tier": "core" if core_eligible else ("full" if full_eligible else "drop"),
    }


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 12")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    counts = {"core": 0, "full": 0, "drop": 0}
    aggregates = {
        "interp_ratio_hist_sum": 0.0,
        "interp_ratio_fut_sum": 0.0,
        "interp_ratio_total_sum": 0.0,
        "grid_interp_ratio_total_sum": 0.0,
        "samples": 0,
    }

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        if df.empty:
            continue
        rows = [process_sample(row, config) for _, row in df.iterrows()]
        result = pd.DataFrame(rows)
        if result.empty:
            continue
        for tier in result["quality_tier"].value_counts().to_dict().items():
            counts[tier[0]] += int(tier[1])
        aggregates["interp_ratio_hist_sum"] += float(result["interp_ratio_hist"].sum())
        aggregates["interp_ratio_fut_sum"] += float(result["interp_ratio_fut"].sum())
        aggregates["interp_ratio_total_sum"] += float(result["interp_ratio_total"].sum())
        aggregates["grid_interp_ratio_total_sum"] += float(result["grid_interp_ratio_total"].sum())
        aggregates["samples"] += len(result)
        write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    sample_count = int(aggregates["samples"])
    denom = max(sample_count, 1)
    write_json(
        summary_path,
        {
            "stage": "12_compute_quality_labels",
            "samples": sample_count,
            "counts": counts,
            "mean_interp_ratio_hist": aggregates["interp_ratio_hist_sum"] / denom,
            "mean_interp_ratio_fut": aggregates["interp_ratio_fut_sum"] / denom,
            "mean_interp_ratio_total": aggregates["interp_ratio_total_sum"] / denom,
            "mean_grid_interp_ratio_total": aggregates["grid_interp_ratio_total_sum"] / denom,
        },
    )
    print(f"core_samples={counts['core']}")


if __name__ == "__main__":
    main()
