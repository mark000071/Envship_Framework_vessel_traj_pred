#!/usr/bin/env python3
"""Stage 08: resample valid underway segments to 20 s."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_utils import (
    circular_interp_deg,
    ensure_dir,
    epoch_align_ceil,
    epoch_align_floor,
    latlon_to_local_xy,
    list_primary_stage_csvs,
    load_config,
    local_xy_to_latlon,
    parse_stage_args,
    read_stage_csv,
    write_json,
    write_stage_csv,
)


def split_for_resample(group: pd.DataFrame, config: dict) -> list[pd.DataFrame]:
    max_gap = int(config["thresholds"]["short_gap_max_seconds"])
    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    pieces = []
    start = 0
    for idx in range(1, len(group)):
        dt_seconds = float((group.loc[idx, "timestamp_utc"] - group.loc[idx - 1, "timestamp_utc"]).total_seconds())
        if dt_seconds > max_gap:
            pieces.append(group.iloc[start:idx].copy())
            start = idx
    pieces.append(group.iloc[start:].copy())
    return [piece for piece in pieces if len(piece) >= 2]


def resample_piece(group: pd.DataFrame, config: dict, suffix: int) -> pd.DataFrame:
    interval = int(config["thresholds"]["resample_interval_seconds"])
    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    start_ts = epoch_align_ceil(group["timestamp_utc"].iloc[0], interval)
    end_ts = epoch_align_floor(group["timestamp_utc"].iloc[-1], interval)
    if end_ts < start_ts:
        return group.iloc[0:0].copy()
    grid = pd.date_range(start=start_ts, end=end_ts, freq=f"{interval}s", tz="UTC")
    ref_lat = float(group["lat"].iloc[0])
    ref_lon = float(group["lon"].iloc[0])
    x_obs, y_obs = latlon_to_local_xy(group["lat"].to_numpy(), group["lon"].to_numpy(), ref_lat, ref_lon)
    obs_ts = group["timestamp_utc"].astype("int64").to_numpy()
    exact_map = {ts: idx for idx, ts in enumerate(obs_ts)}
    rows = []
    new_segment_id = f"{group['segment_id'].iloc[0]}_r{suffix:02d}"

    for ts in grid:
        ts_ns = int(ts.value)
        if ts_ns in exact_map:
            row = group.iloc[exact_map[ts_ns]].copy()
            row["segment_id"] = new_segment_id
            row["is_resampled_point"] = True
            row["is_grid_interpolated_point"] = False
            row["is_gap_imputed_point"] = bool(row.get("is_interpolated_obs", False))
            row["is_interpolated_point"] = row["is_gap_imputed_point"]
            rows.append(row.to_dict())
            continue

        right_idx = int(np.searchsorted(obs_ts, ts_ns, side="right"))
        left_idx = right_idx - 1
        if left_idx < 0 or right_idx >= len(group):
            continue

        left = group.iloc[left_idx]
        right = group.iloc[right_idx]
        total = float((right["timestamp_utc"] - left["timestamp_utc"]).total_seconds())
        if total <= 0:
            continue
        alpha = float((ts - left["timestamp_utc"]).total_seconds()) / total
        x_val = (1 - alpha) * x_obs[left_idx] + alpha * x_obs[right_idx]
        y_val = (1 - alpha) * y_obs[left_idx] + alpha * y_obs[right_idx]
        lat_val, lon_val = local_xy_to_latlon(
            pd.Series([x_val]).to_numpy(), pd.Series([y_val]).to_numpy(), ref_lat, ref_lon
        )
        row = left.copy()
        row["timestamp_utc"] = ts
        row["lat"] = float(lat_val[0])
        row["lon"] = float(lon_val[0])
        if pd.notna(left["sog"]) and pd.notna(right["sog"]) and abs(float(right["sog"]) - float(left["sog"])) <= float(config["thresholds"]["sog_interp_diff_max_knots"]):
            row["sog"] = (1 - alpha) * float(left["sog"]) + alpha * float(right["sog"])
        else:
            row["sog"] = pd.NA
        row["cog"] = circular_interp_deg(left["cog"], right["cog"], alpha)
        row["heading"] = circular_interp_deg(left["heading"], right["heading"], alpha)
        row["segment_id"] = new_segment_id
        row["is_resampled_point"] = True
        row["is_grid_interpolated_point"] = True
        row["is_gap_imputed_point"] = False
        row["is_interpolated_point"] = False
        rows.append(row.to_dict())

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 08")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    total_rows = 0
    total_segments = 0

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        pieces = []
        for _, group in df.groupby("segment_id", sort=False):
            subsegments = split_for_resample(group, config)
            for suffix, subsegment in enumerate(subsegments):
                resampled = resample_piece(subsegment, config, suffix)
                if not resampled.empty:
                    pieces.append(resampled)
        non_empty_pieces = [piece.dropna(axis=1, how="all") for piece in pieces if not piece.empty]
        result = pd.concat(non_empty_pieces, ignore_index=True) if non_empty_pieces else df.iloc[0:0].copy()
        total_rows += len(result)
        total_segments += int(result["segment_id"].nunique()) if not result.empty else 0
        write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    write_json(
        summary_path,
        {"stage": "08_resample_20s", "rows_after": total_rows, "segments_after": total_segments},
    )
    print(f"rows_after={total_rows}")


if __name__ == "__main__":
    main()
