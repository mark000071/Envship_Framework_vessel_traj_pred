#!/usr/bin/env python3
"""Stage 09: second-pass anomaly detection after resampling."""

from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    KNOTS_PER_MPS,
    ensure_dir,
    haversine_m,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    read_stage_csv,
    soft_speed_cap,
    write_json,
    write_stage_csv,
)


def clean_segment(group: pd.DataFrame, config: dict) -> tuple[list[pd.DataFrame], int]:
    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    if len(group) < 3:
        return [group.copy()], 0

    lat_med = group["lat"].rolling(window=3, center=True).median()
    lon_med = group["lon"].rolling(window=3, center=True).median()
    max_offset_m = float(config["thresholds"]["secondary_anomaly_offset_meters"])
    speed_margin = float(config["thresholds"]["secondary_anomaly_speed_margin_knots"])
    ship_cap = soft_speed_cap(group["ship_type"].iloc[0], config) or float(config["thresholds"]["global_hard_speed_cap_knots"])

    to_drop = set()
    for idx in range(1, len(group) - 1):
        if pd.isna(lat_med.iloc[idx]) or pd.isna(lon_med.iloc[idx]):
            continue
        deviation = haversine_m(group.loc[idx, "lat"], group.loc[idx, "lon"], lat_med.iloc[idx], lon_med.iloc[idx])
        dt_prev = float((group.loc[idx, "timestamp_utc"] - group.loc[idx - 1, "timestamp_utc"]).total_seconds())
        dt_next = float((group.loc[idx + 1, "timestamp_utc"] - group.loc[idx, "timestamp_utc"]).total_seconds())
        if dt_prev <= 0 or dt_next <= 0:
            continue
        v_prev = haversine_m(group.loc[idx - 1, "lat"], group.loc[idx - 1, "lon"], group.loc[idx, "lat"], group.loc[idx, "lon"]) / dt_prev * KNOTS_PER_MPS
        v_next = haversine_m(group.loc[idx, "lat"], group.loc[idx, "lon"], group.loc[idx + 1, "lat"], group.loc[idx + 1, "lon"]) / dt_next * KNOTS_PER_MPS
        if deviation > max_offset_m and max(v_prev, v_next) > ship_cap + speed_margin:
            to_drop.add(idx)

    cleaned = group.drop(index=list(to_drop)).reset_index(drop=True)
    if cleaned.empty:
        return [], len(to_drop)

    pieces = []
    start = 0
    for idx in range(1, len(cleaned)):
        dt_seconds = float((cleaned.loc[idx, "timestamp_utc"] - cleaned.loc[idx - 1, "timestamp_utc"]).total_seconds())
        if dt_seconds > 40:
            pieces.append(cleaned.iloc[start:idx].copy())
            start = idx
    pieces.append(cleaned.iloc[start:].copy())

    final_pieces = []
    for piece_idx, piece in enumerate(pieces):
        if piece.empty:
            continue
        piece = piece.copy()
        piece["segment_id"] = f"{piece['segment_id'].iloc[0]}_c{piece_idx:02d}"
        final_pieces.append(piece)
    return final_pieces, len(to_drop)


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 09")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    removed_points = 0
    total_rows = 0

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        cleaned_parts = []
        for _, group in df.groupby("segment_id", sort=False):
            parts, removed = clean_segment(group, config)
            cleaned_parts.extend(parts)
            removed_points += removed
        result = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else df.iloc[0:0].copy()
        total_rows += len(result)
        write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    write_json(
        summary_path,
        {"stage": "09_second_pass_anomaly_check", "rows_after": total_rows, "removed_points": removed_points},
    )
    print(f"rows_after={total_rows}")


if __name__ == "__main__":
    main()
