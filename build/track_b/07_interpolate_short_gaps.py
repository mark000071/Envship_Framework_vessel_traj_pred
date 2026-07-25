#!/usr/bin/env python3
"""Stage 07: interpolate only short gaps <= 120 s."""

from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    circular_interp_deg,
    ensure_dir,
    latlon_to_local_xy,
    list_primary_stage_csvs,
    load_config,
    local_xy_to_latlon,
    parse_stage_args,
    read_stage_csv,
    write_json,
    write_stage_csv,
)


def classify_gap(dt_seconds: float, config: dict) -> str:
    short_max = int(config["thresholds"]["short_gap_max_seconds"])
    medium_max = int(config["thresholds"]["medium_gap_max_seconds"])
    if dt_seconds <= 20:
        return "none"
    if dt_seconds <= short_max:
        return "short"
    if dt_seconds <= medium_max:
        return "medium"
    return "long"


def interpolate_segment(group: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    stats = {"short_gaps": 0, "medium_gaps": 0, "long_gaps": 0, "inserted_points": 0}
    sog_interp_limit = float(config["thresholds"]["sog_interp_diff_max_knots"])
    group = group.sort_values("timestamp_utc").reset_index(drop=True)

    first = group.iloc[0].copy()
    first["gap_class_before"] = "start"
    first["is_interpolated_obs"] = False
    rows.append(first.to_dict())

    for idx in range(1, len(group)):
        prev = group.iloc[idx - 1]
        curr = group.iloc[idx]
        dt_seconds = float((curr["timestamp_utc"] - prev["timestamp_utc"]).total_seconds())
        gap_class = classify_gap(dt_seconds, config)
        if gap_class in {"short", "medium", "long"}:
            stats[f"{gap_class}_gaps"] += 1

        if gap_class == "short":
            ref_lat = float((prev["lat"] + curr["lat"]) / 2.0)
            ref_lon = float((prev["lon"] + curr["lon"]) / 2.0)
            x_pair, y_pair = latlon_to_local_xy(
                pd.Series([prev["lat"], curr["lat"]], dtype="float64").to_numpy(),
                pd.Series([prev["lon"], curr["lon"]], dtype="float64").to_numpy(),
                ref_lat,
                ref_lon,
            )
            t_cursor = prev["timestamp_utc"] + pd.Timedelta(seconds=20)
            while t_cursor < curr["timestamp_utc"]:
                alpha = (t_cursor - prev["timestamp_utc"]).total_seconds() / dt_seconds
                x_val = (1 - alpha) * x_pair[0] + alpha * x_pair[1]
                y_val = (1 - alpha) * y_pair[0] + alpha * y_pair[1]
                lat_val, lon_val = local_xy_to_latlon(
                    pd.Series([x_val]).to_numpy(), pd.Series([y_val]).to_numpy(), ref_lat, ref_lon
                )
                row = prev.copy()
                row["timestamp_utc"] = t_cursor
                row["lat"] = float(lat_val[0])
                row["lon"] = float(lon_val[0])
                if pd.notna(prev["sog"]) and pd.notna(curr["sog"]) and abs(float(curr["sog"]) - float(prev["sog"])) <= sog_interp_limit:
                    row["sog"] = (1 - alpha) * float(prev["sog"]) + alpha * float(curr["sog"])
                else:
                    row["sog"] = pd.NA
                row["cog"] = circular_interp_deg(prev["cog"], curr["cog"], alpha)
                row["heading"] = circular_interp_deg(prev["heading"], curr["heading"], alpha)
                row["gap_class_before"] = "short"
                row["is_interpolated_obs"] = True
                rows.append(row.to_dict())
                stats["inserted_points"] += 1
                t_cursor += pd.Timedelta(seconds=20)

        curr_row = curr.copy()
        curr_row["gap_class_before"] = gap_class
        curr_row["is_interpolated_obs"] = False
        rows.append(curr_row.to_dict())

    return pd.DataFrame(rows), stats


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 07")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    totals = {"short_gaps": 0, "medium_gaps": 0, "long_gaps": 0, "inserted_points": 0}

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        out_parts = []
        for _, group in df.groupby("segment_id", sort=False):
            part, stats = interpolate_segment(group, config)
            out_parts.append(part)
            for key, value in stats.items():
                totals[key] += value
        result = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
        write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    write_json(summary_path, {"stage": "07_interpolate_short_gaps", **totals})
    print(f"inserted_points={totals['inserted_points']}")


if __name__ == "__main__":
    main()
