#!/usr/bin/env python3
"""Stage 10: remove segments too short for 10->10 sampling."""

from __future__ import annotations

from pipeline_utils import ensure_dir, list_primary_stage_csvs, load_config, parse_stage_args, read_stage_csv, write_json, write_stage_csv


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 10")
    config = load_config(args.config)
    ensure_dir(args.output)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    min_points = int(config["thresholds"]["min_resampled_points"])
    min_duration = int(config["thresholds"]["min_segment_duration_seconds"])
    kept_rows = 0
    kept_segments = 0

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        keep_ids = []
        for segment_id, group in df.groupby("segment_id", sort=False):
            duration = float((group["timestamp_utc"].max() - group["timestamp_utc"].min()).total_seconds())
            if len(group) >= min_points and duration >= min_duration:
                keep_ids.append(segment_id)
        result = df.loc[df["segment_id"].isin(keep_ids)].copy()
        kept_rows += len(result)
        kept_segments += len(keep_ids)
        write_stage_csv(result, out_dir / input_path.name)

    summary_path = args.output / ("summary.json" if args.input.is_dir() else f"{args.input.stem}.summary.json")
    write_json(
        summary_path,
        {"stage": "10_filter_short_segments", "rows_after": kept_rows, "segments_after": kept_segments},
    )
    print(f"segments_after={kept_segments}")


if __name__ == "__main__":
    main()
