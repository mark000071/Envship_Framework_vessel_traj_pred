#!/usr/bin/env python3
"""Stage 13: export core/full benchmark release layout."""

from __future__ import annotations

from collections import defaultdict

from pipeline_utils import ensure_dir, list_primary_stage_csvs, load_config, parse_stage_args, read_stage_csv, stable_split_from_mmsi, write_json, write_stage_csv


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 13")
    load_config(args.config)
    ensure_dir(args.output)
    counts = defaultdict(int)

    for input_path in list_primary_stage_csvs(args.input):
        df = read_stage_csv(input_path)
        if df.empty:
            continue
        for version, flag in [("core", "core_eligible"), ("full", "full_eligible")]:
            subset = df.loc[df[flag].fillna(False)].copy()
            if subset.empty:
                continue
            subset["split"] = subset["mmsi"].map(stable_split_from_mmsi)
            for split, split_df in subset.groupby("split", sort=False):
                split_dir = args.output / version / split
                ensure_dir(split_dir)
                out_path = split_dir / input_path.name
                write_stage_csv(split_df, out_path)
                counts[f"{version}_{split}"] += len(split_df)

    summary_path = args.output / ("export_summary.json" if args.input.is_dir() else f"{args.input.stem}.export_summary.json")
    write_json(summary_path, {"stage": "13_export_benchmark", "counts": dict(counts)})
    print(dict(counts))


if __name__ == "__main__":
    main()
