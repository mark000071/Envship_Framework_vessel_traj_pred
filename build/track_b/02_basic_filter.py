#!/usr/bin/env python3
"""Stage 02: basic legality filtering."""

from __future__ import annotations

import pandas as pd

from pipeline_utils import ensure_dir, list_stage_csvs, load_config, parse_stage_args, read_stage_csv, write_json, write_stage_csv


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 02")
    config = load_config(args.config)
    ensure_dir(args.output)
    manifest = []

    for path in list_stage_csvs(args.input):
        df = read_stage_csv(path)
        before = len(df)

        valid_required = (
            df["mmsi"].notna()
            & df["timestamp_utc"].notna()
            & df["lat"].notna()
            & df["lon"].notna()
        )
        valid_range = (
            df["lat"].between(-90, 90, inclusive="both")
            & df["lon"].between(-180, 180, inclusive="both")
        )
        sog_ok = df["sog"].isna() | (df["sog"] >= 0)
        df = df.loc[valid_required & valid_range & sog_ok].copy()

        invalid_cog = df["cog"].notna() & ((df["cog"] < 0) | (df["cog"] >= 360))
        invalid_heading = df["heading"].notna() & ((df["heading"] < 0) | (df["heading"] >= 360))
        df.loc[invalid_cog, "cog"] = pd.NA
        df.loc[invalid_heading, "heading"] = pd.NA

        out_path = args.output / path.name
        write_stage_csv(df, out_path)
        manifest.append(
            {
                "input_file": str(path),
                "output_file": str(out_path),
                "rows_before": before,
                "rows_after": len(df),
                "dropped_rows": before - len(df),
            }
        )

    manifest_path = args.output / ("manifest.json" if args.input.is_dir() else f"{args.input.stem}.manifest.json")
    write_json(manifest_path, {"files": manifest, "stage": "02_basic_filter"})
    print(f"filtered_files={len(manifest)}")


if __name__ == "__main__":
    main()
