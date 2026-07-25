#!/usr/bin/env python3
"""Stage 04: ship-type-aware speed threshold filtering."""

from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    ensure_dir,
    infer_ship_class,
    list_primary_stage_csvs,
    load_config,
    parse_stage_args,
    read_stage_csv,
    soft_speed_cap,
    write_json,
    write_stage_csv,
)


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 04")
    config = load_config(args.config)
    ensure_dir(args.output)
    input_paths = list_primary_stage_csvs(args.input)
    out_dir = args.output / "partitions"
    ensure_dir(out_dir)
    rows_before = 0
    rows_after = 0
    hard_removed = 0
    soft_removed = 0

    for input_path in input_paths:
        df = read_stage_csv(input_path)
        rows_before += len(df)
        df["ship_class"] = df["ship_type"].map(infer_ship_class).astype("string")
        df["shiptype_soft_cap_knots"] = df["ship_type"].map(lambda value: soft_speed_cap(value, config))

        hard_cap = float(config["thresholds"]["global_hard_speed_cap_knots"])
        hard_keep = df["sog"].isna() | (df["sog"] <= hard_cap)
        hard_removed += int((~hard_keep).sum())
        df = df.loc[hard_keep].copy()

        soft_violation = (
            df["shiptype_soft_cap_knots"].notna()
            & df["sog"].notna()
            & (df["sog"] > df["shiptype_soft_cap_knots"])
        )
        soft_removed += int(soft_violation.sum())
        df = df.loc[~soft_violation].copy()
        rows_after += len(df)
        write_stage_csv(df, out_dir / input_path.name)

    write_json(
        args.output / "summary.json",
        {
            "stage": "04_shiptype_speed_filter",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "hard_cap_removed": hard_removed,
            "soft_cap_removed": soft_removed,
        },
    )
    print(f"rows_after={rows_after}")


if __name__ == "__main__":
    main()
