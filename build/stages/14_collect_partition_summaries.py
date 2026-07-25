#!/usr/bin/env python3
"""Collect per-partition summary files into one stage-level summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.summary.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def merge_value(dst: dict, key: str, value: object) -> None:
    if isinstance(value, dict):
        current = dst.setdefault(key, {})
        for sub_key, sub_val in value.items():
            current[sub_key] = current.get(sub_key, 0) + sub_val
    elif isinstance(value, (int, float)):
        dst[key] = dst.get(key, 0) + value


def main() -> None:
    args = parse_args()
    files = sorted(args.stage_dir.glob(args.pattern))
    merged: dict = {}
    weighted = {
        "mean_interp_ratio_hist": 0.0,
        "mean_interp_ratio_fut": 0.0,
        "mean_interp_ratio_total": 0.0,
        "mean_grid_interp_ratio_total": 0.0,
    }
    total_samples = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        merged["stage"] = payload.get("stage", merged.get("stage"))
        samples = int(payload.get("samples", 0))
        if samples > 0:
            total_samples += samples
            for key in weighted:
                if key in payload:
                    weighted[key] += float(payload[key]) * samples
        for key, value in payload.items():
            if key in {"stage", *weighted.keys()}:
                continue
            merge_value(merged, key, value)

    if total_samples > 0:
        merged["samples"] = total_samples
        for key, numerator in weighted.items():
            merged[key] = numerator / total_samples

    output = args.output or args.stage_dir / "summary.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    print(output)


if __name__ == "__main__":
    main()
