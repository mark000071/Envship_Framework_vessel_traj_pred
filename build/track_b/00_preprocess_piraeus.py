#!/usr/bin/env python3
"""Stage 00 — Piraeus preprocessor.

Converts the Piraeus Zenodo dataset (unipi_ais_dynamic_*.csv +
ais_static/unipi_ais_static.csv) into NOAA-style column-named CSV files
that stage 01 will accept via the existing standardize_noaa_chunk path.

Why a separate stage 00:
  Piraeus encodes vessel identity as a SHA256 hash, ship_type lives in a
  separate static lookup table, and timestamps are UNIX milliseconds.
  Rather than adding a third adapter inside pipeline_utils.py, we
  preprocess into the canonical NOAA schema so 01-13 run unchanged.

Adapter logic:
  - vessel_id (sha256 hex) → pseudo-MMSI: int(hex[:8], 16) % 999_999_999 + 1
    (stable per vessel, < 1e-4 collision probability for ~10K vessels)
  - t (epoch ms) → ISO 8601 UTC string (base_date_time)
  - lon/lat/heading/speed/course → longitude/latitude/heading/sog/cog
  - LEFT JOIN static.shiptype (ITU code) → vessel_type
  - status/length/width/draught/imo set to empty (not in source)

Input:  --input <dir> containing the raw .csv file(s)
        --static <path> to ais_static.csv
Output: --output <dir> where preprocessed .csv.gz lands (one per input file,
        optionally split per day for parallel stage-01)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

import pandas as pd


def vessel_id_to_pseudo_mmsi(vid: str) -> int:
    if not isinstance(vid, str) or len(vid) < 8:
        return 0
    return int(vid[:8], 16) % 999_999_999 + 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="dir or single .csv")
    p.add_argument("--static", type=Path, required=True, help="path to ais_static.csv")
    p.add_argument("--output", type=Path, required=True, help="output dir")
    p.add_argument("--split-per-day", action="store_true",
                   help="split output into one file per UTC day (recommended for parallel pipeline)")
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load static lookup (small file, ~10K rows)
    static = pd.read_csv(args.static, dtype={"vessel_id": str})
    static = static[["vessel_id", "shiptype"]].rename(columns={"shiptype": "vessel_type"})
    static_lut = dict(zip(static["vessel_id"], static["vessel_type"]))
    print(f"[00] loaded static lookup: {len(static_lut)} vessels", file=sys.stderr)

    files = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    files = [f for f in files if not f.name.endswith("_synopses.csv")]  # skip synopses
    print(f"[00] {len(files)} input file(s)", file=sys.stderr)

    for src in files:
        print(f"[00] processing {src.name}", file=sys.stderr)
        # Per-day buffers when --split-per-day
        day_handles: dict[str, gzip.GzipFile] = {}
        day_wrote_header: dict[str, bool] = {}
        single_out = args.output / f"piraeus_{src.stem}.csv.gz"
        if not args.split_per_day and single_out.exists():
            single_out.unlink()
        total_in = total_out = 0

        for chunk in pd.read_csv(src, chunksize=args.chunksize, dtype={"vessel_id": str}):
            total_in += len(chunk)
            # transform
            chunk["mmsi"] = chunk["vessel_id"].map(vessel_id_to_pseudo_mmsi)
            chunk["base_date_time"] = pd.to_datetime(
                chunk["t"], unit="ms", utc=True, errors="coerce"
            ).dt.strftime("%Y-%m-%dT%H:%M:%S")
            chunk["vessel_type"] = chunk["vessel_id"].map(static_lut)
            # NOAA-style columns
            out = pd.DataFrame({
                "mmsi": chunk["mmsi"],
                "base_date_time": chunk["base_date_time"],
                "latitude": chunk["lat"],
                "longitude": chunk["lon"],
                "sog": chunk["speed"],
                "cog": chunk["course"],
                "heading": chunk["heading"],
                "status": "",
                "vessel_type": chunk["vessel_type"],
                "length": "",
                "width": "",
                "draft": "",
                "imo": "",
            })
            # drop bad timestamps
            out = out[out["base_date_time"].notna() & (out["base_date_time"] != "NaT")]
            total_out += len(out)

            if args.split_per_day:
                for day, sub in out.groupby(out["base_date_time"].str[:10], dropna=True):
                    out_path = args.output / f"piraeus_{day}.csv.gz"
                    if day not in day_handles:
                        day_handles[day] = gzip.open(out_path, "wt", encoding="utf-8", newline="")
                        day_wrote_header[day] = False
                    sub.to_csv(day_handles[day], index=False, header=not day_wrote_header[day])
                    day_wrote_header[day] = True
            else:
                if not single_out.exists():
                    out.to_csv(single_out, index=False, compression="gzip")
                else:
                    with gzip.open(single_out, "at", encoding="utf-8", newline="") as fh:
                        out.to_csv(fh, index=False, header=False)
        for fh in day_handles.values():
            fh.close()
        print(f"[00] {src.name}: {total_in:,} in → {total_out:,} out "
              f"({len(day_handles) if args.split_per_day else 1} output file(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
