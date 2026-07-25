#!/usr/bin/env python3
"""Stage 10b — Track B pre-filter for the long-horizon (30/60) benchmark.

Reads Track-A stage-10 partitions and:
  (1) re-filters segments by Track B's min_segment_duration_seconds (=7200 s = 2 h)
  (2) optionally patches ship_class / ship_type from an enrichment LUT (Norway)
Writes the filtered partitions to a Track-B-specific output dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_enrichment(path: Path) -> dict[int, str]:
    """JSONL with one VesselFinder record per line; build MMSI -> ship_type_text."""
    if not path or not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if "_error" in rec:
                continue
            t = rec.get("type")
            m = rec.get("mmsi")
            if t and m is not None:
                out[int(m)] = str(t)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",    type=Path, required=True, help="stage-10 partitions dir")
    p.add_argument("--output",   type=Path, required=True, help="track_b stage-10b output dir")
    p.add_argument("--config",   type=Path, required=True)
    p.add_argument("--enrichment", type=Path, default=None,
                   help="JSONL of {mmsi,type,...} to patch ship_type when null")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    min_dur = float(cfg["thresholds"]["min_segment_duration_seconds"])
    enrich = load_enrichment(args.enrichment) if args.enrichment else {}
    out_dir = args.output / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage10b] min_segment_duration_seconds={min_dur}", flush=True)
    print(f"[stage10b] enrichment: {len(enrich)} MMSIs", flush=True)

    parts_in = sorted(args.input.glob("part-*.csv.gz"))
    if not parts_in:
        raise SystemExit(f"no partitions found in {args.input}")

    rows_in_total = rows_out_total = segs_in_total = segs_out_total = patched = 0
    for pf in parts_in:
        df = pd.read_csv(pf, compression="gzip", parse_dates=["timestamp_utc"], low_memory=False)
        rows_in_total += len(df)
        if df.empty:
            continue
        # Patch ship_type before filter so the carried-through column is enriched.
        # Use full-length aligned series to avoid shape-mismatch when masking.
        if enrich:
            mask_null = df["ship_type"].isna()
            if mask_null.any():
                lookup_full = df["mmsi"].where(mask_null).map(enrich)
                got_full = lookup_full.notna()
                df.loc[got_full, "ship_type"] = lookup_full[got_full]
                patched += int(got_full.sum())
        # Refilter by segment duration
        seg_durs = df.groupby("segment_id")["timestamp_utc"].agg(
            lambda s: (s.max() - s.min()).total_seconds())
        keep_segs = seg_durs[seg_durs >= min_dur].index
        segs_in_total += int(df["segment_id"].nunique())
        segs_out_total += len(keep_segs)
        out = df[df["segment_id"].isin(keep_segs)].copy()
        rows_out_total += len(out)
        if out.empty:
            continue
        out.to_csv(out_dir / pf.name, index=False, compression="gzip")

    summary = {
        "stage": "10b_track_b_filter",
        "min_segment_duration_seconds": min_dur,
        "rows_in": rows_in_total,
        "rows_out": rows_out_total,
        "segments_in": segs_in_total,
        "segments_out": segs_out_total,
        "ship_type_rows_patched_from_enrichment": patched,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[stage10b] DONE  rows {rows_in_total:,} -> {rows_out_total:,}  "
          f"segments {segs_in_total:,} -> {segs_out_total:,}  patched={patched:,}", flush=True)


if __name__ == "__main__":
    main()
