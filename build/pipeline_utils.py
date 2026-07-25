#!/usr/bin/env python3
"""Common utilities for the AIS benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

KNOTS_PER_MPS = 1.9438444924406
DMA_COLUMN_MAP = {
    "# Timestamp": "timestamp_utc",
    "Timestamp": "timestamp_utc",
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "Navigational status": "nav_status",
    "Ship type": "ship_type",
    "Length": "length",
    "Width": "width",
    "Draught": "draught",
    "IMO": "imo",
}
CANONICAL_COLUMNS = [
    "mmsi",
    "timestamp_utc",
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "nav_status",
    "ship_type",
    "length",
    "width",
    "draught",
    "imo",
    "source_file",
]
NUMERIC_NA_TOKENS = {"", "Unknown", "Unknown value", "Undefined", "None"}
PRIMARY_EXCLUDE_BASENAMES = {
    "manifest.json",
    "summary.json",
    "segment_summary.csv.gz",
    "segment_labels.csv.gz",
    "underway_only.csv.gz",
    "all_motion.csv.gz",
}
SHIP_CLASS_TO_ID = {
    "unknown": 0,
    "cargo": 1,
    "tanker": 2,
    "bulk": 3,
    "container": 4,
    "passenger": 5,
    "ferry": 6,
    "fishing": 7,
    "tug": 8,
    "service": 9,
}


def parse_stage_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.yaml"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def list_raw_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".zip", ".csv", ".gz"}
    ]
    return paths


def list_stage_csvs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.csv.gz") if p.is_file())


def list_primary_stage_csvs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths = []
    for path in sorted(root.rglob("*.csv.gz")):
        if path.name in PRIMARY_EXCLUDE_BASENAMES:
            continue
        if path.name.startswith("part-") or path.name == "messages.csv.gz" or path.parent == root:
            paths.append(path)
    return paths


def compression_for_path(path: Path) -> str | None:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or suffixes.endswith(".gz"):
        return "gzip"
    if suffixes.endswith(".zip"):
        return "zip"
    return None


def read_csv_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        chunksize=chunksize,
        compression=compression_for_path(path),
        dtype=str,
        keep_default_na=False,
    )


def read_stage_csv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    try:
        return pd.read_csv(
            path,
            compression=compression,
            parse_dates=["timestamp_utc"],
            low_memory=False,
        )
    except ValueError as exc:
        if "Missing column provided to 'parse_dates'" not in str(exc):
            raise
        try:
            return pd.read_csv(path, compression=compression, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_stage_chunks(path: Path, chunksize: int, parse_dates: list[str] | None = None) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        compression="gzip" if path.suffix == ".gz" else None,
        chunksize=chunksize,
        parse_dates=parse_dates or [],
    )


def write_stage_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression="gzip")


def append_stage_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
        compression="gzip",
    )


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(NUMERIC_NA_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text(series: pd.Series, unknown_tokens: set[str] | None = None) -> pd.Series:
    normalized = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    normalized = normalized.replace("", pd.NA)
    if unknown_tokens:
        lowered = normalized.str.lower()
        normalized = normalized.mask(lowered.isin({token.lower() for token in unknown_tokens}))
    return normalized


def standardize_dma_chunk(chunk: pd.DataFrame, source_name: str, config: dict) -> pd.DataFrame:
    renamed = chunk.rename(columns=DMA_COLUMN_MAP)
    standardized = pd.DataFrame()
    for column in CANONICAL_COLUMNS:
        if column == "source_file":
            standardized[column] = pd.Series(source_name, index=renamed.index, dtype="string")
        elif column in renamed.columns:
            standardized[column] = renamed[column]
        else:
            standardized[column] = pd.NA

    standardized["timestamp_utc"] = pd.to_datetime(
        standardized["timestamp_utc"], errors="coerce", utc=True, dayfirst=True
    )
    standardized["mmsi"] = coerce_numeric(standardized["mmsi"]).astype("Int64")
    for column in ["lat", "lon", "sog", "cog", "heading", "length", "width", "draught", "imo"]:
        standardized[column] = coerce_numeric(standardized[column]).astype("Float64")

    standardized["nav_status"] = clean_text(
        standardized["nav_status"],
        {"Unknown value", "Unknown", "Undefined", "Not defined"},
    ).str.lower()
    standardized["ship_type"] = clean_text(
        standardized["ship_type"],
        {"Unknown", "Unknown value", "Undefined"},
    ).str.lower()

    thresholds = config["thresholds"]
    standardized.loc[standardized["sog"] >= thresholds["sog_unavailable_knots"], "sog"] = pd.NA
    standardized.loc[standardized["cog"] >= thresholds["cog_unavailable_degrees"], "cog"] = pd.NA
    standardized.loc[
        standardized["heading"] == thresholds["heading_unavailable_value"], "heading"
    ] = pd.NA
    standardized.loc[standardized["lat"].abs() >= 91, "lat"] = pd.NA
    standardized.loc[standardized["lon"].abs() >= 181, "lon"] = pd.NA

    return standardized


def infer_ship_class(ship_type: object) -> str | None:
    if ship_type is None or pd.isna(ship_type):
        return None
    text = str(ship_type).lower()
    if any(token in text for token in ["cargo", "bulk", "container", "ro-ro cargo"]):
        if "container" in text:
            return "container"
        if "bulk" in text:
            return "bulk"
        return "cargo"
    if "tanker" in text:
        return "tanker"
    if any(token in text for token in ["passenger", "ferry", "cruise"]):
        if "ferry" in text:
            return "ferry"
        return "passenger"
    if "fishing" in text:
        return "fishing"
    if any(token in text for token in ["tug", "pilot", "service", "port tender", "tow"]):
        if "tug" in text or "tow" in text:
            return "tug"
        return "service"
    return None


def ship_type_id(ship_type: object) -> int:
    ship_class = infer_ship_class(ship_type) or "unknown"
    return SHIP_CLASS_TO_ID[ship_class]


def soft_speed_cap(ship_type: object, config: dict) -> float | None:
    ship_class = infer_ship_class(ship_type)
    if ship_class is None:
        return None
    return config.get("ship_type_caps", {}).get(ship_class)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> tuple[np.ndarray, np.ndarray]:
    r = 6_371_000.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * r
    y = (lat_rad - ref_lat_rad) * r
    return x, y


def local_xy_to_latlon(x: np.ndarray, y: np.ndarray, ref_lat: float, ref_lon: float) -> tuple[np.ndarray, np.ndarray]:
    r = 6_371_000.0
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    lat = np.degrees(y / r + ref_lat_rad)
    lon = np.degrees(x / (r * math.cos(ref_lat_rad)) + ref_lon_rad)
    return lat, lon


def partition_id_for_mmsi(mmsi: int, partition_count: int) -> int:
    return int(mmsi) % partition_count


def write_partitioned_chunk(
    chunk: pd.DataFrame,
    temp_dir: Path,
    partition_count: int,
    partition_col: str = "mmsi",
) -> None:
    partitions = chunk[partition_col].astype("Int64") % partition_count
    for part_id, part in chunk.groupby(partitions, dropna=True, sort=False):
        out_path = temp_dir / f"part-{int(part_id):03d}.csv"
        part.to_csv(out_path, mode="a" if out_path.exists() else "w", header=not out_path.exists(), index=False)


def finalize_partition_csvs(temp_dir: Path, output_dir: Path) -> list[Path]:
    ensure_dir(output_dir)
    outputs = []
    for csv_path in sorted(temp_dir.glob("part-*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"] if "timestamp_utc" in csv_path.read_text(encoding="utf-8", errors="ignore").split("\n", 1)[0] else None)
        out_path = output_dir / f"{csv_path.stem}.csv.gz"
        write_stage_csv(df, out_path)
        outputs.append(out_path)
    return outputs


def epoch_align_ceil(ts: pd.Timestamp, interval_seconds: int) -> pd.Timestamp:
    seconds = int(ts.timestamp())
    aligned = ((seconds + interval_seconds - 1) // interval_seconds) * interval_seconds
    return pd.Timestamp(aligned, unit="s", tz="UTC")


def epoch_align_floor(ts: pd.Timestamp, interval_seconds: int) -> pd.Timestamp:
    seconds = int(ts.timestamp())
    aligned = (seconds // interval_seconds) * interval_seconds
    return pd.Timestamp(aligned, unit="s", tz="UTC")


def circular_interp_deg(a0: float | None, a1: float | None, alpha: float) -> float | None:
    if a0 is None or a1 is None or pd.isna(a0) or pd.isna(a1):
        return None
    a0r = math.radians(float(a0))
    a1r = math.radians(float(a1))
    x = (1 - alpha) * math.cos(a0r) + alpha * math.cos(a1r)
    y = (1 - alpha) * math.sin(a0r) + alpha * math.sin(a1r)
    return math.degrees(math.atan2(y, x)) % 360.0


def consecutive_flag_duration_seconds(flags: np.ndarray, timestamps_ns: np.ndarray) -> float:
    longest = 0.0
    run_start = None
    run_end = None
    for idx, flag in enumerate(flags):
        if flag and run_start is None:
            run_start = timestamps_ns[idx]
            run_end = timestamps_ns[idx]
        elif flag:
            run_end = timestamps_ns[idx]
        elif run_start is not None:
            longest = max(longest, (run_end - run_start) / 1e9)
            run_start = None
            run_end = None
    if run_start is not None:
        longest = max(longest, (run_end - run_start) / 1e9)
    return float(longest)


def has_low_displacement_window(
    lats: np.ndarray, lons: np.ndarray, timestamps_ns: np.ndarray, window_seconds: int, min_displacement_m: float
) -> bool:
    j = 0
    n = len(timestamps_ns)
    for i in range(n):
        while j < n and (timestamps_ns[j] - timestamps_ns[i]) / 1e9 < window_seconds:
            j += 1
        if j >= n:
            break
        displacement = haversine_m(lats[i], lons[i], lats[j], lons[j])
        if displacement < min_displacement_m:
            return True
    return False


def stable_split_from_mmsi(mmsi: int) -> str:
    bucket = int(hashlib.md5(str(int(mmsi)).encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def json_dumps_compact(values: object) -> str:
    return json.dumps(values, separators=(",", ":"))


def json_loads(value: str) -> object:
    return json.loads(value)
