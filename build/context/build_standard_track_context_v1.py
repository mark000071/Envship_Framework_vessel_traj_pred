#!/usr/bin/env python3
"""Build context_v1 for standard_track_v1.

Produces a unified context package with:

  environment/
    anchors/          — per-split anchor CSVs with anchor lat/lon recovered from stage10
    features/         — numeric environment descriptors (scene type, navigability, etc.)
    rasters/          — 6-channel 128x128 binary masks + 2 signed-distance maps
    vectors/          — clipped OSM segment vectors in local meters
    osm_cache/        — tile-level OSM JSON cache (reuses existing cache if provided)
    feature_stats.json
    summary.json

  social/
    features/         — per-split social context tables (neighbor count, CPA, TCPA, etc.)
    _snapshot_buckets/ — compact per-timestamp AIS snapshots for neighbor lookup
    summary.json

  augmented/
    {train,val,test}/part-000.csv.gz  — original trajectory rows + env + social columns merged

Pipeline (single pass over stage10):
  1. Load standard_track sample metadata (segment_id, hist_end_ts, mmsi, split).
  2. Scan stage10 partitions:
       a. Build anchor_lookup: (segment_id, ts_epoch) → (lat, lon, sog, cog, heading)
       b. Build snapshot_buckets: for each row whose timestamp matches any target,
          write a compact snapshot (current-state only, no history buffer).
  3. Verify anchor coverage; write anchors CSVs.
  4. Fetch / reuse OSM tiles (0.25° tiles, Overpass API with disk cache).
  5. Build per-sample environment rasters, vectors, descriptors.
  6. Process snapshot buckets: match neighbors within radius, compute CPA/TCPA.
  7. Merge everything; export augmented CSVs.

Bug fixes vs existing builders:
  - hash(ts) replaced with MD5 for deterministic bucket assignment.
  - Timestamp normalisation is consistent between scan and lookup.
  - Memory: rasters allocated per split (not all splits at once).
  - Snapshot schema is compact (current state only) to avoid 20 GB+ storage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0
OVERPASS_URL   = "https://overpass-api.de/api/interpreter"

DEFAULT_TRACK_ROOT   = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets"
    "/multi_type_mini_bench_build/standard_track_v1"
)
DEFAULT_STAGE10_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets"
    "/data_interim/10_minlen_filtered/partitions"
)
DEFAULT_OSM_CACHE_DIR = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets"
    "/mini_benchmark/clean_ship_core_lite_v1/environment_v2/osm_cache/tiles"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track-root",    type=Path, default=DEFAULT_TRACK_ROOT)
    p.add_argument("--stage10-root",  type=Path, default=DEFAULT_STAGE10_ROOT)
    p.add_argument("--osm-cache-dir", type=Path, default=DEFAULT_OSM_CACHE_DIR,
                   help="Existing OSM tile cache dir to reuse (avoids Overpass downloads)")
    p.add_argument("--output-dir",    type=Path, default=None,
                   help="Defaults to track_root/context_v1")
    p.add_argument("--overpass-url",  type=str,  default=OVERPASS_URL)
    # Environment params
    p.add_argument("--patch-radius-m",   type=float, default=5000.0)
    p.add_argument("--grid-size",        type=int,   default=128)
    p.add_argument("--tile-deg",         type=float, default=0.25)
    p.add_argument("--line-sample-step-m", type=float, default=32.0)
    p.add_argument("--request-timeout-s",  type=float, default=90.0)
    p.add_argument("--connect-timeout-s",  type=float, default=20.0)
    p.add_argument("--sleep-between-tiles",type=float, default=0.3)
    p.add_argument("--max-fetch-attempts", type=int,   default=3)
    # Social params
    p.add_argument("--social-radius-m",  type=float, default=3000.0)
    p.add_argument("--max-neighbors",    type=int,   default=10)
    p.add_argument("--min-neighbors-flag", type=int, default=2)
    p.add_argument("--bucket-count",     type=int,   default=128)
    # Misc
    p.add_argument("--skip-stage10-scan", action="store_true",
                   help="Skip stage10 scan (reuse existing anchors + snapshot_buckets)")
    p.add_argument("--skip-environment",  action="store_true")
    p.add_argument("--skip-social",       action="store_true")
    p.add_argument("--log-every",         type=int, default=50000)
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Timestamp utilities (deterministic, cross-format)
# ---------------------------------------------------------------------------

def _parse_ts(ts: object) -> int:
    """Convert any AIS timestamp string to UTC epoch seconds (integer)."""
    import calendar, datetime
    s = str(ts).strip()
    # Try common formats: ISO with T or space separator, with or without timezone
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%d %H:%M:%S+00:00",
    ):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(calendar.timegm(dt.utctimetuple()))
        except ValueError:
            pass
    # Fallback: strip offset manually
    s_clean = s.replace("T", " ").split("+")[0].split(".")[0].strip()
    try:
        dt = datetime.datetime.strptime(s_clean, "%Y-%m-%d %H:%M:%S")
        return int(calendar.timegm(dt.utctimetuple()))
    except Exception:
        return 0


def normalize_ts_str(ts: object) -> str:
    epoch = _parse_ts(ts)
    import datetime
    dt = datetime.datetime.utcfromtimestamp(epoch)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def bucket_id_deterministic(ts_str: str, bucket_count: int) -> int:
    """MD5-based bucket assignment — deterministic across Python processes."""
    return int(hashlib.md5(ts_str.encode("utf-8")).hexdigest(), 16) % bucket_count


# ---------------------------------------------------------------------------
# Load standard_track target samples
# ---------------------------------------------------------------------------

def load_track_samples(track_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (all_samples, by_split)."""
    all_samples: list[dict] = []
    by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        path = track_root / split / "part-000.csv.gz"
        rows = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry = {
                    "sample_id":   row["sample_id"],
                    "split":       split,
                    "mmsi":        int(row["mmsi"]),
                    "segment_id":  row["segment_id"],
                    "ship_type":   row.get("ship_type", ""),
                    "hist_end_ts": normalize_ts_str(row["hist_end_ts"]),
                    "pred_end_ts": normalize_ts_str(row["pred_end_ts"]),
                    "ts_epoch":    _parse_ts(row["hist_end_ts"]),
                }
                rows.append(entry)
                all_samples.append(entry)
        by_split[split] = rows
        log(f"[load] {split}: {len(rows):,} samples")
    log(f"[load] total: {len(all_samples):,} samples")
    return all_samples, by_split


# ---------------------------------------------------------------------------
# Stage10 scan: anchor recovery + snapshot building
# ---------------------------------------------------------------------------

def scan_stage10(
    stage10_root:  Path,
    all_samples:   list[dict],
    snapshot_dir:  Path,
    bucket_count:  int,
    log_every:     int,
) -> dict[tuple[str, int], dict]:
    """Single pass over stage10.

    Returns anchor_lookup: {(segment_id, ts_epoch): {lat, lon, sog, cog, heading}}.
    Also writes compact snapshot bucket files under snapshot_dir.
    """
    # Build lookup sets for fast filtering
    target_seg_epochs: dict[str, set[int]] = defaultdict(set)
    target_epochs: set[int] = set()
    for s in all_samples:
        target_seg_epochs[s["segment_id"]].add(s["ts_epoch"])
        target_epochs.add(s["ts_epoch"])

    log(f"[stage10] target_segments={len(target_seg_epochs):,}  target_timestamps={len(target_epochs):,}")

    # Snapshot bucket writers (compact: only current state, no history buffer)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    SNAP_FIELDS = [
        "timestamp_utc", "mmsi", "segment_id", "ship_type",
        "lat", "lon", "sog", "cog", "heading",
    ]
    snap_handles: dict[int, gzip.GzipFile] = {}
    snap_writers: dict[int, csv.DictWriter] = {}

    def get_snap_writer(bucket: int) -> csv.DictWriter:
        if bucket not in snap_writers:
            path = snapshot_dir / f"snapshots-{bucket:03d}.csv.gz"
            exists = path.exists()
            h = gzip.open(path, "at", encoding="utf-8", newline="")
            w = csv.DictWriter(h, fieldnames=SNAP_FIELDS)
            if not exists:
                w.writeheader()
            snap_handles[bucket] = h
            snap_writers[bucket] = w
        return snap_writers[bucket]

    anchor_lookup: dict[tuple[str, int], dict] = {}
    rows_processed = 0
    t0 = time.time()
    parts = sorted(stage10_root.glob("part-*.csv.gz"))
    log(f"[stage10] scanning {len(parts)} part files")

    for part_idx, part_path in enumerate(parts, start=1):
        log(f"[stage10] part {part_idx}/{len(parts)} {part_path.name}", )
        with gzip.open(part_path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows_processed += 1
                ts_epoch = _parse_ts(row["timestamp_utc"])
                seg_id   = row["segment_id"]
                mmsi_val = int(row["mmsi"])

                # Anchor recovery: target vessel at target timestamp
                if seg_id in target_seg_epochs and ts_epoch in target_seg_epochs[seg_id]:
                    anchor_lookup[(seg_id, ts_epoch)] = {
                        "lat":      float(row["lat"]),
                        "lon":      float(row["lon"]),
                        "sog":      _safe_float(row.get("sog")),
                        "cog":      _safe_float(row.get("cog")),
                        "heading":  _safe_float(row.get("heading")),
                        "ship_type": row.get("ship_type", ""),
                    }

                # Snapshot: any vessel at any target timestamp (for social context)
                if ts_epoch in target_epochs:
                    ts_str = normalize_ts_str(row["timestamp_utc"])
                    bucket = bucket_id_deterministic(ts_str, bucket_count)
                    w = get_snap_writer(bucket)
                    w.writerow({
                        "timestamp_utc": ts_str,
                        "mmsi":      mmsi_val,
                        "segment_id": seg_id,
                        "ship_type": row.get("ship_type", ""),
                        "lat":    row["lat"],
                        "lon":    row["lon"],
                        "sog":    row.get("sog", ""),
                        "cog":    row.get("cog", ""),
                        "heading":row.get("heading", ""),
                    })

                if rows_processed % log_every == 0:
                    elapsed = time.time() - t0
                    log(f"[stage10] rows={rows_processed:,}  anchors_found={len(anchor_lookup):,}"
                        f"  elapsed={elapsed:.0f}s")

    for h in snap_handles.values():
        h.close()

    log(f"[stage10] DONE rows={rows_processed:,}  anchors={len(anchor_lookup):,}"
        f"  elapsed={time.time()-t0:.0f}s")
    return anchor_lookup


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Anchor CSVs
# ---------------------------------------------------------------------------

def tile_id_for_point(lat: float, lon: float, tile_deg: float) -> str:
    tile_lat = math.floor(lat / tile_deg) * tile_deg
    tile_lon = math.floor(lon / tile_deg) * tile_deg
    return f"{tile_lat:+08.3f}_{tile_lon:+09.3f}"


def write_anchor_csvs(
    by_split: dict[str, list[dict]],
    anchor_lookup: dict[tuple[str, int], dict],
    anchors_dir: Path,
    tile_deg: float,
) -> tuple[dict[str, list[dict]], int]:
    """Write per-split anchor CSVs; return enriched rows and missing-anchor count."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    FIELDS = ["sample_id", "split", "mmsi", "segment_id",
              "hist_end_ts", "pred_end_ts", "anchor_lat", "anchor_lon", "tile_id"]
    missing = 0
    all_anchor_rows: list[dict] = []
    enriched_by_split: dict[str, list[dict]] = {}

    for split, samples in by_split.items():
        rows = []
        for s in samples:
            key = (s["segment_id"], s["ts_epoch"])
            anchor = anchor_lookup.get(key)
            if anchor is None:
                missing += 1
                lat, lon = 0.0, 0.0          # placeholder; env will be skipped
            else:
                lat = float(anchor["lat"])
                lon = float(anchor["lon"])
            tile = tile_id_for_point(lat, lon, tile_deg)
            row = dict(s)
            row["anchor_lat"] = lat
            row["anchor_lon"] = lon
            row["tile_id"]    = tile
            row["anchor_ok"]  = anchor is not None
            rows.append(row)
            all_anchor_rows.append(row)

        csv_path = anchors_dir / f"{split}_anchors.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({f: row[f] for f in FIELDS})
        enriched_by_split[split] = rows
        log(f"[anchors] {split}: {len(rows):,}  missing={sum(1 for r in rows if not r['anchor_ok'])}")

    # All-anchors CSV
    with open(anchors_dir / "all_anchors.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in all_anchor_rows:
            writer.writerow({f: row[f] for f in FIELDS if f in row})

    if missing:
        log(f"[anchors] WARNING: {missing:,} samples missing anchor coordinates")
    return enriched_by_split, missing


# ---------------------------------------------------------------------------
# OSM tile fetching (reuses existing disk cache)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureWay:
    osm_id: int
    category: str    # "natural_boundary" | "manmade_boundary"
    subtype:  str
    lat: tuple[float, ...]
    lon: tuple[float, ...]


def _make_overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""[out:json][timeout:240];
(
  way["natural"="coastline"]({south},{west},{north},{east});
  way["waterway"~"riverbank|dock|canal"]({south},{west},{north},{east});
  way["man_made"~"pier|breakwater|groyne|quay"]({south},{west},{north},{east});
  way["landuse"="port"]({south},{west},{north},{east});
);
out geom;""".strip()


def _classify_way(tags: dict) -> tuple[str, str] | None:
    nat = tags.get("natural", "")
    ww  = tags.get("waterway", "")
    mm  = tags.get("man_made", "")
    lu  = tags.get("landuse", "")
    if nat == "coastline":                               return "natural_boundary", "coastline"
    if ww in {"riverbank", "dock", "canal"}:             return "natural_boundary", ww
    if mm in {"pier", "breakwater", "groyne", "quay"}:   return "manmade_boundary", mm
    if lu == "port":                                     return "manmade_boundary", "port_boundary"
    return None


def _parse_ways(payload: dict) -> list[FeatureWay]:
    ways = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        cls = _classify_way(el.get("tags", {}))
        if cls is None:
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        ways.append(FeatureWay(
            osm_id=int(el["id"]),
            category=cls[0], subtype=cls[1],
            lat=tuple(float(n["lat"]) for n in geom),
            lon=tuple(float(n["lon"]) for n in geom),
        ))
    return ways


def fetch_tile(
    tile_id: str,
    tile_points: list[dict],
    cache_dir: Path,
    alt_cache_dir: Path | None,
    args: argparse.Namespace,
) -> list[FeatureWay]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{tile_id}.json"

    # Try local cache first, then the reuse cache (existing env_v2 cache)
    for path in [local_path, (alt_cache_dir / f"{tile_id}.json") if alt_cache_dir else None]:
        if path and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            log(f"[osm] cache_hit {tile_id} (from {path.parent.name})")
            # Copy to local cache if from alt cache
            if path != local_path:
                local_path.write_bytes(path.read_bytes())
            return _parse_ways(payload)

    # Fetch from Overpass
    lats = [float(r["anchor_lat"]) for r in tile_points]
    lons = [float(r["anchor_lon"]) for r in tile_points]
    clat  = 0.5 * (min(lats) + max(lats))
    dlat  = math.degrees(args.patch_radius_m * 1.3 / EARTH_RADIUS_M)
    dlon  = math.degrees(args.patch_radius_m * 1.3 / (EARTH_RADIUS_M * max(math.cos(math.radians(clat)), 1e-6)))
    query = _make_overpass_query(min(lats)-dlat, min(lons)-dlon, max(lats)+dlat, max(lons)+dlon)

    last_exc = None
    for attempt in range(max(1, args.max_fetch_attempts)):
        try:
            log(f"[osm] fetch {tile_id} attempt {attempt+1}")
            resp = requests.post(
                args.overpass_url,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": "EnvShip-Bench/1.0 (https://github.com/mark000071/envship_v2_datasets; research)",
                    "Accept": "application/json, */*",
                },
                timeout=(args.connect_timeout_s, args.request_timeout_s),
            )
            resp.raise_for_status()
            payload = resp.json()
            local_path.write_text(json.dumps(payload), encoding="utf-8")
            log(f"[osm] fetch_ok {tile_id} ways={len(payload.get('elements',[]))}")
            if args.sleep_between_tiles > 0:
                time.sleep(args.sleep_between_tiles)
            return _parse_ways(payload)
        except Exception as exc:
            last_exc = exc
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"OSM fetch failed for {tile_id}: {last_exc}")


def build_tile_cache(
    all_rows: list[dict],
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[FeatureWay]], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        if row["anchor_ok"]:
            groups[row["tile_id"]].append(row)

    osm_cache    = output_root / "environment" / "osm_cache" / "tiles"
    alt_cache    = args.osm_cache_dir if args.osm_cache_dir and args.osm_cache_dir.exists() else None
    cache: dict[str, list[FeatureWay]] = {}
    failures: list[dict] = []
    log(f"[osm] unique_tiles={len(groups):,}  alt_cache_tiles={len(list(alt_cache.glob('*.json'))) if alt_cache else 0}")

    for idx, (tile_id, tile_rows) in enumerate(sorted(groups.items()), start=1):
        log(f"[osm] tile {idx}/{len(groups)} {tile_id} samples={len(tile_rows)}")
        try:
            cache[tile_id] = fetch_tile(tile_id, tile_rows, osm_cache, alt_cache, args)
        except Exception as exc:
            failures.append({"tile_id": tile_id, "error": str(exc)})
            cache[tile_id] = []
            log(f"[osm] tile_failed {tile_id}: {exc}")

    log(f"[osm] done  failed={len(failures)}")
    return cache, failures


# ---------------------------------------------------------------------------
# Environment raster building (adapted from env_v2)
# ---------------------------------------------------------------------------

def _latlon_to_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float):
    x = (np.radians(lon) - math.radians(ref_lon)) * math.cos(math.radians(ref_lat)) * EARTH_RADIUS_M
    y = (np.radians(lat) - math.radians(ref_lat)) * EARTH_RADIUS_M
    return x, y


def _outcode(x: float, y: float, r: float) -> int:
    code = 0
    if x < -r: code |= 1
    elif x > r: code |= 2
    if y < -r: code |= 4
    elif y > r: code |= 8
    return code


def _clip_seg(x0, y0, x1, y1, r):
    o0, o1 = _outcode(x0,y0,r), _outcode(x1,y1,r)
    while True:
        if not (o0|o1): return x0,y0,x1,y1
        if o0&o1: return None
        o = o0 or o1
        if o&8:   x = x0+(x1-x0)*(r-y0)/(y1-y0); y = r
        elif o&4: x = x0+(x1-x0)*(-r-y0)/(y1-y0); y = -r
        elif o&2: y = y0+(y1-y0)*(r-x0)/(x1-x0); x = r
        else:     y = y0+(y1-y0)*(-r-x0)/(x1-x0); x = -r
        if o==o0: x0,y0,o0 = x,y,_outcode(x,y,r)
        else:     x1,y1,o1 = x,y,_outcode(x,y,r)


def _rasterize(segs, r, g, step):
    mask = np.zeros((g,g), dtype=np.uint8)
    for s in segs:
        x0,y0 = map(float,s[0]); x1,y1 = map(float,s[1])
        n = max(2, int(math.ceil(math.hypot(x1-x0,y1-y0)/max(step,1)))+1)
        xs = np.linspace(x0,x1,n); ys = np.linspace(y0,y1,n)
        gx = np.clip(np.floor((xs+r)/(2*r)*g).astype(int), 0, g-1)
        gy = np.clip(np.floor((r-ys)/(2*r)*g).astype(int), 0, g-1)
        mask[gy,gx] = 1
    return mask


def _dilate(mask, px):
    if px <= 0: return mask.copy()
    out = mask.copy()
    for dy in range(-px, px+1):
        for dx in range(-px, px+1):
            if dx*dx+dy*dy > px*px: continue
            sy0=max(0,-dy); sy1=min(mask.shape[0],mask.shape[0]-dy)
            sx0=max(0,-dx); sx1=min(mask.shape[1],mask.shape[1]-dx)
            dy0=max(0,dy); dy1=min(mask.shape[0],mask.shape[0]+dy)
            dx0=max(0,dx); dx1=min(mask.shape[1],mask.shape[1]+dx)
            out[dy0:dy1,dx0:dx1] |= mask[sy0:sy1,sx0:sx1]
    return out


def _flood_fill(barrier):
    h,w = barrier.shape
    water = np.zeros_like(barrier, dtype=np.uint8)
    cy,cx = h//2, w//2
    if barrier[cy,cx]:
        for rad in range(1,5):
            found=False
            for dy in range(-rad,rad+1):
                for dx in range(-rad,rad+1):
                    yy,xx = cy+dy, cx+dx
                    if 0<=yy<h and 0<=xx<w and not barrier[yy,xx]:
                        cy,cx,found = yy,xx,True; break
                if found: break
            if found: break
    if barrier[cy,cx]: return water
    q=[( cy,cx)]; water[cy,cx]=1; head=0
    while head<len(q):
        y,x=q[head]; head+=1
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=ny<h and 0<=nx<w and not water[ny,nx] and not barrier[ny,nx]:
                water[ny,nx]=1; q.append((ny,nx))
    return water


def _udist(mask, r):
    h,w = mask.shape
    ys,xs = np.nonzero(mask)
    cell = (2*r)/max(h,1)
    if len(xs)==0: return np.full((h,w),r,dtype=np.float32)
    gy,gx = np.indices((h,w), dtype=np.float32)
    pts = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    flat_y,flat_x = gy.reshape(-1), gx.reshape(-1)
    out = np.full(flat_y.shape[0], np.inf, dtype=np.float32)
    for i in range(0, len(pts), 512):
        part = pts[i:i+512]
        dy = flat_y[:,None]-part[None,:,0]; dx = flat_x[:,None]-part[None,:,1]
        out = np.minimum(out, np.sqrt(dx*dx+dy*dy).min(axis=1))
    return (out.reshape(h,w)*cell).astype(np.float32)


def _signed_dist(ref_mask, positive_mask, r):
    d = _udist(ref_mask, r)
    s = np.where(positive_mask>0, 1.0, -1.0)
    return (d*s).astype(np.float32)


def _seg_stats(segs):
    total_len, min_dist = 0.0, float("inf")
    for s in segs:
        x0,y0=map(float,s[0]); x1,y1=map(float,s[1])
        total_len += math.hypot(x1-x0,y1-y0)
        for t in np.linspace(0,1,max(2,int(math.hypot(x1-x0,y1-y0)/50)+1)):
            px,py = x0+t*(x1-x0), y0+t*(y1-y0)
            min_dist = min(min_dist, math.hypot(px,py))
    return total_len, (min_dist if math.isfinite(min_dist) else 0.0)


def _scene_type(water_r, nav_r, nearest_shore, manmade_d):
    if manmade_d > 0.02 and nearest_shore < 500.0: return "harbor"
    if water_r < 0.45 or nav_r < 0.3:             return "constrained"
    if nearest_shore < 1200.0:                     return "nearshore"
    return "open_water"


def build_sample_env(row: dict, ways: list[FeatureWay], args: argparse.Namespace):
    """Return (descriptors_dict, masks_6xgxg uint8, signed_shore gxg f16, signed_nav gxg f16, vector_payload)."""
    r  = args.patch_radius_m
    g  = args.grid_size
    st = args.line_sample_step_m
    alat, alon = float(row["anchor_lat"]), float(row["anchor_lon"])

    nat_segs, mm_segs = [], []
    vec_natural, vec_manmade = [], []

    for way in ways:
        lat  = np.asarray(way.lat, dtype=np.float64)
        lon  = np.asarray(way.lon, dtype=np.float64)
        x, y = _latlon_to_xy(lat, lon, alat, alon)
        if x.max() < -r or x.min() > r or y.max() < -r or y.min() > r:
            continue
        clipped = []
        for i in range(1, len(x)):
            c = _clip_seg(float(x[i-1]),float(y[i-1]),float(x[i]),float(y[i]),r)
            if c:
                clipped.append(np.array([[c[0],c[1]],[c[2],c[3]]], dtype=np.float32))
        if not clipped:
            continue
        segs_ref = nat_segs if way.category == "natural_boundary" else mm_segs
        segs_ref.extend(clipped)
        vec_ref  = vec_natural if way.category == "natural_boundary" else vec_manmade
        for s in clipped:
            vec_ref.append({"osm_id": int(way.osm_id), "subtype": way.subtype,
                             "xy": [[round(float(s[0,0]),2),round(float(s[0,1]),2)],
                                    [round(float(s[1,0]),2),round(float(s[1,1]),2)]]})

    nat_mask = _rasterize(nat_segs, r, g, st)
    mm_mask  = _rasterize(mm_segs,  r, g, st)
    barrier  = np.maximum(nat_mask, mm_mask)
    water    = _flood_fill(barrier)
    land     = (1 - water).astype(np.uint8)
    non_nav  = _dilate(barrier, 1)
    geo_nav  = water.copy()
    geo_nav[non_nav > 0] = 0
    if geo_nav[g//2, g//2] == 0 and water[g//2, g//2] > 0:
        geo_nav[g//2, g//2] = 1

    s_shore = _signed_dist(barrier,   water,   r)
    s_nav   = _signed_dist(1-geo_nav, geo_nav, r)

    total = float(g*g)
    nat_len, nearest_nat = _seg_stats(nat_segs)
    mm_len,  nearest_mm  = _seg_stats(mm_segs)
    nearest_barrier = min(nearest_nat if nat_segs else r, nearest_mm if mm_segs else r)
    nearest_shore   = nearest_nat if nat_segs else r
    water_r  = float(water.sum()/total)
    nav_r    = float(geo_nav.sum()/total)
    nat_d    = float(nat_mask.sum()/total)
    mm_d     = float(mm_mask.sum()/total)
    barrier_d= float(barrier.sum()/total)
    scene    = _scene_type(water_r, nav_r, nearest_shore, mm_d)
    ain_w    = int(water[g//2,g//2]>0)
    ain_n    = int(geo_nav[g//2,g//2]>0)
    aon_b    = int(barrier[g//2,g//2]>0)
    qual = min(1.0, max(0.0, 0.25*ain_w + 0.35*ain_n + 0.2*water_r + 0.2*nav_r))

    center_shore = float(min(r, abs(float(s_shore[g//2,g//2]))))

    desc = {
        "sample_id":   row["sample_id"],
        "split":       row["split"],
        "segment_id":  row["segment_id"],
        "hist_end_ts": row["hist_end_ts"],
        "anchor_lat":  round(float(row["anchor_lat"]),6),
        "anchor_lon":  round(float(row["anchor_lon"]),6),
        "tile_id":     row["tile_id"],
        "patch_radius_m": r,
        "grid_size":   g,
        "nearest_shore_dist_m":   round(nearest_shore, 2),
        "nearest_manmade_dist_m": round(nearest_mm if mm_segs else r, 2),
        "nearest_barrier_dist_m": round(nearest_barrier, 2),
        "center_signed_shore_m":  round(center_shore, 2),
        "water_ratio":            round(water_r,   6),
        "navigable_ratio":        round(nav_r,     6),
        "barrier_density":        round(barrier_d, 6),
        "natural_boundary_density": round(nat_d,  6),
        "manmade_boundary_density": round(mm_d,   6),
        "natural_boundary_length_m": round(nat_len,2),
        "manmade_boundary_length_m": round(mm_len, 2),
        "has_natural_boundary":   int(bool(nat_segs)),
        "has_manmade_boundary":   int(bool(mm_segs)),
        "anchor_in_water":        ain_w,
        "anchor_in_navigable":    ain_n,
        "anchor_on_barrier":      aon_b,
        "scene_type":             scene,
        "env_quality_score":      round(qual, 6),
        "scene_open_water":       int(scene=="open_water"),
        "scene_nearshore":        int(scene=="nearshore"),
        "scene_harbor":           int(scene=="harbor"),
        "scene_constrained":      int(scene=="constrained"),
    }

    masks6 = np.stack([land, water, geo_nav, nat_mask, mm_mask, barrier], axis=0).astype(np.uint8)

    vector = {
        "sample_id": row["sample_id"], "split": row["split"],
        "anchor_lat": float(row["anchor_lat"]), "anchor_lon": float(row["anchor_lon"]),
        "tile_id": row["tile_id"],
        "transform": {"patch_radius_m": r, "grid_size": g, "meters_per_pixel": 2*r/g},
        "natural_boundary": vec_natural,
        "manmade_boundary": vec_manmade,
        "barrier": vec_natural + vec_manmade,
    }
    return desc, masks6, s_shore.astype(np.float16), s_nav.astype(np.float16), vector


def build_environment(
    enriched: dict[str, list[dict]],
    ways_cache: dict[str, list[FeatureWay]],
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    env_root = output_root / "environment"
    all_desc: list[dict] = []
    split_summaries = {}

    for split, rows in enriched.items():
        valid_rows = [r for r in rows if r["anchor_ok"]]
        n = len(valid_rows)
        log(f"[env] {split}: building {n:,} samples")

        feat_dir   = env_root / "features" / split
        rast_dir   = env_root / "rasters"  / split
        vec_dir    = env_root / "vectors"  / split
        for d in (feat_dir, rast_dir, vec_dir):
            d.mkdir(parents=True, exist_ok=True)

        masks6  = np.zeros((n, 6, args.grid_size, args.grid_size), dtype=np.uint8)
        s_shore = np.zeros((n, args.grid_size, args.grid_size), dtype=np.float16)
        s_nav   = np.zeros((n, args.grid_size, args.grid_size), dtype=np.float16)
        sample_ids = []
        desc_rows  = []

        with gzip.open(vec_dir / "vectors.jsonl.gz", "wt", encoding="utf-8") as vec_fh:
            for idx, row in enumerate(valid_rows):
                desc, m6, ss, sn, vec = build_sample_env(
                    row, ways_cache.get(row["tile_id"], []), args
                )
                masks6[idx]  = m6
                s_shore[idx] = ss
                s_nav[idx]   = sn
                sample_ids.append(row["sample_id"])
                desc_rows.append(desc)
                all_desc.append(desc)
                vec_fh.write(json.dumps(vec, ensure_ascii=False, separators=(",",":")) + "\n")
                if (idx+1) % args.log_every == 0:
                    log(f"[env] {split}: {idx+1:,}/{n:,}")

        np.save(rast_dir / "sample_ids.npy",        np.asarray(sample_ids, dtype=object), allow_pickle=True)
        np.save(rast_dir / "masks.npy",             masks6,  allow_pickle=False)
        np.save(rast_dir / "signed_dist_shore.npy", s_shore, allow_pickle=False)
        np.save(rast_dir / "signed_dist_nav.npy",   s_nav,   allow_pickle=False)

        with open(feat_dir / "environment_descriptors.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(desc_rows[0].keys()))
            w.writeheader()
            w.writerows(desc_rows)

        scene_counts = Counter(d["scene_type"] for d in desc_rows)
        split_summaries[split] = {
            "count": n,
            "anchor_in_water":     int(sum(d["anchor_in_water"]    for d in desc_rows)),
            "anchor_in_navigable": int(sum(d["anchor_in_navigable"] for d in desc_rows)),
            "scene_counts":        dict(scene_counts),
            "mean_env_quality":    round(float(np.mean([d["env_quality_score"] for d in desc_rows])),4),
        }
        log(f"[env] {split} done: {scene_counts}")

    # Global descriptor CSV + feature stats
    with open(env_root / "all_environment_descriptors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_desc[0].keys()))
        w.writeheader()
        w.writerows(all_desc)

    log1p_max = {}
    train_desc = [d for d in all_desc if d["split"] == "train"]
    for col in ("barrier_density","natural_boundary_density","manmade_boundary_density"):
        vals = [float(d[col]) for d in train_desc]
        log1p_max[col] = float(np.log1p(max(vals))) if vals else 1.0
    stats = {
        "descriptor_columns": [k for k in all_desc[0] if k not in ("sample_id","split","segment_id","hist_end_ts","anchor_lat","anchor_lon","tile_id","scene_type")],
        "log1p_max": log1p_max,
    }
    with open(env_root / "feature_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)

    summary = {
        "config": {"patch_radius_m": args.patch_radius_m, "grid_size": args.grid_size, "tile_deg": args.tile_deg},
        **split_summaries,
    }
    with open(env_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    log(f"[env] ALL DONE  total={len(all_desc):,}")
    return summary


# ---------------------------------------------------------------------------
# Social context
# ---------------------------------------------------------------------------

def _meters_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def _meters_lat() -> float:
    return 110540.0


def _sog_cog_to_vxvy(sog, cog) -> tuple[float, float]:
    try:
        if sog in (None, "", "nan", "None") or cog in (None, "", "nan", "None"):
            return 0.0, 0.0
        s = float(sog) * 0.514444
        r = math.radians(float(cog))
        return s * math.sin(r), s * math.cos(r)
    except (ValueError, TypeError):
        return 0.0, 0.0


def _cpa_tcpa(rx, ry, rvx, rvy):
    v2 = rvx*rvx + rvy*rvy
    if v2 <= 1e-8: return None, None
    tcpa = -((rx*rvx)+(ry*rvy)) / v2
    cx = rx + tcpa*rvx; cy = ry + tcpa*rvy
    return math.sqrt(cx*cx+cy*cy), tcpa


def _density_bin(n: int) -> str:
    if n >= 7:  return "dense"
    if n >= 4:  return "medium"
    if n >= 1:  return "sparse"
    return "isolated"


def process_social_buckets(
    by_split:      dict[str, list[dict]],
    anchor_lookup: dict[tuple[str,int], dict],
    snapshot_dir:  Path,
    bucket_count:  int,
    radius_m:      float,
    max_neighbors: int,
    min_flag:      int,
    log_every_buckets: int = 16,
) -> tuple[dict[str, list[dict]], dict]:
    # Build: ts_str → list of target samples
    ts_to_targets: dict[str, list[dict]] = defaultdict(list)
    for samples in by_split.values():
        for s in samples:
            ts_to_targets[s["hist_end_ts"]].append(s)

    # Bucket-aligned target index
    bucket_to_targets: dict[int, list[dict]] = defaultdict(list)
    for ts_str, targets in ts_to_targets.items():
        b = bucket_id_deterministic(ts_str, bucket_count)
        bucket_to_targets[b].extend(targets)

    rows_by_split: dict[str, list[dict]] = {"train":[], "val":[], "test":[]}
    social_stats: dict[str, dict] = {s: {"n_counts":[], "density": Counter(), "interaction":0, "avail":0}
                                      for s in ("train","val","test")}

    for bucket in range(bucket_count):
        if bucket and bucket % log_every_buckets == 0:
            log(f"[social] bucket {bucket}/{bucket_count}")
        targets = bucket_to_targets.get(bucket, [])
        if not targets:
            continue

        snap_path = snapshot_dir / f"snapshots-{bucket:03d}.csv.gz"
        # Load snapshot for this bucket (compact format: no history)
        snaps_by_ts: dict[str, list[dict]] = defaultdict(list)
        if snap_path.exists():
            with gzip.open(snap_path, "rt", encoding="utf-8", newline="") as fh:
                for snap in csv.DictReader(fh):
                    snaps_by_ts[snap["timestamp_utc"]].append(snap)

        for target in targets:
            ts = target["hist_end_ts"]
            key = (target["segment_id"], target["ts_epoch"])
            anchor = anchor_lookup.get(key)
            split  = target["split"]

            if anchor is None:
                # No anchor → social context unavailable
                row = _empty_social(target)
                rows_by_split[split].append(row)
                social_stats[split]["n_counts"].append(0)
                social_stats[split]["density"]["isolated"] += 1
                continue

            lat0 = float(anchor["lat"])
            lon0 = float(anchor["lon"])
            tvx, tvy = _sog_cog_to_vxvy(anchor.get("sog"), anchor.get("cog"))

            # Find nearby vessels
            snaps = snaps_by_ts.get(ts, [])
            dists = []
            for snap in snaps:
                m = int(snap["mmsi"])
                if m == target["mmsi"]: continue
                slat, slon = float(snap["lat"]), float(snap["lon"])
                dx = (slon - lon0) * _meters_lon(lat0)
                dy = (slat - lat0) * _meters_lat()
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= radius_m:
                    dists.append((dist, dx, dy, snap))
            dists.sort(key=lambda x: x[0])
            chosen = dists[:max_neighbors]

            n_total = len(dists)
            n_used  = len(chosen)
            row = _empty_social(target)
            row["target_global_lat"]      = lat0
            row["target_global_lon"]      = lon0
            row["target_sog_at_hist_end"] = anchor.get("sog")
            row["target_cog_at_hist_end"] = anchor.get("cog")
            row["neighbor_count_total"]   = n_total
            row["neighbor_count_used"]    = n_used
            row["neighbor_density_bin"]   = _density_bin(n_total)
            row["social_context_available"] = 1
            row["interaction_candidate"]  = int(n_total >= min_flag)

            mmsi_list, seg_list, type_list = [], [], []
            dist_list, vx_list, vy_list   = [], [], []
            cpa_list, tcpa_list            = [], []
            # Relative positions (current state at hist_end_ts)
            rel_x_list, rel_y_list = [], []

            for dist, dx, dy, snap in chosen:
                nvx, nvy = _sog_cog_to_vxvy(snap.get("sog"), snap.get("cog"))
                rvx, rvy = nvx - tvx, nvy - tvy
                cpa, tcpa = _cpa_tcpa(dx, dy, rvx, rvy)
                mmsi_list.append(int(snap["mmsi"]))
                seg_list.append(snap["segment_id"])
                type_list.append(snap["ship_type"])
                dist_list.append(round(dist, 2))
                vx_list.append(round(rvx, 4))
                vy_list.append(round(rvy, 4))
                cpa_list.append(round(cpa, 2) if cpa is not None else None)
                tcpa_list.append(round(tcpa, 2) if tcpa is not None else None)
                rel_x_list.append(round(dx, 2))
                rel_y_list.append(round(dy, 2))

            row["neighbor_mmsi_json"]          = json.dumps(mmsi_list)
            row["neighbor_segment_id_json"]    = json.dumps(seg_list)
            row["neighbor_ship_type_json"]     = json.dumps(type_list)
            row["neighbor_distance_m_json"]    = json.dumps(dist_list)
            row["neighbor_rel_x_m_json"]       = json.dumps(rel_x_list)
            row["neighbor_rel_y_m_json"]       = json.dumps(rel_y_list)
            row["neighbor_rel_vx_mps_json"]    = json.dumps(vx_list)
            row["neighbor_rel_vy_mps_json"]    = json.dumps(vy_list)
            row["neighbor_cpa_m_json"]         = json.dumps(cpa_list)
            row["neighbor_tcpa_s_json"]        = json.dumps(tcpa_list)
            row["min_neighbor_distance_m"]     = min(dist_list) if dist_list else None
            row["mean_neighbor_distance_m"]    = round(sum(dist_list)/len(dist_list), 2) if dist_list else None
            valid_cpa  = [c for c in cpa_list  if c is not None]
            valid_tcpa = [abs(t) for t in tcpa_list if t is not None]
            row["min_cpa_m"]      = min(valid_cpa)  if valid_cpa  else None
            row["min_abs_tcpa_s"] = min(valid_tcpa) if valid_tcpa else None

            rows_by_split[split].append(row)
            st = social_stats[split]
            st["n_counts"].append(n_total)
            st["density"][_density_bin(n_total)] += 1
            st["interaction"] += row["interaction_candidate"]
            st["avail"]       += 1

    summaries = {}
    for split, st in social_stats.items():
        nc = st["n_counts"]
        summaries[split] = {
            "num_samples":              len(nc),
            "social_context_available": st["avail"],
            "interaction_candidate_count": st["interaction"],
            "density_bin_counts":       dict(st["density"]),
            "mean_neighbor_count":      round(float(np.mean(nc)), 4) if nc else 0.0,
            "samples_with_neighbors":   int(sum(c>0 for c in nc)),
        }
        log(f"[social] {split}: {summaries[split]}")
    return rows_by_split, summaries


def _empty_social(target: dict) -> dict:
    return {
        **target,
        "target_global_lat": None, "target_global_lon": None,
        "target_sog_at_hist_end": None, "target_cog_at_hist_end": None,
        "neighbor_count_total": 0, "neighbor_count_used": 0,
        "neighbor_density_bin": "isolated",
        "neighbor_mmsi_json": "[]", "neighbor_segment_id_json": "[]",
        "neighbor_ship_type_json": "[]", "neighbor_distance_m_json": "[]",
        "neighbor_rel_x_m_json": "[]", "neighbor_rel_y_m_json": "[]",
        "neighbor_rel_vx_mps_json": "[]", "neighbor_rel_vy_mps_json": "[]",
        "neighbor_cpa_m_json": "[]", "neighbor_tcpa_s_json": "[]",
        "min_neighbor_distance_m": None, "mean_neighbor_distance_m": None,
        "min_cpa_m": None, "min_abs_tcpa_s": None,
        "social_context_available": 0, "interaction_candidate": 0,
    }


def export_social(
    rows_by_split: dict[str, list[dict]],
    social_root: Path,
    summaries: dict,
    args: argparse.Namespace,
) -> None:
    social_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for split, rows in rows_by_split.items():
        feat_dir = social_root / "features" / split
        feat_dir.mkdir(parents=True, exist_ok=True)
        if not rows:
            continue
        fields = list(rows[0].keys())
        with open(feat_dir / "social_features.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        all_rows.extend(rows)
        log(f"[social] {split}: {len(rows):,} rows written")
    # Summarize
    summary = {
        "config": {
            "radius_m": args.social_radius_m, "max_neighbors": args.max_neighbors,
            "min_neighbors_flag": args.min_neighbors_flag,
            "bucket_count": args.bucket_count,
        },
        **summaries,
    }
    with open(social_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


# ---------------------------------------------------------------------------
# Augmented CSV export (trajectory + env + social merged)
# ---------------------------------------------------------------------------

def export_augmented(
    track_root:    Path,
    env_root:      Path,
    social_root:   Path,
    augmented_root: Path,
) -> None:
    augmented_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        traj_path   = track_root   / split / "part-000.csv.gz"
        env_desc_path = env_root / "features" / split / "environment_descriptors.csv"
        soc_feat_path = social_root / "features" / split / "social_features.csv"
        out_path    = augmented_root / split / "part-000.csv.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build lookup tables
        env_lookup: dict[str, dict] = {}
        if env_desc_path.exists():
            with open(env_desc_path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    env_lookup[row["sample_id"]] = row

        soc_lookup: dict[str, dict] = {}
        if soc_feat_path.exists():
            with open(soc_feat_path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    soc_lookup[row["sample_id"]] = row

        env_extra_cols  = [c for c in (list(next(iter(env_lookup.values())).keys()) if env_lookup else [])
                           if c not in {"sample_id","split","segment_id","hist_end_ts"}]
        soc_extra_cols  = [c for c in (list(next(iter(soc_lookup.values())).keys()) if soc_lookup else [])
                           if c not in {"sample_id","split","mmsi","segment_id","ship_type",
                                        "hist_end_ts","pred_end_ts","ts_epoch"}]

        with gzip.open(traj_path, "rt", encoding="utf-8", newline="") as src, \
             gzip.open(out_path,   "wt", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            traj_fields = reader.fieldnames or []
            all_fields  = traj_fields + env_extra_cols + soc_extra_cols
            writer = csv.DictWriter(dst, fieldnames=all_fields)
            writer.writeheader()
            for row in reader:
                sid = row["sample_id"]
                if sid in env_lookup:
                    for c in env_extra_cols:
                        row[c] = env_lookup[sid].get(c, "")
                if sid in soc_lookup:
                    for c in soc_extra_cols:
                        row[c] = soc_lookup[sid].get(c, "")
                writer.writerow(row)

        log(f"[augmented] {split}: written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_root = args.output_dir or (args.track_root / "context_v1")
    output_root.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log(f"[main] track_root  = {args.track_root}")
    log(f"[main] output_root = {output_root}")
    log(f"[main] stage10     = {args.stage10_root}")

    # ── Step 1: Load target samples ──────────────────────────────────────────
    all_samples, by_split = load_track_samples(args.track_root)

    # ── Step 2: Stage10 scan ─────────────────────────────────────────────────
    snapshot_dir = output_root / "social" / "_snapshot_buckets"
    anchor_cache_path = output_root / "environment" / "anchor_lookup.json"

    if args.skip_stage10_scan and anchor_cache_path.exists():
        log("[main] loading cached anchor_lookup")
        with open(anchor_cache_path) as fh:
            raw = json.load(fh)
        anchor_lookup = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}
    else:
        anchor_lookup = scan_stage10(
            args.stage10_root, all_samples, snapshot_dir,
            args.bucket_count, args.log_every,
        )
        anchor_cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {f"{k[0]}|{k[1]}": v for k, v in anchor_lookup.items()}
        with open(anchor_cache_path, "w") as fh:
            json.dump(serialisable, fh)
        log(f"[main] anchor_lookup cached to {anchor_cache_path}")

    # ── Step 3: Write anchor CSVs ────────────────────────────────────────────
    anchors_dir = output_root / "environment" / "anchors"
    enriched, missing = write_anchor_csvs(by_split, anchor_lookup, anchors_dir, args.tile_deg)
    flat_anchors = [r for rows in enriched.values() for r in rows]

    # ── Step 4: OSM tiles ────────────────────────────────────────────────────
    if not args.skip_environment:
        ways_cache, failures = build_tile_cache(flat_anchors, output_root, args)
        with open(output_root / "environment" / "failed_tiles.json", "w") as fh:
            json.dump(failures, fh, indent=2)

        # ── Step 5: Environment rasters / vectors / descriptors ─────────────
        env_summary = build_environment(enriched, ways_cache, output_root, args)
    else:
        log("[main] skip_environment=True")
        env_summary = {}

    # ── Step 6: Social context ───────────────────────────────────────────────
    if not args.skip_social:
        social_rows, social_summaries = process_social_buckets(
            by_split, anchor_lookup, snapshot_dir,
            args.bucket_count, args.social_radius_m,
            args.max_neighbors, args.min_neighbors_flag,
        )
        export_social(social_rows, output_root / "social", social_summaries, args)
    else:
        log("[main] skip_social=True")
        social_summaries = {}

    # ── Step 7: Augmented CSV export ─────────────────────────────────────────
    export_augmented(
        args.track_root,
        output_root / "environment",
        output_root / "social",
        output_root / "augmented",
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "config": {
            "track_root":   str(args.track_root),
            "output_root":  str(output_root),
            "stage10_root": str(args.stage10_root),
            "patch_radius_m": args.patch_radius_m,
            "grid_size":    args.grid_size,
            "tile_deg":     args.tile_deg,
            "social_radius_m":   args.social_radius_m,
            "max_neighbors":     args.max_neighbors,
            "min_neighbors_flag":args.min_neighbors_flag,
            "bucket_count":      args.bucket_count,
        },
        "anchor_missing": missing,
        "environment":   env_summary,
        "social":        social_summaries,
        "total_elapsed_s": round(time.time() - t_total, 1),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[main] ALL DONE  elapsed={time.time()-t_total:.0f}s  output={output_root}")


if __name__ == "__main__":
    main()
