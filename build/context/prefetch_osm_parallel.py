"""Parallel OSM tile pre-fetcher for the NOAA context build.

Downloads any not-yet-cached tiles into the existing cache dir, using a
small thread pool that rotates across multiple Overpass mirrors. The
build_standard_track_context_v1.py script will then see them as cache hits
on its next pass and skip all networking.
"""
from __future__ import annotations
import argparse, json, math, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

HEADERS = {
    "Content-Type": "text/plain; charset=utf-8",
    "User-Agent":   "EnvShip-Bench/1.0 (https://github.com/mark000071/envship_v2_datasets; research)",
    "Accept":       "application/json, */*",
}
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]


def make_query(south, west, north, east, timeout=240):
    return f"""[out:json][timeout:{timeout}];
(
  way["natural"="coastline"]({south},{west},{north},{east});
  way["waterway"~"riverbank|dock|canal"]({south},{west},{north},{east});
  way["man_made"~"pier|breakwater|groyne|quay"]({south},{west},{north},{east});
  way["landuse"="port"]({south},{west},{north},{east});
);
out geom;""".strip()


def tile_id_to_bbox(tile_id: str, tile_deg: float, patch_radius_m: float):
    # tile_id format: "+033.250_-0078.250"  (tile lower-left lat/lon)
    lat = float(tile_id.split("_")[0])
    lon = float(tile_id.split("_")[1])
    # Add margin in degrees for the patch radius
    EARTH_R = 6371000.0
    dlat = math.degrees(patch_radius_m * 1.3 / EARTH_R)
    dlon = math.degrees(patch_radius_m * 1.3 / (EARTH_R * max(math.cos(math.radians(lat)), 1e-6)))
    s = lat - dlat
    n = lat + tile_deg + dlat
    w = lon - dlon
    e = lon + tile_deg + dlon
    return s, w, n, e


def fetch_one(tile_id: str, cache_dir: Path, tile_deg: float, patch_r: float,
              mirror_idx: int, timeout: int = 60, max_attempts: int = 3):
    out = cache_dir / f"{tile_id}.json"
    if out.exists() and out.stat().st_size > 0:
        return tile_id, "cached", 0
    s, w, n, e = tile_id_to_bbox(tile_id, tile_deg, patch_r)
    q = make_query(s, w, n, e)
    for attempt in range(max_attempts):
        url = MIRRORS[(mirror_idx + attempt) % len(MIRRORS)]
        try:
            r = requests.post(url, data=q.encode("utf-8"), headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                # Validate it's JSON
                json.loads(r.content)
                out.write_bytes(r.content)
                return tile_id, "ok", attempt + 1
            elif r.status_code in (429, 503, 504):
                time.sleep(1.5 + attempt)
            else:
                # 4xx — try next mirror
                pass
        except Exception:
            time.sleep(1 + attempt)
    return tile_id, "failed", max_attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Where the build expects cached tiles (.../osm_cache/tiles)")
    ap.add_argument("--anchors-dir", type=Path, required=True,
                    help=".../context_v1/environment/anchors/ — read tile_id list")
    ap.add_argument("--tile-deg",   type=float, default=0.25)
    ap.add_argument("--patch-radius-m", type=float, default=5000.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique tile_ids from anchors CSVs
    import csv
    tile_ids = set()
    for f in args.anchors_dir.glob("*_anchors.csv"):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("tile_id"):
                    tile_ids.add(row["tile_id"])
    print(f"Found {len(tile_ids):,} unique tile_ids")

    # Filter: only those not already cached
    todo = [t for t in tile_ids if not (args.cache_dir / f"{t}.json").exists()]
    print(f"  already cached: {len(tile_ids) - len(todo):,}")
    print(f"  to fetch:       {len(todo):,}")
    if not todo:
        print("Nothing to do."); return

    t0 = time.time()
    ok = 0; failed = 0; cached_already = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, t, args.cache_dir, args.tile_deg,
                              args.patch_radius_m, i % len(MIRRORS))
                   for i, t in enumerate(todo)]
        for i, f in enumerate(as_completed(futures), start=1):
            tid, status, attempts = f.result()
            if status == "ok":     ok += 1
            elif status == "cached": cached_already += 1
            else:                    failed += 1
            if i % 25 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i:5d}/{len(todo)}]  ok={ok}  failed={failed}  "
                      f"rate={rate:.1f}/s  ETA={eta/60:.0f}m")

    print(f"\nDone in {(time.time() - t0)/60:.1f} min.  ok={ok}  failed={failed}")
    print(f"Tile cache now: {len(list(args.cache_dir.glob('*.json'))):,} files")


if __name__ == "__main__":
    main()
