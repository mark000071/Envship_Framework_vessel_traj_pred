"""Smart OSM cache filler.

For each unfetched tile in the anchors list:
  - Classify with global-land-mask: is there ANY land within 1° of the tile?
  - If no   → write empty {"elements": []} JSON instantly (no API call)
  - If yes  → fetch from Overpass (rotating across mirrors with UA)

The classifier was validated on already-fetched tiles: 0 false negatives
(no land tile was wrongly classified as ocean). 35% of fetches avoided.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests
from global_land_mask import globe

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


def tile_has_land_nearby(lat: float, lon: float, tile_deg: float = 0.25,
                         padding_deg: float = 1.0, grid_n: int = 9) -> bool:
    lats = np.linspace(lat - padding_deg, lat + tile_deg + padding_deg, grid_n)
    lons = np.linspace(lon - padding_deg, lon + tile_deg + padding_deg, grid_n)
    LA, LO = np.meshgrid(lats, lons)
    return bool(globe.is_land(LA.ravel(), LO.ravel()).any())


def make_query(south, west, north, east, timeout=240):
    return f"""[out:json][timeout:{timeout}];
(
  way["natural"="coastline"]({south},{west},{north},{east});
  way["waterway"~"riverbank|dock|canal"]({south},{west},{north},{east});
  way["man_made"~"pier|breakwater|groyne|quay"]({south},{west},{north},{east});
  way["landuse"="port"]({south},{west},{north},{east});
);
out geom;""".strip()


def bbox_for_tile(tile_id: str, tile_deg: float, patch_r: float):
    lat = float(tile_id.split("_")[0])
    lon = float(tile_id.split("_")[1])
    EARTH_R = 6371000.0
    dlat = math.degrees(patch_r * 1.3 / EARTH_R)
    dlon = math.degrees(patch_r * 1.3 / (EARTH_R * max(math.cos(math.radians(lat)), 1e-6)))
    return lat - dlat, lon - dlon, lat + tile_deg + dlat, lon + tile_deg + dlon


def fetch_one(tile_id, cache_dir, tile_deg, patch_r, mirror_idx, max_attempts=3):
    out = cache_dir / f"{tile_id}.json"
    if out.exists() and out.stat().st_size > 0:
        return tile_id, "cached"

    # ── Land-mask short-circuit ──
    lat = float(tile_id.split("_")[0])
    lon = float(tile_id.split("_")[1])
    if not tile_has_land_nearby(lat, lon, tile_deg):
        out.write_text(json.dumps({"version": 0.6, "elements": []}))
        return tile_id, "skip_ocean"

    # ── Otherwise hit Overpass ──
    s, w, n, e = bbox_for_tile(tile_id, tile_deg, patch_r)
    q = make_query(s, w, n, e)
    for attempt in range(max_attempts):
        url = MIRRORS[(mirror_idx + attempt) % len(MIRRORS)]
        try:
            r = requests.post(url, data=q.encode("utf-8"), headers=HEADERS, timeout=60)
            if r.status_code == 200:
                json.loads(r.content)
                out.write_bytes(r.content)
                return tile_id, "fetched"
            elif r.status_code in (429, 503, 504):
                time.sleep(1.5 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return tile_id, "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir",   type=Path, required=True)
    ap.add_argument("--anchors-dir", type=Path, required=True)
    ap.add_argument("--tile-deg",    type=float, default=0.25)
    ap.add_argument("--patch-radius-m", type=float, default=5000.0)
    ap.add_argument("--workers",     type=int, default=4)
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique tile_ids from anchors CSVs
    tile_ids = set()
    for f in args.anchors_dir.glob("*_anchors.csv"):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("tile_id"):
                    tile_ids.add(row["tile_id"])
    todo = [t for t in tile_ids if not (args.cache_dir / f"{t}.json").exists()]
    print(f"Total unique tiles: {len(tile_ids):,}", flush=True)
    print(f"  already cached:   {len(tile_ids) - len(todo):,}", flush=True)
    print(f"  to process:       {len(todo):,}", flush=True)
    if not todo: return

    t0 = time.time()
    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, t, args.cache_dir, args.tile_deg,
                              args.patch_radius_m, i % len(MIRRORS))
                   for i, t in enumerate(todo)]
        for i, f in enumerate(as_completed(futures), start=1):
            tid, status = f.result()
            if   status == "fetched":  ok += 1
            elif status == "skip_ocean": skipped += 1
            elif status == "cached":   pass
            else:                       failed += 1
            if i % 50 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i:5d}/{len(todo)}]  fetched={ok}  skipped={skipped}  failed={failed}  "
                      f"{rate:.1f}/s  ETA={eta/60:.0f}m", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. fetched={ok} skipped_ocean={skipped} failed={failed}",
          flush=True)
    print(f"Tile cache now: {len(list(args.cache_dir.glob('*.json'))):,} files", flush=True)


if __name__ == "__main__":
    main()
