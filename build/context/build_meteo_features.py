#!/usr/bin/env python3
"""EnvShip-Bench v2 — Phase 1+2 builder for weather/sea-state/port/TSS context.

Produces 15 new scalar columns per sample (see CC_prompt_lab/prompt_add_wather_wave_prot.md
for the full schema). Reads existing anchor CSVs, queries
Open-Meteo Archive + Marine APIs (free, ERA5-derived) plus Overpass for
OSM seamarks, caches everything to parquet/geojson, and writes
augmented main CSVs.

Idempotent — re-running skips cached cells, geometry pulls, and CSVs
that already carry the new columns.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent.parent
CACHE_DIR     = ROOT / "_meteo_cache"
WEATHER_DIR   = CACHE_DIR / "weather"
MARINE_DIR    = CACHE_DIR / "marine"
OSM_DIR       = CACHE_DIR / "osm"
SUMMARY_PATH  = CACHE_DIR / "meteo_build_summary.json"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MARINE_URL  = "https://marine-api.open-meteo.com/v1/marine"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

WEATHER_VARS = [
    "wind_speed_10m", "wind_direction_10m",
    "temperature_2m", "surface_pressure", "cloud_cover",
]
MARINE_VARS = [
    "wave_height", "wave_direction", "wave_period",
    "swell_wave_height",
]

GRID_DEG = 0.25  # ERA5-matched

NEW_COLS = [
    "met_wind_speed_mps", "met_wind_dir_deg", "met_wind_rel_heading_deg",
    "met_temperature_c", "met_pressure_hpa", "met_cloud_cover_pct",
    "sea_wave_height_m", "sea_wave_dir_deg", "sea_wave_period_s", "sea_swell_wave_height_m",
    "port_nearest_dist_km", "port_nearest_name",
    "in_fairway", "dist_to_fairway_m", "in_tss",
]

SUBSETS = {
    "DMA":     {
        "root":         "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets",
        "hf_dir_a":     "track_a_short-term_Cross-domain_Datasets/dma_track_v1",
        "hf_dir_b":     "track_b_medium-term_Cross-domain_Datasets/dma/standard_track_v1",
    },
    "NOAA":    {
        "root":         "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets",
        "hf_dir_a":     "track_a_short-term_Cross-domain_Datasets/noaa_track_v1",
        "hf_dir_b":     "track_b_medium-term_Cross-domain_Datasets/noaa/standard_track_v1",
    },
    "Piraeus": {
        "root":         "Piraeus_ship_trajectory_datasets",
        "hf_dir_a":     "track_a_short-term_Cross-domain_Datasets/piraeus_track_v1",
        "hf_dir_b":     "track_b_medium-term_Cross-domain_Datasets/piraeus/standard_track_v1",
    },
    "Norway":  {
        "root":         "norway_ship_trajectory_datasets",
        "hf_dir_a":     "track_a_short-term_Cross-domain_Datasets/norway_track_v1",
        "hf_dir_b":     "track_b_medium-term_Cross-domain_Datasets/norway/standard_track_v1",
    },
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell_of(lat: float, lon: float, grid: float = GRID_DEG) -> tuple[float, float]:
    return (round(lat / grid) * grid, round(lon / grid) * grid)


def cell_id(lat: float, lon: float) -> str:
    return f"{lat:+.3f}_{lon:+.3f}"


def month_key(ts: str) -> str:
    """ts in form 2025-09-04T09:47:20+00:00 → '2025-09'."""
    return ts[:7]


def month_bounds(ym: str) -> tuple[str, str]:
    from datetime import date, timedelta
    y, m = ym.split("-")
    first = date(int(y), int(m), 1)
    nxt = date(int(y)+1, 1, 1) if m == "12" else date(int(y), int(m)+1, 1)
    last = nxt - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    rl1, rl2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rl1)*math.cos(rl2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def signed_angle_deg(a: float, b: float) -> float:
    """Shortest signed angle a-b in [-180, 180]."""
    d = (a - b + 540.0) % 360.0 - 180.0
    return d


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def _load_anchors_from(anchors_csv: Path) -> list[dict]:
    out = []
    if not anchors_csv.exists():
        return out
    with anchors_csv.open() as fh:
        for row in csv.DictReader(fh):
            try:
                lat = float(row["anchor_lat"]); lon = float(row["anchor_lon"])
                if lat == 0 and lon == 0:
                    continue
                out.append({
                    "sample_id":   row["sample_id"],
                    "lat":         lat,
                    "lon":         lon,
                    "ts":          row["hist_end_ts"],
                })
            except Exception:
                pass
    return out


def load_anchors(root: str) -> list[dict]:
    """Load Track-A anchors for a subset (sample_id → lat/lon/ts)."""
    return _load_anchors_from(Path(root) / "multi_type_mini_bench_build/standard_track_v1/context_v1/environment/anchors/all_anchors.csv")


def load_anchors_track_b(subset_name: str) -> list[dict]:
    """Load Track-B anchors for a subset."""
    return _load_anchors_from(ROOT / "track_b" / subset_name / "multi_type_mini_bench_build/standard_track_v1/context_v1/environment/anchors/all_anchors.csv")


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------

def _save_pq(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fetch_open_meteo(url: str, lat: float, lon: float, start: str, end: str,
                     hourly: list[str], retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame | None:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(60); continue
            r.raise_for_status()
            j = r.json()
            h = j.get("hourly", {})
            if not h or "time" not in h:
                return None
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            time.sleep(sleep_s)
            return df
        except Exception as e:
            last_err = e
            time.sleep(min(30.0, 2 ** attempt))
    log(f"[open-meteo] FAIL ({lat:.3f},{lon:.3f},{start}): {last_err}")
    return None


def fetch_cell_months(cell_months: set[tuple[float,float,str]],
                      base_dir: Path, url: str, hourly: list[str],
                      max_workers: int = 4) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for lat, lon, ym in cell_months:
        cache = base_dir / ym / f"{cell_id(lat, lon)}.parquet"
        if cache.exists() and cache.stat().st_size > 0:
            continue
        pending.append((lat, lon, ym, cache))
    if not pending:
        return
    log(f"[fetch] {base_dir.name}: {len(pending):,} cell-months pending")
    t0 = time.time()
    done = 0
    def _task(lat, lon, ym, cache):
        start, end = month_bounds(ym)
        df = fetch_open_meteo(url, lat, lon, start, end, hourly)
        if df is not None:
            _save_pq(df, cache)
        return df is not None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_task, *p) for p in pending]
        for f in as_completed(futures):
            done += 1
            if done % 50 == 0:
                eta = (time.time()-t0)/done * (len(pending)-done)
                log(f"  {done}/{len(pending)} done  eta {eta:.0f}s")
    log(f"[fetch] {base_dir.name} done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# OSM ports + fairways via Overpass
# ---------------------------------------------------------------------------

def overpass_query(query: str, retries: int = 3) -> dict | None:
    last_err = None
    for url in OVERPASS_URLS:
        for attempt in range(retries):
            try:
                r = requests.post(url, data=query.encode("utf-8"),
                                  headers={"Content-Type":"text/plain",
                                           "User-Agent":"EnvShip-Bench/v2 meteo build"},
                                  timeout=120)
                if r.status_code == 429:
                    time.sleep(60); continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(min(20.0, 5 * (attempt + 1)))
    log(f"[overpass] FAILED: {last_err}")
    return None


def fetch_osm_layers(name: str, bbox: tuple[float,float,float,float],
                     anchor_cells: set | None = None) -> dict:
    """Return {'ports': [{...}], 'fairways': [{...}], 'tss': [{...}]}.

    For subsets with a wide bbox (NOAA), use the actual anchor 0.25° cells
    grouped into 5° super-bins → only query bboxes where samples exist.
    Avoids hundreds of empty-ocean chunks.
    """
    s, w, n, e = bbox
    layers = {"ports": [], "fairways": [], "tss": []}
    out_dir = OSM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{name.lower()}.geojson"
    if cache.exists() and cache.stat().st_size > 100:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    # Default: one bbox covering everything.
    boxes = [(s, w, n, e)]
    is_wide = (e - w) > 30 or (n - s) > 30
    if is_wide and anchor_cells:
        # Group 0.25° cells into 5° bins; one bbox per non-empty bin (with small pad).
        bins: dict[tuple[float,float], list[tuple[float,float]]] = defaultdict(list)
        for la, lo in anchor_cells:
            bins[(round(la/5)*5, round(lo/5)*5)].append((la, lo))
        boxes = []
        for _, cells in bins.items():
            lats = [c[0] for c in cells]; lons = [c[1] for c in cells]
            boxes.append((min(lats) - 0.25, min(lons) - 0.25,
                          max(lats) + 0.5,  max(lons) + 0.5))
        log(f"[osm:{name}] wide bbox → {len(boxes)} anchor-clustered sub-boxes")
    elif is_wide:
        # Fallback: uniform split
        lat_steps = max(1, int(math.ceil((n - s) / 10)))
        lon_steps = max(1, int(math.ceil((e - w) / 10)))
        boxes = []
        for i in range(lat_steps):
            for j in range(lon_steps):
                ss = s + (n - s) * i / lat_steps
                nn = s + (n - s) * (i + 1) / lat_steps
                ww = w + (e - w) * j / lon_steps
                ee = w + (e - w) * (j + 1) / lon_steps
                boxes.append((ss, ww, nn, ee))
        log(f"[osm:{name}] wide bbox → {len(boxes)} uniform chunks")

    for idx, (ss, ww, nn, ee) in enumerate(boxes, 1):
        log(f"[osm:{name}] chunk {idx}/{len(boxes)}  bbox=({ss:.2f},{ww:.2f},{nn:.2f},{ee:.2f})")
        # 1) ports
        q_port = f"""[out:json][timeout:120];
(
  way["harbour"~"yes|sea|marina|fishing"]({ss},{ww},{nn},{ee});
  way["seamark:type"="harbour"]({ss},{ww},{nn},{ee});
  way["landuse"="harbour"]({ss},{ww},{nn},{ee});
  node["harbour"~"yes|sea|marina"]({ss},{ww},{nn},{ee});
  node["seamark:type"="harbour"]({ss},{ww},{nn},{ee});
);
out tags center 200;"""
        j = overpass_query(q_port)
        if j:
            for el in j.get("elements", []):
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lon = el.get("lon") or el.get("center", {}).get("lon")
                if lat is None or lon is None: continue
                tags = el.get("tags", {})
                layers["ports"].append({
                    "lat": lat, "lon": lon,
                    "name": tags.get("name", ""),
                    "harbour": tags.get("harbour", ""),
                    "osm_id": el.get("id"),
                })
        # 2) fairways
        q_fw = f"""[out:json][timeout:120];
(
  way["seamark:type"="fairway"]({ss},{ww},{nn},{ee});
  relation["seamark:type"="fairway"]({ss},{ww},{nn},{ee});
);
out geom 500;"""
        j = overpass_query(q_fw)
        if j:
            for el in j.get("elements", []):
                if el.get("type") == "way":
                    geom = el.get("geometry", [])
                    if len(geom) >= 2:
                        coords = [(g["lat"], g["lon"]) for g in geom]
                        layers["fairways"].append({"type":"way","coords":coords,
                                                   "tags":el.get("tags",{}),
                                                   "osm_id":el.get("id")})
        # 3) TSS
        q_tss = f"""[out:json][timeout:120];
(
  way["seamark:type"~"separation_zone|separation_line|separation_boundary"]({ss},{ww},{nn},{ee});
  relation["seamark:type"~"separation_zone|separation_line|separation_boundary"]({ss},{ww},{nn},{ee});
);
out geom 500;"""
        j = overpass_query(q_tss)
        if j:
            for el in j.get("elements", []):
                if el.get("type") == "way":
                    geom = el.get("geometry", [])
                    if len(geom) >= 2:
                        coords = [(g["lat"], g["lon"]) for g in geom]
                        layers["tss"].append({"type":"way","coords":coords,
                                              "tags":el.get("tags",{}),
                                              "osm_id":el.get("id")})

    cache.write_text(json.dumps(layers))
    log(f"[osm:{name}]  ports={len(layers['ports'])}  fairways={len(layers['fairways'])}  tss={len(layers['tss'])}")
    return layers


# ---------------------------------------------------------------------------
# Build local-meter STRtree from layers for distance queries
# ---------------------------------------------------------------------------

def build_geom_index(layers: dict) -> dict:
    """Return {'port_tree': STRtree, 'port_items': [...], 'fairway_tree': STRtree,
              'fairway_items': [...], 'tss_tree': STRtree, 'tss_items': [...]}.
    Geometries are in (lon, lat) deg space (shapely treats them as planar).
    Distance computed via haversine in the query.
    """
    out = {}
    # ports — points
    port_pts = []
    for p in layers.get("ports", []):
        port_pts.append((Point(p["lon"], p["lat"]), p))
    if port_pts:
        out["port_tree"]  = STRtree([g for g,_ in port_pts])
        out["port_items"] = [m for _,m in port_pts]
    else:
        out["port_tree"], out["port_items"] = None, []
    # fairways — lines
    fw_items = []
    for f in layers.get("fairways", []):
        coords = [(lo, la) for la, lo in f["coords"]]
        try:
            fw_items.append((LineString(coords), f))
        except Exception:
            continue
    if fw_items:
        out["fairway_tree"]  = STRtree([g for g,_ in fw_items])
        out["fairway_items"] = [m for _,m in fw_items]
    else:
        out["fairway_tree"], out["fairway_items"] = None, []
    # TSS — polygons or lines
    tss_items = []
    for t in layers.get("tss", []):
        coords = [(lo, la) for la, lo in t["coords"]]
        try:
            if coords[0] == coords[-1] and len(coords) >= 4:
                tss_items.append((Polygon(coords), t))
            else:
                tss_items.append((LineString(coords), t))
        except Exception:
            continue
    if tss_items:
        out["tss_tree"]  = STRtree([g for g,_ in tss_items])
        out["tss_items"] = [m for _,m in tss_items]
    else:
        out["tss_tree"], out["tss_items"] = None, []
    return out


# ---------------------------------------------------------------------------
# Cell-month → cell × hour lookup
# ---------------------------------------------------------------------------

def load_grid_for_subset(base_dir: Path, cell_months: set) -> dict:
    """Return {(round(lat,3), round(lon,3)) → DataFrame indexed by hour}."""
    out = {}
    for lat, lon, ym in cell_months:
        cache = base_dir / ym / f"{cell_id(lat,lon)}.parquet"
        if not cache.exists():
            continue
        try:
            df = pd.read_parquet(cache)
        except Exception:
            continue
        df = df.set_index("time")
        key = (round(lat, 3), round(lon, 3))
        if key in out:
            out[key] = pd.concat([out[key], df]).sort_index()
            out[key] = out[key][~out[key].index.duplicated(keep='last')]
        else:
            out[key] = df
    return out


# ---------------------------------------------------------------------------
# Per-sample column compute
# ---------------------------------------------------------------------------

def per_sample_columns(sid: str, lat: float, lon: float, ts_str: str,
                       wx: dict, mr: dict,
                       geom: dict, cog_deg: float | None) -> dict:
    """Return a dict of the 15 new columns for one sample."""
    out = {c: None for c in NEW_COLS}
    # Cell + hour
    cell = cell_of(lat, lon)
    key = (round(cell[0],3), round(cell[1],3))
    ts = pd.Timestamp(ts_str).tz_convert("UTC").floor("h") if pd.Timestamp(ts_str).tzinfo else \
         pd.Timestamp(ts_str, tz="UTC").floor("h")
    # Weather
    df_w = wx.get(key)
    if df_w is not None and ts in df_w.index:
        row = df_w.loc[ts]
        ws_kmh = row.get("wind_speed_10m")
        if ws_kmh is not None and not pd.isna(ws_kmh):
            out["met_wind_speed_mps"] = float(ws_kmh) / 3.6
        wd = row.get("wind_direction_10m")
        if wd is not None and not pd.isna(wd):
            out["met_wind_dir_deg"] = float(wd)
            if cog_deg is not None:
                out["met_wind_rel_heading_deg"] = signed_angle_deg(float(wd), cog_deg)
        for src, dst in [("temperature_2m","met_temperature_c"),
                         ("surface_pressure","met_pressure_hpa"),
                         ("cloud_cover","met_cloud_cover_pct")]:
            v = row.get(src)
            if v is not None and not pd.isna(v):
                out[dst] = float(v)
    # Marine
    df_m = mr.get(key)
    if df_m is not None and ts in df_m.index:
        row = df_m.loc[ts]
        for src, dst in [("wave_height","sea_wave_height_m"),
                         ("wave_direction","sea_wave_dir_deg"),
                         ("wave_period","sea_wave_period_s"),
                         ("swell_wave_height","sea_swell_wave_height_m")]:
            v = row.get(src)
            if v is not None and not pd.isna(v):
                out[dst] = float(v)
    # Ports
    if geom["port_tree"] is not None:
        pt = Point(lon, lat)
        idxs = geom["port_tree"].query_nearest(pt)
        if len(idxs):
            best = None; best_d = float("inf")
            for ii in idxs[:5]:
                p = geom["port_items"][int(ii)]
                d = haversine_km(lat, lon, p["lat"], p["lon"])
                if d < best_d: best_d, best = d, p
            if best is not None:
                out["port_nearest_dist_km"] = float(best_d)
                out["port_nearest_name"]    = best.get("name","")
    # Fairway
    if geom["fairway_tree"] is not None:
        pt = Point(lon, lat)
        idxs = geom["fairway_tree"].query_nearest(pt)
        if len(idxs):
            # Distance from point to nearest fairway line (in deg space then converted)
            best_geom = geom["fairway_items"][int(idxs[0])]
            line = LineString([(lo, la) for la, lo in best_geom["coords"]])
            np_pt = line.interpolate(line.project(pt))
            d_km  = haversine_km(lat, lon, np_pt.y, np_pt.x)
            d_m   = d_km * 1000.0
            out["dist_to_fairway_m"] = float(d_m)
            out["in_fairway"]        = bool(d_m <= 100.0)
    else:
        out["in_fairway"] = False
    # TSS
    in_tss = False
    if geom["tss_tree"] is not None:
        pt = Point(lon, lat)
        idxs = geom["tss_tree"].query(pt, predicate="intersects")
        if len(idxs):
            in_tss = True
    out["in_tss"] = bool(in_tss)
    return out


# ---------------------------------------------------------------------------
# CSV augmentation
# ---------------------------------------------------------------------------

def cog_from_row(row: dict) -> float | None:
    """Return the last history COG in degrees, computed from
    hist_cog_sin_json / hist_cog_cos_json arrays."""
    try:
        s = json.loads(row["hist_cog_sin_json"])
        c = json.loads(row["hist_cog_cos_json"])
        sin = s[-1]; cos = c[-1]
        deg = (math.degrees(math.atan2(sin, cos)) + 360.0) % 360.0
        return deg
    except Exception:
        return None


def augment_csv(csv_in: Path, csv_out: Path, wx: dict, mr: dict, geom: dict,
                anchor_map: dict) -> dict:
    """Return stats {rows, nulls_per_col, written}."""
    nulls = {c: 0 for c in NEW_COLS}
    n_total = 0
    with gzip.open(csv_in, "rt", encoding="utf-8", newline="") as ih:
        reader = csv.DictReader(ih)
        base_fields = list(reader.fieldnames or [])
        # Drop already-present new cols (idempotent re-run)
        out_fields = [f for f in base_fields if f not in NEW_COLS] + NEW_COLS
        with gzip.open(csv_out, "wt", encoding="utf-8", newline="") as oh:
            writer = csv.DictWriter(oh, fieldnames=out_fields)
            writer.writeheader()
            for row in reader:
                n_total += 1
                sid = row["sample_id"]
                anchor = anchor_map.get(sid)
                if anchor is None:
                    new = {c: "" for c in NEW_COLS}
                else:
                    cog = cog_from_row(row)
                    cols = per_sample_columns(sid, anchor["lat"], anchor["lon"],
                                              anchor["ts"], wx, mr, geom, cog)
                    new = {}
                    for c in NEW_COLS:
                        v = cols.get(c)
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            new[c] = ""
                            nulls[c] += 1
                        else:
                            new[c] = v
                base = {k: row.get(k, "") for k in out_fields if k not in NEW_COLS}
                base.update(new)
                writer.writerow(base)
    return {"rows": n_total, "nulls": nulls, "written": str(csv_out)}


# ---------------------------------------------------------------------------
# Subset driver
# ---------------------------------------------------------------------------

def process_subset(name: str) -> dict:
    cfg = SUBSETS[name]
    root_a = Path(cfg["root"])
    log(f"=== {name} ===")
    anchors = load_anchors(cfg["root"])
    if not anchors:
        log(f"  no anchors → skip"); return {"name": name, "skipped": True}
    log(f"  anchors loaded: {len(anchors):,}")
    cell_months = set()
    bbox = [180,90,-180,-90]  # w,s,e,n
    for a in anchors:
        c = cell_of(a["lat"], a["lon"])
        cell_months.add((c[0], c[1], month_key(a["ts"])))
        bbox[0] = min(bbox[0], a["lon"]); bbox[1] = min(bbox[1], a["lat"])
        bbox[2] = max(bbox[2], a["lon"]); bbox[3] = max(bbox[3], a["lat"])
    log(f"  unique cell-months: {len(cell_months):,}")
    # Fetch weather + marine
    fetch_cell_months(cell_months, WEATHER_DIR / name, ARCHIVE_URL, WEATHER_VARS, max_workers=4)
    fetch_cell_months(cell_months, MARINE_DIR  / name, MARINE_URL,  MARINE_VARS,  max_workers=4)
    # Load grids
    wx = load_grid_for_subset(WEATHER_DIR / name, cell_months)
    mr = load_grid_for_subset(MARINE_DIR  / name, cell_months)
    log(f"  loaded wx cells: {len(wx)}  marine cells: {len(mr)}")
    # OSM layers + STRtree
    pad = 0.5
    anchor_cells = {(c[0], c[1]) for (c0, c1, _) in cell_months for c in [(c0, c1)]}
    layers = fetch_osm_layers(
        name,
        (bbox[1]-pad, bbox[0]-pad, bbox[3]+pad, bbox[2]+pad),
        anchor_cells=anchor_cells,
    )
    geom = build_geom_index(layers)
    anchor_map = {a["sample_id"]: a for a in anchors}
    # Augment Track A
    summary = {"name": name, "track_a": {}, "track_b": {}}
    for split in ("train","val","test"):
        in_path  = root_a / f"multi_type_mini_bench_build/standard_track_v1/{split}/part-000.csv.gz"
        if not in_path.exists():
            log(f"  TrackA/{split} missing — skip"); continue
        out_tmp = in_path.with_suffix(".csv.gz.tmp")
        stats = augment_csv(in_path, out_tmp, wx, mr, geom, anchor_map)
        out_tmp.replace(in_path)
        summary["track_a"][split] = stats
        log(f"  TrackA/{split}: {stats['rows']:,} rows, "
            f"null_wind={stats['nulls']['met_wind_speed_mps']}, "
            f"null_wave={stats['nulls']['sea_wave_height_m']}, "
            f"null_port_km={stats['nulls']['port_nearest_dist_km']}")
    # Track B (has its own anchors)
    tb_root = ROOT / "track_b" / name / "multi_type_mini_bench_build/standard_track_v1"
    if tb_root.exists():
        anchors_b = load_anchors_track_b(name)
        anchor_map_b = {a["sample_id"]: a for a in anchors_b}
        # Add any Track-B cells/months that we did not fetch before
        extra = set()
        for a in anchors_b:
            c = cell_of(a["lat"], a["lon"]); extra.add((c[0], c[1], month_key(a["ts"])))
        new_cm = extra - cell_months
        if new_cm:
            log(f"  TrackB extra cell-months: {len(new_cm):,}")
            fetch_cell_months(new_cm, WEATHER_DIR/name, ARCHIVE_URL, WEATHER_VARS, max_workers=4)
            fetch_cell_months(new_cm, MARINE_DIR/name,  MARINE_URL,  MARINE_VARS,  max_workers=4)
            wx.update(load_grid_for_subset(WEATHER_DIR/name, new_cm))
            mr.update(load_grid_for_subset(MARINE_DIR/name,  new_cm))
        log(f"  TrackB anchors loaded: {len(anchors_b):,}")
        for split in ("train","val","test"):
            in_path = tb_root / split / "part-000.csv.gz"
            if not in_path.exists():
                continue
            out_tmp = in_path.with_suffix(".csv.gz.tmp")
            stats = augment_csv(in_path, out_tmp, wx, mr, geom, anchor_map_b)
            out_tmp.replace(in_path)
            summary["track_b"][split] = stats
            log(f"  TrackB/{split}: {stats['rows']:,} rows, "
                f"null_wind={stats['nulls']['met_wind_speed_mps']}, "
                f"null_port={stats['nulls']['port_nearest_dist_km']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS.keys()))
    p.add_argument("--skip-fetch", action="store_true",
                   help="Re-use existing parquet/geojson cache, only re-run merge.")
    args = p.parse_args()

    summaries = {}
    for s in args.subsets:
        if s not in SUBSETS:
            log(f"unknown subset {s}"); continue
        summaries[s] = process_subset(s)
    SUMMARY_PATH.write_text(json.dumps(summaries, indent=2, default=str))
    log(f"build summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
