"""Dataset loader for EnvShip-Bench Standard Track.

Loads trajectory CSV shards into numpy arrays (no PyTorch dependency required).
PyTorch Dataset wrapper provided if torch is available.
"""

from __future__ import annotations

import csv, gzip, json
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Fast shard reader
# ---------------------------------------------------------------------------

def _parse_f32(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float32)


def _classify_ship(ship_type: str, ship_class: str) -> str:
    full = f"{ship_type} {ship_class}".lower()
    if any(k in full for k in ("cargo", "tanker", "bulk", "container")): return "cargo_tanker"
    if any(k in full for k in ("passenger", "ferry")):                    return "passenger_ferry"
    if "fishing" in full:                                                  return "fishing"
    if any(k in full for k in ("tug", "service", "pilot")):               return "tug_service"
    if any(k in full for k in ("sailing", "pleasure", "yacht")):          return "sailing_leisure"
    return "other"


def load_split(
    track_root: str | Path,
    split: str,
    use_augmented: bool = False,
    max_samples: int = 0,
) -> dict[str, np.ndarray]:
    """Load a split into numpy arrays.

    Returns a dict with keys:
      hist     : (N, 30, 6)  float32  [x, y, vx, vy, sog, t_of_day_sin]
      future   : (N, 30, 2)  float32  [x, y]
      meta     : dict with string arrays (sample_id, ship_group, difficulty_tier, scene_type, ...)
    """
    track_root = Path(track_root)
    if use_augmented:
        csv_path = track_root / "context_v1" / "augmented" / split / "part-000.csv.gz"
    else:
        csv_path = track_root / split / "part-000.csv.gz"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    hists, futures = [], []
    meta: dict[str, list] = {
        "sample_id": [], "mmsi": [], "segment_id": [],
        "ship_group": [], "difficulty_tier": [],
        "scene_type": [], "avg_speed_knots": [],
        "interp_ratio_total": [],
    }

    # Load difficulty_tier from metadata report if available
    meta_path = track_root / "reports" / f"{split}_selected_metadata.csv"
    diff_tier_map: dict[str, str] = {}
    if meta_path.exists():
        with open(meta_path, "r", newline="", encoding="utf-8") as mf:
            for row in csv.DictReader(mf):
                if "difficulty_tier" in row:
                    diff_tier_map[row["sample_id"]] = row["difficulty_tier"]

    n = 0
    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hx = _parse_f32(row["hist_x_json"])      # (30,)
            hy = _parse_f32(row["hist_y_json"])      # (30,)
            fx = _parse_f32(row["fut_x_json"])       # (30,)
            fy = _parse_f32(row["fut_y_json"])       # (30,)

            # Derive velocity from position differences (20 s interval)
            dt = 20.0
            vx = np.gradient(hx, dt)                 # (30,)
            vy = np.gradient(hy, dt)                 # (30,)
            sog = _parse_f32(row["hist_sog_json"])   # (30,) in knots

            # Time-of-day sine as a temporal feature
            tod_sin = _parse_f32(row["hist_time_of_day_sin_json"])

            hist = np.stack([hx, hy, vx, vy, sog, tod_sin], axis=-1)  # (30, 6)
            fut  = np.stack([fx, fy], axis=-1)                         # (30, 2)

            hists.append(hist)
            futures.append(fut)

            sg = _classify_ship(row.get("ship_type",""), row.get("ship_class",""))
            meta["sample_id"].append(row["sample_id"])
            meta["mmsi"].append(int(row.get("mmsi", 0)))
            meta["segment_id"].append(row.get("segment_id", ""))
            meta["ship_group"].append(sg)
            dt = diff_tier_map.get(row["sample_id"], row.get("difficulty_tier", ""))
            meta["difficulty_tier"].append(dt)
            meta["scene_type"].append(row.get("scene_type", "open_water"))
            meta["avg_speed_knots"].append(float(row.get("avg_speed_knots", 0) or 0))
            meta["interp_ratio_total"].append(float(row.get("interp_ratio_total", 0) or 0))

            n += 1
            if max_samples > 0 and n >= max_samples:
                break

    return {
        "hist":   np.stack(hists),    # (N, 30, 6)
        "future": np.stack(futures),  # (N, 30, 2)
        "meta":   {k: np.asarray(v) for k, v in meta.items()},
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset (optional)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    class ShipTrajectoryDataset(Dataset):
        """PyTorch Dataset wrapping load_split()."""

        def __init__(
            self,
            track_root: str | Path,
            split: str,
            use_augmented: bool = False,
            max_samples: int = 0,
        ):
            data = load_split(track_root, split, use_augmented, max_samples)
            self.hist   = torch.from_numpy(data["hist"])    # (N, 30, 6)
            self.future = torch.from_numpy(data["future"])  # (N, 30, 2)
            self.meta   = data["meta"]

        def __len__(self) -> int:
            return len(self.hist)

        def __getitem__(self, idx: int) -> dict:
            return {
                "hist":       self.hist[idx],     # (30, 6)
                "future":     self.future[idx],   # (30, 2)
                "sample_id":  str(self.meta["sample_id"][idx]),
                "ship_group": str(self.meta["ship_group"][idx]),
                "scene_type": str(self.meta["scene_type"][idx]),
            }

except ImportError:
    pass  # torch not available — numpy-only mode is fine
