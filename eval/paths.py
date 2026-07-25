"""Single source of truth for dataset / output locations.

All data paths are resolved relative to the ``ENVSHIP_DATA_ROOT`` environment
variable so the framework runs against a dataset downloaded from Hugging Face
without editing any source. Arrange the download as::

    $ENVSHIP_DATA_ROOT/
        track_a/{dma,noaa,piraeus,norway}/standard_track_v1/   # 10 min -> 10 min
        track_b/{dma,noaa,piraeus,norway}/standard_track_v1/   # 30 min -> 60 min

Each ``standard_track_v1/`` holds the split parquet/JSONL tables plus the
``context_v1/`` sub-tree (environment rasters/SDF, social neighbor tables).

Override the root for a single run either by exporting the env var::

    export ENVSHIP_DATA_ROOT=/path/to/envship_v2_datasets

or, for the training entrypoint, with ``--track-root /explicit/path``.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ENVSHIP_DATA_ROOT", "data")).expanduser()
CKPT_DIR  = Path(os.environ.get("ENVSHIP_CKPT_DIR", "checkpoints")).expanduser()
RES_DIR   = Path(os.environ.get("ENVSHIP_RES_DIR", "results")).expanduser()

SOURCES = ("dma", "noaa", "piraeus", "norway")


def track_root(source: str, track: str = "a") -> Path:
    """Canonical dataset root for one jurisdiction and track ('a' or 'b')."""
    return DATA_ROOT / f"track_{track}" / source / "standard_track_v1"


# Track A (10 min -> 10 min) and Track B (30 min -> 60 min) roots per source.
SOURCE_ROOTS: dict[str, Path]   = {s: track_root(s, "a") for s in SOURCES}
SOURCE_ROOTS_B: dict[str, Path] = {s: track_root(s, "b") for s in SOURCES}

# DMA Track A is the canonical single-source default used by the trainer.
DEFAULT_TRACK_ROOT = SOURCE_ROOTS["dma"]
