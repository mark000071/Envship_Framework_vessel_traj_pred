"""Local-disk memmap caches for large per-sample context arrays.

Activated by env var LAZY_CTX=1. Instead of materialising (N, 6, 128, 128)
raster stacks or (N, 2, 128, 128) SDF stacks in RAM (8-14 GB on the DMA and
pooled tracks), the gather is streamed in chunks into a local .npy cache under
CACHE_DIR, and the returned object is a numpy memmap opened in copy-on-write
mode. Downstream code (TensorDataset / torch.from_numpy / slicing) works
unchanged: values are bit-identical to the eager path; the OS pages data in
on demand.

Caches are keyed by (path, split/tag, key, N) and are reused across seeds and
models, so the gather cost is paid once per split.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("LAZY_CTX_DIR", "/var/tmp/kdd_ctx_cache"))
CHUNK = 2048


def lazy_enabled() -> bool:
    return os.environ.get("LAZY_CTX") == "1"


def _cache_path(tag: str, key: str, n: int, shape: tuple, dtype) -> Path:
    h = hashlib.sha1(f"{tag}|{key}|{n}|{shape}|{np.dtype(dtype).str}".encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}_{n}_{h}.npy"


def _open_cow(path: Path) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="c")


def gather_to_cache(tag: str, key: str, idx: list[int], sources, shape: tuple,
                    dtype) -> np.memmap:
    """Gather rows `idx` (with -1 meaning all-zeros) from `sources` into a
    local cache file and return it as a copy-on-write memmap.

    sources: one mmap array (raster) or a tuple of two (sdf shore, nav) that
    are stacked along a new leading channel axis per sample.
    """
    n = len(idx)
    path = _cache_path(tag, key, n, shape, dtype)
    if path.exists():
        return _open_cow(path)
    tmp = path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(n,) + shape)
    idx_arr = np.asarray(idx, dtype=np.int64)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        ids = idx_arr[i0:i1]
        valid = ids >= 0
        if isinstance(sources, tuple):
            shore, nav = sources
            buf = np.zeros((i1 - i0,) + shape, dtype=dtype)
            if valid.any():
                buf[valid, 0] = shore[ids[valid]]
                buf[valid, 1] = nav[ids[valid]]
            out[i0:i1] = buf
        else:
            buf = np.zeros((i1 - i0,) + shape, dtype=dtype)
            if valid.any():
                buf[valid] = sources[ids[valid]]
            out[i0:i1] = buf
    out.flush()
    del out
    os.replace(tmp, path)
    return _open_cow(path)


def concat_to_cache(tag: str, key: str, arrays) -> np.memmap:
    """Concatenate arrays along axis 0 into a local cache file (chunked copy)."""
    n = sum(len(a) for a in arrays)
    shape = arrays[0].shape[1:]
    dtype = arrays[0].dtype
    path = _cache_path(tag, key, n, shape, dtype)
    if path.exists():
        return _open_cow(path)
    tmp = path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(n,) + shape)
    pos = 0
    for a in arrays:
        for i0 in range(0, len(a), CHUNK):
            i1 = min(i0 + CHUNK, len(a))
            out[pos + i0:pos + i1] = a[i0:i1]
        pos += len(a)
    out.flush()
    del out
    os.replace(tmp, path)
    return _open_cow(path)
