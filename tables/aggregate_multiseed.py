#!/usr/bin/env python3
"""Aggregate per-seed result JSONs into 5-seed mean/std summaries for the
three main tables.

Each learned model is trained under seeds 1-5; this script reads the per-seed
result files and reports mean and standard deviation of ADE / FDE (and the
stratified ADE columns for Table 3) over those seeds. Results are read from
$ENVSHIP_RES_DIR (default: ./results), laid out as

    <res>/track_a/in_domain/<src>/<model>_seed<k>.json
    <res>/track_a/<pool>/<model>_seed<k>.json
    <res>/track_b/in_domain/<SRC>/<model>_seed<k>.json

Writes summary.json next to this script.
"""
import json, glob, os, re
from pathlib import Path
import numpy as np

RES = Path(os.environ.get("ENVSHIP_RES_DIR", "results"))
SEEDS = [1, 2, 3, 4, 5]

A_MODELS = ["lstm_2l", "gru_2l", "bilstm_2l", "tcn", "transformer_nar",
            "lstm_social_pool", "lstm_social_attn", "lstm_social_attn_v5",
            "lstm_env_raster", "lstm_env_desc", "lstm_env_desc_v2",
            "lstm_env_binary_spatial_attn", "lstm_env_sdf",
            "lstm_env_spatial_attn", "lstm_social_env_v2", "lstm_social_env_sdf"]
X_MODELS = ["tcn", "lstm_env_sdf", "lstm_social_env_sdf"]
B_MODELS = ["lstm_2l", "gru_2l", "bilstm_2l", "tcn", "transformer_nar",
            "lstm_social_pool", "lstm_env_raster",
            "lstm_env_binary_spatial_attn", "lstm_env_sdf", "lstm_social_env_sdf"]
POOLS = ["combined", "loso_dma", "loso_noaa", "loso_piraeus", "loso_norway"]
XDOM_SOURCES = ["dma", "noaa", "piraeus", "norway"]
STRATA = ["ADE_3min", "ADE_6min", "ADE_easy", "ADE_medium", "ADE_hard",
          "ADE_open_water", "ADE_nearshore", "ADE_harbor"]


def collect(pattern, keys):
    """Return {key: {seed: value}} for seeds in SEEDS across matching files."""
    out = {k: {} for k in keys}
    for p in sorted(glob.glob(str(pattern))):
        m = re.search(r"_seed(\d+)\.json$", p)
        if not m:
            continue
        seed = int(m.group(1))
        if seed not in SEEDS:
            continue
        d = json.loads(Path(p).read_text())
        for k in keys:
            if d.get(k) is not None:
                out[k][seed] = float(d[k])
    return out


def stats(per_seed):
    v = [per_seed[s] for s in SEEDS if s in per_seed]
    if not v:
        return None
    return {"mean": round(float(np.mean(v)), 2),
            "std": round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0,
            "n": len(v)}


def cell(per_seed_by_key, keys):
    out = {}
    for k in keys:
        st = stats(per_seed_by_key[k])
        if st is not None:
            out[k] = st
    return out or None


summary = {"protocol": "mean/std over seeds 1-5",
           "table3": {}, "table4": {}, "table5": {}}

# Table 3: DMA Track A in-domain (headline + stratified)
for m in A_MODELS:
    c = collect(RES / "track_a" / "in_domain" / "dma" / f"{m}_seed*.json",
                keys=["ADE", "FDE"] + STRATA)
    e = cell(c, ["ADE", "FDE"] + STRATA)
    if e:
        summary["table3"][m] = e

# Table 4: Track B in-domain, DMA + NOAA
for src in ["DMA", "NOAA"]:
    summary["table4"][src] = {}
    for m in B_MODELS:
        c = collect(RES / "track_b" / "in_domain" / src / f"{m}_seed*.json",
                    keys=["ADE", "FDE"])
        e = cell(c, ["ADE", "FDE"])
        if e:
            summary["table4"][src][m] = e

# Table 5: cross-domain pools, per-source ADE
for pool in POOLS:
    summary["table5"][pool] = {}
    for m in X_MODELS:
        keys = [f"ADE_src_{s}" for s in XDOM_SOURCES]
        c = collect(RES / "track_a" / pool / f"{m}_seed*.json", keys=keys)
        e = cell(c, keys)
        if e:
            summary["table5"][pool][m] = e

out = Path(__file__).parent / "summary.json"
out.write_text(json.dumps(summary, indent=1))
print(f"wrote {out}")
print(f"table3: {len(summary['table3'])}/16 models")
for src in summary["table4"]:
    print(f"table4/{src}: {len(summary['table4'][src])}/10 models")
for pool in summary["table5"]:
    print(f"table5/{pool}: {len(summary['table5'][pool])}/3 models")
