#!/usr/bin/env python3
"""Render LaTeX table bodies from summary.json (5-seed mean/std).

Produces gen_table3.tex / gen_table4.tex / gen_table5.tex with the learned-model
rows. Deterministic physics / classical-ML rows are added separately in the paper.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "summary.json")))
t3, t4, t5 = S["table3"], S["table4"], S["table5"]

STRATA = ["ADE_3min", "ADE_6min", "ADE_easy", "ADE_medium", "ADE_hard",
          "ADE_open_water", "ADE_nearshore", "ADE_harbor"]


def pm(e, bold=False):
    if e is None:
        return "---"
    txt = f"{e['mean']:.1f}" if not e.get("std") else f"{e['mean']:.1f}$\\pm${e['std']:.1f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def mean_only(e):
    return "---" if e is None else f"{e['mean']:.1f}"


# ---- Table 3: DMA Track A ----
NAMES3 = [("lstm_2l", "LSTM-2L"), ("gru_2l", "GRU-2L"), ("bilstm_2l", "BiLSTM-2L"),
          ("tcn", "TCN"), ("transformer_nar", "Transformer-NAR"),
          ("lstm_social_pool", "LSTM+Social-Pool"), ("lstm_social_attn", "LSTM+Social-Attn"),
          ("lstm_social_attn_v5", "LSTM+Social-AttnV5"), ("lstm_env_raster", "LSTM+Env-Raster"),
          ("lstm_env_desc", "LSTM+Env-Desc"), ("lstm_env_desc_v2", "LSTM+Env-Desc-v2"),
          ("lstm_env_binary_spatial_attn", "LSTM+EnvBinary-SpatialAttn"),
          ("lstm_env_sdf", "LSTM+Env-SDF"), ("lstm_env_spatial_attn", "LSTM+Env-SpatialAttn"),
          ("lstm_social_env_v2", "LSTM+Social+Env-v2"), ("lstm_social_env_sdf", "LSTM+Social+Env-SDF")]
present = [k for k, _ in NAMES3 if k in t3]
best_ade = min(present, key=lambda k: t3[k]["ADE"]["mean"]) if present else None
best_fde = min(present, key=lambda k: t3[k]["FDE"]["mean"]) if present else None
rows = []
for k, n in NAMES3:
    e = t3.get(k)
    if e is None:
        continue
    cells = [pm(e["ADE"], k == best_ade), pm(e["FDE"], k == best_fde)]
    cells += [mean_only(e.get(s)) for s in STRATA]
    nm = f"\\textbf{{{n}}}" if k in (best_ade, best_fde) else n
    rows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
open(os.path.join(HERE, "gen_table3.tex"), "w").write("\n".join(rows) + "\n")

# ---- Table 4: Track B ----
NAMES4 = [("lstm_2l", "LSTM-2L"), ("gru_2l", "GRU-2L"), ("bilstm_2l", "BiLSTM-2L"),
          ("tcn", "TCN"), ("transformer_nar", "Transformer-NAR"),
          ("lstm_social_pool", "LSTM+Social-Pool"), ("lstm_env_raster", "LSTM+Env-Raster"),
          ("lstm_env_binary_spatial_attn", "LSTM+EnvBin-SpAttn"),
          ("lstm_env_sdf", "LSTM+Env-SDF"), ("lstm_social_env_sdf", "LSTM+Soc+Env-SDF")]
best4 = {}
for src in ("DMA", "NOAA"):
    avail = [k for k, _ in NAMES4 if k in t4.get(src, {})]
    best4[src] = min(avail, key=lambda k: t4[src][k]["ADE"]["mean"]) if avail else None
rows = []
for k, n in NAMES4:
    c = []
    for src in ("DMA", "NOAA"):
        e = t4.get(src, {}).get(k)
        c += [pm(e["ADE"], k == best4[src]) if e else "---", pm(e["FDE"]) if e else "---"]
    rows.append(f"{n} & {c[0]} & {c[1]} & & {c[2]} & {c[3]} \\\\")
open(os.path.join(HERE, "gen_table4.tex"), "w").write("\n".join(rows) + "\n")

# ---- Table 5: cross-domain matrix ----
POOL_NAMES = [("combined", "Combined (all 4)"), ("loso_dma", "LOSO (no DMA)"),
              ("loso_noaa", "LOSO (no NOAA)"), ("loso_piraeus", "LOSO (no Piraeus)"),
              ("loso_norway", "LOSO (no Norway)")]
NAMES5 = [("tcn", "TCN"), ("lstm_env_sdf", "LSTM+Env-SDF"), ("lstm_social_env_sdf", "LSTM+Soc+Env-SDF")]
HOLD = {"loso_dma": "dma", "loso_noaa": "noaa", "loso_piraeus": "piraeus", "loso_norway": "norway"}
rows = []
for pool, pname in POOL_NAMES:
    block = []
    for k, n in NAMES5:
        e = t5.get(pool, {}).get(k)
        cells = []
        for dom in ("dma", "noaa", "piraeus", "norway"):
            if HOLD.get(pool) == dom:
                cells.append("---")
            elif e is None or f"ADE_src_{dom}" not in e:
                cells.append("---")
            else:
                cells.append(pm(e[f"ADE_src_{dom}"]))
        block.append(f"  & {n} & " + " & ".join(cells) + " \\\\")
    block[0] = f"\\multirow{{3}}{{*}}{{{pname}}}" + block[0][1:]
    rows.append("\n".join(block))
open(os.path.join(HERE, "gen_table5.tex"), "w").write("\n\\midrule\n".join(rows) + "\n")

print("wrote gen_table{3,4,5}.tex")
