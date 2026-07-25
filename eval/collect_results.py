#!/usr/bin/env python3
"""Collect all result JSONs and produce a combined comparison table."""

from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RES_DIR   = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/results")
OLD_RJSON = Path("/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/eval/baseline_results.json")


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

def load_all() -> list[dict]:
    results = []
    # Old analytic baselines
    if OLD_RJSON.exists():
        old = json.loads(OLD_RJSON.read_text())
        seen = set()
        for r in old:
            nm = r.get("model","")
            if nm == "Constant Velocity" and nm not in seen:
                r2 = dict(r); r2["model"] = "CV (w=3)"; results.append(r2)
                seen.add(nm)
            elif nm in ("Constant Acceleration", "Dead Reckoning"):
                results.append(r)
            # Skip broken Transformer from old run
    # New results
    _SKIP = {"all_results.json", "social_dense_eval.json", "results_table.md"}
    for jpath in sorted(RES_DIR.glob("*.json")):
        if jpath.name in _SKIP: continue
        try:
            r = json.loads(jpath.read_text())
            results.append(r)
        except Exception as e:
            print(f"  [warn] {jpath.name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    # Analytic
    "CV (w=3)", "Dead Reckoning", "Constant Acceleration",
    # ML
    "Random Forest", "XGBoost",
    # Sequence DL
    "LSTM-2L", "LSTM-4L", "GRU-2L", "BiLSTM-2L", "TCN",
    "Transformer-3L", "Transformer-6L",
    # TrAISformer
    "TrAISformer (local)",
    # Context
    "LSTM+Social-Pool", "LSTM+Social-Attn",
    "LSTM+Env-Desc", "LSTM+Env-Raster",
    "LSTM+Social+Env",
]


def sort_key(r: dict) -> int:
    nm = r.get("model","")
    try:
        return MODEL_ORDER.index(nm)
    except ValueError:
        return 999


def print_table(results: list[dict]) -> str:
    results = sorted(results, key=sort_key)
    rows = []
    header = f"{'Model':<28} {'ADE':>7} {'FDE':>7} {'@3m':>6} {'@6m':>6} | {'Easy':>7} {'Med':>7} {'Hard':>7} | {'O.Water':>7} {'NearSh':>7} {'Harbor':>7}"
    sep    = "-" * len(header)
    rows.append(header); rows.append(sep)
    for r in results:
        nm  = r.get("model","?")[:27]
        ade = r.get("ADE","-"); fde = r.get("FDE","-")
        a3  = r.get("ADE_3min","-"); a6 = r.get("ADE_6min","-")
        ea  = r.get("ADE_easy","-"); me = r.get("ADE_medium","-"); ha = r.get("ADE_hard","-")
        ow  = r.get("ADE_open_water","-"); ns = r.get("ADE_nearshore","-"); hb = r.get("ADE_harbor","-")
        rows.append(f"{nm:<28} {ade:>7} {fde:>7} {a3:>6} {a6:>6} | {ea:>7} {me:>7} {ha:>7} | {ow:>7} {ns:>7} {hb:>7}")
    return "\n".join(rows)


def build_markdown(results: list[dict]) -> str:
    results = sorted(results, key=sort_key)
    cols = ["model","ADE","FDE","ADE_3min","ADE_6min",
            "ADE_easy","ADE_medium","ADE_hard",
            "ADE_open_water","ADE_nearshore","ADE_harbor",
            "ADE_cargo_tanker","ADE_fishing","ADE_sailing_leisure","ADE_passenger_ferry"]
    hdr = "| " + " | ".join(c.replace("ADE_","") for c in cols) + " |"
    sep = "|" + "|".join([":---:"]*(len(cols))) + "|"
    rows = [hdr, sep]
    for r in results:
        vals = [str(r.get(c,"-")) for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all()
    print(f"\nCollected {len(results)} model results\n")
    print(print_table(results))
    md = build_markdown(results)
    print("\n\n=== MARKDOWN TABLE ===")
    print(md)

    combined = {"results": results}
    out = RES_DIR / "all_results.json"
    out.write_text(json.dumps(combined, indent=2))
    print(f"\nSaved to {out}")

    # Also write a markdown file
    md_out = RES_DIR / "results_table.md"
    md_out.write_text(f"# EnvShip-Bench Baseline Results\n\n{md}\n")
    print(f"Markdown table saved to {md_out}")
