# Construction pipeline

The dataset is built from raw AIS in two phases: cleaning raw messages into resampled
tracks (the core), then cutting prediction windows and attaching context. All stages live
under `build/`.

## Stages (build/stages/)

| Stage | What it does |
|-------|--------------|
| 01 standardize_fields | map source columns to a common schema (mmsi, lat, lon, sog, cog, ts, ship_type) |
| 02 basic_filter | drop rows with invalid coordinates, timestamps, or speeds |
| 03 sort_dedup | sort by (mmsi, ts) and drop duplicate reports |
| 04 shiptype_speed_filter | keep the target ship classes; cap implausible speeds |
| 05 segment_tracks | split each vessel into continuous voyages on time/gap breaks |
| 06 remove_anchorage | drop stationary anchorage/moored spans |
| 07 interpolate_short_gaps | linearly fill short reporting gaps |
| 08 resample_20s | resample every voyage to a fixed 20 s grid |
| 09 second_pass_anomaly_check | remove residual kinematic outliers |
| 10 filter_short_segments | drop voyages too short for a full window |
| 11 make_sliding_windows | cut fixed observation/prediction windows |
| 12 compute_quality_labels | difficulty / scene / neighbor-density labels per window |
| 13 export_benchmark | write train/val/test with vessel-disjoint splits |
| 14 collect_partition_summaries | per-split counts and coverage stats |

Stages 01-10 produce the core (clean 20 s tracks); 11-14 produce the benchmark windows.

## Track B (build/track_b/)

Same structure with 30 min / 60 min windows. `stage_10b_track_b_filter.py` applies the
extra segment-duration prefilter needed for the longer horizon. `00_preprocess_piraeus.py`
adapts the Piraeus raw format before stage 01.

## Context (build/context/)

- `prefetch_osm_parallel.py`, `smart_osm_filler.py` — download and backfill OSM coastline
  tiles for the regions covered by the anchors.
- `build_standard_track_context_v1.py` — for each window, crop a ±5 km patch at the anchor,
  rasterize the coastline (128×128), compute the shore/navigable signed-distance fields, and
  gather up to 10 neighbors within 3 km with CPA/TCPA.
- `build_meteo_features.py` — attach anchor-time ERA5 weather and wave-state scalars.
- `stage_17_osm_temporal_consistency.py` — flag windows whose trajectory crosses land on the
  OSM snapshot used to build the context (guards against port geometry that post-dates the AIS).

The context patch is fixed at the anchor and does not move with the prediction, so its
usefulness is bounded by how long the vessel stays inside ±5 km (nearly the whole Track A
horizon, a fraction of Track B).

## Notes

The scripts carry absolute paths from the original build host. They document how the release
was produced and are not a one-command rebuild; point them at your own raw AIS and output
directories to re-run.
