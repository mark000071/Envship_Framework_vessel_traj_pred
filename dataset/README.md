# Dataset

The data is hosted on Hugging Face, not in this repository:

https://huggingface.co/datasets/mark000071/envship_v2_datasets

Download it and point the code at it with `ENVSHIP_DATA_ROOT` (see `../eval/paths.py`).
This folder only documents the expected layout so the training and evaluation scripts
find each split and its context.

## Layout

```
$ENVSHIP_DATA_ROOT/
  track_a/                         10 min -> 10 min
    dma/standard_track_v1/
    noaa/standard_track_v1/
    piraeus/standard_track_v1/
    norway/standard_track_v1/
  track_b/                         30 min -> 60 min
    dma/standard_track_v1/
    noaa/standard_track_v1/
    piraeus/standard_track_v1/
    norway/standard_track_v1/
```

Each `standard_track_v1/` holds:

```
train/  val/  test/      part-000.csv.gz   prediction windows + inline flag columns
context_v1/              environmental rasters / SDF and social neighbor tables
  augmented/               windows merged with context features
  environment/rasters/     128x128 coastline masks + shore/nav signed-distance fields
  environment/features/    scalar scene descriptors
  social/                  neighbor descriptors and snapshot buckets
```

Trajectories are stored in vessel-centric metric coordinates. The environment patch is a
128x128 grid over a +/-5 km window at the anchor (78 m/px); the social context covers a 3 km
radius with up to 10 neighbors. See the dataset card on Hugging Face for the full column
reference, licensing, and quality flags.
