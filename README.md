# EnvShip

Code for EnvShip, a framework for context-aware and cross-region vessel trajectory
forecasting. It covers the full path from raw AIS to trained models: the construction
pipeline that turns raw messages into fixed prediction windows, the environmental and
social context builders, and the training/evaluation framework with 21 reference models
across four jurisdictions (DMA, NOAA, Piraeus, Norway) and two horizons.

Two tracks:

- Track A: 10 min observation, 10 min prediction (30 + 30 points at 20 s).
- Track B: 30 min observation, 60 min prediction (90 + 180 points at 20 s).

Trajectories are in vessel-centric metric coordinates. Every window is paired with a
coastline raster and signed-distance field (±5 km, cropped at the anchor), a 3 km social
neighborhood with CPA/TCPA descriptors, and anchor-time weather and sea-state scalars.

## Layout

```
eval/               training + evaluation framework
  train_dl.py         deep models: train + test on one source/track
  train_ml.py         Random Forest / XGBoost baselines
  train_k3.py         K=3 multimodal head (Track B)
  run_baselines.py    physics baselines (CV / DR / CA)
  eval_loso_holdout.py    leave-one-source-out transfer
  zero_shot_transfer.py   zero-shot cross-region
  dataset.py, context_dataset.py, multi_source_dataset.py   loaders
  normalizer.py, metrics.py, lazy_context.py                utilities
  paths.py            data-root resolution (ENVSHIP_DATA_ROOT)
  models/             model implementations + MODEL_REGISTRY
  baselines/          physics + reference sequence baselines
build/              raw AIS -> core -> track construction
  stages/             01..14 staged pipeline (Track A / core)
  track_b/            Track B variant (30/60 min windows)
  context/            environment + social context, OSM download, weather
  build_standard_track_v1.py   core -> track windows
  pipeline_utils.py
tables/             result aggregation (5-seed mean/std, LaTeX bodies)
scripts/            run launchers
dataset/            expected data layout + pointer to Hugging Face (no data here)
docs/               pipeline notes
```

## Install

```bash
git clone https://github.com/mark000071/Envship_Framework_vessel_traj_pred.git
cd Envship_Framework_vessel_traj_pred
pip install -r requirements.txt
export ENVSHIP_DATA_ROOT=/path/to/envship_v2_datasets
```

Data is hosted separately on Hugging Face:
https://huggingface.co/datasets/mark000071/envship_v2_datasets

Arrange the download as `track_{a,b}/{dma,noaa,piraeus,norway}/standard_track_v1/`; see
`eval/paths.py` for the exact rules.

## Training and evaluation

Run from the repo root (the package is `eval`):

```bash
# one deep model on DMA Track A
python -m eval.train_dl --model lstm_env_sdf --seed 1 --epochs 80 --min-epochs 32

# another source / track
python -m eval.train_dl --model tcn --seed 1 \
    --track-root "$ENVSHIP_DATA_ROOT/track_b/noaa/standard_track_v1"

# classical-ML and physics baselines
python -m eval.train_ml
python -m eval.run_baselines

# cross-region transfer
python -m eval.eval_loso_holdout --holdout piraeus
python -m eval.zero_shot_transfer

# multimodal K=3 head (Track B)
python -m eval.train_k3 --model lstm_env_sdf --seed 1
```

`--min-epochs` suspends early stopping until the teacher-forcing anneal ends; without it,
some seeds stop in a worse basin. Each learned model is run under seeds 1-5 and reported as
mean/std; `tables/aggregate_multiseed.py` collects the per-seed results and
`tables/make_tables.py` renders the LaTeX bodies. `scripts/run.sh` wraps a small end-to-end
example.

Models (keys in `eval/models/__init__.py`):

| Family | Keys |
|---|---|
| Trajectory-only | lstm_2l, gru_2l, bilstm_2l, tcn, transformer_nar |
| Social | lstm_social_pool, lstm_social_attn, lstm_social_attn_v5 |
| Environmental | lstm_env_raster, lstm_env_desc, lstm_env_desc_v2, lstm_env_binary_spatial_attn, lstm_env_sdf, lstm_env_spatial_attn |
| Social + Env | lstm_social_env_v2, lstm_social_env_sdf |

## Dataset construction

`build/` reproduces the dataset from raw AIS. The staged pipeline under `build/stages/`
runs in order:

```
01 standardize fields      08 resample to 20 s
02 basic filter            09 second-pass anomaly check
03 sort + dedup            10 filter short segments
04 shiptype + speed        11 make sliding windows
05 segment tracks          12 quality labels
06 remove anchorage        13 export benchmark
07 interpolate short gaps  14 partition summaries
```

Stages 01-10 take raw messages to clean 20 s tracks (the core); 11-14 cut the prediction
windows and export the benchmark. `build/track_b/` is the 30/60 min variant.
`build/context/` builds the per-window context: `prefetch_osm_parallel.py` and
`smart_osm_filler.py` fetch OSM tiles, `build_standard_track_context_v1.py` rasterizes the
coastline and computes the signed-distance fields and social neighbors, and
`build_meteo_features.py` attaches ERA5 weather and wave state. `stage_17_osm_temporal_consistency.py`
flags windows whose trajectory leaves the water on the OSM snapshot used for the context.

The build scripts use absolute paths from the original run and are meant as a reference for
how the released data was produced, not as a turnkey rebuild. See `docs/PIPELINE.md`.

## License

MIT (see `LICENSE`). The dataset carries its own composite terms on Hugging Face.
