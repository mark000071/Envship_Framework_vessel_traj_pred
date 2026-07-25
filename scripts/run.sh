#!/usr/bin/env bash
# Minimal end-to-end example: train a few models on DMA Track A and print results.
# Assumes ENVSHIP_DATA_ROOT points at the downloaded dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ENVSHIP_DATA_ROOT:?set ENVSHIP_DATA_ROOT to the dataset root}"

SEED="${SEED:-1}"
MODELS="${MODELS:-tcn lstm_env_sdf lstm_social_env_sdf}"

for m in $MODELS; do
  echo "=== $m (seed $SEED) ==="
  python -m eval.train_dl --model "$m" --seed "$SEED" \
      --epochs 80 --min-epochs 32 --batch-size 256 --lr 5e-4
done

# physics + classical-ML reference points
python -m eval.run_baselines
python -m eval.train_ml
