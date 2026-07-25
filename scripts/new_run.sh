#!/usr/bin/env bash
set -e
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="experiments/runs/$TS"
mkdir -p "$RUN_DIR/checkpoints"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
python3 -m pip freeze > "$RUN_DIR/env.txt" 2>/dev/null || pip freeze > "$RUN_DIR/env.txt" 2>/dev/null || true
date -Iseconds > "$RUN_DIR/start_time.txt"
hostname > "$RUN_DIR/host.txt"
nvidia-smi -L > "$RUN_DIR/gpu.txt" 2>/dev/null || echo "no gpu" > "$RUN_DIR/gpu.txt"
if [ -n "$1" ] && [ -f "$1" ]; then
  cp "$1" "$RUN_DIR/config.yaml"
fi
echo "$RUN_DIR"
