#!/usr/bin/env bash
# Usage: bash scripts/run_in_tmux.sh <session_name> <run_dir> "<command>"
set -e
SESSION="$1"
RUN_DIR="$2"
CMD="$3"
if [ -z "$SESSION" ] || [ -z "$RUN_DIR" ] || [ -z "$CMD" ]; then
  echo "Usage: $0 <session_name> <run_dir> \"<command>\""
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Choose another name or kill it."
  exit 1
fi
mkdir -p "$RUN_DIR"
tmux new-session -d -s "$SESSION" \
  "($CMD) > >(tee $RUN_DIR/stdout.log) 2> >(tee $RUN_DIR/stderr.log >&2); \
   date -Iseconds > $RUN_DIR/end_time.txt; \
   exec bash"
echo "Started tmux session: $SESSION"
echo "Run dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
echo "Tail logs:   tail -f $RUN_DIR/stdout.log"
