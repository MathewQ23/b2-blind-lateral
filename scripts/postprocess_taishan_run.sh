#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
run_dir="${1:?Pass a completed run directory}"
if [ -f "$run_dir/recording_disabled.json" ]; then
  printf '%s\n' 'Postprocessing disabled by user for this run; existing files preserved.'
  exit 0
fi
test -f "$run_dir/training_finished.json"
.venv/bin/python scripts/check_history.py "$run_dir" --read-all > "$run_dir/history_integrity.json"
for stride in 50; do
  for world in 0 1 2 3; do
    .venv/bin/python scripts/render_selected_history.py "$run_dir" --world "$world" --stride "$stride"
  done
done
printf '%s\n' 'Four 50x actual-training overview videos exported (full original-speed export dropped by user request; raw trajectories retained); this does not imply policy acceptance.' > "$run_dir/videos_finished.txt"
