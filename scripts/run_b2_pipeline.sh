#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
run_dir="${1:?Pass an absolute B2 run directory}"
budget="${2:-60000000}"
task_version="${B2_TASK_VERSION:-b2_payload_v1}"
exploration_profile="${B2_EXPLORATION_PROFILE:-standard}"
mkdir -p "$run_dir"
child_pid=""
stop_pipeline() {
  if [ -n "$child_pid" ]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" || true
  fi
  exit 0
}
trap stop_pipeline INT TERM
resume=()
if [ -f "$run_dir/checkpoints/latest.pkl" ]; then
  resume+=(--resume)
  task_version="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_version"])' "$run_dir/config.json")"
  exploration_profile="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("exploration_profile","standard"))' "$run_dir/config.json")"
fi
if [ ${#resume[@]} -eq 0 ] && [ -n "${3:-}" ]; then resume+=(--warm-start "$3"); fi
while true; do
  .venv/bin/python -u -m stairs.train_taishan --run "$run_dir" --task-version "$task_version" --exploration-profile "$exploration_profile" --num-envs 512 --horizon 64 --start-stage "${B2_START_STAGE:-0}" --max-stage-steps "${B2_MAX_STAGE_STEPS:-12000000}" --steps "$budget" "${resume[@]}" >> "$run_dir/train.log" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$run_dir/train.pid"
  training_exit=0
  wait "$child_pid" || training_exit=$?
  child_pid=""
  if [ "$training_exit" -ne 0 ]; then break; fi
  stop_reason="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$run_dir/training_finished.json")"
  if [ "$stop_reason" != "curriculum_restart" ]; then break; fi
  resume=(--resume)
done
printf '%s\n' "$training_exit" > "$run_dir/training_exit_code.txt"
if [ "$training_exit" -eq 0 ]; then
  .venv/bin/python -u scripts/evaluate_taishan.py "$run_dir/checkpoints/final.pkl" --task-version "$task_version" --output "$run_dir/evaluation_target" > "$run_dir/evaluation.log" 2>&1 &
  child_pid=$!
  evaluation_exit=0
  wait "$child_pid" || evaluation_exit=$?
  child_pid=""
  printf '%s\n' "$evaluation_exit" > "$run_dir/evaluation_exit_code.txt"
fi
bash scripts/postprocess_taishan_run.sh "$run_dir" > "$run_dir/render.log" 2>&1 &
child_pid=$!
wait "$child_pid"
child_pid=""
printf '%s\n' 'B2 pipeline finished. Inspect exit codes and independent evaluation; process completion does not imply acceptance.' > "$run_dir/pipeline_finished.txt"
