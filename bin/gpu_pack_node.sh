#!/usr/bin/env bash
# Usage: gpu_pack_node.sh [-h]
# Dedicate one exclusive outer claim to packed untimed inner-spool work.
# Environment: PROJECT (default: my-project), PLAYBOOK_ROOT,
# PLAYBOOK_PROJECT_ROOT, PLAYBOOK_ENV_FILE, GPU_SPOOL_DIR, TELEMETRY_DIR,
# GPU_WAIT_SECONDS, GPU_SLOT_MAX_TASKS, PLAYBOOK_TELEMETRY_MANAGED.
set -uo pipefail

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac

PROJECT=${PROJECT:-my-project}
PLAYBOOK_ROOT=${PLAYBOOK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PLAYBOOK_PROJECT_ROOT=${PLAYBOOK_PROJECT_ROOT:-$PWD}
PLAYBOOK_ENV_FILE=${PLAYBOOK_ENV_FILE:-}
GPU_SPOOL_DIR=${GPU_SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq_gpu}
TELEMETRY_DIR=${TELEMETRY_DIR:-/scratch/${USER:?}/$PROJECT/telemetry}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-600}
[ -n "$PLAYBOOK_PROJECT_ROOT" ] && cd "$PLAYBOOK_PROJECT_ROOT"
[ -z "$PLAYBOOK_ENV_FILE" ] || source "$PLAYBOOK_ENV_FILE"

gpu_count=$(nvidia-smi -L 2>/dev/null | awk 'END {print NR+0}')
(( gpu_count >= 1 )) || { echo "gpu_pack_node: no GPUs visible" >&2; exit 75; }
waited=0
while :; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 >= 1500 {n++} END {print n+0}')
  [ "$busy" -eq 0 ] && break
  (( waited >= GPU_WAIT_SECONDS )) && { echo "gpu_pack_node: GPUs stayed busy" >&2; exit 75; }
  sleep 15
  waited=$((waited + 15))
done

telemetry_pid=
if [ "${PLAYBOOK_TELEMETRY_MANAGED:-0}" != 1 ]; then
  bash "$PLAYBOOK_ROOT/bin/gpu_telemetry.sh" "$TELEMETRY_DIR" 30 >/dev/null 2>&1 &
  telemetry_pid=$!
fi
declare -a slot_pids=()
stopping=0
start_slot() {
  local gpu=$1
  GPU_SPOOL_DIR=$GPU_SPOOL_DIR bash "$PLAYBOOK_ROOT/bin/gpu_slot_worker.sh" "$gpu" &
  slot_pids[$gpu]=$!
}
kill_bounded() {
  local pid
  for pid in "$@"; do [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true; done
  for _ in {1..8}; do
    alive=0
    for pid in "$@"; do [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  for pid in "$@"; do [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true; done
  for pid in "$@"; do [ -n "$pid" ] && wait "$pid" 2>/dev/null || true; done
}
terminate() { stopping=1; kill_bounded "${slot_pids[@]:-}"; }
cleanup() {
  terminate
  if [ -n "$telemetry_pid" ]; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap 'cleanup; exit 130' TERM INT
trap cleanup EXIT

for ((gpu = 0; gpu < gpu_count; gpu++)); do start_slot "$gpu"; done
printf 'GPU_PACK_NODE host=%s slots=%s\n' "$(hostname -s)" "$gpu_count"
rc=0
while :; do
  [ "$stopping" -eq 1 ] && break
  alive=0
  for ((gpu = 0; gpu < gpu_count; gpu++)); do
    pid=${slot_pids[$gpu]:-}
    if [ -n "$pid" ]; then
      state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
      case "$state" in
        ''|*Z*) wait "$pid" || rc=1; slot_pids[$gpu]= ;;
        *) alive=$((alive + 1)) ;;
      esac
    fi
  done
  pending=$(find "$GPU_SPOOL_DIR/pending" -maxdepth 1 -type f -name '*.sh' 2>/dev/null | awk 'END {print NR+0}')
  if [ "$alive" -eq 0 ] && { [ "$pending" -eq 0 ] || [ "${GPU_SLOT_MAX_TASKS:-0}" -gt 0 ]; }; then break; fi
  if [ "$pending" -gt 0 ] && [ "${GPU_SLOT_MAX_TASKS:-0}" -eq 0 ]; then
    for ((gpu = 0; gpu < gpu_count; gpu++)); do
      [ -n "${slot_pids[$gpu]:-}" ] || start_slot "$gpu"
    done
  fi
  sleep 5
done
trap - TERM INT EXIT
cleanup
printf 'GPU_PACK_EXIT host=%s rc=%s\n' "$(hostname -s)" "$rc"
exit "$rc"
