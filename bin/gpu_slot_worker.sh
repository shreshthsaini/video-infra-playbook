#!/usr/bin/env bash
# Usage: gpu_slot_worker.sh [-h] PHYSICAL_GPU_INDEX
# Repeatedly claim untimed inner-spool tasks for one physical GPU.
# Environment: PROJECT (default: my-project), GPU_SPOOL_DIR, GPU_SLOT_MAX_IDLE_MIN,
# GPU_SLOT_POLL_SECONDS, GPU_SLOT_FREE_MIB, GPU_SLOT_MAX_TASKS,
# PLAYBOOK_WORKER_STREAM_PREFIX.
set -uo pipefail

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help|'') usage; [ -n "${1:-}" ] && exit 0 || exit 2;; esac
gpu_index=$1
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo "gpu_slot_worker: invalid GPU index" >&2; exit 2; }

PROJECT=${PROJECT:-my-project}
GPU_SPOOL_DIR=${GPU_SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq_gpu}
MAX_IDLE_MIN=${GPU_SLOT_MAX_IDLE_MIN:-15}
POLL_SECONDS=${GPU_SLOT_POLL_SECONDS:-15}
FREE_MIB=${GPU_SLOT_FREE_MIB:-1500}
MAX_TASKS=${GPU_SLOT_MAX_TASKS:-0}
mkdir -p "$GPU_SPOOL_DIR"/{pending,running,done,failed}

host=$(hostname -s)
job=${SLURM_JOB_ID:-x}
proc=${SLURM_PROCID:-0}
idle_seconds=0
tasks_done=0
child=
claim=

cleanup() {
  if [ -n "$child" ]; then
    kill -TERM -- "-$child" 2>/dev/null || true
    for _ in {1..8}; do kill -0 "$child" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    child=
  fi
  if [ -n "$claim" ] && [ -f "$claim" ]; then
    base=$(basename "$claim" | sed 's/\.sh\..*/.sh/')
    mv "$claim" "$GPU_SPOOL_DIR/pending/$base" 2>/dev/null || true
  fi
}
terminate() { cleanup; trap - TERM INT EXIT; exit 130; }
trap terminate TERM INT
trap cleanup EXIT

printf 'GPU_SLOT_WORKER host=%s job=%s proc=%s physical_gpu=%s\n' "$host" "$job" "$proc" "$gpu_index"
while :; do
  [ -f "$GPU_SPOOL_DIR/STOP" ] && break
  (( MAX_TASKS > 0 && tasks_done >= MAX_TASKS )) && break
  used=$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "$used" in ''|*[!0-9]*) used=999999;; esac
  if (( used >= FREE_MIB )); then
    sleep "$POLL_SECONDS"
    idle_seconds=$((idle_seconds + POLL_SECONDS))
    continue
  fi

  claimed=0
  for task in "$GPU_SPOOL_DIR"/pending/*.sh; do
    [ -e "$task" ] || break
    base=$(basename "$task")
    claim=$GPU_SPOOL_DIR/running/${base}.${job}.${proc}.gpu${gpu_index}
    mv "$task" "$claim" 2>/dev/null || { claim=; continue; }
    claimed=1
    idle_seconds=0
    log=$GPU_SPOOL_DIR/done/${base}.log
    printf 'GPU_SLOT_RUN task=%s host=%s physical_gpu=%s\n' "$base" "$host" "$gpu_index"
    (
      export CUDA_VISIBLE_DEVICES=$gpu_index
      export PLAYBOOK_PHYSICAL_GPU_INDEX=$gpu_index
      export PLAYBOOK_WORKER_STREAM_ID=${PLAYBOOK_WORKER_STREAM_PREFIX:-${job}-${proc}}-gpu${gpu_index}
      exec setsid bash "$claim"
    ) >"$log" 2>&1 &
    child=$!
    wait "$child"; rc=$?
    child=
    if [ "$rc" -eq 0 ]; then
      mv "$claim" "$GPU_SPOOL_DIR/done/$base"
      printf 'GPU_SLOT_OK task=%s host=%s physical_gpu=%s\n' "$base" "$host" "$gpu_index"
    else
      mv "$claim" "$GPU_SPOOL_DIR/failed/$base"
      mv "$log" "$GPU_SPOOL_DIR/failed/${base}.log" 2>/dev/null || true
      printf 'GPU_SLOT_FAIL task=%s host=%s physical_gpu=%s rc=%s\n' "$base" "$host" "$gpu_index" "$rc"
    fi
    claim=
    tasks_done=$((tasks_done + 1))
  done
  if [ "$claimed" -eq 0 ]; then
    sleep "$POLL_SECONDS"
    idle_seconds=$((idle_seconds + POLL_SECONDS))
    if (( MAX_IDLE_MIN > 0 && idle_seconds >= MAX_IDLE_MIN * 60 )); then
      printf 'GPU_SLOT_IDLE_EXIT host=%s physical_gpu=%s idle_min=%s\n' "$host" "$gpu_index" "$MAX_IDLE_MIN"
      break
    fi
  fi
done
trap - TERM INT EXIT
cleanup
