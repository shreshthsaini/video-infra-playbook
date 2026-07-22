#!/usr/bin/env bash
# Usage: worker_loop.sh [-h]
# Drain an atomic node-level task spool, optionally with untimed scoring slots.
# Environment: PROJECT (default: my-project), SPOOL_DIR, MAX_IDLE_MIN,
# POLL_SECONDS, SLOTS, UNTIMED_ONLY, UNTIMED_REGEX, REAP_EVERY_LOOPS,
# PLAYBOOK_ACTIVE_JOBS, SLURM_USER.
set -euo pipefail

usage() {
  sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-my-project}
SPOOL_DIR=${SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq}
MAX_IDLE_MIN=${MAX_IDLE_MIN:-30}
POLL_SECONDS=${POLL_SECONDS:-60}
SLOTS=${SLOTS:-1}
UNTIMED_ONLY=${UNTIMED_ONLY:-0}
UNTIMED_REGEX=${UNTIMED_REGEX:-'(^|[-_.])(score|eval|index|analysis|rescore)([-_.]|$)'}
REAP_EVERY_LOOPS=${REAP_EVERY_LOOPS:-10}

for value in "$MAX_IDLE_MIN" "$POLL_SECONDS" "$SLOTS" "$REAP_EVERY_LOOPS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "worker_loop: numeric setting required" >&2; exit 2; }
done
(( SLOTS >= 1 && POLL_SECONDS >= 1 && REAP_EVERY_LOOPS >= 1 )) || exit 2

mkdir -p "$SPOOL_DIR"/{pending,running,done,failed,names}
worker_children=()
cleanup_workers() {
  local pid
  for pid in "${worker_children[@]:-}"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${worker_children[@]:-}"; do
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
  done
}
terminate_workers() {
  cleanup_workers
  trap - TERM INT EXIT
  exit 130
}
trap terminate_workers TERM INT
trap cleanup_workers EXIT

if (( SLOTS > 1 )) && [ -z "${WORKER_SLOT:-}" ]; then
  for ((slot = 2; slot <= SLOTS; slot++)); do
    WORKER_SLOT=$slot SLOTS=1 UNTIMED_ONLY=1 bash "$0" &
    worker_children+=("$!")
  done
  export WORKER_SLOT=1 SLOTS=1 UNTIMED_ONLY=1
fi

host=$(hostname -s)
job=${SLURM_JOB_ID:-x}
proc=${SLURM_PROCID:-0}
slot=${WORKER_SLOT:-1}
printf 'WORKER_LOOP host=%s job=%s proc=%s slot=%s untimed_only=%s spool=%s\n' \
  "$host" "$job" "$proc" "$slot" "$UNTIMED_ONLY" "$SPOOL_DIR"

reap_stale() {
  local active_raw active_jobs claim file owner base target
  if [ -n "${PLAYBOOK_ACTIVE_JOBS+x}" ]; then
    active_jobs=$PLAYBOOK_ACTIVE_JOBS
  else
    if ! active_raw=$(squeue -u "${SLURM_USER:-$USER}" -h -o %A 2>/dev/null); then
      echo "WORKER_REAP skipped: scheduler query failed" >&2
      return 0
    fi
    active_jobs=$(printf '%s\n' "$active_raw" | sort -u | tr '\n' ' ')
  fi
  for claim in "$SPOOL_DIR"/running/*.sh.*; do
    [ -e "$claim" ] || break
    file=$(basename "$claim")
    owner=$(printf '%s\n' "$file" | sed -nE 's/^.*\.sh\.([0-9]+)\..*$/\1/p')
    [ -n "$owner" ] || continue
    case " $active_jobs " in *" $owner "*) continue ;; esac
    base=$(printf '%s\n' "$file" | sed 's/\.sh\..*/.sh/')
    target=$SPOOL_DIR/pending/$base
    if [ -e "$target" ] || [ -e "$SPOOL_DIR/done/$base" ] || [ -e "$SPOOL_DIR/failed/$base" ]; then
      mv "$claim" "$SPOOL_DIR/failed/${base}.orphan-${owner}-$(date +%s)"
    else
      mv "$claim" "$target"
      printf 'WORKER_REAP recovered=%s owner_job=%s\n' "$base" "$owner"
    fi
  done
}

idle_seconds=0
loop_count=0
while :; do
  loop_count=$((loop_count + 1))
  if (( loop_count % REAP_EVERY_LOOPS == 1 )); then reap_stale; fi
  claimed=0
  for task in "$SPOOL_DIR"/pending/*.sh; do
    [ -e "$task" ] || break
    base=$(basename "$task")
    if [ "$UNTIMED_ONLY" = 1 ] && ! [[ "$base" =~ $UNTIMED_REGEX ]]; then continue; fi
    claim="$SPOOL_DIR/running/${base}.${job}.${proc}.${slot}"
    mv "$task" "$claim" 2>/dev/null || continue
    claimed=1
    idle_seconds=0
    log="$SPOOL_DIR/done/${base}.log"
    printf 'WORKER_RUN task=%s host=%s time=%s\n' "$base" "$host" "$(date '+%F %T')"
    if bash "$claim" >"$log" 2>&1; then
      mv "$claim" "$SPOOL_DIR/done/$base"
      printf 'WORKER_OK task=%s host=%s\n' "$base" "$host"
    else
      rc=$?
      mv "$claim" "$SPOOL_DIR/failed/$base"
      mv "$log" "$SPOOL_DIR/failed/${base}.log" 2>/dev/null || true
      printf 'WORKER_FAIL task=%s host=%s rc=%s\n' "$base" "$host" "$rc"
    fi
  done
  if [ "$claimed" -eq 0 ]; then
    sleep "$POLL_SECONDS"
    idle_seconds=$((idle_seconds + POLL_SECONDS))
    if (( MAX_IDLE_MIN > 0 && idle_seconds >= MAX_IDLE_MIN * 60 )); then
      printf 'WORKER_IDLE_EXIT host=%s idle_min=%s\n' "$host" "$MAX_IDLE_MIN"
      break
    fi
  fi
done

trap - TERM INT EXIT
cleanup_workers
