#!/usr/bin/env bash
# Usage: reap_gpu_spool.sh [-h]
# Recover inner-spool claims whose numeric owner job is absent from Slurm.
# Fail closed on scheduler transport errors.
# Environment: PROJECT (default: my-project), GPU_SPOOL_DIR,
# PLAYBOOK_ACTIVE_JOBS.
set -euo pipefail

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac
PROJECT=${PROJECT:-my-project}
GPU_SPOOL_DIR=${GPU_SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq_gpu}
mkdir -p "$GPU_SPOOL_DIR"/{pending,running,done,failed}

if [ -n "${PLAYBOOK_ACTIVE_JOBS+x}" ]; then
  active_jobs=$PLAYBOOK_ACTIVE_JOBS
else
  if ! active_raw=$(squeue -u "${SLURM_USER:-$USER}" -h -o %A 2>/dev/null); then
    echo "GPU_SPOOL_REAP skipped: scheduler query failed" >&2
    exit 1
  fi
  active_jobs=$(printf '%s\n' "$active_raw" | sort -u | tr '\n' ' ')
fi

recovered=0
duplicates=0
stamp=$(date +%s)
for claim in "$GPU_SPOOL_DIR"/running/*.sh.*; do
  [ -e "$claim" ] || break
  file=$(basename "$claim")
  owner=$(printf '%s\n' "$file" | sed -nE 's/^.*\.sh\.([0-9]+)\..*$/\1/p')
  [ -n "$owner" ] || continue
  case " $active_jobs " in *" $owner "*) continue;; esac
  base=$(printf '%s\n' "$file" | sed 's/\.sh\..*/.sh/')
  if [ -e "$GPU_SPOOL_DIR/pending/$base" ] || [ -e "$GPU_SPOOL_DIR/done/$base" ] || [ -e "$GPU_SPOOL_DIR/failed/$base" ]; then
    mv "$claim" "$GPU_SPOOL_DIR/failed/${base}.orphan-${owner}-${stamp}"
    duplicates=$((duplicates + 1))
  else
    mv "$claim" "$GPU_SPOOL_DIR/pending/$base"
    recovered=$((recovered + 1))
  fi
done
printf 'GPU_SPOOL_REAP recovered=%s duplicate_orphans=%s\n' "$recovered" "$duplicates"
