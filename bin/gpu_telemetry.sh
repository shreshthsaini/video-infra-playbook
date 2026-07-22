#!/usr/bin/env bash
# Usage: gpu_telemetry.sh [outdir] [interval-seconds]
# Write per-host nvidia-smi CSV samples until signalled.
# Environment: PROJECT (default: my-project), TELEMETRY_DIR,
# TELEMETRY_INTERVAL.
set -uo pipefail

usage() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

PROJECT=${PROJECT:-my-project}
TELEMETRY_DIR=${1:-${TELEMETRY_DIR:-/scratch/${USER:?}/$PROJECT/telemetry}}
INTERVAL=${2:-${TELEMETRY_INTERVAL:-30}}
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "gpu_telemetry: interval must be positive" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "gpu_telemetry: nvidia-smi not found" >&2; exit 127; }

mkdir -p "$TELEMETRY_DIR"
host=$(hostname -s)
output=$TELEMETRY_DIR/${host}_job${SLURM_JOB_ID:-na}_$(date +%Y%m%d%H%M%S).csv
printf '%s\n' 'timestamp,mem_used_mib,mem_total_mib,util_pct,power_w' >"$output"
printf 'GPU_TELEMETRY output=%s interval=%s\n' "$output" "$INTERVAL"
while :; do
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g' >>"$output" || true
  sleep "$INTERVAL"
done
