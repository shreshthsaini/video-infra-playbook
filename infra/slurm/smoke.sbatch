#!/usr/bin/env bash
# Usage: bash infra/slurm/smoke.sbatch -h
# Submit: sbatch infra/slurm/smoke.sbatch
# Environment: PROJECT, PLAYBOOK_ROOT, PROJECT_ROOT, PROJECT_SCRATCH,
# TELEMETRY_DIR, PLAYBOOK_SMOKE_COMMAND.
# Required: replace __GPU_PARTITION__, __ACCOUNT__, and __SCRATCH_LOG_DIR__.
# Calibrate a full cycle and reserve reported timings for batch-one passes.
#SBATCH --job-name=fleetcraft-smoke
#SBATCH --partition=__GPU_PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=__SCRATCH_LOG_DIR__/%x-%j.out

set -euo pipefail
usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac

PROJECT=${PROJECT:-my-project}
PLAYBOOK_ROOT=${PLAYBOOK_ROOT:-$SLURM_SUBMIT_DIR}
PROJECT_ROOT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-/scratch/${USER:?}/$PROJECT}
TELEMETRY_DIR=${TELEMETRY_DIR:-$PROJECT_SCRATCH/telemetry}
PLAYBOOK_SMOKE_COMMAND=${PLAYBOOK_SMOKE_COMMAND:-nvidia-smi -L}
mkdir -p "$PROJECT_SCRATCH/logs" "$TELEMETRY_DIR"
bash "$PLAYBOOK_ROOT/bin/gpu_telemetry.sh" "$TELEMETRY_DIR" 30 &
telemetry_pid=$!
cleanup() {
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup TERM INT EXIT
cd "$PROJECT_ROOT"
bash -lc "$PLAYBOOK_SMOKE_COMMAND"
