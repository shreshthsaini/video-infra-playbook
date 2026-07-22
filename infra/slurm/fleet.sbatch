#!/usr/bin/env bash
# Usage: bash infra/slurm/fleet.sbatch -h
# Submit: sbatch [-N NODES] infra/slurm/fleet.sbatch
# Environment: PROJECT, PLAYBOOK_ROOT, PROJECT_SCRATCH, SPOOL_DIR,
# GPU_SPOOL_DIR, TELEMETRY_DIR, MAX_IDLE_MIN, WANDB_MODE.
# Required: replace __GPU_PARTITION__, __ACCOUNT__, and __SCRATCH_LOG_DIR__.
# Keep the queue declared, telemetry active, and release idle capacity.
#SBATCH --job-name=fleetcraft-fleet
#SBATCH --partition=__GPU_PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=__SCRATCH_LOG_DIR__/%x-%j.out

set -euo pipefail
usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac

PROJECT=${PROJECT:-my-project}
PLAYBOOK_ROOT=${PLAYBOOK_ROOT:-$SLURM_SUBMIT_DIR}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-/scratch/${USER:?}/$PROJECT}
export PROJECT PLAYBOOK_ROOT PROJECT_SCRATCH
export SPOOL_DIR=${SPOOL_DIR:-$PROJECT_SCRATCH/taskq}
export GPU_SPOOL_DIR=${GPU_SPOOL_DIR:-$PROJECT_SCRATCH/taskq_gpu}
export TELEMETRY_DIR=${TELEMETRY_DIR:-$PROJECT_SCRATCH/telemetry}
export MAX_IDLE_MIN=${MAX_IDLE_MIN:-30}
export WANDB_MODE=${WANDB_MODE:-online}
export PLAYBOOK_TELEMETRY_MANAGED=1
mkdir -p "$SPOOL_DIR" "$GPU_SPOOL_DIR" "$TELEMETRY_DIR" "$PROJECT_SCRATCH/logs"

node_worker() {
  set -uo pipefail
  bash "$PLAYBOOK_ROOT/bin/gpu_telemetry.sh" "$TELEMETRY_DIR" 30 &
  telemetry_pid=$!
  cleanup() {
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  }
  trap cleanup TERM INT EXIT
  bash "$PLAYBOOK_ROOT/bin/worker_loop.sh"
  rc=$?
  trap - TERM INT EXIT
  cleanup
  return "$rc"
}
export -f node_worker
srun --ntasks-per-node=1 bash -c node_worker
