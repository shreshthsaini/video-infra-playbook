#!/usr/bin/env bash
# Usage: bash infra/slurm/devnode.sbatch -h
# Submit: sbatch infra/slurm/devnode.sbatch
# Environment: PROJECT, PLAYBOOK_ROOT, PROJECT_SCRATCH, TELEMETRY_DIR.
# Required: replace __GPU_PARTITION__, __ACCOUNT__, and __SCRATCH_LOG_DIR__.
# Keep telemetry active and cancel the job when useful work ends.
#SBATCH --job-name=fleetcraft-dev
#SBATCH --partition=__GPU_PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=__SCRATCH_LOG_DIR__/%x-%j.out

set -euo pipefail
usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac

PROJECT=${PROJECT:-my-project}
PLAYBOOK_ROOT=${PLAYBOOK_ROOT:-$SLURM_SUBMIT_DIR}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-/scratch/${USER:?}/$PROJECT}
TELEMETRY_DIR=${TELEMETRY_DIR:-$PROJECT_SCRATCH/telemetry}
mkdir -p "$PROJECT_SCRATCH/logs" "$TELEMETRY_DIR"
bash "$PLAYBOOK_ROOT/bin/gpu_telemetry.sh" "$TELEMETRY_DIR" 30 &
telemetry_pid=$!
cleanup() {
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup TERM INT EXIT
printf 'DEVNODE_READY host=%s job=%s\n' "$(hostname -s)" "$SLURM_JOB_ID"
while sleep 60; do
  date -Is >"$PROJECT_SCRATCH/logs/devnode-${SLURM_JOB_ID}.heartbeat"
done
