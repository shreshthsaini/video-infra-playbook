#!/usr/bin/env bash
# Usage: allsmi_view.sh [extra all-smi flags, e.g. -i 2]
# Live all-smi view across every node of every running Slurm job you own.
#
# Why this wrapper exists: all-smi does NOT discover Slurm allocations on its
# own; you must hand it the node list. Two field lessons are encoded here:
#  - Node discovery uses squeue plus scontrol, which work from login AND
#    compute nodes.
#  - Clusters recycle node names. A recycled name returns with a different
#    SSH host key, and all-smi's default strict host-key checking then
#    SILENTLY drops that node from the view (the symptom reads as "not all
#    my GPUs show up"). On a trusted intra-cluster fabric we accept any key.
#  - A per-node connect timeout keeps one dead or slow node from stalling
#    the whole view.
set -o pipefail

command -v all-smi >/dev/null || { echo "all-smi not found on PATH (https://github.com/lablup/all-smi)" >&2; exit 1; }

NODES=$(squeue -u "$USER" -h -t R -o %N 2>/dev/null \
          | xargs -n1 scontrol show hostnames 2>/dev/null \
          | sort -u)
[ -z "$NODES" ] && { echo "No running jobs (no allocated nodes)."; exit 1; }

TARGETS=$(echo "$NODES" | sed "s/^/$USER@/" | paste -sd,)

exec all-smi view \
  --ssh "$TARGETS" \
  --ssh-strict-host-key no \
  --ssh-timeout-secs 8 \
  "$@"
