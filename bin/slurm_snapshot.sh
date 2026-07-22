#!/usr/bin/env bash
# Usage: slurm_snapshot.sh [-h] [--user NAME] [--gpu] [--cluster]
# Report assigned and pending jobs, live partition QOS limits, and optionally
# classify GPUs through one bounded srun step per owned allocation.
# Environment: PLAYBOOK_PARTITION, PLAYBOOK_GPU_PARTITION_RE,
# PLAYBOOK_IDLE_UTIL, PLAYBOOK_LOADED_MIB.
set -uo pipefail

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }
snapshot_user=${SLURM_USER:-${USER:-$(whoami)}}
do_gpu=0
do_cluster=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --user) [ "$#" -ge 2 ] || exit 2; snapshot_user=$2; shift 2 ;;
    --gpu) do_gpu=1; shift ;;
    --no-gpu) do_gpu=0; shift ;;
    --cluster) do_cluster=1; shift ;;
    *) echo "slurm_snapshot: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

partition=${PLAYBOOK_PARTITION:-${SLURM_JOB_PARTITION:-}}
gpu_partition_re=${PLAYBOOK_GPU_PARTITION_RE:-'.*'}
idle_util=${PLAYBOOK_IDLE_UTIL:-5}
loaded_mib=${PLAYBOOK_LOADED_MIB:-2000}

if ! run_raw=$(squeue -u "$snapshot_user" -t RUNNING -h -o '%D|%P|%i|%N' 2>/dev/null); then
  echo "slurm_snapshot: squeue query failed" >&2
  exit 1
fi
pend_raw=$(squeue -u "$snapshot_user" -t PENDING -h -o '%D|%P|%i|%r' 2>/dev/null || true)

printf 'SLURM_SNAPSHOT user=%s host=%s time=%s\n' "$snapshot_user" "$(hostname -s)" "$(date '+%F %T %Z')"
printf '\nASSIGNED\n'
if [ -n "$run_raw" ]; then
  printf '%s\n' "$run_raw" | awk -F'|' '{nodes[$2]+=$1; jobs[$2]++} END {for (p in nodes) printf "  %s nodes=%d jobs=%d\n",p,nodes[p],jobs[p]}' | sort
else
  echo "  none"
fi
printf '\nPENDING\n'
if [ -n "$pend_raw" ]; then
  printf '%s\n' "$pend_raw" | awk -F'|' '{nodes[$2]+=$1; reasons[$4]++} END {for (p in nodes) printf "  %s nodes=%d\n",p,nodes[p]; for (r in reasons) printf "  reason=%s jobs=%d\n",r,reasons[r]}' | sort
else
  echo "  none"
fi

qos=
if [ -n "$partition" ]; then
  qos=$(scontrol show partition "$partition" 2>/dev/null | tr ' ' '\n' | awk -F= '/^QoS=/{print $2; exit}')
fi
qos=${qos:-unknown}
limits=
if [ "$qos" != unknown ]; then
  limits=$(sacctmgr -nP show qos format=MaxTRESPU,MaxJobsPU,MaxSubmitPU where name="$qos" 2>/dev/null | head -1)
fi
printf '\nLIMITS partition=%s qos=%s\n' "${partition:-unset}" "$qos"
if [ -n "$limits" ]; then
  printf '  MaxTRESPU|MaxJobsPU|MaxSubmitPU=%s\n' "$limits"
else
  echo "  unavailable"
fi

if [ "$do_gpu" -eq 1 ]; then
  printf '\nGPU SNAPSHOT util_below=%s loaded_above_mib=%s\n' "$idle_util" "$loaded_mib"
  probe_output=
  while IFS='|' read -r _nodes job_partition job_id node_list; do
    [ -n "$job_id" ] || continue
    [[ "$job_partition" =~ $gpu_partition_re ]] || continue
    one=$(srun --overlap --jobid="$job_id" --nodelist="$node_list" --ntasks-per-node=1 \
      bash -c 'host=$(hostname -s); nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", *" -v h="$host" "{if (\$1>u) u=\$1; if (\$2>m) m=\$2; n++} END {printf \"%s|%d|%d|%d\\n\",h,u,m,n}"' 2>/dev/null || true)
    probe_output="${probe_output}${one}"$'\n'
  done <<<"$run_raw"
  if [ -z "${probe_output//$'\n'/}" ]; then
    echo "  no reachable owned GPU allocations"
  else
    printf '%s' "$probe_output" | awk -F'|' -v ut="$idle_util" -v lm="$loaded_mib" '
      NF<4 {next}
      $2>=ut {state="working"}
      $2<ut && $3>=lm {state="reserved-idle"}
      $2<ut && $3<lm {state="free-idle"}
      {printf "  host=%s state=%s max_util=%s max_mem_mib=%s gpus=%s\n",$1,state,$2,$3,$4}'
  fi
fi

if [ "$do_cluster" -eq 1 ]; then
  printf '\nCLUSTER IDLE\n'
  sinfo -h -o '%P|%t|%D' 2>/dev/null | awk -F'|' '$2=="idle" || $2=="mix" {gsub(/\*/,"",$1); printf "  partition=%s state=%s nodes=%s\n",$1,$2,$3}' | sort
fi
