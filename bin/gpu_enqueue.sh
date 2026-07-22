#!/usr/bin/env bash
# Usage: gpu_enqueue.sh "command using logical cuda:0" task-name
# Atomically publish one untimed, single-GPU task to the packed inner spool.
# Environment: PROJECT (default: my-project), GPU_SPOOL_DIR.
set -euo pipefail

usage() { sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help|'') usage; [ -n "${1:-}" ] && exit 0 || exit 2;; esac
[ "$#" -eq 2 ] || { usage >&2; exit 2; }
command_text=$1
name=$2
PROJECT=${PROJECT:-my-project}
GPU_SPOOL_DIR=${GPU_SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq_gpu}

[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "gpu_enqueue: unsafe task name: $name" >&2
  exit 2
}
mkdir -p "$GPU_SPOOL_DIR"/{pending,running,done,failed,names}
identity=$GPU_SPOOL_DIR/names/$name
if ! mkdir "$identity" 2>/dev/null; then
  echo "gpu_enqueue: task identity already reserved: $identity" >&2
  exit 1
fi
output=$GPU_SPOOL_DIR/pending/$name.sh
temporary=$GPU_SPOOL_DIR/pending/.$name.tmp.$$
cleanup() {
  [ -e "$output" ] || rmdir "$identity" 2>/dev/null || true
  [ -e "$temporary" ] && rm -f "$temporary"
}
trap cleanup EXIT
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
  printf 'exec bash -lc %q\n' "$command_text"
} >"$temporary"
chmod 700 "$temporary"
mv "$temporary" "$output"
trap - EXIT
printf 'gpu-enqueued: %s\n' "$output"
