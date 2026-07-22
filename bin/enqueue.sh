#!/usr/bin/env bash
# Usage: enqueue.sh "command" [task-name]
# Atomically publish one node-level task with durable identity across states.
# Environment: PROJECT (default: my-project), SPOOL_DIR,
# PLAYBOOK_PROJECT_ROOT, PLAYBOOK_ENV_FILE.
set -euo pipefail

usage() {
  sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in -h|--help|'') usage; [ -n "${1:-}" ] && exit 0 || exit 2;; esac
COMMAND=$1
NAME=${2:-task-$(date +%s)-$RANDOM}
PROJECT=${PROJECT:-my-project}
SPOOL_DIR=${SPOOL_DIR:-/scratch/${USER:?}/$PROJECT/taskq}
PLAYBOOK_PROJECT_ROOT=${PLAYBOOK_PROJECT_ROOT:-$PWD}
PLAYBOOK_ENV_FILE=${PLAYBOOK_ENV_FILE:-}

[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "enqueue: unsafe task name: $NAME" >&2
  exit 2
}

mkdir -p "$SPOOL_DIR"/{pending,running,done,failed,names}
identity=$SPOOL_DIR/names/$NAME
if ! mkdir "$identity" 2>/dev/null; then
  echo "enqueue: task identity already reserved: $identity" >&2
  exit 1
fi
output=$SPOOL_DIR/pending/$NAME.sh
temporary=$SPOOL_DIR/pending/.$NAME.tmp.$$
cleanup() {
  [ -e "$output" ] || rmdir "$identity" 2>/dev/null || true
  [ -e "$temporary" ] && rm -f "$temporary"
}
trap cleanup EXIT
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
  printf 'cd -- %q\n' "$PLAYBOOK_PROJECT_ROOT"
  if [ -n "$PLAYBOOK_ENV_FILE" ]; then printf 'source %q\n' "$PLAYBOOK_ENV_FILE"; fi
  printf 'exec bash -lc %q\n' "$COMMAND"
} >"$temporary"
chmod 700 "$temporary"
mv "$temporary" "$output"
trap - EXIT
printf 'enqueued: %s\n' "$output"
