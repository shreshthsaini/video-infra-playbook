#!/usr/bin/env bash
# Usage: login_guard.sh [-h] [--reap-git]
# Protect a shared login node process and SSH budget before fan-out.
# Environment: LOGIN_WARN_PROCS, LOGIN_STOP_PROCS, LOGIN_PROCESS_CAP,
# LOGIN_WARN_SSH, LOGIN_STOP_SSH.
set -uo pipefail

usage() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; }
REAP=0
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --reap-git) REAP=1 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac

account_user=${SLURM_USER:-${USER:-$(whoami)}}
WARN_PROCS=${LOGIN_WARN_PROCS:-25}
STOP_PROCS=${LOGIN_STOP_PROCS:-40}
PROCESS_CAP=${LOGIN_PROCESS_CAP:-100}
WARN_SSH=${LOGIN_WARN_SSH:-2}
STOP_SSH=${LOGIN_STOP_SSH:-3}
host=$(hostname -s 2>/dev/null)
case "$host" in login*) on_login=yes;; *) on_login=no;; esac

count_procs() { ps -u "$account_user" --no-headers 2>/dev/null | awk 'END {print NR+0}'; }
count_named() { pgrep -u "$account_user" -x "$1" 2>/dev/null | awk 'END {print NR+0}'; }
processes=$(count_procs)
ssh_count=$(count_named ssh)
git_count=$(count_named git)

if [ "$REAP" -eq 1 ] && [ "$git_count" -gt 0 ]; then
  pkill -u "$account_user" -x git 2>/dev/null || true
  sleep 1
  processes=$(count_procs)
  git_count=$(count_named git)
fi

printf 'LOGIN_GUARD host=%s login=%s procs=%s/%s ssh=%s git=%s\n' \
  "$host" "$on_login" "$processes" "$PROCESS_CAP" "$ssh_count" "$git_count"
if [ "$processes" -ge "$STOP_PROCS" ] || [ "$ssh_count" -ge "$STOP_SSH" ]; then
  echo "verdict: NO-GO; stop, reap, and do not fan out"
  exit 2
fi
if [ "$processes" -ge "$WARN_PROCS" ] || [ "$ssh_count" -ge "$WARN_SSH" ]; then
  echo "verdict: CAUTION; reduce footprint and do not fan out"
  exit 1
fi
echo "verdict: GO"
