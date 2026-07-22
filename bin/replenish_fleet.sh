#!/usr/bin/env bash
# Usage: replenish_fleet.sh [-h]
# Submit one or more bounded fleet renewal lanes from a login node.
# Environment: FLEET_TEMPLATE, FLEET_NODES, RENEWAL_LANES, RENEWAL_HOURS,
# PLAYBOOK_BALANCE_OK=1, PLAYBOOK_ROOT. Check the account balance before every
# invocation.
set -euo pipefail

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0;; '') ;; *) usage >&2; exit 2;; esac

case "$(hostname -s)" in login*) ;; *)
  echo "replenish_fleet: run on a login node; batch one internal SSH command from compute" >&2
  exit 2
esac
[ "${PLAYBOOK_BALANCE_OK:-0}" = 1 ] || {
  echo "replenish_fleet: inspect /usr/local/etc/taccinfo, then set PLAYBOOK_BALANCE_OK=1" >&2
  exit 2
}

FLEET_TEMPLATE=${FLEET_TEMPLATE:-infra/slurm/fleet.sbatch}
FLEET_NODES=${FLEET_NODES:-1}
RENEWAL_LANES=${RENEWAL_LANES:-1}
RENEWAL_HOURS=${RENEWAL_HOURS:-12}
[[ "$FLEET_NODES" =~ ^[1-9][0-9]*$ && "$RENEWAL_LANES" =~ ^[1-9][0-9]*$ && "$RENEWAL_HOURS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( RENEWAL_LANES <= 4 )) || { echo "replenish_fleet: at most four renewal lanes" >&2; exit 2; }
[ -f "$FLEET_TEMPLATE" ] || { echo "replenish_fleet: missing $FLEET_TEMPLATE" >&2; exit 2; }

PLAYBOOK_ROOT=${PLAYBOOK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
"$PLAYBOOK_ROOT/bin/login_guard.sh"

for ((lane = 0; lane < RENEWAL_LANES; lane++)); do
  if [ "$lane" -eq 0 ]; then
    sbatch -N "$FLEET_NODES" "$FLEET_TEMPLATE"
  else
    begin_hours=$((lane * RENEWAL_HOURS))
    sbatch --begin="now+${begin_hours}hours" -N "$FLEET_NODES" "$FLEET_TEMPLATE"
  fi
  if (( lane + 1 < RENEWAL_LANES )); then sleep 10; fi
done
printf 'scheduled %s fleet lane(s), %s node(s) each\n' "$RENEWAL_LANES" "$FLEET_NODES"
