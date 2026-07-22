---
name: tacc-operations
description: Load before TACC scheduler submissions, login-node work, queue inspection, interactive compute sessions, allocation accounting, SSH, or fleet renewal.
---

# TACC Operations

## When to use

- Submit, inspect, renew, or cancel Slurm jobs on Vista or a similar TACC system.
- Work from a compute node when `sbatch` is blocked locally.
- Start a durable interactive agent or development node.
- Diagnose account limits, idle allocations, or login-node risk.

## Procedure

1. Run `bin/login_guard.sh` before fan-out or submission. Stop and reap at a no-go verdict.
2. Read the live account balance on a login node with `/usr/local/etc/taccinfo`. Apply a hard burn-stop threshold well before exhaustion.
3. Inspect queue and QOS state with `bin/slurm_snapshot.sh --no-gpu`. Use the GPU probe only when needed and keep its concurrency at three or less.
4. Submit a small number of multi-node fleet jobs, not one task per job. Batch multiple scheduler commands into one login SSH connection.
5. From a compute node, use plain internal SSH:
   ```bash
   ssh LOGIN_NODE 'sbatch /absolute/path/to/job.sbatch'
   ```
6. Do not add `BatchMode=yes` to that submission route. It disables the authentication method that works inside the cluster.
7. Start development sessions with `infra/slurm/devnode.sbatch`, then attach from a login terminal and work inside tmux.
8. Wait at least ten seconds between repeated scheduler or SSH polls. Prefer one snapshot or scheduler notification.
9. Track every background PID, terminate it, and `wait` before the shell exits. Keep the account-wide login process count below 40.
10. Cancel unused allocations as soon as declared work ends.

## Rules (hard)

- Never run builds, encoding, data processing, model loads, or agent sessions on login nodes, because their CPU, virtual memory, and process budget are shared and small. [memory-facts]
- Stop launching at 40 login processes, because the measured hard ceiling is 100 across all sessions. [memory-facts]
- Use at most three concurrent login SSH connections, because fan-out can lock the account out. [memory-facts]
- Never SSH once per item in a loop, because connections consume the shared process budget. [memory-facts]
- Wait at least ten seconds between scheduler polls, because tight loops burden login and scheduler services. [memory-facts]
- Check the allocation balance before every submission, because Slurm reserves nodes times walltime immediately. [FL G21]
- Stop new GPU submissions at a hard threshold well before exhaustion, because late reservations can exceed the project allocation. [memory-facts]
- Use neutral job names, because operational names should describe work rather than resource possession.

## Pitfalls seen in production

- Symptom: every new shell fails with `exec request failed on channel 0`. Cause: the shared 100-process ceiling was exhausted. Fix: connect through another login node and reap offending processes. [memory-facts]
- Symptom: a heavy import or download dies without GPU use. Cause: login nodes cap virtual memory at 8 GB. Fix: move the work to a compute node. [VG1 b1_rescore.py:281; memory-facts]
- Symptom: internal submission reports keyboard-interactive permission denied. Cause: `BatchMode=yes` disabled the working path. Fix: use plain `ssh LOGIN_NODE 'sbatch ...'`. [memory-facts]
- Symptom: the account consumes more allocation than expected at submission. Cause: nodes times requested walltime were reserved immediately. Fix: size walltime realistically and check balance first. [memory-facts; FL G21]
- Symptom: an allocated node burns time after the queue drains. Cause: idle tail is unbounded. Fix: configure a bounded idle exit or cancel after useful work ends. [FL G26]

## Pointers

- Related skills: `slurm-fleets-and-spools.md`, `storage-and-caches.md`, `telemetry-and-monitoring.md`.
- Scripts: `bin/login_guard.sh`, `bin/slurm_snapshot.sh`, `bin/replenish_fleet.sh`.
- Templates: `infra/slurm/fleet.sbatch`, `infra/slurm/devnode.sbatch`, `infra/slurm/smoke.sbatch`.
- Compendium themes: 4, 8, 9, and 10.
