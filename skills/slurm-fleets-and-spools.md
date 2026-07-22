---
name: slurm-fleets-and-spools
description: Load when allocating durable Slurm worker fleets, publishing resumable tasks, packing per-GPU work, or recovering stale spool claims.
---

# Slurm Fleets and Spools

## When to use

- Run many independent GPU tasks without consuming one scheduler job per task.
- Keep an allocation useful across generation, scoring, and evaluation waves.
- Pack untimed single-GPU work safely onto heterogeneous multi-GPU nodes.
- Recover claims after walltime, cancellation, or worker failure.

## Procedure

1. Check account balance, queue limits, and login safety before submission:
   ```bash
   bin/login_guard.sh
   bin/slurm_snapshot.sh --no-gpu
   /usr/local/etc/taccinfo
   ```
2. Set a project name and scratch spool. Keep queues off home and work:
   ```bash
   export PROJECT=my-project
   export SPOOL_DIR="/scratch/$USER/$PROJECT/taskq"
   ```
3. Copy `infra/slurm/fleet.sbatch`, fill the marked account and partition fields, and freeze every script it will load.
4. Submit one multi-node allocation after the balance check. Resize nodes at submit time instead of creating an array:
   ```bash
   sbatch -N 8 infra/slurm/fleet.sbatch
   ```
5. Publish idempotent outer tasks with stable names:
   ```bash
   bin/enqueue.sh 'bash experiments/run_one.sh --item 17' item-0017
   ```
6. Publish useful, already approved work before the queue drains. Let workers retry briefly through dependency gaps, then exit after the configured idle bound.
7. For untimed, independent single-GPU tasks, publish with `bin/gpu_enqueue.sh` and enqueue one `bin/gpu_pack_node.sh` bridge claim in the outer spool.
8. Run `bin/reap_gpu_spool.sh` from a bounded controller pass after walltime boundaries. Treat scheduler-query failure as no permission to requeue.
9. Inspect `pending`, `running`, `done`, and `failed` together. Validate output payloads before treating a done script as complete.
10. Cancel the allocation when no frozen useful work remains. Allocated nodes are reusable capacity, not a reason to invent work.

## Rules (hard)

- Use one multi-node fleet instead of one job per item, because job slots are scarcer than nodes. [FL G13]
- Claim tasks with atomic rename, because concurrent shared-filesystem appends and non-atomic ownership duplicate work. [VG1 worker_loop.sh:53-70]
- Reserve task identity across every queue state, because two controllers can publish the same logical task concurrently. [VSAO gpu_enqueue.sh:23-48]
- Fail closed when `squeue` fails, because an empty response is not proof that live claims are stale. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Keep timed tasks exclusive, because co-location corrupts reported latency and FPS. [VG1 worker_loop.sh:4-18,56-57]
- Derive physical GPU count at runtime, because packed claims can land on one, two, or three-GPU nodes. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]
- Kill and reap complete task process groups, because grandchildren can retain GPUs after wrappers exit. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Release capacity when declared work ends, because allocation is not scientific utilization. [FL G26]

## Pitfalls seen in production

- Symptom: pending inner tasks exist but nodes stay idle. Cause: no outer bridge claim exposes the nested spool. Fix: enqueue a bounded `gpu_pack_node.sh` claim. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: mixed one, two, and three-GPU nodes are underfilled. Cause: bridge count assumes a fixed three-GPU shape. Fix: bound claims by observed free physical GPUs. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]
- Symptom: four tasks remain in running after walltime. Cause: only workers performed stale reaping. Fix: add a controller-side, fail-closed reaper pass. [VG1 NOTES-a1-agent.md:95-98]
- Symptom: a pilot dies in an NCCL watchdog. Cause: a common worker claimed the pilot rank's GPU. Fix: defer incompatible tasks or isolate roles in disjoint Slurm steps. [FL G22]
- Symptom: cancelling a worker releases the entire fleet. Cause: the worker itself is the Slurm step. Fix: target recorded child PIDs or the exact failed step. [FL G20]
- Symptom: a node stays partially idle beside a long scorer. Cause: each slot claimed only once. Fix: let slots claim repeatedly and revive exited siblings while useful work remains. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

## Pointers

- Related skills: `gpu-utilization-batching.md`, `async-rl-fleets.md`, `tacc-operations.md`, `provenance-and-repro.md`.
- Scripts: `bin/worker_loop.sh`, `bin/enqueue.sh`, `bin/replenish_fleet.sh`, `bin/gpu_enqueue.sh`, `bin/gpu_pack_node.sh`, `bin/gpu_slot_worker.sh`, `bin/reap_gpu_spool.sh`.
- Templates: `infra/slurm/fleet.sbatch`.
- Compendium themes: 4, 6, 8, 10, and 12.
