---
name: multi-node-training
description: Load when launching or recovering torchrun, FSDP, DDP, NCCL, multi-role Slurm steps, distributed checkpoints, or training supervisors.
---

# Multi-Node Training

## When to use

- Launch torchrun across held nodes or diagnose rendezvous and NCCL failures.
- Choose FSDP sharding for one-GPU or multi-GPU nodes.
- Build a retrying supervisor without dueling ownership or stale-log decisions.
- Use surplus nodes without changing a registered training world size.

## Procedure

1. Freeze the registered world size, global batch, accumulation, sharding mode, backend, and prompt grid before launch.
2. Inspect the physical node shape. On one-GPU nodes, choose full FSDP sharding rather than an intra-node hybrid strategy.
3. Select one master host and c10d rendezvous endpoint. Publish one rank-zero action after every time-sensitive barrier so all ranks take the same decision.
4. Give the supervisor a nonblocking `flock` and a launch epoch. Archive or rotate rank logs at each fresh launch or adoption.
5. Use exact PID records and Slurm step IDs for cleanup. Never kill by a broad pattern that can match the controlling shell.
6. Match current-epoch signatures for CUDA OOM, `ChildFailedError`, `RendezvousConnectionError`, `DistNetworkError`, NCCL watchdog failure, and scheduler termination.
7. Isolate `TRITON_CACHE_DIR` and generated sources by host, run, role, and retry. Reuse content-addressed graph caches only sequentially on the same node.
8. Define a durable checkpoint as a completed gather and save plus explicit training-loop confirmation. Store optimizer, RNG, sampler, step, branch, probe, and compute-ledger state for exact resume.
9. If ranks hang after one fails, cancel the exact Slurm step and preserve the parent allocation for recovery.
10. Keep validated training on its original ranks. Start surplus-node evaluation in a disjoint Slurm step after a ready marker, then exit evaluators only after producer done and queue empty.

## Rules (hard)

- Preserve validated world size and global batch, because adding ranks changes the registered optimization trajectory. [FL G43]
- Use one supervisor lock, because timed-out launch commands can leave hidden competing owners. [FL G24]
- Scope signatures to the current launch epoch, because old crash lines can kill a healthy resumed run. [FL G23]
- Isolate generated compiler sources, because concurrent shared writes can corrupt code and collapse rendezvous. [FL G27]
- Cancel exact steps and recorded processes, because broad cleanup can destroy held allocations and healthy groups. [FL G20; FL G31]
- Treat weight-only checkpoints as partial, because exact resume also requires optimizer, RNG, sampler, and protocol state. [FL G33]
- Freeze dynamically loaded launch and worker files before submission, because mixed on-disk script versions can invalidate completed seeds. [FL G44]

## Pitfalls seen in production

- Symptom: FSDP reaches 93 GB on one-GPU GH200 nodes. Cause: hybrid sharding degenerates into replication. Fix: use full sharding across nodes. [FL G19]
- Symptom: a sequencer waits forever after distributed network failure. Cause: its classifier only knows CUDA and child failures. Fix: include rendezvous and distributed-network signatures. [FL G25]
- Symptom: a healthy adopted run is repeatedly restarted. Cause: stale `ChildFailedError` remains in the current log. Fix: compare signature time with the launch epoch or archive logs. [FL G23]
- Symptom: five launches churn without useful progress. Cause: two or three supervisors own the same run after a timeout. Fix: acquire `flock` and census processes after every launch attempt. [FL G24]
- Symptom: surviving ranks wait forever after peer OOM. Cause: they remain blocked in a collective. Fix: cancel only the failed Slurm step. [FL G31]
- Symptom: DDP fails at reducer.cpp with static graph plus `no_sync`. Cause: CFG double-forward and checkpointing conflict with chunked accumulation. Fix: use a parity-tested manual gradient reduction when the standard reducer cannot express the graph. [VSAO SMOKE.md:1497-1512]
- Symptom: a visible checkpoint cannot resume. Cause: FSDP gathers or state writes were incomplete. Fix: require save completion and demonstrated continuation. [FL G34]

## Pointers

- Related skills: `slurm-fleets-and-spools.md`, `gpu-memory-hierarchy.md`, `async-rl-fleets.md`, `storage-and-caches.md`, `provenance-and-repro.md`.
- Template: `infra/slurm/fleet.sbatch`.
- Compendium themes: 2, 5, 6, 9, 10, and 12.
