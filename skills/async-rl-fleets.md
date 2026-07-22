---
name: async-rl-fleets
description: Load when operating asynchronous rollout producers and trainers, trajectory queues, lag controls, packed identities, transactional batches, or safe RL shutdown.
---

# Async RL Fleets

## When to use

- Couple rollout production to one or more trainer consumers.
- Bound policy lag and backlog in an asynchronous post-training run.
- Pack rollout or evaluation work without duplicating deterministic streams.
- Resume, abort, or clean up a trainer while preserving valid state.

## Procedure

1. Separate the executable task spool from the trajectory queue. Give each independent lifecycle and recovery rules.
2. Publish each trajectory by writing a temporary file and atomically renaming it into `trajq/pending` only after validation.
3. Include run ID, policy version, physical slot, logical CUDA binding, and unique `worker_stream_id` in every trajectory and heartbeat.
4. Seed prompts from `worker_stream_id`, not host plus logical `cuda:0`, because multiple packed slots share the logical name.
5. Set a frozen high-water backlog cap, a lower refill threshold, and a maximum policy-version lag. Mark throttled workers as healthy.
6. Add capacity only for a live trainer below low water. Estimate producer and consumer rates, then add at most one bounded node per controller pass.
7. Let rank zero claim and validate complete update groups before sharding. Reject duplicate deterministic identities before policy movement.
8. If validation leaves a batch short, return valid claims to pending and quarantine duplicates. Record failed allocation cost even when no update commits.
9. Write policy checkpoints and full trainer state atomically. Include critic, reward normalization, counters, optimizer, and accounting state.
10. On ABORT, finish the current bounded unit, save state, and exit distinctly. Use bounded TERM, poll, then KILL on the process group, while leaving unrelated fleet allocations intact.

## Rules (hard)

- Bound backlog and version lag, because unconsumable trajectories waste compute and become stale. [VSAO RUNBOOK-RL.md:176-178,261-262]
- Separate physical slot from logical device identity, because every packed worker may see `cuda:0`. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]
- Audit duplicates before policy movement, because rollback after an optimizer update cannot restore the registered trajectory. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]
- Treat a batch as a durable transaction, because partial claims must not silently change group semantics. [VSAO DESIGN.md:8-24]
- Require the declared terminal checkpoint, because an outer done task does not prove trainer completion. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Keep run shutdown separate from allocation shutdown, because a shared fleet can continue useful unrelated work. [VSAO RUNBOOK-RL.md:243-253]
- Bound cleanup time and kill the process group, because signal-ignoring descendants can hold a GPU indefinitely. [VSAO SMOKE.md:2097-2124]

## Pitfalls seen in production

- Symptom: seed 6 repeats trajectory streams. Cause: two physical GPUs share host plus logical-device identity. Fix: assign and persist unique `worker_stream_id` values. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]
- Symptom: rollout nodes remain allocated but intentionally idle. Cause: the backlog reached 160 while the trainer consumed more slowly. Fix: mark throttled, release at high water, and replenish below 80. [VSAO ensure_rollout_capacity.sh:28-30]
- Symptom: a replacement never starts despite stale work. Cause: an old claim on an incompatible node shape suppresses capacity. Fix: require fresh compatible heartbeats. [VSAO report-vsao.md, Trajectory queues]
- Symptom: a task is done but version 37 is absent. Cause: the wrapper ended at version 30 near a fleet boundary. Fix: validate the terminal checkpoint and restore one resumable claim. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: a cleanup holds 24 GB at zero utilization. Cause: an unbounded wait on a child that ignores signals. Fix: bounded TERM, grace poll, then KILL. [VSAO SMOKE.md:2097-2124]
- Symptom: two trainers take over the same stale lock. Cause: both pass the stale test simultaneously. Fix: use nonce, jitter, recheck, and durable queue evidence. [VSAO report-vsao.md, Trainer single-instance lock]

## Pointers

- Related skills: `slurm-fleets-and-spools.md`, `multi-node-training.md`, `telemetry-and-monitoring.md`, `provenance-and-repro.md`.
- Scripts: `bin/gpu_enqueue.sh`, `bin/gpu_pack_node.sh`, `bin/gpu_slot_worker.sh`, `bin/reap_gpu_spool.sh`.
- Compendium themes: 4, 5, 6, 8, and 12.
