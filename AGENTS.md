# Fleetcraft Agent Guide

This is the single agent entry point for the repository.

## Iron rules

- Protect login access. Stay below 40 processes, use at most three login SSH connections, never SSH per item, and keep heavy work on compute nodes.
- Use one multi-node fleet with a resumable spool. Supply only declared work, bound idle waits, and release nodes when useful work ends.
- Calibrate one complete workload cycle before enqueueing at scale, then freeze the per-family batch table.
- Start telemetry on every GPU host and track every real project in WandB.
- Keep reported latency and FPS in exclusive batch-one timed passes.
- Check the live allocation balance before every submission. Apply a hard burn-stop threshold well before exhaustion.
- Preserve validated training world size and use surplus nodes only in isolated, dependency-safe roles.
- Keep heavy data and caches on scratch, persistent code and evidence on work, and only small configuration in home.
- Treat done markers as claims until canonical payloads, hashes, checkpoints, and terminal accounting validate them.
- Use no em dash or section symbol in prose, comments, or generated artifacts.

## Task routing

| Task | Read |
|---|---|
| Fleet jobs, outer queues, packed GPU queues, stale claims | `skills/slurm-fleets-and-spools.md` |
| Batch calibration, throughput, co-location, timing | `skills/gpu-utilization-batching.md` |
| HBM, caches, activation peaks, OOM, offload | `skills/gpu-memory-hierarchy.md` |
| torchrun, FSDP, DDP, NCCL, supervisors, checkpoints | `skills/multi-node-training.md` |
| Rollout and trainer fleets, lag, identity, ABORT | `skills/async-rl-fleets.md` |
| GPU logging, rolling utilization, compute plots | `skills/telemetry-and-monitoring.md` |
| TACC login safety, Slurm, allocation balance, tmux nodes | `skills/tacc-operations.md` |
| UV, aarch64 wheels, SDPA, Triton, native builds | `skills/env-setup-aarch64.md` |
| Video decode, encode, HDR, FFmpeg, torchcodec | `skills/video-io.md` |
| HOME, WORK, SCRATCH, caches, atomic shared I/O | `skills/storage-and-caches.md` |
| Manifests, hashes, frozen grids, release gates | `skills/provenance-and-repro.md` |

Humans should read `README.md` first.
