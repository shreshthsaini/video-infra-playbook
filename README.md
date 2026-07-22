# Fleetcraft

The craft of keeping GPU fleets honestly busy on shared HPC.

![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![status: field-tested](https://img.shields.io/badge/status-field--tested-brightgreen.svg)
![systems: Slurm | A100 | H100 | GH200 | GB200](https://img.shields.io/badge/systems-Slurm%20%7C%20A100%20%7C%20H100%20%7C%20GH200%20%7C%20GB200-555555.svg)

Fleetcraft is an evidence-backed operating library for GPU video training, inference, evaluation, and asynchronous post-training on shared Slurm systems. It turns lessons from production projects into reusable queue workers, telemetry, scheduler templates, analysis utilities, and task-focused operating guides. For the narrative behind the design, read the companion post, [Fleetcraft: GPU Infrastructure Notes from Three Video-Generation Projects](https://shreshthsaini.github.io/blogs/gpu-infra-for-video-gen.html).

## What is inside

```text
.
├── AGENTS.md          Agent rules and task router
├── bin/               Resumable spools, workers, telemetry, and safety tools
├── docs/              Evidence compendium, case studies, hardware notes, and sources
├── infra/
│   ├── python/        Analysis and portable video I/O utilities
│   └── slurm/         Fleet, development-node, and smoke templates
├── skills/            Task-focused operational procedures
├── .gitignore         Generated output and Python cache exclusions
├── LICENSE            MIT license
└── README.md          Human entry point
```

## Quickstart for humans

1. Clone the repository.

   ```bash
   git clone https://github.com/shreshthsaini/fleetcraft.git
   cd fleetcraft
   ```

2. Pick the file in `skills/` that matches your task. Start with `skills/slurm-fleets-and-spools.md` for a queue-driven fleet.

3. Copy `infra/slurm/fleet.sbatch`, replace `__GPU_PARTITION__`, `__ACCOUNT__`, and `__SCRATCH_LOG_DIR__`, then set a project and scratch root.

   ```bash
   export PROJECT=my-project
   export PROJECT_SCRATCH="/scratch/$USER/$PROJECT"
   sbatch -N 4 infra/slurm/fleet.sbatch
   ```

4. Publish stable, declared work to the shared spool.

   ```bash
   bin/enqueue.sh 'bash experiments/run_one.sh --item 17' item-0017
   ```

5. Watch the fleet with per-host telemetry and rolling summaries.

   ```bash
   export TELEMETRY_DIR="$PROJECT_SCRATCH/telemetry"
   infra/python/gpu_telemetry_summary.py --telemetry-dir "$TELEMETRY_DIR" --window-seconds 300
   bin/slurm_snapshot.sh --no-gpu
   ```

## Quickstart for agents

Read `AGENTS.md`, route the task through its table, and follow the selected skill before changing or running anything.

## `bin/` catalog

| Script | What it does | Typical invocation |
|---|---|---|
| `enqueue.sh` | Atomically publishes one node-level task | `bin/enqueue.sh 'bash run.sh' task-001` |
| `gpu_enqueue.sh` | Publishes one untimed single-GPU task | `bin/gpu_enqueue.sh 'python score.py' score-001` |
| `gpu_pack_node.sh` | Bridges an outer claim to packed per-GPU work | `bin/enqueue.sh 'bin/gpu_pack_node.sh' gpu-pack-001` |
| `gpu_slot_worker.sh` | Drains inner-spool work on one physical GPU | `bin/gpu_slot_worker.sh 0` |
| `gpu_telemetry.sh` | Records per-host GPU utilization, memory, and power | `bin/gpu_telemetry.sh "$TELEMETRY_DIR" 30` |
| `login_guard.sh` | Checks shared login-node process and SSH pressure | `bin/login_guard.sh` |
| `reap_gpu_spool.sh` | Recovers stale inner-spool claims fail-closed | `bin/reap_gpu_spool.sh` |
| `replenish_fleet.sh` | Schedules bounded fleet renewal lanes | `PLAYBOOK_BALANCE_OK=1 bin/replenish_fleet.sh` |
| `slurm_snapshot.sh` | Summarizes jobs, limits, and optional GPU state | `bin/slurm_snapshot.sh --gpu` |
| `worker_loop.sh` | Drains the durable outer task spool | `bin/worker_loop.sh` |

Every shell utility supports `-h` or `--help`. Inspect its header before use.

## `skills/` catalog

| Skill | Read it when |
|---|---|
| `slurm-fleets-and-spools.md` | Building durable fleets, queues, packed GPU work, or recovery |
| `gpu-utilization-batching.md` | Calibrating batches, throughput, co-location, or timing |
| `gpu-memory-hierarchy.md` | Diagnosing HBM, caches, activation peaks, OOM, or offload |
| `multi-node-training.md` | Running torchrun, FSDP, DDP, NCCL, supervisors, or checkpoints |
| `async-rl-fleets.md` | Coordinating rollout and trainer rates, identity, lag, or aborts |
| `telemetry-and-monitoring.md` | Logging GPUs, interpreting rolling activity, or plotting usage |
| `tacc-operations.md` | Protecting login nodes and operating Slurm on TACC systems |
| `env-setup-aarch64.md` | Building UV environments and native aarch64 dependencies |
| `video-io.md` | Decoding, encoding, handling HDR, or stabilizing video metrics |
| `storage-and-caches.md` | Placing data and caches or making shared I/O atomic |
| `provenance-and-repro.md` | Freezing protocols, validating completion, or releasing evidence |

## Documentation and reproducible analysis

`docs/COMPENDIUM.md` is the append-only production evidence ledger. `docs/CASE-STUDIES.md` traces the source projects, `docs/HARDWARE-NOTES.md` explains the hardware implications, and `docs/sources/` preserves public references plus anonymized telemetry.

The telemetry figures and compute summary regenerate from `docs/sources/telemetry` into the ignored `out/` directory:

```bash
python3 infra/python/plot_blog_telemetry.py --output-dir out/telemetry
python3 infra/python/plot_compute_usage.py \
  --telemetry-dir docs/sources/telemetry/forcing-laws \
  --output-dir out/compute --tag forcing-laws
```

## Status and contributing

Fleetcraft is a work in progress. The infrastructure is still evolving, and I
am actively working to make it better, more generalizable, and more usable
beyond the systems it grew up on. Feel free to contribute: issues,
corrections, hardened scripts, new skills, and ports to other schedulers are
all welcome.

## Provenance and honesty

Fleetcraft is evidence-first. Measured claims point to preserved records, estimates are labeled as estimates, and completion requires canonical payloads rather than optimistic markers. The library is field-tested, but it remains a work in progress toward better useful utilization across changing workloads and systems.

## Adaptation and permission

Some examples reflect TACC paths, commands, and operating constraints. Adapt partitions, accounts, storage roots, limits, and authentication to your site. Ask your cluster or infrastructure operators for permission before running fleet automation, telemetry, or login-node tooling on shared systems.

## Acknowledgments

Developed through work on TACC Vista and Lonestar6 via the Institute for Foundations of Machine Learning (IFML).

## Citation

```bibtex
@misc{saini2026fleetcraftblog,
  author       = {Shreshth Saini},
  title        = {The Systems Layer Nobody Teaches You: GPU Infrastructure for Video Generation},
  year         = {2026},
  howpublished = {\url{https://shreshthsaini.github.io/blogs/gpu-infra-for-video-gen.html}}
}
```

```bibtex
@misc{saini2026fleetcraft,
  author       = {Shreshth Saini},
  title        = {Fleetcraft},
  year         = {2026},
  howpublished = {\url{https://github.com/shreshthsaini/fleetcraft}}
}
```
