---
name: storage-and-caches
description: Load when placing repositories, environments, model data, checkpoints, queues, telemetry, compiler caches, or paper-critical evidence across HOME, WORK, and SCRATCH.
---

# Storage and Caches

## When to use

- Choose a durable or regenerable location for project data.
- Configure Hugging Face, Torch, Triton, UV, or temporary caches.
- Design queue, checkpoint, and manifest I/O on shared filesystems.
- Diagnose corrupt downloads, compiler races, or slow checkpoint resumes.

## Procedure

1. Classify each artifact as configuration, persistent code, paper-critical evidence, or heavy regenerable state.
2. Put small configuration in `/home1/$USER`-equivalent home storage, repositories and compact evidence on work, and heavy data on scratch.
3. Set project scratch explicitly:
   ```bash
   export PROJECT=my-project
   export PROJECT_SCRATCH="/scratch/$USER/$PROJECT"
   ```
4. Point HF, dataset, transformer, Torch, Triton, WandB, and temporary caches to scratch. Keep UV environment paths at the site's stable work symlinks when configured.
5. Namespace generated compiler sources by host, run, role, and retry. Reuse graph caches only where writes cannot overlap.
6. Publish queue tasks, trajectories, checkpoints, and shared feature caches through same-directory temporary writes plus atomic rename.
7. Validate cache schema, source fingerprint, model or backbone, filenames, tensor shape, and finite values before reuse.
8. Archive only the records, manifests, hashes, and compact results needed to rederive every paper number on persistent work storage.
9. Never move or rename a path referenced by a queued or running job. Copy archival evidence after the consumer closes it.
10. Diagnose large checkpoint reads with CPU time and I/O progress. Limit concurrent heavy readers when scratch throughput collapses.

## Rules (hard)

- Keep environments, datasets, weights, checkpoints, and caches out of home, because its quota is small. [memory-facts]
- Treat scratch as purge-eligible and regenerable, because idle files can disappear after roughly ten days. [memory-facts]
- Preserve the paper-critical crux on work, because raw scratch artifacts are not an archive. [memory-facts]
- Use atomic rename instead of shared append for ownership, because concurrent Lustre writers can race or tear records. [VG1 worker_loop.sh:53-70]
- Make effective cache variables explicit, because `HF_HUB_CACHE` can override `HF_HOME`. [VG1 report-videogen1.md, Storage layout]
- Never relocate live job inputs, because stable path identity is part of the scheduler contract. [VG1 report-videogen1.md, Storage layout]
- Isolate concurrent generated sources, because shared Triton writes can corrupt compilation. [FL G27]

## Pitfalls seen in production

- Symptom: a worker redownloads Wan-14B and corrupts shard 00005. Cause: its cache contract differs from the interactive shell. Fix: quarantine the shard and pin all workers to one verified cache. [VG1 report-videogen1.md, Storage layout]
- Symptom: compiler failures appear as malformed source and rendezvous collapse. Cause: active nodes share generated Triton files. Fix: isolate generated-source roots per node and retry. [FL G27]
- Symptom: an evaluation project fails on a dead GRiT URL. Cause: scorer files were fetched on demand. Fix: mirror and validate the complete cache tree once. [FL G9]
- Symptom: a 12 to 15 GB resume looks hung for 20 to 48 minutes. Cause: single-threaded unpickle is CPU-active. Fix: distinguish advancing CPU time from D-state I/O stalls. [VSAO report-vsao.md, Vista GB200 side]
- Symptom: six concurrent GB200 runs fall to kilobytes per second. Cause: scratch I/O saturation. Fix: cap heavy readers or runs at the measured four to five stable shape. [VSAO report-vsao.md, Vista GB200 side]
- Symptom: a shared teacher cache is readable but wrong. Cause: content metadata was not validated. Fix: lock creation, atomically replace, and verify schema plus fingerprint. [FL G28]

## Pointers

- Related skills: `env-setup-aarch64.md`, `slurm-fleets-and-spools.md`, `provenance-and-repro.md`, `video-io.md`.
- Compendium themes: 9, 10, 11, and 12.
