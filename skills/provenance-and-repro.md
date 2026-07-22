---
name: provenance-and-repro
description: Load when defining completion, freezing experiment grids, building manifests, validating checkpoints, archiving evidence, or preparing a release gate.
---

# Provenance and Reproducibility

## When to use

- Decide whether a project, evaluation, checkpoint, or paper result is complete.
- Build append-only records, hash manifests, numeric gates, and compact evidence archives.
- Resume or recover work without rerunning valid components.
- Change scripts or prose while jobs and result pipelines are active.

## Procedure

1. Freeze the scientific protocol: prompts, seeds, arms, schedules, backend, batch policy, timed-pass policy, expected outputs, and escalation rules.
2. Assign stable identities to tasks, trajectories, samples, and comparison cells. Enforce identity across pending, running, done, and failed states.
3. Write append-only per-worker JSONL shards or atomic per-item records. Re-glob on read, tolerate only a torn final live line, and deduplicate by stable key.
4. Record source paths, SHA-256 hashes, stack versions, host or architecture, job IDs, timing status, and configuration fingerprints with every result.
5. Define payload validators for every done marker. Verify counts, schema, finite numbers, path containment, non-symlinks where required, hashes, and terminal score files.
6. Separate weight-only, resumable full state, terminal checkpoint, evaluation complete, paper gate complete, and scheduler accounting archived.
7. Freeze dynamically loaded scripts before submission. If a live project must change, identify whether code was already parsed or will be read later by children.
8. Recover only invalid components. Preserve valid videos, probes, scores, and manifests, then rerun the smallest failed stage.
9. Generate headline numbers from canonical records into a machine-readable numeric file. Make figures and prose warn or fail on drift.
10. Run semantic release audit after mechanical tests. Capture terminal Slurm accounting only after allocations end, then rerun the final gate.

## Rules (hard)

- Treat done as a claim, because completion requires validated canonical payloads. [FL G39]
- Derive counts from filenames and records rather than self-report, because a process cannot independently attest its own work. [FL G36]
- Hash evidence sources, because readable paths can change after a result is summarized. [FL G36]
- Freeze grids and backends before execution, because post-hoc protocol changes invalidate comparisons. [FL D6; FL D8]
- Preserve every valid component during recovery, because regeneration wastes compute and can introduce new variation. [FL G46]
- Keep exact-resume state complete, because weights alone omit optimizer, RNG, sampler, and protocol history. [FL G33]
- Do not infer permission to publish from local completion, because publication is a separate user decision. [FL D9a]

## Pitfalls seen in production

- Symptom: a readable marker points to a missing probe or truncated checkpoint. Cause: the marker was trusted without payload validation. Fix: validate the canonical artifact before retry or release. [FL G39]
- Symptom: a manifest's top-level counts agree but nested fit rows are empty. Cause: internal consistency replaced evidence validation. Fix: reject malformed nested records. [FL G36]
- Symptom: task state says done but terminal trainer version is missing. Cause: wrapper completion and scientific completion diverged. Fix: validate the declared checkpoint and enqueue one resume. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: a project finishes both seeds but loads mixed code. Cause: a child read a script edited after submission. Fix: freeze every dynamically loaded controller and worker file. [FL G44]
- Symptom: release reports final accounting while the allocation still runs. Cause: terminal scheduler evidence is unavailable in-job. Fix: mark preterminal state, capture accounting after exit, and rerun gates. [FL G45]
- Symptom: a paper builds cleanly but contains stale claims. Cause: reference checks replaced semantic audit. Fix: run both mechanical and semantic gates in order. [FL G42]
- Symptom: an invalid drift score causes full regeneration. Cause: recovery was stage-blind. Fix: keep valid videos, teacher probes, and FVD, then rerun only drift. [FL G46]

## Pointers

- Related skills: every operational skill, especially `storage-and-caches.md`, `telemetry-and-monitoring.md`, and `multi-node-training.md`.
- Phase 1 evidence: `docs/COMPENDIUM.md`, `docs/CASE-STUDIES.md`, and `docs/sources/` after source promotion.
- Compendium themes: 7, 8, 10, and 12.
