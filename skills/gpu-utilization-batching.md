---
name: gpu-utilization-batching
description: Load when calibrating batch size, diagnosing low GPU occupancy, separating timed measurements from bulk throughput, or choosing safe co-location.
---

# GPU Utilization and Batching

## When to use

- Choose inference batch, training microbatch, or gradient accumulation.
- Explain low memory occupancy or intermittent `nvidia-smi` utilization.
- Increase project throughput without invalidating latency measurements.
- Decide whether duplicate processes should share a GPU.

## Procedure

1. Define workload families by model, resolution, frames, denoise steps, decode path, scorer, and training phase.
2. Run one representative task per family through its full cycle. Include model load, denoise, decode, scoring, backward, optimizer step two, probe, and save where applicable.
3. Start `bin/gpu_telemetry.sh "$TELEMETRY_DIR" 30` before the trial and record framework allocated and reserved peaks at phase boundaries.
4. Sweep batches conservatively, such as 1, 2, 4, then intermediate values near the limit. Stop after the first OOM and retain the last stable complete-cycle point.
5. Target roughly 70 to 85 percent VRAM with 10 to 15 percent headroom, while also checking host RAM, scratch I/O, and complete-cycle throughput.
6. Compare throughput at each stable batch. Keep batch one when the path is already compute-saturated or batching changes semantics.
7. Freeze a per-family table before enqueueing the project. Record batch, expected peak, limiting phase, and exception rationale.
8. Run reported latency or FPS in exclusive batch-one passes. Mark bulk records untimed in manifests.
9. Preserve one independent RNG generator per prompt and seed inside batches. Compare batched and single outputs before mixing any study arm.
10. Use `infra/python/gpu_telemetry_summary.py` over at least one complete rolling window, then join the result with spool and scheduler state.

## Rules (hard)

- Calibrate before scale-out, because mid-project tuning changes work shapes and wastes completed spooling. [FL G15]
- Observe decode and optimizer step two, because first forward or first step can miss the true peak. [FL G14; VSAO HANDOVER.md:150-154]
- Keep batch-one timing exclusive, because co-located or batched timing changes the reported system. [FL G14; VG1 worker_loop.sh:4-18]
- Measure throughput rather than infer it from free HBM, because a compute-saturated model may slow under batching. [VG1 README.md:105]
- Do not fill spare GPUs with undeclared endpoints, because scientific validity outranks occupancy. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Freeze batching within a study arm, because batch arithmetic and shared cache decisions can change outputs. [VG1 RELEASE.md:250-255]

## Pitfalls seen in production

- Symptom: batch one uses only 30 GB, but larger batches do not improve throughput. Cause: Wan was compute-saturated at 32K tokens. Fix: keep batch one after the measured 0.92 to 0.90 videos-per-minute anti-result. [VG1 README.md:105; VG1 PROGRESS.md:42]
- Symptom: denoising looks safe but decode OOMs. Cause: batch-four VAE decode peaked at 91 GB after 48 to 54 GB denoising. Fix: calibrate the full cycle and use batch four only on the measured 60-second shape. [FL G14]
- Symptom: CUDA memory fits but the process prints `Killed`. Cause: fp32 pixel staging exhausted host cgroup memory. Fix: reduce decode batch or chunk and offload samples. [FL G15]
- Symptom: two co-located LTX workers underperform one. Cause: compute contention. Fix: retain one worker per GPU for that family. [VSAO CALIBRATION.md:50-56]
- Symptom: GB200 shows only 42 percent memory occupancy despite 85 to 100 percent utilization. Cause: the path is compute-bound with no batching route. Fix: accept the measured exception and optimize the real bottleneck. [VSAO vista/CALIBRATION.md:9,18-19]
- Symptom: zero-utilization samples appear between bursts. Cause: sequential decode, write, synchronization, or queue waits. Fix: judge complete-cycle rolling activity, not one point sample. [FL G14; FL G16]

## Pointers

- Related skills: `gpu-memory-hierarchy.md`, `telemetry-and-monitoring.md`, `provenance-and-repro.md`.
- Scripts: `bin/gpu_telemetry.sh`, `infra/python/gpu_telemetry_summary.py`, `infra/python/plot_compute_usage.py`.
- Compendium themes: 1, 2, 3, 7, and 8.
