---
name: telemetry-and-monitoring
description: Load when instrumenting GPU jobs, interpreting utilization and memory, producing rolling fleet summaries, or preserving compute evidence.
---

# Telemetry and Monitoring

## When to use

- Add monitoring to a real training, inference, or evaluation job.
- Distinguish working, reserved-idle, free-idle, and unreachable GPUs.
- Diagnose low utilization across asynchronous workload phases.
- Produce immutable usage summaries and post-hoc plots.

## Procedure

1. Set a scratch telemetry directory per project or run:
   ```bash
   export PROJECT=my-project
   export TELEMETRY_DIR="/scratch/$USER/$PROJECT/telemetry"
   ```
2. Start one logger on every GPU host near the top of the Slurm step:
   ```bash
   bin/gpu_telemetry.sh "$TELEMETRY_DIR" 30 &
   telemetry_pid=$!
   trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
   ```
3. Stream training loss, step, branch state, allocated and peak VRAM, and compute tags to WandB. Keep real projects online or synchronized later.
4. Write per-item manifests for inference with batch, elapsed generation time, timed-pass status, host, job, and stable item identity.
5. Use point probes only to classify the current phase. Do not infer long-window efficiency from one sample.
6. Summarize a complete recent window:
   ```bash
   python3 infra/python/gpu_telemetry_summary.py --telemetry-dir "$TELEMETRY_DIR" --window-seconds 300 --output summary.json
   ```
7. Read active-sample fraction, mean utilization, and mean memory fraction together. Join them with allocated GPU count, spool depth, trainer state, and model-loading state.
8. Produce post-hoc plots and JSON evidence:
   ```bash
   python3 infra/python/plot_compute_usage.py --telemetry-dir "$TELEMETRY_DIR" --output-dir out/compute --tag project
   ```
9. Archive compact CSV or hash-bound summaries needed for paper claims on persistent work storage.

## Rules (hard)

- Start telemetry on every host, because partial host coverage can hide idle or failed devices. [FL G16]
- Keep real projects in WandB, because GPU allocation without step and queue context is not auditable progress. [memory-facts]
- Interpret GPU-Util as kernel-active time, because it is not occupancy, SM coverage, or model FLOP utilization.
- Use rolling and point views together, because video pipelines alternate denoise, decode, score, synchronization, and I/O. [FL G16]
- Keep timed batch-one records separate, because utilization optimization must not contaminate reported latency. [FL G14]
- Hash live telemetry prefixes before citing them, because a logger can append after the diagnostic is captured. [VSAO gpu_telemetry_summary.py]

## Pitfalls seen in production

- Symptom: 47 GPUs are allocated but only 12 work. Cause: both scientifically valid spools drained. Fix: publish frozen repeatability work or release capacity. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: five-minute active fraction is 23.6 percent. Cause: queue visibility and supply failed, not calibrated tasks. Fix: restore compatible work and remeasure, which reached 81.5 percent across 41 GPUs. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: memory is high while utilization is zero. Cause: a trainer waits for a complete 64-record transaction. Fix: inspect pending records and heartbeats before killing it. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: a scorer appears idle in a point sample. Cause: it holds about 11 GB and runs in bursts. Fix: classify it over a full rolling window. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]
- Symptom: utilization swings from zero to 100 percent. Cause: normal load, denoise, decode, encode, or write phases. Fix: optimize time-averaged useful throughput. [FL G16]
- Symptom: a summary misses two GPUs. Cause: not every active queue or host was enumerated. Fix: state both allocated nodes and physical GPUs, then validate host coverage. [VSAO report-vsao.md, Telemetry]

## Live fleet views with all-smi

[all-smi](https://github.com/lablup/all-smi) (Lablup, Apache-2.0) gives a live
nvidia-smi-style view across many nodes at once. It complements, and does not
replace, the durable CSV logging above.

Setup caveats learned in production:

1. all-smi does not discover your Slurm allocation. Build the node list
   yourself from the scheduler: `squeue -u $USER -h -t R -o %N | xargs -n1
   scontrol show hostnames | sort -u`, then pass it as
   `--ssh user@node1,user@node2,...`. `bin/allsmi_view.sh` wraps exactly this.
2. Clusters recycle node names. A recycled name comes back with a different
   SSH host key, and all-smi's default strict host-key checking silently
   drops that node from the view; the symptom reads as missing GPUs, not as
   an error. On a trusted intra-cluster fabric run with
   `--ssh-strict-host-key no`.
3. Set a per-node connect timeout (`--ssh-timeout-secs 8`) so one dead or
   slow node cannot stall the whole view.

## Pointers

- Related skills: `gpu-utilization-batching.md`, `slurm-fleets-and-spools.md`, `provenance-and-repro.md`.
- Scripts: `bin/allsmi_view.sh`, `bin/gpu_telemetry.sh`, `bin/slurm_snapshot.sh`, `infra/python/gpu_telemetry_summary.py`, `infra/python/plot_compute_usage.py`.
- Compendium theme: 8.
