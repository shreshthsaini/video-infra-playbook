# Case Studies

These cases trace each project's workload, infrastructure failures, repairs,
measured evidence, and lessons that transfer to later video projects.

## 1. CachedSearch: deterministic inference at fleet scale

### Workload and hardware

CachedSearch combined training-free transformer caching with test-time search.
Each rollout used one Vista GH200, while the fleet sharded prompts, seeds, and
variants across nodes. [VG1 RELEASE.md:182-187,240-257]

The project covered Wan, CogVideoX, HunyuanVideo, and LTX pipelines. It
coupled generation with verifier scoring, cache sweeps, full-compute recommit,
and paper extraction from append-only records. [VG1 RELEASE.md:188-229]

### Five defining infrastructure problems

1. **Empty HBM looked like batching headroom.** Wan2.1-1.3B used about 30 GB
   at batch one, but full throughput fell from 0.92 to 0.90 videos per minute
   under batching. Cached throughput rose only 8 percent, and batch eight
   OOMed. [VG1 RELEASE.md:93,192; VG1 README.md:105]

   The model was compute-saturated at batch one for 32K tokens. A sweep over
   batches one, two, four, six, and eight established batch one as the measured
   production point. [VG1 PROGRESS.md:42; VG1 bench_batch.py:1-37]

2. **Co-location could corrupt reported latency.** Generation and gate tasks
   produced reported speedups and cost terms, so they ran exclusively, one
   per node. Only untimed scoring and analysis could opt into two slots.
   [VG1 worker_loop.sh:4-18,56-57]

   Fixed task-class reservations once left 16 nodes idle after one class
   drained. The corrected policy let single-slot nodes claim any task while
   retaining timed-task exclusivity. [VG1 report-videogen1.md,
   Measurement-integrity policy]

3. **Cache state had hidden branch semantics.** Wan needed two independent
   branch states for sequential classifier-free guidance calls. CogVideoX,
   LTX, and guidance-distilled Hunyuan needed one. [VG1
   NOTES-e4-agent.md:63-68; VG1 NOTES-gate5-agent.md:143-148]

   The wrong two-branch default silently halved step counters and corrupted
   warmup, cooldown, and skips. Per-model assertions converted this into a
   fast failure. [VG1 NOTES-e4-agent.md:63-68]

4. **Resume required solver state, not only latents.** Correct pause and
   resume serialized CachedTransformer branches plus UniPC model-output,
   timestep, order, sample, and step-index state. [VG1
   NOTES-e6-agent.md:8-26]

   With that state, a CPU pause at step eight and resume through step 20 was
   bit-exact. A latent stash plus a begin index was not. [VG1
   NOTES-e6-agent.md:8-26]

5. **Direct scheduler use did not scale.** Workers atomically renamed tasks
   through pending, running, done, and failed directories. A stale-claim
   reaper returned work owned by dead Slurm jobs. [VG1
   worker_loop.sh:15,34-70]

   Four walltime-killed claims still stranded and were manually requeued.
   The project remained resumable, but the incident showed why recovery
   cannot depend only on a busy worker reaching its local reaper. [VG1
   NOTES-a1-agent.md:95-98]

### What was built

- A 16-node, 16-task, 12-hour fleet used one job slot. Even a 64-node VBench
  sweep remained one scheduler job, with long single workers for stragglers.
  [VG1 PROGRESS.md:49]

- Candidate JSONL rows recorded seed, latency, verifier score, tau, and every
  branch's compute and skip counts. Readers re-globbed shards, tolerated torn
  final lines, and deduplicated stable keys. [VG1 RELEASE.md:198-229]

- Deterministic full rollouts were computed once for all tau arms. Recommit
  regenerated a selected winner exactly on the fixed stack. [VG1
  RELEASE.md:240-257]

- Batched output showed about 0.6 reduction-order drift and a batch-averaged
  skip schedule, so no arm mixed batched and single execution. Cross-stack
  bitwise equality was not claimed. [VG1 RELEASE.md:194-196,244-255]

- The I/O path used torchcodec for reads, bounded x264 threads for writes,
  and an explicit zero-to-one float conversion before uint8. [VG1
  video_io.py:94-115]

### Numbers that prove the result

- Wan2.1-1.3B at 480 by 832, 81 frames, and 50 steps ran in 68.3 seconds full
  and 34.7 seconds cached, a 1.97x speedup with 26 skips of 50.
  [VG1 RELEASE.md:188-196]

- CogVideoX-5B measured 91.3 versus 44.3 seconds, or 2.06x. Wan2.2-TI2V-5B
  measured 213 versus 104 seconds, or 2.05x. [VG1
  NOTES-gate5-agent.md:121-161]

- HunyuanVideo-13B measured 87 to 97 seconds full and 39.5 cached, or 2.19x.
  LTX-Video-2B measured 10.8 seconds steady full and 4.6 cached, or 2.3 to
  2.63x. [VG1 NOTES-gate5-agent.md:121-161]

- At tau 0.10, 25, 50, and 100 steps produced skip fractions of 32, 51, and
  66 percent and speedups of 1.41, 1.97, and 2.80x. [VG1 PROGRESS.md:148]

- Best-of-eight full search cost 547 seconds. Cached recommit cost 346 seconds
  and retained 94.7 percent of the gain at 63.3 percent of the cost.
  [VG1 report-videogen1.md, Wall-clock numbers]

- The project used about 414 GH200 GPU-hours of pure generation and produced
  more than 31,000 scored rollouts. [VG1 PROGRESS.md:38-43,146]

### What transfers

- Benchmark batching before equating free memory with free throughput. Keep
  timed batch-one passes exclusive. [VG1 README.md:105; VG1
  worker_loop.sh:4-18]

- Make cache branch identity, solver state, and fixed-stack determinism part
  of the checkpoint contract. [VG1 NOTES-e4-agent.md:63-68; VG1
  NOTES-e6-agent.md:8-26]

- Scale independent scientific units through a fleet and idempotent spool,
  then persist the compact evidence crux. CachedSearch archived 515 files
  totaling 39,376,860 bytes while leaving 89 GB of reproducible videos on
  scratch. [VG1 report-videogen1.md, Storage layout]

## 2. Forcing Laws / Drift Atlas: distributed distillation on GH200

### Workload and hardware

Forcing Laws studied teacher-distillation post-training with long-horizon
inference, multi-node Self-Forcing training, checkpoint evaluation, and a
fail-closed paper evidence chain. [FL D5; FL D9]

The validated topology used six one-GPU GH200 nodes with torchrun and full
FSDP sharding. SDPA and FlexAttention stayed fixed as one experimental
control. [FL D8; FL G4; FL G19]

### Five defining infrastructure problems

1. **The replacement launcher had hidden seams.** It had to merge three YAML
   layers, reproduce upstream CLI-injected keys, set working WandB identity,
   provide an existing prompt file, and resolve the initial checkpoint.
   [FL G19]

   Hybrid full sharding assumed eight GPUs per node. On one-GPU GH200 nodes it
   became replication and reached 93 GB, so training required full sharding.
   [FL G19]

2. **Memory peaks moved by phase and tier.** A vendored inference call without
   no-grad retained about 80 GB of autograd state. Adapter guards removed the
   graph without changing inference. [FL G6]

   Batch four used 48 to 54 GB during denoising and peaked at 91 GB in VAE
   decode of four 60-second clips. It produced about three to four times the
   earlier batch-one throughput. [FL G14]

   At 240 seconds, about 18.5 GB of fp32 host pixels per sample made batch four
   a host-cgroup OOM and batch two the safe shape. [FL G15]

   A batch-four probe also retained about 24 GiB of KV and cross-attention
   cache per GPU before batch-two training. Releasing the owning objects, not
   only the allocator, restored headroom. [FL G30]

3. **The crash-taxonomy week exposed supervisor flaws.** An unguarded spool
   task stole a pilot rank and triggered an NCCL watchdog after five hours at
   u=0.155. Killing worker_loop by pattern then ended two six-node fleets.
   [FL G20; FL G22]

   A stale ChildFailedError line nearly killed a healthy adopted run. Later,
   two to three dueling supervisors churned five launches after a timed-out
   background command still ran. [FL G23; FL G24]

   Flock enforced one supervisor, process censuses followed every launch, and
   crash signatures became relative to the current launch epoch. [FL G23;
   FL G24]

   A rendezvous death emitted RendezvousConnectionError and DistNetworkError,
   which the original CUDA-focused classifier missed. Ghost re-adoption and a
   broad pre-kill across all hosts completed the taxonomy. [FL G25]

4. **Distributed state outlived failed ranks.** Three OOMed ranks left three
   peers inside a collective. Cancelling only the failed Slurm step preserved
   the allocation and let the controller continue. [FL G31]

   Shared generated compiler sources caused malformed-code symptoms and
   rendezvous collapse. Source caches became node, run, and retry specific,
   while safe graph reuse stayed sequential and node-local. [FL G27; FL G32]

   A visible checkpoint was not durable until gathers and save confirmation
   completed. Weight-only state also lacked optimizer, RNG, sampler, branch,
   probe, and compute-ledger state. [FL G33; FL G34]

5. **Compute completion did not close evidence.** The gate validated five
   manifests, 20 planned paired evaluations, two recovered paired evaluations,
   canonical grids, scorer provenance, postprocessing, crux, tests, paper
   audit, accounting, and fleet release. [FL G29]

   It rejected readable but incomplete manifests, stale paper claims,
   malformed nested records, path aliases, and missing hashes. [FL G36;
   FL G37; FL G39]

### What was built

- A fleet and spool held nodes across generation, scoring, and training. GPU
  tasks waited when pilots owned the hardware, and cleanup targeted exact
  pidfiles rather than fleet workers. [FL G13; FL G20; FL G22]

- Every GPU host wrote 30-second memory, utilization, and power samples.
  Training streamed step metrics to WandB, and inference wrote per-video
  timing manifests. [FL G16]

- Retry decisions came from rank zero after time-sensitive barriers. Every
  rank consumed the same action and collision-safe quarantine handled stale
  claim copies. [FL G40]

- Closeout preserved the six-rank training world size. Four surplus nodes ran
  evaluation in a disjoint Slurm step behind a ready marker and exited only
  after producer completion plus an empty queue. [FL G43]

- Evaluation stages had separate one-hour generation, one-hour FVD, 30-minute
  drift, and five-minute termination bounds. Valid components survived a
  partial retry. [FL G35; FL G46]

- Dynamically loaded controller, worker, retry, and launch scripts were frozen
  before submission. Terminal Slurm accounting was captured only after the
  allocation ended, then release gates reran. [FL G44; FL G45]

### Numbers that prove the result

- A GH200 smoke measured a 50-step, 81-frame, 480p teacher at 88 to 90 seconds
  and 18.9 GB peak. The Self-Forcing student took 7.5 seconds. [FL G7]

- One DMD rollout training step at batch one with gradient checkpointing and
  a 1.3B real-score override peaked at 52.7 GB. [FL G7]

- A 500-video encoder A/B moved aesthetic quality from 0.7803 to 0.7506,
  about three points, while imaging quality moved 0.005 and subject
  consistency 0.0001. [FL G11]

- Official VBench aggregation produced 85.37 Quality, 81.01 Semantic, and
  84.50 Total, each within plus or minus 0.3 of the cited values. [FL G12]

- The crash-heavy main project consumed a modest fraction of its allocation
  because most turbulence was orchestration rather than full recomputation.
  [FL G25]

- The final three-allocation record remained bounded to a small fraction of
  the project allocation. [FL D14]

### What transfers

- Calibrate the complete model-load, rollout, decode, backward, optimizer,
  probe, and save cycle before scaling. [FL G14; FL G15; FL G30]

- Give supervisors, log epochs, rendezvous actions, and cleanup scopes explicit
  identity. Process grep alone is not a control plane. [FL G23; FL G24;
  FL G40]

- Preserve validated world size, isolate generated compiler sources, and
  define checkpoints by complete state plus a completed save boundary.
  [FL G27; FL G33; FL G34; FL G43]

- Separate compute success, semantic release audit, and scheduler accounting.
  Release idle nodes when registered work ends. [FL G26; FL G42; FL G45]

## 3. VSAO: asynchronous RL on a heterogeneous 47-GPU fleet

### Workload and hardware

VSAO used asynchronous rollout producers and serial trainers. Workers sampled
the newest policy, stored per-step log probabilities, decoded and scored in
process, and atomically published trajectory records. [VSAO DESIGN.md:8-24]

The LS6 peak was 47 GPUs across 18 nodes: 12 three-GPU A100 nodes, two A100
development nodes, three A100-small nodes, and one two-GPU H100 node.
[VSAO COORDINATION.md:267-270]

Vista added four nodes with four 185 GiB GB200 devices each for 81-frame Wan
and CogVideoX work that did not fit the A100-40GB path. [VSAO
vista/README.md:355-357]

### Five defining infrastructure problems

1. **Node ownership hid idle GPUs.** A full A100 allocation contained three
   devices, but many evaluations used one. The outer node spool could not see
   an independent per-GPU queue. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

   VSAO added a packed outer claim with one slot worker per physical GPU. The
   node remained exclusive to the pack while slots atomically drained untimed
   inner work. [VSAO RUNBOOK-RL.md:56-59]

   Bridge fan-out initially divided free GPUs by three and underfilled mixed
   one, two, and three-GPU nodes. One bounded claim per observed free GPU
   restored 35 of 45 working A100-class GPUs. [VSAO
   GPU-UTILIZATION-LEARNINGS 2026-07-19]

2. **Rollout supply could outrun training.** Wan capped backlog at 160 records
   and refilled below 80. The trainer rejected lag beyond eight policy
   versions. [VSAO ensure_rollout_capacity.sh:28-30; VSAO
   RUNBOOK-RL.md:176-178]

   A full Wan rollout node produced about 154 records per hour, while one
   trainer consumed about 96. Capacity therefore followed queue deficit, not
   equal node counts per run. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

3. **Packed logical identity duplicated trajectories.** Different physical
   slots both appeared as cuda:0 and replayed the same deterministic prompt
   stream for seed 6. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

   Three duplicate pairs entered the first 64-record batch, and three more
   pending records duplicated consumed records. Six were quarantined and 90
   unique records remained. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

   Physical slot became provenance, while unique worker_stream_id controlled
   deterministic seeding and appeared in every trajectory and heartbeat. The
   trainer rejected duplicates before policy movement. [VSAO
   GPU-UTILIZATION-LEARNINGS 2026-07-19]

4. **Task completion and trainer completion diverged.** Wan seed 3204 had a
   done task and exact version-30 state, but version 37 was absent after a
   fleet boundary. An idempotent helper enqueued one resume. [VSAO
   GPU-UTILIZATION-LEARNINGS 2026-07-21]

   Exact resume combined the newest policy with critic, normalizer, counters,
   and optimizer state. ABORT finished the current unit, saved state, and left
   fleet allocations available. [VSAO RUNBOOK-RL.md:243-253; VSAO
   vista/README.md:265-272]

   Bounded TERM then KILL replaced an unbounded wait that held one node at
   24 GB and zero percent utilization. Lock takeover used nonce, jitter, and
   durable queue evidence. [VSAO SMOKE.md:2097-2124; VSAO
   report-vsao.md, Trainer single-instance lock]

5. **Allocated capacity was not useful capacity.** One July 21 audit found 47
   allocated GPUs but only 12 working and 35 truly free. [VSAO
   GPU-UTILIZATION-LEARNINGS 2026-07-21]

   A five-minute, 470-sample window measured 23.6 percent active samples,
   23.45 percent mean utilization, and 20.1 percent memory occupancy because
   both spools had drained. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

   Recovery froze 24 evaluator panels, hash-bound 1,536 source videos, and
   queued 48 independent reruns already justified as repeatability evidence.
   Activity recovered to 81.5 percent across 41 GPUs. [VSAO
   GPU-UTILIZATION-LEARNINGS 2026-07-21]

### What was built

- Adaptive rollout wrappers used low and high watermarks, fresh heartbeats,
  version guards, and explicit throttled status. [VSAO
  ensure_rollout_capacity.sh:28-30; VSAO RUNBOOK-RL.md:176-178]

- Each trajectory used temporary-write plus atomic rename. A complete trainer
  update was a validated 64-record transaction with duplicate audit and
  rollback to pending when short. [VSAO DESIGN.md:8-24; VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

- GRPO and SAO used manual data parallel reduction after DDP static graph plus
  no-sync crashed. Rank zero preserved whole-group advantage and checkpoint
  semantics. [VSAO SMOKE.md:1497-1512,2041-2069]

- Telemetry combined point classification with a five-minute all-host view of
  active fraction, mean utilization, and memory fraction. Immutable JSON tied
  each conclusion to source CSV hashes. [VSAO gpu_telemetry_summary.py]

### Numbers that prove the result

- A verified 47-GPU window measured 91.8 percent active samples, 90.3 percent
  mean utilization, and 70.7 percent memory occupancy. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- The latest full-shape trace used 474 samples and measured 92.4 percent
  active, 91.8 percent mean utilization, and 79.3 percent memory occupancy.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- A100 per-worker rates were 364.2 rollouts per hour for LTX, 51.3 for Wan,
  and 37.6 for tiled CogVideoX. Their three-GPU node rates were about 1,090,
  154, and 113. [VSAO CALIBRATION.md:50-56]

- Three-GPU GRPO reduced step time from 1,893 to 639 seconds, a 2.96x speedup.
  SAO-Wan took 2,437 to 2,492 seconds per 64-record update. [VSAO
  report-vsao.md, DP trainer design]

- GB200 Wan produced 104.9 rollouts per hour per worker and 8,019 corpus
  records in about four hours and 50 minutes. [VSAO
  vista/CALIBRATION.md:9,18-19; VSAO vista/SMOKE.md:73-75]

- GB200 used only about 42 percent of 185 GiB for that path, but six or more
  concurrent 81-frame runs saturated scratch. Stable operation used about
  four to five runs, roughly one per node. [VSAO vista/CALIBRATION.md:9,18-19;
  VSAO report-vsao.md, Vista GB200 side]

### What transfers

- Couple producer capacity to lag and queue watermarks. More trajectories are
  waste once they cannot enter a valid update. [VSAO
  ensure_rollout_capacity.sh:28-30]

- Separate physical slot, logical device, and deterministic stream identity.
  Audit duplicates before policy movement and keep rollback accounting.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- Require terminal checkpoints, complete optimizer-aware state, and bounded
  process cleanup. A done wrapper is not proof of trainer completion.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Use point and rolling telemetry together. Fill spare capacity only with
  frozen useful evidence, and release it when no such work exists.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Calibrate storage alongside HBM. On GB200, shared scratch throughput set a
  lower packing ceiling than device memory. [VSAO report-vsao.md,
  Vista GB200 side]
