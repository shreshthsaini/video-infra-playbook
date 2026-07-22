# Cross-Project Video Infrastructure Compendium

This compendium correlates operational evidence from CachedSearch, Forcing
Laws / Drift Atlas, and VSAO by system behavior rather than repository. Each
list entry ends with its provenance.

## 1. GPU memory: where VRAM actually goes

Video workloads do not have one stable memory footprint. Weights establish a
floor, while activations, attention caches, decode staging, and optimizer state
create phase-specific peaks. The most useful calibration therefore follows a
complete generation or training cycle, not merely model load or the first
forward pass. Across all three projects, memory ownership mattered as much as
allocator state because a live Python reference could keep tens of GiB resident.

### Resident model state

- Wan2.1-1.3B used about 30 GB on one GH200 at batch one in CachedSearch.
  This was a practical baseline, not a complete statement of the peak for
  every resolution, frame count, or decode phase. [VG1 RELEASE.md:93,192]

- CogVideoX-5B used about 25 GB with weights in bf16 end to end. Its VAE
  followed a model-specific dtype exception described in Theme 7.
  [VG1 NOTES-e4-agent.md:43-46]

- HunyuanVideo-13B occupied about 43 GB in weights on GH200. The breakdown
  was about 25 GB for the 13B transformer, about 16 GB for the Llama-8B text
  encoder, plus CLIP-L and the VAE. [VG1 RELEASE.md:192-193; VG1
  NOTES-gate5-agent.md:58-60,126]

- The VideoScore verifier contained 8.27B parameters and occupied about
  17 GB in bf16. A scoring model can therefore be a first-class memory
  resident rather than a negligible postprocessing process. [VG1
  NOTES-evalstack-agent.md:40]

- On an A100-40GB, the Hunyuan transformer alone occupied 23.88 GB for
  12.82B bf16 parameters, while all Hunyuan weights together occupied
  38.55 GB. Every phase exceeded 40 GB without about 14 GB of Llama text
  encoder offload. [VSAO SMOKE.md:721-741]

### Activations and optimizer state

- A direct call into vendored Self-Forcing inference without no-grad built an
  autograd graph through the rollout. The graph consumed about 80 GB for 21
  latent frames and four steps on a 1.3B model, enough to OOM a 95 GB GH200.
  [FL G6]

- One Wan critic transition with about 14,000 tokens retained about 30 GiB
  of saved activations when the critic backbone lacked activation
  checkpointing. Enabling it reduced measured peak memory from 38.91 GiB to
  19.32 GiB. [VSAO SMOKE.md:1136-1176]

- The passing VSAO end-to-end critic had silently inherited gradient
  checkpointing from the policy before deepcopy. A separately constructed
  critic did not, which explains why apparently equivalent code paths had
  very different activation envelopes. [VSAO SMOKE.md:1136-1176]

- An optimizer can move the true steady-state peak to the second update.
  SAO-Wan at 33 frames passed its first step, then OOMed after optimizer-state
  materialization and resumed transactionally with policy microbatch reduced
  from four to two. [VSAO HANDOVER.md:150-154]

- Rank-zero-only services can create a distributed memory asymmetry. VSAO
  rank zero held about 18.6 GiB of policy state, about 11 GiB of critic state,
  about 9.6 GiB of T5 state, and activations, reaching 38.49 of 39.49 GiB.
  [VSAO SMOKE.md:2071-2088]

- Encoding the finite prompt pool once, sharing it bitwise across ranks, and
  dropping T5, the pipeline, and the VAE reduced VSAO rank-zero residency to
  about 28 to 29 GiB. It also removed about 0.5 seconds of per-step prompt
  encoding. [VSAO SMOKE.md:2071-2088]

### KV caches and horizon-scaled buffers

- Drift Atlas rCM used a full-length KV cache with no sliding window. At 240
  latents it required about 70 GB per sample, so batch one was the correct
  60-second operating point even on GH200. [FL G15]

- The Self-Forcing, Rolling, and LongLive families used windowed KV state at
  about 4 GB per sample and calibrated to batches 12 through 16 for the
  60-second atlas. Causal Forcing chunk and frame variants used about 7 GB per
  sample and calibrated to batch eight. [FL G15]

- A measurement probe used chunks of four while training used batch two. Its
  retained 30-layer KV and cross-attention cache consumed about 24 GiB per
  GPU, so the next training step tried to allocate a new batch-two cache while
  the batch-four cache was still live. [FL G30]

- Calling an allocator clear did not resolve that probe-cache incident. The
  owning Python object had to release its tensors after every rollout chunk
  before the intended HBM headroom returned. [FL G30]

- CachedSearch cache state also had semantic ownership. Correct pause and
  resume required every CachedTransformer branch state plus the UniPC
  multistep scheduler internals, not just latent tensors and a begin index.
  [VG1 NOTES-e6-agent.md:8-26]

### Decode staging and measured ceilings

- Drift Atlas batch four used 48 to 54 GB during denoising, then peaked at
  91 GB during VAE decode of four 60-second clips. Decode staging, not the
  DiT forward, set the safe maximum. [FL G14]

- A 240-second decode staged about 18.5 GB of fp32 pixels per sample in host
  memory. Batch four implied about 74 GB plus copies and triggered the Slurm
  cgroup OOM killer, while batch two was the calibrated host-memory ceiling.
  [FL G15]

- LongLive decoded a full video on the GPU and then created a second
  normalized fp32 copy of about 17 to 18 GB. Sequential fragmentation caused
  a CUDA OOM after three successful 240-second videos. [FL G17]

- The durable fix chunked and offloaded every sample, including batch one,
  and freed video tensors between calls. Horizon-scaled buffers need
  per-sample release even when the nominal batch is one. [FL G17]

- VSAO calibration on A100-40GB measured Wan at 32.68 GB torch peak and
  34,945 MiB nvidia-smi peak, or 85 percent. CogVideoX with VAE tiling used
  24.35 GB torch peak and 26,829 MiB nvidia-smi peak, or 65 percent.
  [VSAO CALIBRATION.md:50-56]

- VSAO calibration measured LTX at 22.35 GB torch peak and 31,177 MiB
  nvidia-smi peak on A100, or 76 percent. Hunyuan used 48.74 GB torch peak
  and 51,119 MiB nvidia-smi peak on H100, which allowed only one worker per
  80 GB GPU. [VSAO CALIBRATION.md:50-56]

- Wan 81-frame rollout on GB200 peaked at 78,752 MiB by nvidia-smi and
  32.8 GiB by torch. This was only about 42 percent of the 185 GiB device,
  but the path was compute-bound and had no available batching path.
  [VSAO vista/CALIBRATION.md:9,18-19]

- During live 81-frame SAO on GB200, the trainer used 55,865 MiB on GPU zero
  and seven rollout GPUs used about 35,447 MiB each while reporting 85 to
  100 percent utilization. Low memory fraction did not imply idle compute.
  [VSAO vista/SMOKE.md:126-127]

## 2. OOM taxonomy and diagnosis

The word OOM hides several different failures. A CUDA allocator exception, a
bare process kill, a login-node virtual-memory failure, and a second-step
optimizer peak require different responses. Diagnosis should begin with the
failure signature and phase boundary, then compare live object ownership,
host memory, and device memory. Shrinking the scientific task before locating
the failing resource can conceal the actual defect.

### Distinguish the failure domains

- A Python traceback ending in torch.cuda.OutOfMemoryError identifies the
  CUDA class. A bare Killed message with no traceback identifies the host or
  Slurm cgroup class in the documented Drift Atlas incidents. [FL G15]

- The Vista login nodes impose an 8 GB virtual-memory limit. Heavy torch or
  ImageReward import chains and the Hugging Face Xet download path can fail
  there even when no GPU is involved. [VG1 b1_rescore.py:281; memory-facts]

- Login-node process exhaustion is separate from memory exhaustion. The
  measured per-user process ceiling is 100, shared across sessions, and the
  operational stop threshold is 40 processes. [memory-facts]

- Wan CachedSearch batch eight produced a CUDA OOM during its batching
  sweep. The harness caught torch.cuda.OutOfMemoryError after measuring
  batches one, two, four, six, and eight. [VG1 bench_batch.py:1-37]

- CogVideoX could run untiled on GH200, but VAE tiling was the documented
  smaller-card fallback. Hunyuan enabled tiling because its 3D VAE decode was
  the peak-memory phase. [VG1 gen_cog.py:55-56; VG1 gen_hunyuan.py:29,68]

### Phase-specific CUDA failures

- A VSAO A100 trainer completed step one before failing at 38.51 of 39.49 GiB
  inside the Wan transformer forward. Splitting policy, critic, and rewards
  over three A100s fixed placement without shrinking the task. [VSAO
  SMOKE.md:577-582,638-641]

- The repaired split placed policy on cuda:0, critic on cuda:1, and reward
  models on cuda:2. GPU zero then used 33,321 MiB and retained 7.1 GiB of
  headroom. [VSAO SMOKE.md:577-582,638-641]

- The critic-pretraining OOM occurred at microbatch one despite expandable
  segments, embedding-cache use, and VAE offload. Missing activation
  checkpointing, not allocator fragmentation, was the root cause.
  [VSAO SMOKE.md:1136-1176]

- GRPO data parallel training retained policy microbatch two on A100-40GB.
  Static state was about 21.5 GiB, while microbatch-four activations added
  about 23 GiB and exceeded the device; the measured microbatch-two envelope
  was 33.3 to 34.1 GiB. [VSAO report-vsao.md, GPU memory learnings]

- A100 rank zero failed because prompt encoding and support models were
  rank-specific residents. Removing those services from every rank was a
  topology repair, not a batch-size workaround. [VSAO SMOKE.md:2071-2088]

- Self-Forcing FSDP with hybrid_full sharding assumed an eight-GPU node.
  Vista had one GPU per GH200 node, so intra-node sharding degenerated into
  full replication and reached 93 GB. Full sharding was required. [FL G19]

- Vendored inference code that omitted no-grad generated an accidental
  autograd graph. Decorating adapter calls with no-grad or inference mode
  eliminated the roughly 80 GB graph without changing inference semantics.
  [FL G6]

### Host and cgroup failures

- Drift Atlas 240-second batch four exhausted host memory because decoded
  fp32 pixels scaled with horizon. The remedy was batch two and all-node host
  memory, not CUDA allocator tuning. [FL G15]

- A Self-Forcing batch-12 top-up host-OOMed at 60 seconds when co-resident
  allocations differed from the calibration state. Finisher tasks therefore
  rechecked batch size against current node state and used batch six.
  [FL G17]

- Glibc malloc arenas can size themselves from the 144 physical Grace cores
  and reserve about 70 GB of virtual address space. MALLOC_ARENA_MAX=2 is a
  login-node protection, not a GPU-memory setting. [memory-facts]

- Unbounded x264 or x265 threading can cross the login process limit and
  yield a zero-byte file. The observable artifact looks like failed I/O, but
  the resource failure is process count. [memory-facts]

### Temporal signatures and recovery

- Passing one optimizer step is not proof that training fits. The second
  step can materialize optimizer state and become the steady-state maximum.
  [VSAO HANDOVER.md:150-154]

- A multi-rank CUDA failure can leave surviving ranks blocked in a
  collective. In the affected run, three ranks OOMed while three torchrun
  agents remained alive until only the failed Slurm step was cancelled. [FL G31]

- Cancelling the failed step, rather than the parent allocation, preserved
  the other nodes for recovery. Future launches also required bad-exit propagation
  and bounded retry-preamble timeouts. [FL G31]

- A slow GB200 resume can look like a hang. Unpickling a 12 to 15 GB trainer
  state took 20 to 48 minutes in one CPU thread; about 99 percent CPU with
  advancing user time meant progress, while near-zero CPU in D state with
  frozen reads indicated a real stall. [VSAO vista/README.md:355-357;
  memory-facts]

- Repeatedly killing a CPU-active resume resets the unpickle and creates a
  false restart cascade. Partial resume from policy plus critic re-warm was
  the practical iteration path when exact full-state load was unnecessary.
  [VSAO report-vsao.md, Vista GB200 side]

## 3. Batching and utilization doctrine

Batching is an empirical operating-point decision, not a universal throughput
switch. The reliable sequence is to calibrate a full workload cycle, freeze a
per-family table, and only then scale out. A project should aim for high but
safe memory occupancy while keeping reported latency in exclusive batch-one
runs. Compute saturation, unavailable batching paths, or scientific
comparability can justify a measured exception.

### Calibration before scale-out

- Drift Atlas formalized a three-stage order: test one task per workload
  family at trial batches, observe denoise and decode phases, freeze the
  family table, then enqueue the project. Mid-project tuning had already
  cost hours of re-spooling. [FL G15]

- The operating target was about 70 to 85 percent VRAM, leaving about 10 to
  15 percent headroom. The target applied to inference batching and training
  microbatch or gradient accumulation. [FL G14]

- Forcing Laws calibrated windowed Self-Forcing, Rolling, and LongLive at
  batches 12 through 16, Causal Forcing chunk and frame at batch eight,
  CF++ first-chunk mechanics at no more than batch four, and rCM and CausVid
  at batch one. [FL G15]

- CF++ had a first-chunk KV path that was not batch safe at eight. CausVid
  had a vendored KV cache hard-wired for batch one. These were implementation
  constraints, not reasons to co-locate duplicate compute-heavy processes.
  [FL G15]

- The 60-second Drift Atlas safe maximum was batch four because VAE decode
  peaked at 91 GB. Moving from the old 30 GB batch-one shape delivered about
  three to four times the bulk throughput. [FL G14]

- The 240-second operating point was batch two because host decode staging,
  rather than VRAM, became the limiting resource. Calibration must cover
  every relevant memory tier. [FL G15]

### Anti-results and justified exceptions

- CachedSearch found that batching Wan candidates did not raise full-model
  throughput: 0.92 videos per minute at batch one became 0.90. Cached
  throughput improved only 8 percent, and batch eight OOMed. [VG1
  README.md:105; VG1 PROGRESS.md:42]

- The CachedSearch interpretation was compute saturation at batch one for
  Wan-1.3B with 32K tokens. Idle VRAM was not usable throughput, so the study
  retained batch one. [VG1 README.md:105; VG1 PROGRESS.md:42]

- VSAO LTX on H100 produced 625.6 rollouts per hour with one worker. Two
  co-located workers produced about 578 rollouts per hour combined, so
  co-location was throughput parity or worse rather than a gain.
  [VSAO CALIBRATION.md:50-56; VSAO report-vsao.md, Quotable numbers]

- GB200 Wan at 81 frames remained at about 42 percent memory occupancy because
  the 50-step path was compute-bound and did not expose a batching route.
  The measured throughput was 104.9 rollouts per hour per worker.
  [VSAO vista/CALIBRATION.md:9,18-19]

- Scientifically useful occupancy is stricter than constant allocation. VSAO
  explicitly rejected new efficacy endpoints, metrics, and hyperparameters
  as filler when idle devices appeared. [VSAO GPU-UTILIZATION-LEARNINGS
  2026-07-21]

### Measurement integrity

- Every reported FPS or latency value in Drift Atlas came from a dedicated,
  exclusive batch-one pass. Bulk generation was untimed and batched.
  [FL G14]

- CachedSearch classified generation and gate tasks as timed. They ran one
  per node, while only untimed scoring and analysis could opt into two slots.
  Zero dual-slot timed claims meant no latency record was contaminated.
  [VG1 worker_loop.sh:4-18,56-57]

- CachedSearch never used multi-GPU parallelism within a rollout. It scaled
  across prompt, seed, and variant shards, with one GH200 per rollout.
  [VG1 RELEASE.md:182-187]

- VSAO calibration kept batch one for in-process VideoAlign scoring while
  scaling via one worker per physical GPU where the memory envelope allowed.
  [VSAO CALIBRATION.md:50-56]

- Wan on A100 produced 51.3 rollouts per hour per worker and about 154 per
  three-GPU node. The older two-generator plus one-scorer shape produced
  about 103 per node and remained only an OOM fallback. [VSAO
  CALIBRATION.md:50-56]

- CogVideoX on A100 produced 37.6 rollouts per hour per worker and about 113
  per three-GPU node with VAE tiling. Hunyuan on H100 produced 26.8 per hour
  with one worker. [VSAO CALIBRATION.md:50-56]

- Trainer batching can change optimization statistics without changing
  intake throughput. VSAO measured about 13.75 seconds per LTX-A100 record
  and about 84 seconds per Wan-H100 record; batch 64 improved gradient signal
  rather than records per second. [VSAO SMOKE.md:1927-1955]

- VSAO's effective batch had previously been 8 to 16 times below the target:
  arms ran batches 8 to 12 where the SAO paper used 128 rollouts per update
  and DanceGRPO used 192. [VSAO SMOKE.md:1927-1955]

### Reproducibility within batching

- Drift Atlas gave each batched prompt and seed pair its own torch.Generator
  so the initial noise matched single-sample runs. This preserved the frozen
  grid while accepting ordinary batched arithmetic. [FL G14]

- CachedSearch found about 0.6 kernel reduction-order drift between batched
  and single runs. Its adaptive-cache skip indicator also became averaged
  across the batch, producing one shared skip schedule. [VG1
  RELEASE.md:250-255]

- Because batching changed CachedSearch numerics and cache decisions, one
  study arm could not mix batched and unbatched outputs. A speed optimization
  that changes the experimental unit is a protocol change. [VG1
  RELEASE.md:250-255]

- Residual zero-utilization windows during Drift Atlas were accepted when
  they corresponded to sequential decode and write phases. The project
  optimized complete-cycle throughput, not a permanently maxed dashboard.
  [FL G14]

## 4. Fleet orchestration

All three projects converged on the same scheduler abstraction: acquire a
small number of durable multi-node allocations, then feed them through a
resumable task spool. This decouples scientific work from queue latency and
conserves scarce job slots. VSAO extended the pattern with a nested per-GPU
spool, shape-aware routing, and adaptive rollout capacity. The fleet is useful
only when each idle resource can see compatible, already justified work.

### Job shape and the outer spool

- Vista gh allowed 40 submitted jobs and about 20 running jobs per user, with
  at most 96 nodes. These limits made job slots scarcer than nodes.
  [memory-facts; VG1 PROGRESS.md:49]

- The preferred shape was one fleet job requesting many nodes, with one
  worker per node draining a shared spool. Forcing Laws replaced an initial
  32-element single-node array with this shape. [FL G13]

- CachedSearch used pending, running, done, and failed directories on
  scratch. A worker claimed a task by atomic rename, ran it, and moved the
  script and log to the terminal state. [VG1 worker_loop.sh:15,53-70]

- A stale-claim reaper checked whether the owning Slurm job was still
  running before returning work to pending. Four CachedSearch tasks were
  nevertheless stranded at a walltime boundary and needed manual recovery,
  which motivated controller-side reaping as well. [VG1 worker_loop.sh:34-46;
  VG1 NOTES-a1-agent.md:95-98]

- Reapers must fail closed when a scheduler query fails. An empty squeue
  result caused by scheduler trouble is not proof that every running claim is
  stale. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- CachedSearch used a 16-node, 16-task, 12-hour fleet in one job slot and
  long single workers for stragglers. Its separate 64-node, five-hour VBench
  sweep still expressed many nodes in one scheduler job. [VG1 PROGRESS.md:49]

- VSAO ran a multi-node A100 fleet within a 48-hour walltime, while a
  separate H100 trainer allocation and vm-small controller covered roles
  that required different hardware. [VSAO report-vsao.md, Scheduler patterns]

- The queue-driven contract keeps workers useful through successive declared
  phases. CachedSearch and Forcing Laws queued scoring behind generation and
  published the next wave before the current spool drained. [VG1
  workers_n16_12h.sbatch:6-8; FL G18]

- An allocation is not an instruction to consume the full reservation after
  useful work ends. Forcing Laws used a short bounded idle tail, then returned
  unused capacity. [FL G26]

- Slurm reserves nodes times walltime at submission. Every submission needs
  a balance check, and the account-wide burn stop forbids new GPU-node
  requests at a hard threshold well before exhaustion. [memory-facts; FL G21]

### Priority, safety, and task visibility

- A common fleet worker can claim any compatible pending task. During a live
  multi-node pilot, an unguarded generation task took one rank's GPU and the
  NCCL watchdog killed a five-hour training run at u=0.155. [FL G22]

- The safe coexistence rule was to gate spooled GPU tasks on current memory
  use, or move them to a deferred directory while training owned the devices.
  Killing the fleet worker itself terminated the Slurm step and cost two
  six-node fleets. [FL G20; FL G22]

- Deliberate restarts can move claims to failed and silently empty pending.
  Recovery must immediately return resumable claims to pending and remove
  superseded variants so workers cannot execute the wrong shape. [FL G13]

- Queue depth alone does not prove capacity can see work. A node-level worker
  cannot discover an inner per-GPU task until the controller publishes a
  bridge claim into the outer spool. [VSAO GPU-UTILIZATION-LEARNINGS
  2026-07-21]

- Prefix ordering was part of VSAO scheduling semantics. Pretraining prefixes
  sorted before RL, which sorted before evaluation; an m-rl prefix would have
  preempted a declared pretraining chain and was rejected. [VSAO
  report-vsao.md, Scheduler patterns]

- A full-node task must match its assigned hardware shape. A trainer needing
  one update of about 40 minutes plus load time was rejected from a two-hour
  development allocation and reserved for the full A100 partition.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

### The nested per-GPU spool

- VSAO's outer spool owned whole nodes. One outer packed claim launched one
  slot worker per physical GPU, and each slot atomically claimed an untimed
  single-GPU task from taskq_gpu. [VSAO RUNBOOK-RL.md:56-59]

- The packed node remained one exclusive outer claim, so per-GPU evaluations
  could not collide with a trainer that required the whole node. [VSAO
  RUNBOOK-RL.md:56-59]

- Bridge claims were bounded by observed free GPUs, pending inner tasks, and
  a cap of 12. A surplus claim exited after one idle minute and returned the
  node to the ordinary spool. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Dividing free GPUs by three assumed every node was a full A100 shape and
  underfilled a mixed fleet. Counting one potential bridge per observed free
  GPU restored 35 of 45 working A100-class GPUs while models were still
  loading. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- A packed claim can land on a one-GPU A100-small node, a two-GPU H100 node,
  or a three-GPU A100 node. The slot layer must derive physical capacity at
  runtime rather than hard-code a node shape. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

- One-task-per-slot packing created holes when a long held-out scorer shared
  a node with short tasks. Slots now keep claiming while useful work remains,
  and the pack revives exited siblings until the global inner queue is empty
  for one minute. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Every per-GPU task ran in its own process group. Shutdown targeted the
  process group so a Python grandchild could not retain a GPU after its shell
  wrapper exited. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- An outer pack had to reap every slot child before releasing its node. Three
  briefly orphaned slot shells on one compute host demonstrated that an empty inner
  queue did not prove a clean process tree. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

### Adaptive supply and renewal

- Rollout nodes were burst capacity. The controller started replacements
  below a frozen low-water mark, produced until a high-water cap, then
  returned the node to the common spool. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Wan's high-water cap was 160 records and its low-water mark was half the
  cap. This bounded staleness and prevented supply from outrunning trainer
  intake indefinitely. [VSAO ensure_rollout_capacity.sh:28-30]

- Under the calibrated three-worker topology, a Wan A100 node produced about
  154 records per hour while one trainer consumed about 96 records per hour.
  Capacity therefore had to be balanced by queue deficit, not equal node
  counts per run. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- When a free full node appeared, VSAO temporarily increased capacity only
  for the most undersupplied live trainer below its low-water mark. The
  helper added at most one bounded node per call. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

- A 12-node A100 allocation used independent eight-node and four-node renewal
  lanes. Each lane needed an exact-shape queued replacement because an
  eight-node renewal could not replace an expiring four-node lane under the
  12-node cap. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- The peak VSAO shape was 47 GPUs across 18 nodes: 12 full A100 nodes, two
  A100 development nodes, three A100-small nodes, and one H100 node.
  Utilization accounting had to state both nodes and physical GPUs.
  [VSAO COORDINATION.md:267-270; VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- A foreground agent watcher was not durable. VSAO moved endpoint detection
  into an idempotent one-pass check in the existing ten-minute controller,
  backed by a locked seen-state ledger. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

## 5. Multi-node training

Multi-node failures often live outside the model code. Rendezvous health,
supervisor identity, crash-log epochs, sharding assumptions, compiler caches,
and checkpoint boundaries all became correctness conditions in Forcing Laws.
VSAO added evidence that some training graphs cannot be expressed safely with
standard DDP settings. The common answer is explicit ownership: one launch
decision, one supervisor, scoped cleanup, complete state, and a preserved
scientific world size.

### Launch and rendezvous failure modes

- Forcing Laws launched torchrun across held GH200 nodes with a c10d
  rendezvous on the selected master. The validated Self-Forcing training
  topology used one GPU per node and full FSDP sharding. [FL G19]

- A run at u=0.29 died with RendezvousConnectionError and DistNetworkError.
  The original supervisor matched only ChildFailedError and unhandled CUDA,
  so its sequencer waited forever. [FL G25]

- Crash-signature coverage therefore included rendezvous and distributed
  network failures. Signature matching was necessary, but not sufficient,
  because logs persist across launch epochs. [FL G25]

- A transient ChildFailedError could remain in the current log after
  torchrun recovered and resumed training. Re-reading that old line on every
  poll drove the retry counter toward killing a healthy run. [FL G23]

- A valid signature must be newer than the current launch or adoption epoch,
  or old logs must be archived at adoption. A search over an undifferentiated
  current log is not a state machine. [FL G23]

- Fresh log modification time plus a lingering process caused ghost
  re-adoption of a dead run after an operator injected a crash signature.
  Clean relaunch required archiving rank logs or switching to a controlled
  manual path. [FL G25]

- A distributed retry needed one rank-zero decision after each time-sensitive
  barrier. Every rank consumed the same published action instead of checking
  freshness independently. [FL G40]

- When scheduler time was unavailable, a log-derived deadline included a
  conservative startup-lag guard. Collision-safe quarantine also prevented a
  stale claim and its pending copy from both surviving. [FL G40]

### Supervisor ownership and cleanup

- Two or three run_waves supervisors once managed the same groups after a
  timed-out launch command completed later in the background. They repeatedly
  pre-killed one another's runs and caused five launch churns. [FL G24]

- Every supervisor launch then acquired a nonblocking flock on a shared lock
  and was followed by a process census on both the fleet node and session
  host. A timed-out background command was treated as possibly executed.
  [FL G24]

- Pattern-based killing was unsafe because the pattern could match the shell
  performing the kill. Exact process IDs from an anchored process search were
  required. [FL G24]

- A second manual training launch on a shared allocation triggered a launcher
  pre-kill across every job host and killed the first group. Only one manual
  launch was permitted unless host scope was explicitly separated. [FL G25]

- An OOMed Slurm step left surviving ranks in a collective. Cancelling the
  exact step, rather than the job, recovered the controller while preserving
  the parent allocation. [FL G31]

- Fleet workers themselves were Slurm steps. Killing worker_loop by pattern
  completed the fleet job, so extra workers recorded dedicated pidfiles and
  cleanup targeted only those recorded processes. [FL G20]

### Distributed memory and gradient mechanics

- Hybrid FSDP sharding assumed multiple GPUs inside a node. On Vista's
  one-GPU GH200 nodes it became replication and reached 93 GB, while full
  sharding distributed state across nodes correctly. [FL G19]

- DDP with static_graph enabled and no_sync crashed on the first backward at
  reducer.cpp:1660. CFG double-forward plus activation checkpointing required
  the static graph, while chunked accumulation required no_sync, so the
  combination could not express the workload. [VSAO SMOKE.md:1497-1512]

- VSAO replaced DDP with manual gradient reduction. GRPO accumulated locally,
  then all-reduced about 2.6 GB of bf16 policy gradients once per step using
  an average. [VSAO SMOKE.md:1497-1512; VSAO report-vsao.md, DP trainer design]

- Rank zero claimed complete GRPO groups and computed DanceGRPO advantages
  before sharding. That preserved group semantics while distributing the
  expensive backward pass. [VSAO report-vsao.md, DP trainer design]

- SAO used unequal shards because 64 transitions do not divide evenly across
  three ranks. Chunk losses were weighted by global transition count and
  gradients were reduced by sum. [VSAO SMOKE.md:2041-2069]

- The three-GPU GRPO implementation reduced one step from 1,893 seconds to
  639 seconds, a 2.96x speedup. [VSAO report-vsao.md, DP trainer design]

- The SAO world-one hooks were inert so the H100 flagship stayed
  byte-identical. A distributed optimization should not perturb the validated
  single-device control path. [VSAO SMOKE.md:2041-2069]

### Compiler and checkpoint boundaries

- Sharing generated Triton sources across active nodes produced malformed
  source symptoms and a rendezvous collapse. Each node and retry attempt
  received an isolated generated-source cache. [FL G27]

- The content-addressed TorchInductor graph cache could persist only across
  sequential work on the same node. Training and evaluation also used
  separate node-local cache roles. [FL G32]

- A checkpoint file appearing on disk did not mean an FSDP save had finished.
  Durability required completion of gathers, the save routine's confirmation,
  a training-loop confirmation, and demonstrated continuation. [FL G34]

- A weight-only checkpoint lacked Adam moments, per-rank RNG, sampler
  position, step, branch history, probe state, and the FLOP ledger. It was
  partial evidence, not an exact resume point. [FL G33]

- Preserving a registered six-rank world size outweighed filling four surplus
  nodes with more training ranks. The extra nodes became an isolated
  evaluation Slurm step behind a readiness barrier. [FL G43]

- Evaluation workers stayed within the project namespace, consumed only
  validated outputs, and exited after both a producer-done marker and an
  empty queue. Temporary dependency-driven idle time preserved the registered
  optimization path. [FL G43]

## 6. Async RL orchestration

Asynchronous RL couples two rates: rollout production and trainer consumption.
The queue between them must bound policy lag, preserve unique trajectory
identity, and make each update transactional. Fleet mechanics must also
distinguish a normal backpressure exit from abort, crash, and walltime loss.
VSAO's mature design treated checkpoints, heartbeats, locks, and queue records
as one distributed protocol.

### Producer and consumer topology

- A rollout worker polled the latest policy checkpoint, sampled a video with
  per-step rollout log probabilities, decoded and scored in process, then
  wrote a TrajectoryRecord. [VSAO DESIGN.md:8-24]

- Each trajectory first landed in a temporary file and became visible through
  atomic rename into trajq/pending. A trainer claimed it by rename into its
  consumed state. [VSAO DESIGN.md:8-24]

- The trainer recomputed current-policy log probabilities, applied the DIS
  ratio and hard mask, used a critic with generalized advantage estimation,
  performed two critic steps per policy step, and checkpointed periodically.
  [VSAO DESIGN.md:8-24]

- VSAO used two distinct queues on scratch: a task queue for executable work
  and a trajectory queue for training data. Conflating their lifecycle rules
  would mix scheduler recovery with optimization transactionality. [VSAO
  DESIGN.md:8-24]

### Lag, backpressure, and liveness

- Wan SAO capped backlog at 160 records and used a low-water threshold of 80.
  A rollout node released its whole claim at the cap, and the controller
  replenished only below low water. [VSAO ensure_rollout_capacity.sh:28-30]

- A trainer-side policy-version lag guard allowed at most eight versions.
  The DIS hard mask absorbed trajectories generated during temporary trainer
  gaps. [VSAO RUNBOOK-RL.md:176-178,261-262]

- The trajectory queue was version-guarded but not run-guarded. Another model
  family could not point its workers at a live run's queue merely because
  record versions looked acceptable. [VSAO RUNBOOK-RL.md:176-178,261-262]

- Heartbeats recorded starting, running, or throttled. Throttled was a normal
  state when rollout supply exceeded trainer intake, not evidence of a dead
  worker. [VSAO report-vsao.md, Trajectory queues]

- Replacement logic used heartbeats newer than three minutes. Old claims on
  incompatible one-GPU or two-GPU nodes did not suppress a required
  three-GPU replacement. [VSAO report-vsao.md, Trajectory queues]

- Reaching the frozen backlog cap was a marked successful drain. Scheduler
  signals, an ABORT request, and unmarked child failures remained distinct
  terminal causes. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

### Packed identity and transactional batches

- Inside a packed slot, two different physical GPUs both appeared as logical
  cuda:0. Host plus logical-device identity therefore collided for two
  same-run workers. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- The collision replayed deterministic prompt streams in seed 6. Three
  duplicate pairs entered its first claimed 64-record batch, and three more
  pending records duplicated already consumed records. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

- The corrected identity contract separated physical slot provenance from
  logical CUDA binding and assigned every producer a unique worker_stream_id.
  That stream ID participated in prompt seeding and was stored in each
  trajectory and heartbeat. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- The trainer rejected repeated deterministic trajectory identities before
  forming a batch. If rejection made a transaction short, valid claims
  returned to pending and the duplicate remained excluded. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-19]

- Recovery happened before any policy movement. Six duplicate records were
  quarantined, 90 unique records remained, and version one used exactly 64
  audited version-zero records. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- The failed duplicate-stream allocation was charged 9,309 GPU-seconds.
  Recovery retained the cost even though the invalid update never committed.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-19]

- Task publication also enforced identity across pending, running, done, and
  failed states. An atomic directory reservation ensured that two controller
  passes could not publish the same task name. [VSAO gpu_enqueue.sh:23-48]

### Checkpoint and shutdown protocol

- A trainer was complete only when its declared terminal checkpoint existed.
  Wan seed 3204 had a done outer task and version 30 with optimizer state, but
  it still needed a resume because version 37 was absent. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Exact resume loaded policy from the newest policy_v file and critic, reward
  normalization, step counter, and optimizer states from the atomically
  written trainer_state_latest file. [VSAO report-vsao.md, Checkpoint durability]

- Early trainer states omitted optimizer state. Later drivers round-tripped
  it for all four families. A policy-only recovery also required an explicit
  accounting bootstrap so cost did not silently reset to zero. [VSAO
  vista/README.md:265-272]

- An ABORT marker let rollout workers finish their current video and release
  nodes normally. The trainer handled termination by finishing its step,
  saving policy and state, and exiting with code three. [VSAO
  RUNBOOK-RL.md:243-253]

- ABORT did not cancel fleet allocations because unrelated useful tasks could
  still use them. Run shutdown and allocation shutdown remained separate
  operations. [VSAO RUNBOOK-RL.md:243-253]

- Cleanup used bounded TERM, a grace-period poll, then KILL for survivors. An
  earlier unbounded wait on a signal-ignoring child had held a node at 24 GB
  and zero percent GPU utilization. [VSAO SMOKE.md:2097-2124]

- A trainer claim with a log stale for more than 60 minutes was eligible for
  cleanup, safely above the slowest legitimate step of about 1,930 seconds.
  [VSAO report-vsao.md, ABORT and shutdown safety]

- The trainer single-instance lock used exit-on-lock, not a warm standby. A
  contender yielded only when it saw both a fresh foreign lock and durable
  queue evidence. [VSAO report-vsao.md, Trainer single-instance lock]

- Lock acquisition used a nonce, jitter, and recheck to close a simultaneous
  stale-takeover race in which two nodes had both passed the same stale test.
  [VSAO report-vsao.md, Trainer single-instance lock]

- Shutdown code resolved inherited YAML configurations through the project
  loader. Literal parsing once missed a run-specific ABORT marker because
  run_name came from an extends base file. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- On Vista, trajq_dir had to be a direct child of the project scratch root
  because the driver derived scratch_root from its parent. Deeper nesting
  broke evaluation spooling and the ABORT path. [VSAO report-vsao.md,
  Vista GB200 side]

## 7. Determinism, dtype, and numerics

Determinism is infrastructure because it decides which artifacts can be
reused, compared, and resumed. CachedSearch depended on exact fixed-stack
recommit behavior, while VSAO needed stable trajectory identity and explicit
tolerance for bf16 kernel effects. Dtype rules were component-specific rather
than global. Every project therefore recorded its stack, froze protocol
boundaries, and separated reproducible small offsets from true stochastic
failures.

### Determinism as a systems contract

- CachedSearch computed full rollouts once and reused them across every tau
  sweep arm. Tau did not affect a full-compute rollout, so recomputation would
  have wasted GPU time without adding evidence. [VG1 RELEASE.md:240-257]

- Recommit regenerated the selected winner from its seed with full compute.
  On a fixed hardware and software stack the output was bit-identical, which
  made explore-cheap-then-commit-full quality preserving. [VG1
  RELEASE.md:240-257]

- CachedSearch explicitly did not claim bitwise equality across aarch64 GH200
  and x86 A100 stacks. Latencies and reference outputs had to be regenerated
  for a new stack. [VG1 RELEASE.md:194-196,244-249]

- A pause at step eight followed by resume to step 20 was bit-exact only after
  serializing cache branch state and all relevant UniPC multistep internals.
  Latents plus set_begin_index were insufficient. [VG1
  NOTES-e6-agent.md:8-26]

- UniPC state included model outputs, timestep list, lower-order count, last
  sample, current order, and step index. Treating it as FlowMatchEuler would
  have silently broken the corrector. [VG1 NOTES-e6-agent.md:8-26]

- CachedTransformer's default two branches matched Wan's sequential CFG
  calls. Single-call CogVideoX, LTX, and guidance-distilled Hunyuan required
  one branch or the stream alternated between unrelated cache states.
  [VG1 NOTES-e4-agent.md:63-68; VG1 NOTES-gate5-agent.md:143-148]

- The wrong branch count halved each state's perceived step counter and
  corrupted warmup, cooldown, and skip decisions without necessarily
  crashing. Per-model assertions turned a silent numerical bug into a fast
  failure. [VG1 NOTES-e4-agent.md:63-68]

- Batched CachedSearch was not bit-identical to single inference. About 0.6
  reduction-order drift and a batch-averaged skip indicator required arms to
  remain internally consistent. [VG1 RELEASE.md:250-255]

- Drift Atlas seeded each sample independently inside a batch. This preserved
  the same initial noise as single-sample generation while accepting normal
  differences from batched kernel reduction order. [FL G14]

### Dtype rules for video pipelines

- Wan used a bf16 pipeline with its VAE kept in fp32. Running video VAEs in
  bf16 was a measured quality trap rather than a harmless memory shortcut.
  [VG1 README.md:100; VG1 gen.py:14-16]

- CogVideoX was the exception. Its decode_latents path did not cast latents to
  the VAE dtype, so an fp32 VAE hard-failed on a dtype mismatch and the VAE
  stayed bf16. [VG1 NOTES-e4-agent.md:43-46]

- Wan returned float32 frames in the interval from zero to one. Direct uint8
  conversion silently produced nearly all-zero images, so the shared helper
  multiplied and clipped before conversion. [VG1 README.md:98; VG1
  video_io.py:94-98]

- That uint8 mistake would also have driven ImageReward scores to zero. A
  representation bug in video output can masquerade as a model-quality
  result. [VG1 README.md:98]

- Diffusers cast TeaCache's timestep embedding to bf16 while official Wan
  kept fp32. The resulting relative-L1 indicator offset was about 1e-3,
  negligible relative to the 0.08 threshold but still documented.
  [VG1 NOTES-a1-agent.md:32-34]

### Stable offsets and tolerances

- VSAO found that a second model instance had a deterministic per-instance,
  per-record chain-logp offset as large as plus or minus 3e-4 per chain.
  Storage-address-sensitive bf16 kernel tiling, not random execution, caused
  the shift. [VSAO SMOKE.md:1896-1916]

- The offset mattered for DPO because beta multiplies same-video logp
  differences. For DIS and GRPO, the relevant window scale made effects near
  1e-5 negligible. [VSAO SMOKE.md:1896-1916]

- DPO calibration measured a pair-margin noise floor near 1.5e-4. Averaging
  eight pairs per step reduced it to about 5e-5, and the calibrated beta was
  1000. [VSAO SMOKE.md:1896-1916]

- A one-time version-zero reference-logp cache removed the second resident
  model. The driver spot-validated supplied cache values at version zero with
  tolerance 1e-5. [VSAO SMOKE.md:1896-1916]

- A100 LTX produced a final-step bf16 ratio of 0.98881 against a frozen 0.99
  canary threshold. The approved tolerance was 0.99 to 1.01 per step, while
  all four GB200 families fell between 0.99309 and 1.00003. [VSAO
  SMOKE.md:2269-2272; VSAO vista/SMOKE.md:50-58]

- VSAO observed bf16 DiT tensor-norm movement of at most about 1e-4 over 80
  transitions. This supported the version-lag argument but did not eliminate
  explicit lag guards. [VSAO SMOKE.md:1705]

- The finite prompt pool was encoded once and shared bitwise across ranks.
  Besides saving memory, this removed rank-specific text-encoder numerics from
  the distributed training comparison. [VSAO SMOKE.md:2071-2088]

### Backend control

- CachedSearch used PyTorch SDPA without flash-attn. Paired cache speedups
  used the same backend in both arms, so the ratios were treated as
  backend-insensitive while absolute latency remained GH200 plus SDPA
  specific. [VG1 source agent guide:19]

- Forcing Laws pinned SDPA and FlexAttention as one experimental control.
  Per-configuration backend switching would have confounded a relative study
  more than a consistent slower backend. [FL D8; FL G4]

- A vendored SDPA fallback once dropped the text key-padding mask. Five call
  sites were routed through the corrected dispatcher, demonstrating that a
  nominally equivalent backend still requires semantic parity tests.
  [FL G5]

## 8. Telemetry and measurement

Allocated hardware, loaded models, active kernels, and useful scientific work
are four different states. Point probes are valuable for attribution, but an
asynchronous video pipeline must also be measured over a rolling window that
includes sampling, decode, scoring, synchronization, and I/O. Telemetry became
most actionable when it was immutable, all-host, and joined to queue state.
The VSAO recovery from 23.6 percent to 81.5 percent is the clearest example. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

### What to collect

- The account-wide doctrine starts a 30-second nvidia-smi CSV logger on every
  host in every real GPU job. Samples include memory, utilization, and power.
  [memory-facts; FL G16]

- Training streams per-step losses, branch state, VRAM allocation and peak,
  and compute tags to WandB. Long inference and evaluation projects use a
  persistent fleet monitor rather than disabling tracking. [memory-facts;
  FL G16]

- Inference manifests record per-video generation time, batch, and whether
  the result came from a timed pass. This makes later analysis reject bulk
  throughput records from latency claims. [FL G16]

- CachedSearch predates the telemetry-everywhere policy. It measured peak
  VRAM in process with torch.cuda.max_memory_allocated and wall time around
  the pipeline call, without an explicit synchronize before stop.
  [VG1 bench_batch.py:1-37]

- CachedSearch's strongest telemetry was its append-only candidate JSONL.
  Each row carried seed, latency, verifier score, tau, and per-branch cache
  skip and compute counters. [VG1 RELEASE.md:198-229]

### Point and rolling views

- A point sample separates GPUs currently computing, GPUs holding project
  memory between kernels, and genuinely free GPUs. It must not label every
  zero-utilization device free. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- A scorer can hold about 11 GB and execute in bursts after generators finish.
  A synchronized trainer can pause between phases and then drive every device
  at full utilization. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- VSAO's rolling summary defaulted to 300 seconds. Active-sample fraction was
  the fraction of samples with utilization at least 5 percent, reported with
  mean utilization and mean memory fraction per host and in aggregate.
  [VSAO gpu_telemetry_summary.py]

- The summary inferred physical GPU count from logger bursts, selected the
  latest file for each host, and wrote schema-version-two JSON through a
  temporary file and atomic replacement. [VSAO gpu_telemetry_summary.py]

- Each immutable diagnostic included a SHA-256 prefix tied to its source CSV.
  Operational conclusions therefore remained linked to the exact live trace.
  [VSAO gpu_telemetry_summary.py]

- One historical point probe covered only 45 of the 47 A100-class and H100
  GPUs. Every utilization report now enumerates all active GPU queues and
  states both allocated nodes and allocated GPUs. [VSAO report-vsao.md,
  Telemetry]

### The utilization recovery

- On July 21, VSAO held 47 GPUs but a point probe found only 12 working and
  35 genuinely free. A five-minute, 470-sample window measured 23.6 percent
  active samples, 23.45 percent mean utilization, and 20.1 percent mean
  memory occupancy. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- The cause was not sick calibrated tasks. Both spools had drained, so held
  capacity could not see any scientifically useful work. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Recovery froze 24 exact LTX panels, bound 1,536 source videos by hash, and
  queued 48 independent held-out and VBench reruns. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- The recovery window observed 41 physical GPUs across 16 hosts and reached
  81.5 percent active samples. The added work was already declared evaluator
  repeatability evidence, not a new efficacy endpoint. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- An earlier full-shape window across 47 GPUs measured 91.8 percent active
  samples, 90.3 percent mean utilization, and 70.7 percent mean memory
  occupancy. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- A later full-shape trace included 474 physical-GPU samples and measured
  92.4 percent active samples, 91.8 percent mean utilization, and 79.3 percent
  memory occupancy. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- In that later trace, one trainer host held 76.8 percent memory without
  kernels. Forty-one pending records and a healthy rollout node showed that
  it was waiting for a complete 64-record transaction, not abandoned.
  [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Another 47-GPU window measured 79.0 percent active samples and 78.69 percent
  mean utilization, with 36 GPUs computing, nine holding models, and zero
  unassigned free GPUs. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

### Interpreting and preserving measurements

- The warmed-up VSAO target was at least 78 percent rolling active samples.
  This target complemented, rather than replaced, the 70 to 80 percent VRAM
  calibration target. [VSAO GPU-UTILIZATION-LEARNINGS 2026-07-21]

- Forcing Laws interpreted zero-to-100 percent swings as workload phases such
  as denoise, decode, encode, and load. Health meant time-averaged utilization,
  useful queue depth, and absence of abandoned nodes. [FL G16]

- Reported latency and FPS stayed in batch-one exclusive passes even when
  bulk fleet utilization improved through batching. Measurement integrity was
  never traded for an attractive occupancy graph. [FL G14; VG1
  worker_loop.sh:4-18]

- Controller logs joined utilization, working and total GPU counts, free idle
  count, pending work, and allocation balance. Telemetry without scheduler and
  queue context was insufficient for diagnosis. [VSAO report-vsao.md,
  Telemetry]

- Post-hoc Forcing Laws analysis produced time-averaged utilization, memory,
  energy per node, and fleet aggregate. This converted operational traces
  into paper-auditable compute evidence. [FL G16]

## 9. Environments on aarch64

Vista's aarch64 nodes made dependency provenance part of experimental
reproducibility. A package name alone did not identify a compatible wheel, and
one transitive reinstall could replace a working CUDA build with a CPU ABI.
The durable approach used uv environments on work-backed stable paths,
scratch caches, explicit indexes and compilers, and isolated evaluation
stacks where dependency constraints genuinely conflicted.

### Environment placement and package sources

- New Python environments use uv at the stable path under
  `$WORK/library/uv-envs`. The uv cache uses the sibling library path, which
  currently resolves to scratch. [memory-facts]

- Environment builds and large torch installs run on compute nodes. The login
  node's 8 GB virtual-memory cap makes it unsuitable even when installation is
  CPU-only. [memory-facts]

- CachedSearch pinned Python 3.11.13, torch 2.11.0+cu128, and torchvision from
  the PyTorch cu128 index. [VG1 report-videogen1.md, Environment pins]

- On aarch64, PyPI torchvision was a CPU ABI build. It produced operator
  torchvision::nms does not exist and could surface as an unrelated
  transformers failure to import UMT5EncoderModel. [memory-facts; VG1
  report-videogen1.md, Environment pins]

- Both torch and torchvision therefore came from the cu128 PyTorch index.
  Packages such as image-reward could pull PyPI torchvision back in, so
  torchvision was re-pinned last. [memory-facts; VG1
  report-videogen1.md, Environment pins]

- CachedSearch needed transformers 4.49.0 exactly. WanPipeline required at
  least 4.49, while ImageReward's vendored BLIP depended on an API removed in
  later versions. [VG1 report-videogen1.md, Environment pins]

- VBench used a separate uv environment with transformers 4.33 and a decord
  compatibility path. Isolating evaluation prevented its older dependency
  constraints from destabilizing generation. [VG1
  report-videogen1.md, Environment pins]

### Attention and compiler constraints

- CachedSearch and Forcing Laws had no external flash-attn package on Vista.
  They used PyTorch SDPA, with FlexAttention only where the causal training
  mask required it. [VG1 source agent guide:19; FL G4]

- On LS6, the available flash-attn wheel required GLIBC_2.32 while the system
  provided 2.28. VSAO used the parity-tested SDPA fallback. [VSAO
  SMOKE.md:7,156-161]

- Triton and torch.compile on Vista required CC=gcc and CXX=g++. The TACC
  NVIDIA module exported nvc, which Triton's CUDA utility build could not use.
  [FL G7]

- TRITON_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR belonged on scratch or
  node-local temporary storage. Generated source was isolated by node, run,
  and retry attempt. [FL G7; FL G27]

- The NVIDIA module leaked nvc-specific headers through CPATH and related
  include variables. Native gcc builds first unset CPATH, C_INCLUDE_PATH,
  CPLUS_INCLUDE_PATH, and INCLUDE. [memory-facts; FL G10]

- Detectron2 had no aarch64 wheel. Its source build also required
  CUDAHOSTCXX=g++, gcc and g++, and a pinned TORCH_CUDA_ARCH_LIST.
  [memory-facts; FL G10]

- VBench split 12 detectron2-free dimensions from four GRiT-backed dimensions
  so one difficult native dependency could not block all scoring. The GRiT
  set included object class, multiple objects, color, and spatial relation.
  [FL G10]

### Cache, download, and wrapper hygiene

- Hugging Face Xet could OOM a Vista login node. The safe path disabled Xet,
  capped download workers at four, pre-downloaded on suitable hardware, and
  set offline mode in jobs. [memory-facts]

- Hugging Face cache variables have precedence. A profile-level HF_HUB_CACHE
  overrode project HF_HOME in CachedSearch, so every worker needed the same
  explicit cache contract. [VG1 report-videogen1.md, Storage layout]

- VBench's upstream checkpoint downloads included a dead GRiT URL. Mirroring
  the complete dc-ai/vbench_pretrained_models layout once made scoring
  network-independent. [FL G9]

- PyTorch 2.6 and later default weights-only loading rejected old pickled
  scorers. Evaluation jobs set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 only where
  those known artifacts required it. [FL G10]

- V-JEPA-2 required transformers 4.53 or newer, which conflicted with both
  existing environments. Forcing Laws loaded it through its upstream torch
  hub and recorded the actual backbone, with any fallback explicit.
  [FL G8]

- A vendored package touched CUDA during import, so CPU-node unit tests had to
  guard the import boundary. Importability is not automatically independent
  of hardware context. [FL G4]

- LS6 GPU wrappers changed to the project directory, sourced the full
  environment, and then executed the task. A bare remote shell could land in
  home and fail with missing modules despite a correct shared environment.
  [VSAO report-vsao.md, Scheduler patterns]

- Vista's cu130 aarch64 stack lacked an NPP wheel, so VSAO used CPU decode
  with vendored FFmpeg rather than assuming GPU NVDEC support from the driver.
  [VSAO vista/SMOKE.md:32-36]

## 10. Storage, caches, and I/O

The three-filesystem layout is an availability design, not mere housekeeping.
Small code and paper-critical evidence remain persistent, while model weights,
videos, checkpoints, queues, and caches use purge-eligible scratch. A project
must be resumable from scratch without pretending scratch is an archive.
Shared filesystems also need special treatment for compiler sources, cache
precedence, large checkpoint reads, and files referenced by live jobs.

### Placement and retention

- Home has a 23 GB quota and is reserved for configuration and small launchers.
  Environments, model weights, datasets, checkpoints, and large caches do not
  belong there. [memory-facts]

- Work provides 1 TB of persistent storage for repositories, small vendored
  runtime libraries, and the evidence crux needed to reproduce paper numbers.
  [memory-facts]

- Scratch is effectively large but purge-eligible after roughly ten idle
  days. Heavy datasets, weights, checkpoints, logs, videos, telemetry, task
  queues, and caches are treated as regenerable there. [memory-facts]

- The purge guard copies paper-critical records and manifests into the
  repository on work before they age out. It does not move every raw artifact
  into persistent storage. [memory-facts]

- CachedSearch archived 515 measured-result files totaling 39,376,860 bytes
  inside its repository and byte-verified the archive. [VG1
  report-videogen1.md, Storage layout]

- The CachedSearch paper could be re-derived from that work archive. Its
  89 GB of raw videos remained on scratch because fixed-stack deterministic
  seeds could regenerate them. [VG1 report-videogen1.md, Storage layout]

- Never move a file referenced by a queued or running Slurm job. Stable path
  identity is part of the job contract even if the destination appears
  semantically equivalent. [VG1 report-videogen1.md, Storage layout]

### Cache consistency and corruption

- A CachedSearch worker lacking the profile cache setting re-downloaded a
  Wan-14B shard into the project cache and corrupted shard 00005. The shard
  was quarantined and 14B runs were pinned to the known profile cache.
  [VG1 report-videogen1.md, Storage layout]

- HF_HUB_CACHE can override HF_HOME. Every job wrapper must make the effective
  cache explicit rather than infer it from one interactive shell.
  [VG1 report-videogen1.md, Storage layout]

- VBench's scorer dependencies were mirrored as one complete cache tree after
  an upstream URL failed. Partial on-demand downloads were too brittle for a
  multi-node scoring project. [FL G9]

- Generated Triton source trees cannot be shared by concurrent nodes. They
  are scratch data, but their namespace must still be node-local and specific
  to a run and retry. [FL G27]

- Content-addressed TorchInductor graph artifacts can be reused sequentially
  on the same node. Training and evaluation retain different job-local cache
  roots to avoid cross-role races. [FL G32]

- The shared V-JEPA teacher feature cache contained 384 rows. Creation used a
  file lock and atomic same-directory replacement, then validated schema,
  backbone, filenames, source directory and fingerprint, shape, and finite
  values. [FL G28]

### Queue and checkpoint I/O

- File spools use atomic rename for claims because concurrent append to one
  shared file is fragile on Lustre. Durable identity spans every queue state,
  not just the pending filename. [VG1 worker_loop.sh:53-70; VSAO
  gpu_enqueue.sh:23-48]

- CachedSearch append-only JSONL readers re-globbed all shards, tolerated a
  torn final line in a live file, and deduplicated by stable key with the last
  record winning. [VG1 RELEASE.md:198-229]

- VSAO wrote trajectories to temporary files and renamed them atomically into
  pending. Trainers then claimed records by rename, which made a batch a set
  of durable file identities. [VSAO DESIGN.md:8-24]

- VSAO checkpoint resume separated policy files from trainer state. The
  latest trainer state carried critic, normalizer, counters, and eventually
  optimizer states, and was replaced atomically. [VSAO
  vista/README.md:265-272]

- A 12 to 15 GB GB200 trainer-state file could take 20 to 48 minutes to
  unpickle in one CPU thread. Large-file liveness required process CPU and I/O
  evidence rather than wall-clock impatience. [VSAO report-vsao.md,
  Vista GB200 side]

- Six or more concurrent 81-frame VSAO runs saturated scratch I/O. One load
  fell to 3 KB/s, while about four to five runs, roughly one per node, were
  stable. [VSAO report-vsao.md, Vista GB200 side]

- Two 81-frame runs per GB200 node caused most instability despite ample GPU
  memory. Scratch throughput, not HBM, set the useful packing ceiling.
  [VSAO report-vsao.md, Vista GB200 side]

- Identical GB200 resumes took 40 to 48 minutes on two compute hosts but about
  two minutes on two peers. The suspected host-specific CPU or I/O degradation
  remained unresolved and was mitigated by exclusion.
  [VSAO report-vsao.md, Vista GB200 side]

### Login-node I/O and process safety

- Heavy downloads run away from login nodes. Xet, Rust, uv, and tokenizer
  thread pools can exhaust login virtual memory or process budget before the
  payload is usable. [memory-facts]

- A login node supports editing, light Git, scheduler commands, and brief SSH
  only. Encoding, building, data transformation, model loading, and long-lived
  sessions belong on compute nodes. [memory-facts]

- At most three concurrent SSH connections may touch a login node, and SSH
  must never be launched once per task in a loop. Multiple scheduler actions
  are batched into one connection. [memory-facts]

- Scheduler polls wait at least ten seconds. Every background process must be
  reaped because all sessions share the 100-process per-user limit.
  [memory-facts]

- From a compute node, plain SSH to a login node can submit a job. Adding
  BatchMode=yes disables the working authentication route and produces a
  misleading permission failure. [memory-facts]

## 11. Video I/O

Video I/O can alter quality metrics, exhaust process limits, or silently lose
dynamic range. The canonical Vista stack pairs torchcodec with a complete
vendored FFmpeg and bounded encoder threads. Decode and encode policy must be
frozen across comparison arms because video files are not neutral containers
for every metric. HDR detection, dtype conversion, and codec dependencies are
therefore experiment setup, not cleanup details.

### Decode stack

- Vista's system FFmpeg is an EL9 free build without H.264 or HEVC support.
  Its binary also lacks libunwind.so.8 in the documented environment.
  [memory-facts]

- The working stack used a vendored FFmpeg 7.1 shared build on work and added
  its libraries to LD_LIBRARY_PATH. [memory-facts]

- Torchcodec decode on the cu128 stack required the PyTorch-index torchcodec
  wheel, nvidia-npp-cu12 from the same index, and both vendored FFmpeg and NPP
  library directories on LD_LIBRARY_PATH. [memory-facts]

- The PyPI torchcodec wheel linked CUDA 13 nvrtc and was incompatible with the
  cu128 environment. Wheel provenance mattered even when package versions
  appeared similar. [VG1 report-videogen1.md, Video I/O]

- CachedSearch measured torchcodec at about 156 milliseconds for 33 frames at
  480p on Grace. ImageIO remained a fallback rather than the primary reader.
  [VG1 report-videogen1.md, Video I/O]

- Vista VSAO on cu130 could not obtain an aarch64 NPP wheel. It used CPU
  decode plus vendored FFmpeg instead of a partially working NVDEC path.
  [VSAO vista/SMOKE.md:32-36]

### HDR and dtype safety

- Torchcodec 0.11 and 0.11.1+cu128 silently flattened HDR and 10-bit inputs to
  8-bit uint8. Readers inspected HDR metadata and warned before decode.
  [memory-facts; VG1 report-videogen1.md, Video I/O]

- Real float32 HDR decode required torchcodec 0.14 or newer, which was not
  available on the cu128 aarch64 index during these projects. [memory-facts]

- Wan output arrived as float32 in the zero-to-one interval. The conversion
  helper scaled and clipped before uint8 conversion to prevent silent black
  video. [VG1 video_io.py:94-98]

- VAE dtype remained a per-model contract. Wan used fp32 for quality, while
  CogVideoX required bf16 because its decoder did not reconcile latent and
  VAE dtypes. [VG1 gen.py:14-16; VG1 NOTES-e4-agent.md:43-46]

### Encode and evaluation sensitivity

- CachedSearch wrote 8-bit x264 with four encoder threads. The writer raised
  an error when the output file was empty. [VG1 video_io.py:101-115]

- On a 144-core Grace host, x264 auto-threading could exceed the shared
  process limit and silently leave a zero-byte MP4. x265 required one pool and
  one frame thread for the same reason. [memory-facts]

- An A/B test on 500 identical videos changed VBench aesthetic quality by
  about three points between imageio-q8 and ffmpeg-crf23: 0.7803 versus
  0.7506. [FL G11]

- In the same A/B, imaging quality shifted by 0.005 and subject consistency by
  0.0001. One encoder was therefore fixed for all arms so paired differences
  canceled the codec effect. [FL G11]

- VBench's official aggregate used per-dimension normalization and
  Total=(4*Quality+Semantic)/5. Raw dimension averages could not be compared
  with published aggregate scores. [FL G12]

- The official aggregation produced Quality 85.37, Semantic 81.01, and Total
  84.50, compared with published values 85.07, 81.28, and 84.31. The deltas
  were plus 0.30, minus 0.27, and plus 0.19. [FL G12]

- VBench wrote a full_info metadata JSON before actual score completion. Only
  the terminal score JSON counted as a result. [FL G10]

- VBench's prompt list contained ten duplicate texts across dimensions.
  Consequently 4,720 unique files represented complete 946 by five coverage,
  not missing generation. [FL G9]

- Decord had no aarch64 wheel, so the official VBench environment used a
  decord-to-OpenCV compatibility shim. Torch 2.6 weights-only behavior also
  required an explicit safe-globals repair for AMT-S. [VG1
  report-videogen1.md, Video I/O]

## 12. Process and provenance

The final lesson is procedural: a successful process exit, a done marker, and
a readable manifest are claims, not evidence. Release gates must derive facts
from canonical payloads, hashes, checkpoints, telemetry, and terminal
scheduler records. They should fail closed on stale prose as well as missing
files. Scientific scope remains frozen even when spare hardware or partial
artifacts tempt expansion.

### Completion is evidence-backed

- Forcing Laws completion required exact manifests, planned evaluations,
  recovered evaluations, canonical file grids, scorer provenance,
  postprocessing, a valid crux, clean tests and paper audit, terminal
  scheduler accounting, and fleet release. Partial usefulness did not satisfy
  the gate. [FL G29]

- A readable completion marker could point to a missing probe trajectory, a
  truncated checkpoint, or the wrong schedule. Retry and enqueue decisions
  validated the payload behind the marker. [FL G39]

- Self-reported counts were not independent compute evidence. The finalizer
  derived hosts from canonical filenames, validated rows and samples, rejected
  symlinks and escaping paths, and verified SHA-256. [FL G36]

- Empty or malformed nested fit and validation records were unresolved
  evidence. Internal consistency of a top-level manifest did not convert them
  into success. [FL G36]

- VSAO likewise treated a done task as insufficient when its terminal
  checkpoint was absent. The controller restored exactly one resumable claim
  if neither a pending nor running owner existed. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

- CachedSearch re-globbed raw JSONL shards and deduplicated records by stable
  key. Paper numbers came from records, not from task filenames or worker exit
  codes. [VG1 RELEASE.md:198-229]

### Frozen provenance and numeric gates

- CachedSearch's ci_numbers.json stored every headline point estimate and its
  95 percent prompt-bootstrap interval with 10,000 bootstrap draws. Confidence
  intervals never changed the underlying point estimate. [VG1
  report-videogen1.md, Telemetry and monitoring]

- Figure scripts carried frozen expected values and warned on drift. Body
  display could hide intervals through one macro, while the appendix retained
  the complete values. [VG1 report-videogen1.md, Telemetry and monitoring]

- VSAO immutable utilization diagnostics recorded schema version, time window,
  hosts, aggregate statistics, and a hash of source telemetry. [VSAO
  gpu_telemetry_summary.py]

- The VSAO final provenance gate covered 40 comparison summaries and 232
  hashed sources for a 472-test suite. [VSAO report-vsao.md, Quotable numbers]

- The final VSAO efficacy result was negative despite a functioning systems
  stack: VisionReward-Video changed by minus 0.088 for Wan version 37, while
  LTX was null. Occupancy did not justify rewriting the scientific outcome.
  [VSAO report-vsao.md, Quotable numbers]

- Forcing Laws required paired evaluation on a frozen prompt and seed grid.
  Its 90 percent interval half-width target was at most one third of the
  smallest adjacent-rung effect, with automatic seed escalation from three to
  five to eight. [FL D6]

- Algebraic fit and scientific validation were separate states. Seven
  half-predict-half tests completed, six did not beat last observation, and
  five placed the exponent at its lower bound, so the fit remained descriptive
  and unvalidated. [FL D11]

### Release and recovery discipline

- A clean PDF build checked references and file integrity but did not replace
  a semantic audit for stale claims and generated result cells. Both ran in
  order before release success. [FL G42]

- Outcome-neutral prose could be corrected early, while endpoint values and
  schedule rows stayed generated and fail closed until measurements arrived.
  [FL G41]

- Release gates rejected stale run counts, unscored cells, and descriptions
  of completed work as future work, not just obvious TODO markers. [FL G37]

- An evaluation chain with generation, FVD, and drift scoring bounded each
  stage independently: one hour for generation, one hour for FVD, 30 minutes
  for drift, and five minutes for forced termination. [FL G35]

- Recovery changed only the invalid component. When videos, teacher probes,
  and FVD were valid, Forcing Laws reran drift scoring with one-second windows
  and exactly five segments rather than regenerating everything. [FL G46]

- A running allocation could not archive its own terminal Slurm accounting.
  It published a preterminal crux with scheduler_accounting_archived false,
  then a post-job process captured accounting and reran release gates.
  [FL G45]

- Compiler, controller, worker, retry, and launch files loaded dynamically by
  a live project were frozen before submission. One project completed both
  seeds, then failed when a child loaded a mixed on-disk script state.
  [FL G44]

- Parsed shell functions and later child-script reads have different mutation
  boundaries. Live hardening targeted only files a running controller would
  actually load and preserved one shared decision across ranks. [FL G38]

- A weight-only intermediate checkpoint remained useful partial evidence but
  could not be promoted to exact resume. Missing optimizer and RNG state would
  change the registered trajectory. [FL G33]

- Forcing Laws registered a narrow five-arm closeout rather than retroactively
  claiming its broader original proposal. The six-node, 48-hour registration
  requested a small fraction of the project allocation. [FL D12]

- The final recorded execution used three allocations and remained bounded to
  a small fraction of the project allocation. [FL D14]

- That registered closeout preserved a six-rank training world size and used
  surplus nodes only for dependency-safe evaluation. Scientific validity
  outranked nominal node occupancy. [FL G43]

- Forcing Laws used the phrase distillation post-training for its scope. The
  broader phrase post-training includes regimes that the project did not
  study. [FL D9]

- A local paper, report, and crux did not authorize public release, arXiv, or
  a public repository. Publication remained a separate user decision.
  [FL D9a]

- VSAO's recurrence audit applied the same scope rule operationally. When no
  frozen useful work existed, the correct action was to release excess
  capacity rather than invent endpoints or run dummy kernels. [VSAO
  GPU-UTILIZATION-LEARNINGS 2026-07-21]

## Open questions

None. Source-level unresolved investigations are labeled in their entries.
