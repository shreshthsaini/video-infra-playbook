# NVIDIA Datacenter GPU Architecture and Memory Hierarchy: Fact Sheet

Compiled 2026-07-21. Every bullet carries its source URL. Numbers marked [computed] are arithmetic on cited datasheet figures. Numbers I could not confirm against a primary source are marked [UNVERIFIED].

## 1. Memory hierarchy

### The general picture (fastest/smallest to slowest/largest)
- Registers: 256 KB register file per SM on both A100 and H100 (64K 32-bit registers per SM). A100: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ ; H100: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Shared memory / L1 (on-chip SRAM, per SM, software-configurable split), then a GPU-wide L2 cache, then off-chip HBM ("global memory"), then host memory reached over PCIe (or NVLink-C2C on Grace superchips).

### A100 (Ampere, 2020)
- 108 SMs; each SM has 192 KB of combined shared memory + L1 data cache, of which up to 164 KB is configurable as shared memory. Source: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- L2 cache: 40 MB (6.7x larger than V100). Source: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- A100 40 GB: HBM2, 1,555 GB/s. Source: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- A100 80 GB: HBM2e, 1,935 GB/s (PCIe) / 2,039 GB/s (SXM). Source: https://www.nvidia.com/en-us/data-center/a100/
- Tensor Core peak (per NVIDIA, dense | sparse): BF16 and FP16 312 | 624 TFLOPS; TF32 156 | 312 TFLOPS. Sources: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ and https://www.nvidia.com/en-us/data-center/a100/
- Host link: PCIe Gen4, 64 GB/s (bidirectional). NVLink 3: 600 GB/s per GPU. Source: https://www.nvidia.com/en-us/data-center/a100/

### H100 SXM (Hopper, 2022)
- 132 SMs; shared memory configurable up to 228 KB per SM (out of a 256 KB combined L1/shared array per SM, per the Hopper whitepaper); L2 cache 50 MB. Source: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ (whitepaper: https://resources.nvidia.com/en-us-tensor-core)
- 80 GB HBM3 (5 stacks), 3.35 TB/s. Sources: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ (capacity, "over 3 TB/s") and https://www.nvidia.com/en-us/data-center/h100/ (3.35 TB/s)
- Tensor Core peak, sparse | dense (the NVIDIA product page lists only the sparse number with a "with sparsity" footnote; the GH200 datasheet prints both): TF32 989 | 494 TFLOPS; BF16 and FP16 1,979 | 990 TFLOPS; FP8 3,958 | 1,979 TFLOPS. Sources: https://www.nvidia.com/en-us/data-center/h100/ and the GH200 datasheet spec table https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf
  - Trap to avoid: "989 TFLOPS" appears twice in H100 tables with different meanings. For TF32 it is the WITH-SPARSITY number (dense TF32 is 494). For BF16/FP16 the dense number is ~989-990 and the sparse number is 1,979. Structured 2:4 sparsity numbers are essentially never achieved by ordinary dense training/inference; always quote dense.
  - Peak numbers assume maximum boost clocks; sustained attention kernels top out well below peak even when perfectly tuned (FlashAttention-3 reports 740 TFLOPS FP16 as "75% utilization" of H100, treated as near the practical ceiling). Source: https://pytorch.org/blog/flashattention-3/
- New Hopper hardware relevant to kernels: TMA (Tensor Memory Accelerator) transfers large blocks between global and shared memory with hardware address generation ("frees up registers"); thread block clusters let thread blocks on multiple SMs cooperate; distributed shared memory allows "direct SM-to-SM communications" (one block reading another block's SMEM within a cluster). Sources: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ and https://pytorch.org/blog/flashattention-3/
- NVLink 4: 900 GB/s per GPU, described by NVIDIA as 7x PCIe Gen5 (so PCIe Gen5 x16 is 128 GB/s bidirectional). Sources: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ and https://www.nvidia.com/en-us/data-center/h100/
- H100 NVL (PCIe form factor): 94 GB HBM3, 3.9 TB/s, BF16 1,671 TFLOPS sparse (835.5 dense [computed]). Source: https://www.nvidia.com/en-us/data-center/h100/

### H200 (Hopper refresh, 2024)
- Same Hopper GPU with more/faster memory: 141 GB HBM3e, 4.8 TB/s. Source: https://www.nvidia.com/en-us/data-center/h200/
- Compute peaks identical to H100 SXM: BF16/FP16 1,979 TFLOPS with sparsity (so ~990 dense), FP8 3,958 with sparsity. Source: https://www.nvidia.com/en-us/data-center/h200/
- NVLink 900 GB/s, TDP up to 700 W configurable. Source: https://www.nvidia.com/en-us/data-center/h200/
- 141 GB is the usable figure; 144 GB is physically installed (yield headroom). Source: https://www.tomshardware.com/news/nvidia-reveals-gh200-grace-hopper-gpu-with-141gb-of-hbm3e

### GH200 Grace Hopper superchip (Grace CPU + H100 GPU on one module)
All from the official GH200 datasheet: https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf
- GPU memory: 96 GB HBM3 at up to 4 TB/s, or 144 GB HBM3e at up to 4.9 TB/s (the HBM3e variant is often quoted as 141 GB usable).
- CPU: 72 Arm Neoverse V2 cores; L1 64 KB i + 64 KB d per core, L2 1 MB per core, L3 114 MB; up to 480 GB LPDDR5X with ECC at up to 512 GB/s.
- NVLink-C2C between Grace and Hopper: 900 GB/s bidirectional (450 GB/s per direction, per https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/), 7x PCIe Gen5.
- "Up to 624 GB of fast-access memory" per superchip (480 LPDDR5X + 144 HBM3e); module TDP programmable 450 W to 1000 W (CPU + GPU + memory).
- Coherence model: hardware-coherent memory over NVLink-C2C with Address Translation Services; "The CPU and GPU can share a single per-process page table, enabling all CPU and GPU threads to access all system-allocated memory"; the GPU can oversubscribe its HBM and directly use CPU memory; each Hopper GPU can address up to 608 GB within a superchip (Extended GPU Memory). Source: https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/
- Dense TFLOPS on the GH200 datasheet match H100 SXM: BF16 990 dense, FP8 1,979 dense.

### B200 / GB200 (Blackwell, 2024-2025)
- Launch materials: dual-die design, 208 B transistors, 192 GB HBM3e, 8 TB/s. Source: https://wccftech.com/nvidia-blackwell-gpu-architecture-official-208-billion-transistors-5x-ai-performance-192-gb-hbm3e-memory/ (secondary; NVIDIA's own shipping spec sheets below are what to quote)
- Shipping DGX B200 (8x B200, air-cooled): 1,440 GB total HBM3e = 180 GB per GPU, 64 TB/s total = 8 TB/s per GPU [computed]; FP8 72 PFLOPS per system with sparsity footnote "Dense performance is 1/2 sparse" = 9 PFLOPS sparse / 4.5 PFLOPS dense FP8 per GPU [computed]; FP4 144 | 72 PFLOPS (sparse | dense) per system = 9 PFLOPS dense FP4 per GPU [computed]; 2x NVSwitch, 14.4 TB/s aggregate NVLink = 1.8 TB/s per GPU (NVLink 5). Source: https://www.nvidia.com/en-us/data-center/dgx-b200/
  - B200 dense BF16 is 2.25 PFLOPS per GPU [computed as half of dense FP8, consistent with NVIDIA's precision ladder; not printed on the DGX page itself].
- GB200 superchip = 1 Grace CPU + 2 Blackwell GPUs, NVLink-C2C coherent link (same 900 GB/s C2C as Grace Hopper). GB200 NVL72 rack = 36 Grace CPUs + 72 GPUs on one NVLink domain: 13.4 TB HBM3e total (~186 GB per GPU [computed]), 576 TB/s total (8 TB/s per GPU [computed]), NVLink 130 TB/s aggregate; Grace side: 72 Neoverse V2 cores per CPU, up to 480 GB LPDDR5X at up to 512 GB/s. Per-GPU dense peaks [computed from NVL72 totals with the sparse|dense footnote]: FP4 10 PFLOPS, FP8 5 PFLOPS, BF16 2.5 PFLOPS (the GB200 variant runs hotter/faster than air-cooled B200). Source: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- Second-generation Transformer Engine: adds FP4 (for inference) alongside FP8 (training), with microscaling formats. Source: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVLink 5: 1.8 TB/s per GPU, 18 links. Source: https://www.nvidia.com/en-us/data-center/nvlink/

## 2. Arithmetic intensity and the roofline

- Definitions (NVIDIA's own performance guide): arithmetic intensity = "the ratio of algorithm implementation operations and the number of bytes accessed"; a processor's ops:byte ratio = "the ratio of a processor's math and memory bandwidths". "An algorithm is math limited on a given processor if the algorithm's arithmetic intensity is higher than the processor's ops:byte ratio", and memory limited if lower. Source: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
- Worked example in the guide: V100 has 125 FP16 Tensor TFLOPS, ~900 GB/s HBM, 3.1 TB/s L2 bandwidth, giving an ops:byte ratio between 40 and 139 depending on where operands come from. Source: same URL.
- Ops:byte ratios at dense BF16 vs HBM [computed from the datasheet numbers in Section 1]:
  - A100 80GB SXM: 312e12 / 2.039e12 ≈ 153 FLOP/byte
  - H100 SXM: 989.4e12 / 3.35e12 ≈ 295 FLOP/byte (FP8: ~590)
  - H200: 990e12 / 4.8e12 ≈ 206 FLOP/byte
  - GH200 HBM3e: 990e12 / 4.9e12 ≈ 202 FLOP/byte
  - B200: 2.25e15 / 8e12 ≈ 281 FLOP/byte
  Reading: a kernel must do roughly 150-300 BF16 FLOPs per byte of HBM traffic to be compute-bound; anything less is bandwidth-bound. FLOPS have grown much faster than bandwidth across generations, so fusion/tiling matter more every generation.
- Horace He's three-regime framing ("Making Deep Learning Go Brrrr From First Principles"): time goes to compute, memory bandwidth, or overhead. A100 example: 312 TFLOPS matmul vs 1.5 TB/s bandwidth vs only 19.5 TFLOPS for non-matmul (CUDA-core) ops; elementwise chains like x.cos().cos() are pure bandwidth (fusion halves memory traffic for a 2x win); "in the time that Python can perform a single FLOP, an A100 could have chewed through 9.75 million FLOPS" (overhead regime, hidden by async execution when kernels are big enough). Source: https://horace.io/brrr_intro.html
- "Typical reduction operations have a low arithmetic intensity and thus are memory limited" (softmax, layernorm, most activations). Source: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
- Workload implications for video models (interpretation grounded in the sources above, not a separate measured claim):
  - Attention: naive attention is memory-bound because the N x N score matrix makes HBM round-trips; FlashAttention removes that traffic and turns attention into a mostly matmul-bound kernel (Section 3). Video DiTs make N huge (N scales with frames x height x width / patch volume), so the N^2 term dominates both FLOPs and, if unfused, HBM traffic.
  - VAE decode: convolutions over near-pixel-resolution activation maps have large activation footprints relative to their FLOPs; decode tends to be bandwidth- and VRAM-limited, which is why frameworks ship tiled/sliced VAE decode (e.g. enable_vae_tiling / enable_vae_slicing in Hugging Face Diffusers: https://huggingface.co/docs/diffusers/optimization/memory).
  - Diffusion denoising loops repeat many moderate-size kernels per step, so overhead and non-matmul (norm, embedding, elementwise) time also matters; this is the regime where fusion and CUDA graphs pay off (Horace He, same URL).

## 3. FlashAttention and PyTorch SDPA

### FlashAttention (2022, Dao et al.)
- Core idea: IO-aware exact attention. "Uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM"; the N x N attention matrix is never materialized in HBM; online softmax computes the normalization incrementally per tile, and the backward pass recomputes attention on the fly instead of storing it. HBM memory usage becomes linear in N instead of quadratic. Source: https://arxiv.org/abs/2205.14135
- Reported gains: 15% end-to-end on BERT-large (seq 512), 3x on GPT-2 (seq 1K), 2.4x on long-range arena (1K-4K); proven "optimal for a range of SRAM sizes" in HBM accesses. Source: https://arxiv.org/abs/2205.14135

### FlashAttention-2 (2023)
- Three changes: (1) fewer non-matmul FLOPs in the online-softmax rescaling; (2) parallelize over the sequence-length dimension so even a single head spans multiple thread blocks (better occupancy for long sequences / small batch); (3) partition work between warps within a block to cut shared-memory traffic. Source: https://arxiv.org/abs/2307.08691
- Reported: ~2x over FlashAttention-1; 50-73% of A100 peak FLOP/s for the kernel; 225 TFLOP/s per A100 (72% model FLOPs utilization) in end-to-end GPT training. Source: https://arxiv.org/abs/2307.08691
- FA2 is the algorithm behind PyTorch's "flash attention" SDPA backend. Source: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html

### FlashAttention-3 (2024, Hopper-specific)
- Exploits Hopper hardware: WGMMA (warpgroup async Tensor Core instructions), TMA (async global-to-shared copies that free registers), and FP8 Tensor Cores. Source: https://pytorch.org/blog/flashattention-3/
- Warp specialization: producer warpgroups issue TMA loads while consumer warpgroups compute (asynchrony between data movement and math). Source: https://pytorch.org/blog/flashattention-3/ and https://arxiv.org/abs/2407.08608
- Pingpong scheduling: with 2 warpgroups, barriers arrange that one warpgroup's softmax runs while the other's GEMMs run; measured 570 -> 620 TFLOPS (FP16 fwd, head dim 128, seq 8K), plus intra-warpgroup GEMM/softmax pipelining to 640-660 TFLOPS. Source: https://pytorch.org/blog/flashattention-3/
- FP8: block quantization plus incoherent processing (random-sign Hadamard transform on Q/K to spread outliers, per QuIP) gives 2.6x lower quantization error than baseline FP8 attention. Source: https://arxiv.org/abs/2407.08608 and https://pytorch.org/blog/flashattention-3/
- Reported: 1.5-2.0x over FA2 on H100, up to 740 TFLOPS FP16 = 75% of H100 peak; FP8 close to 1.2 PFLOPS. Source: https://arxiv.org/abs/2407.08608
- Code lives in the same repo as FA2: https://github.com/Dao-AILab/flash-attention

### PyTorch scaled_dot_product_attention (SDPA)
- One API, multiple backends: FlashAttention-2, Memory-Efficient Attention (the xFormers kernel), and a C++ "math" fallback that matches the reference formulation; on CUDA it "attempts to automatically select the most optimal implementation based on the inputs". A cuDNN-based backend also exists on recent CUDA builds (listed in torch.nn.attention.SDPBackend as CUDNN_ATTENTION). Source: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
- Control: torch.nn.attention.sdpa_kernel() context manager is "the preferred mechanism" to restrict backends (global toggles enable_flash_sdp / enable_mem_efficient_sdp / enable_math_sdp also exist); when a fused kernel is rejected (dtype, head dim, mask type, strides), PyTorch warns with the reasons. Source: same URL.

## 4. NVLink, NVSwitch, multi-node

- NVLink per-GPU bandwidth by generation (all NVIDIA totals are bidirectional): NVLink 2 (V100) 300 GB/s (https://www.nvidia.com/en-us/data-center/v100/); NVLink 3 (A100) 600 GB/s (https://www.nvidia.com/en-us/data-center/a100/); NVLink 4 (H100/H200) 900 GB/s over 18 links; NVLink 5 (Blackwell) 1,800 GB/s over 18 links; NVLink 6 announced at 3,600 GB/s. Source for gens 4-6: https://www.nvidia.com/en-us/data-center/nvlink/
- NVSwitch makes NVLink all-to-all within a node (or rack): DGX B200 uses 2 NVSwitch chips for 14.4 TB/s aggregate; GB200 NVL72 puts 72 GPUs on one NVLink-5 switch fabric at 130 TB/s aggregate. Sources: https://www.nvidia.com/en-us/data-center/dgx-b200/ and https://www.nvidia.com/en-us/data-center/nvlink/
- Cross-node fabric on reference H100 systems: DGX H100 has 8 single-port ConnectX-7 NICs at up to 400 Gb/s each (InfiniBand NDR or Ethernet), i.e. one 400 Gb/s rail per GPU, 3.2 Tb/s per node, plus 2 dual-port ConnectX-7 for storage/management. Sources: https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf and the DGX H100 datasheet https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf
- HPE Slingshot (Frontier, El Capitan, Perlmutter class systems) runs 200 Gb/s per NIC port (Slingshot-11). [UNVERIFIED against an HPE primary page in this pass]
- Perspective: NVLink 4 at 900 GB/s is ~18x one 400 Gb/s (50 GB/s) InfiniBand rail [computed]; hence tensor/sequence parallelism stays inside the NVLink domain and only data/pipeline-parallel traffic crosses the IB fabric.
- NCCL all-reduce algorithms: rings deliver full bandwidth but "latency scales linearly with the number of GPUs, preventing scaling above hundreds of GPUs"; NCCL 2.4 added double binary trees with "full bandwidth and a logarithmic latency even lower than 2D ring latency" (each rank sends/receives at most half the data twice, matching ring bandwidth optimality); measured up to 180x latency improvement at 24k GPUs on Summit; NCCL auto-selects and "switches back to rings when that pattern results in greater bandwidth". Source: https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
- Why gradient sync overlaps compute (PyTorch DDP): the Reducer "organizes parameter gradients into buckets, and reduces one bucket at a time"; "when gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket"; autograd hooks fire during backward, so "DDP's performance advantage comes from overlapping allreduce collectives with computations during backwards". Source: https://docs.pytorch.org/docs/2.13/notes/ddp.html
- Rendezvous in torchrun/TorchElastic: "a distributed synchronization primitive with peer discovery". It is a barrier: nodes block until at least min nodes join (waits briefly for stragglers, completes immediately at max); on completion all members agree on membership and each node gets a rank in [0, world_size), and "these ranks are not stable, in the sense that the same node can be assigned a different rank in the next (re-)rendezvous"; only one worker group can exist per job (exclusivity); node failures trigger re-rendezvous of the survivors. Source: https://docs.pytorch.org/docs/2.13/elastic/rendezvous.html

## 5. GPU monitoring

- What nvidia-smi "GPU-Util" actually is (NVML utilization.gpu): "Percent of time over the past sample period during which one or more kernels was executing on the GPU", with a sample period between 1/6 s and 1 s. It is kernel-active wall-clock fraction, NOT occupancy, NOT SM coverage, NOT FLOP efficiency. Source: nvidia-smi manual, https://docs.nvidia.com/deploy/nvidia-smi/index.html
- Famous pitfall: a kernel using a single SM that runs continuously reports 100% GPU-Util (an H100 doing this has ~0.7% SM efficiency); NCCL-heavy phases can pin GPU-Util near 100% while SM activity/occupancy sits at 10-20%. Sources: https://arthurchiao.art/blog/understanding-gpu-performance/ and https://leimao.github.io/blog/NVIDIA-NVML-GPU-Statistics/
- Corollary: "100% util" is fully compatible with a memory-bandwidth-bound (or even overhead-bound) job. For real efficiency, use DCGM profiling metrics: SM activity (DCGM_FI_PROF_SM_ACTIVE), SM occupancy (DCGM_FI_PROF_SM_OCCUPANCY), Tensor Core activity (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE), DRAM activity (DCGM_FI_PROF_DRAM_ACTIVE), or compute MFU at the framework level. Sources: https://developer.nvidia.com/dcgm and https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html
- Memory accounting: PyTorch's caching allocator means nvidia-smi memory != tensor memory. torch.cuda.memory_allocated() tracks tensor-occupied bytes, memory_reserved() tracks the allocator's pool, and "the unused memory managed by the allocator will still show as if used in nvidia-smi"; empty_cache() releases only unused cached blocks. Source: https://docs.pytorch.org/docs/2.13/notes/cuda.html
- MIG (Multi-Instance GPU): partitions one GPU into up to 7 fully isolated instances, each with "its own high-bandwidth memory, cache, and compute cores" and fault isolation; supported on A100/H100 (introduced with Ampere) through Hopper and Blackwell (e.g. GB200: 2x93 GB, 4x46 GB, or 7x23 GB instances). Sources: https://www.nvidia.com/en-us/technologies/multi-instance-gpu/ and https://docs.nvidia.com/datacenter/tesla/mig-user-guide/

## 6. aarch64 / Grace specifics

- The superchip model: Grace (72-core Arm Neoverse V2, LPDDR5X) is cache-coherent with the GPU over NVLink-C2C at 900 GB/s; system-allocated (malloc) memory is directly GPU-accessible via a shared page table, and the GPU can spill/oversubscribe into CPU LPDDR5X at C2C speed instead of PCIe speed. This changes offload economics: C2C at 900 GB/s vs PCIe Gen5 x16 at 128 GB/s bidirectional [computed, 7x claim per NVIDIA]. Sources: GH200 datasheet https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf and https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/
- Binary compatibility: "the very same containers, application binaries, and operating systems that run on other Arm products run on Grace Hopper without modification" (i.e. standard linux/arm64; nothing exotic at the ISA level). Source: GH200 datasheet, same URL.
- The real friction is Python wheels, not the ISA:
  - PyTorch: plain `pip install torch` from PyPI historically gave CPU-only builds on aarch64; CUDA-enabled aarch64 wheels first shipped through download.pytorch.org indexes (official arm64+CUDA support arriving around PyTorch 2.7), and only PyTorch 2.11 made CUDA-enabled aarch64 wheels installable directly from PyPI, explicitly motivated by GH200/GB200/GB300 users. Sources: https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/ and https://github.com/pytorch/pytorch/issues/160162
  - flash-attn: no official prebuilt aarch64 wheels; on GH200 you build from source (long compile, needs bounded MAX_JOBS to avoid OOM; community wheels exist for specific torch/CUDA combos). Sources: https://github.com/Dao-AILab/flash-attention/issues/1866 and https://github.com/Dao-AILab/flash-attention/issues/2036
  - The same wheel gap pattern repeats across the ecosystem (Triton and dependent packages lagged on aarch64; the vLLM+PyTorch aarch64 effort was about closing exactly this). Source: https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/
  - Practical default on Grace systems: NVIDIA NGC PyTorch containers, which ship arm64 builds with matched CUDA/torch/flash-attn stacks. Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch [general reference]
- GB200 keeps the same model at Blackwell scale: 1 Grace + 2 Blackwell GPUs per superchip, coherent C2C, LPDDR5X up to 480 GB / 512 GB/s per Grace. Source: https://www.nvidia.com/en-us/data-center/gb200-nvl72/

## Sources (best 15)

1. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ - Hopper deep dive: SM/SMEM/L2 sizes, TMA, thread block clusters, distributed shared memory.
2. https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ - A100 deep dive: 192 KB L1/SMEM (164 KB usable), 40 MB L2, HBM2 numbers.
3. https://www.nvidia.com/en-us/data-center/h100/ - H100 SXM/NVL spec table with the "with sparsity" footnotes.
4. https://www.nvidia.com/en-us/data-center/h200/ - H200 spec table: 141 GB HBM3e, 4.8 TB/s.
5. https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf - official GH200 datasheet: both HBM variants, LPDDR5X, C2C, and the rare sparse|dense TFLOPS table.
6. https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/ - C2C coherence, ATS shared page table, Extended GPU Memory.
7. https://www.nvidia.com/en-us/data-center/gb200-nvl72/ - GB200 NVL72 spec table (HBM3e totals, NVLink 5, FP4/FP8, 2nd-gen Transformer Engine).
8. https://www.nvidia.com/en-us/data-center/dgx-b200/ - shipping B200 per-GPU numbers via system totals; NVSwitch aggregate.
9. https://www.nvidia.com/en-us/data-center/nvlink/ - NVLink/NVSwitch generation table.
10. https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html - NVIDIA's own roofline/arithmetic-intensity doctrine.
11. https://horace.io/brrr_intro.html - the three-regime (compute/bandwidth/overhead) mental model with A100 numbers.
12. https://arxiv.org/abs/2205.14135 , https://arxiv.org/abs/2307.08691 , https://arxiv.org/abs/2407.08608 - FlashAttention 1/2/3 papers.
13. https://pytorch.org/blog/flashattention-3/ - FA3 mechanics with measured TFLOPS per technique (warp specialization, pingpong).
14. https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html - SDPA backends and dispatch control; plus https://docs.pytorch.org/docs/2.13/notes/ddp.html and https://docs.pytorch.org/docs/2.13/elastic/rendezvous.html for DDP overlap and rendezvous.
15. https://docs.nvidia.com/deploy/nvidia-smi/index.html and https://arthurchiao.art/blog/understanding-gpu-performance/ - what GPU-Util really measures and why it misleads.
