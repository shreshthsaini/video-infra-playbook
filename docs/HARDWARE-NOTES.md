# Hardware Notes

This cheat sheet combines vendor specifications from
[`gpu-arch-factsheet.md`](sources/gpu-arch-factsheet.md) with measured envelopes
from the three source projects. Capacity figures use vendor decimal GB or TB/s.
Project measurements retain the units reported by their tools.

## Reading the numbers

The working hierarchy is registers, configurable shared memory and L1, GPU-wide
L2, HBM, then host memory. Host traffic crosses PCIe on conventional GPU systems
or a coherent NVLink-C2C link on Grace superchips. A100 and H100 each expose a
256 KB register file per SM, or 64K 32-bit registers. Sources:
[Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/),
[Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

Quote dense Tensor Core throughput for ordinary training and inference. NVIDIA
tables often lead with structured 2:4 sparse throughput, which is twice the
dense rate but is not available to an ordinary dense model. H100 is especially
easy to misread: 989 TFLOPS means sparse TF32 but approximately dense BF16.
Sources: [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/),
[GH200 datasheet](https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf).

PyTorch allocator counters and `nvidia-smi` answer different questions.
`memory_allocated()` counts live tensor bytes, `memory_reserved()` includes the
caching allocator pool, and `nvidia-smi` also sees reserved and non-PyTorch
allocations. `empty_cache()` releases unused cached blocks, not tensors that a
Python object still owns. Source:
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.13/notes/cuda.html).

## A100 40 GB

### Architecture

| Field | Value |
|---|---|
| HBM | 40 GB HBM2 at 1,555 GB/s |
| L2 | 40 MB |
| SMEM and L1 | 192 KB combined per SM, up to 164 KB configurable shared memory |
| SM count | 108 |
| Register file | 256 KB per SM |
| Dense Tensor Core peak | BF16 or FP16 312 TFLOPS; TF32 156 TFLOPS |
| Sparse footnote value | BF16 or FP16 624 TFLOPS; TF32 312 TFLOPS |
| Host and fabric links | PCIe Gen4 at 64 GB/s bidirectional; NVLink 3 at 600 GB/s per GPU |

Sources: [NVIDIA Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
and [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/).

### Project shape and measured envelopes

VSAO ran three A100-40GB GPUs per full LS6 node, plus one-GPU development and
small-node shapes. Wan calibration measured 32.68 GB torch peak and 34,945 MiB
by `nvidia-smi`, or 85 percent. Tiled CogVideoX measured 24.35 GB torch and
26,829 MiB, or 65 percent. LTX measured 22.35 GB torch and 31,177 MiB, or
76 percent. [VSAO CALIBRATION.md:50-56]

For GRPO, about 21.5 GiB of static state plus microbatch-four activations of
about 23 GiB exceeded the card. Microbatch two stayed at 33.3 to 34.1 GiB.
[VSAO report-vsao.md, GPU memory learnings] A repaired three-device trainer
placed policy, critic, and rewards separately, leaving 7.1 GiB headroom on the
policy GPU at 33,321 MiB. [VSAO SMOKE.md:577-582,638-641]

Practical note: the 40 GB capacity makes activation checkpointing, support-model
placement, VAE tiling, and complete-cycle calibration first-order choices.

## H100 80 GB SXM

### Architecture

| Field | Value |
|---|---|
| HBM | 80 GB HBM3 at 3.35 TB/s |
| L2 | 50 MB |
| SMEM and L1 | 256 KB combined per SM, up to 228 KB configurable shared memory |
| SM count | 132 |
| Register file | 256 KB per SM |
| Dense Tensor Core peak | BF16 or FP16 about 990 TFLOPS; TF32 494 TFLOPS; FP8 1,979 TFLOPS |
| Sparse footnote value | BF16 or FP16 1,979 TFLOPS; TF32 989 TFLOPS; FP8 3,958 TFLOPS |
| Host and fabric links | PCIe Gen5 x16 at 128 GB/s bidirectional; NVLink 4 at 900 GB/s per GPU |

Sources: [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),
[Hopper whitepaper](https://resources.nvidia.com/en-us-tensor-core),
[NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), and the
[GH200 datasheet](https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf).

Hopper adds TMA for asynchronous global-to-shared transfers, thread-block
clusters, and distributed shared memory. These features are the basis of
FlashAttention-3 warp specialization and overlap. Sources:
[Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),
[FlashAttention-3](https://pytorch.org/blog/flashattention-3/).

### Project shape and measured envelopes

VSAO ran two H100 GPUs per LS6 node. Hunyuan used 48.74 GB torch peak and
51,119 MiB by `nvidia-smi`, so one worker per 80 GB GPU was the calibrated
shape. [VSAO CALIBRATION.md:50-56] One LTX worker produced 625.6 rollouts per
hour, while two co-located workers produced about 578 per hour combined.
Co-location reduced throughput despite available memory. [VSAO CALIBRATION.md:50-56]

Practical note: H100 raises the dense BF16 compute-to-HBM ratio sharply, so
fusion and tiling matter more even though raw bandwidth also increases.

## GH200 96 GB

### Architecture

The projects used the GH200 variant with 96 GB HBM3 at up to 4 TB/s. The
Hopper GPU retains the 50 MB L2, up to 228 KB shared memory per SM, and 256 KB
register file per SM described above. Dense peaks match H100 SXM: BF16 about
990 TFLOPS and FP8 1,979 TFLOPS. The corresponding sparse figures are about
1,979 and 3,958 TFLOPS. Source:
[GH200 datasheet](https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf).

Each superchip has a 72-core Arm Neoverse V2 Grace CPU. Grace provides 64 KB
instruction plus 64 KB data L1 and 1 MB L2 per CPU core, 114 MB shared L3,
and up to 480 GB ECC LPDDR5X at up to 512 GB/s. NVLink-C2C connects Grace and
Hopper at 900 GB/s bidirectional, or 450 GB/s per direction. Hardware
coherence, Address Translation Services, and a shared per-process page table
let the GPU address system memory directly. Sources:
[GH200 datasheet](https://download.boston.co.uk/downloads/0/5/8/0586c659-27bf-4c16-b8b0-0df7822468b2/grace-hopper-superchip-datasheet-2705455.pdf),
[Grace Hopper architecture](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/).

### Project shape and measured envelopes

Vista exposed one GH200 GPU per node. CachedSearch Wan used about 30 GB at
batch one, CogVideoX about 25 GB, and HunyuanVideo about 43 GB in weights.
[VG1 RELEASE.md:93,192-193] Forcing Laws measured 48 to 54 GB during denoising
and 91 GB during batch-four VAE decode of four 60-second clips. [FL G14] A
240-latent rCM KV cache used about 70 GB per sample. [FL G15] A Self-Forcing
training smoke peaked at 52.7 GB with gradient checkpointing. [FL G7]

Practical note: coherent Grace memory improves offload economics compared with
PCIe, but LPDDR5X remains far below the HBM bandwidth. Oversubscription can fit
a workload while making it slower or moving pressure to host memory. The
aarch64 platform also has fewer compatible Python wheels, so SDPA and isolated
UV environments were the reliable controls.

## H200 141 GB

### Architecture

H200 is a Hopper refresh with 141 GB usable HBM3e at 4.8 TB/s. The physical
installation is 144 GB, with 141 GB exposed as usable capacity. Its compute
peaks match H100 SXM, so quote dense BF16 or FP16 at about 990 TFLOPS and dense
FP8 at 1,979 TFLOPS. Sparse table values are 1,979 and 3,958 TFLOPS. NVLink is
900 GB/s and configurable TDP reaches 700 W. Sources:
[NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/),
[installed versus usable capacity](https://www.tomshardware.com/news/nvidia-reveals-gh200-grace-hopper-gpu-with-141gb-of-hbm3e).

Because NVIDIA describes H200 as the same Hopper GPU with more and faster
memory, use the Hopper cache organization for planning: 50 MB L2 and up to
228 KB shared memory per SM in a 256 KB combined L1 and shared array. Source:
[Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

### Project shape and measured envelopes

None of the three source projects ran on H200. Do not transfer a GH200 or
H100 batch table without a complete-cycle calibration. The larger HBM can fit
more resident state, while the lower dense BF16 ops:byte ratio than H100 can
shift some kernels toward compute limitation.

## GB200-class 185 GiB project devices

### Architecture

The Vista project reported four 185 GiB GB200 devices per node. NVIDIA's
NVL72 shipping specification gives 13.4 TB HBM3e across 72 GPUs, approximately
186 GB per GPU, and 576 TB/s aggregate, or 8 TB/s per GPU. Computed per-GPU
dense peaks from the NVL72 table are BF16 2.5 PFLOPS, FP8 5 PFLOPS, and FP4
10 PFLOPS. Sparse values are twice those dense figures under NVIDIA's footnote.
NVLink 5 provides 1.8 TB/s per GPU and 130 TB/s aggregate across the rack.
Sources: [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/),
[NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/).

One GB200 superchip combines one Grace CPU and two Blackwell GPUs. Grace
provides up to 480 GB LPDDR5X at up to 512 GB/s, and the coherent NVLink-C2C
model follows Grace Hopper. [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

[CHECK] The staged primary-source pass did not verify a shipping GB200 L2
capacity or configurable shared-memory-per-SM value. Do not substitute rumor
or an earlier Blackwell launch figure.

### Project shape and measured envelopes

Wan 81-frame rollout measured 78,752 MiB by `nvidia-smi` but 32.8 GiB by torch,
about 42 percent of the project device. [VSAO vista/CALIBRATION.md:9,18-19]
During live SAO, the trainer used 55,865 MiB on GPU zero and seven rollout GPUs
used about 35,447 MiB each while reporting 85 to 100 percent utilization.
[VSAO vista/SMOKE.md:126-127]

Two 81-frame runs per device were unstable despite abundant HBM. Six or more
concurrent runs saturated shared scratch I/O, while about four to five runs,
roughly one per node, were stable. [VSAO report-vsao.md, Vista GB200 side]

Practical note: low memory fraction was not free throughput. The Wan path was
compute-bound without an exposed batching route, and storage throughput set a
lower packing ceiling than HBM.

## Roofline and video workload implications

Arithmetic intensity is FLOPs per byte moved. A kernel is compute-limited when
its arithmetic intensity exceeds the processor's dense compute-to-bandwidth
ratio and memory-limited when it is below. Source:
[NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html).

| Card | Dense BF16 divided by HBM bandwidth | Interpretation |
|---|---:|---|
| A100 40 GB | about 201 FLOP/byte | Computed from 312 TFLOPS and 1.555 TB/s. |
| H100 80 GB SXM | about 295 FLOP/byte | Computed from about 990 TFLOPS and 3.35 TB/s. |
| GH200 96 GB | about 248 FLOP/byte | Computed from about 990 TFLOPS and 4.0 TB/s. |
| H200 141 GB | about 206 FLOP/byte | Computed from about 990 TFLOPS and 4.8 TB/s. |
| GB200 NVL72 GPU | about 313 FLOP/byte | Computed from 2.5 PFLOPS and 8 TB/s. |

Naive attention materializes an `N x N` score matrix and makes large HBM
round trips. FlashAttention tiles exact attention through on-chip SRAM, uses
online softmax, and avoids materializing that matrix in HBM, reducing memory
growth from quadratic to linear in sequence length. Sources:
[FlashAttention](https://arxiv.org/abs/2205.14135),
[FlashAttention-2](https://arxiv.org/abs/2307.08691),
[FlashAttention-3](https://arxiv.org/abs/2407.08608).

VAE decode operates on near-pixel-resolution activations and often becomes
bandwidth, VRAM, or host-staging limited. Tiling and slicing reduce the peak at
the cost of more launches and boundaries. Source:
[Diffusers memory optimization](https://huggingface.co/docs/diffusers/optimization/memory).

Denoise loops repeat matmul plus normalization, embedding, and elementwise
kernels many times. They can move among compute, bandwidth, and launch-overhead
regimes. Fusion and CUDA graphs help overhead and elementwise chains, while
attention tiling attacks HBM traffic. Source:
[Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html).

Treat roofline values as upper-bound classification aids, not utilization
claims. `nvidia-smi` reports the percentage of its sample period with any
kernel executing, not FLOP efficiency. Source:
[nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
