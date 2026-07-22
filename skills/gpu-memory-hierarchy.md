---
name: gpu-memory-hierarchy
description: Load when sizing video workloads across GPU architectures, diagnosing memory peaks, reasoning about attention or decode bottlenecks, or planning offload.
---

# GPU Memory Hierarchy

## When to use

- Estimate whether a video model, KV cache, decode, or training step fits.
- Distinguish tensor ownership from allocator reservation and `nvidia-smi` memory.
- Choose between tiling, chunking, checkpointing, sharding, and host offload.
- Compare A100 (40 GB), H100 (80 GB), GH200 (96 GB), H200 (141 GB), and GB200-class devices.

## Procedure

1. Inventory memory by owner: weights and support models, optimizer state, activations, KV or cross-attention caches, decoded pixels, and framework workspaces.
2. Map access to the hierarchy: registers and shared memory or L1 per SM, GPU-wide L2, HBM, then PCIe host memory or coherent Grace LPDDR5X through NVLink-C2C.
3. Use `docs/HARDWARE-NOTES.md` for cited capacity, bandwidth, cache, dense compute, host-link, and roofline values.
4. Calculate horizon-scaled buffers explicitly. Include frames, spatial tokens, layers, heads, dtype bytes, branches, and batch.
5. Measure a complete phase trace. Record `torch.cuda.memory_allocated()`, `memory_reserved()`, maximum allocated, and `nvidia-smi` memory together.
6. If `empty_cache()` does not reduce residency, find live Python owners and delete or scope them before blaming fragmentation.
7. Classify the failure signature: CUDA allocator exception, bare cgroup kill, login virtual-memory failure, or process exhaustion.
8. Apply the narrow repair: inference mode for accidental graphs, activation checkpointing for saved activations, full FSDP for distributed state, tiling or chunking for decode, or explicit component placement for resident models.
9. Recalibrate through optimizer step two and sequential long-horizon samples. Fragmentation and state materialization can appear late.
10. Treat Grace memory oversubscription as an explicit performance experiment. Coherent access does not make LPDDR5X equal to HBM.

## Rules (hard)

- Quote dense Tensor Core peaks, because vendor sparse figures assume structured 2:4 sparsity that ordinary workloads rarely satisfy.
- Size the complete cycle, because decode, probes, and optimizer state can exceed the denoise or first-step peak. [FL G14; FL G30]
- Release owning objects before allocator cleanup, because cached blocks cannot free live tensors. [FL G30]
- Distinguish host and device OOM signatures, because changing GPU batch does not repair every cgroup failure. [FL G15]
- Preserve model-specific dtype contracts, because a global bf16 rule can damage quality or hard-fail decode. [VG1 gen.py:14-16; VG1 NOTES-e4-agent.md:43-46]
- Measure throughput after offload, because capacity gained through a slower tier can shift the bottleneck.

## Pitfalls seen in production

- Symptom: a 1.3B inference path OOMs near 95 GB. Cause: vendored code built an accidental autograd graph of about 80 GB. Fix: apply `no_grad` or inference mode at the adapter boundary. [FL G6]
- Symptom: training OOMs after a probe despite allocator cleanup. Cause: a live batch-four, 30-layer cache retained about 24 GiB. Fix: release the probe object after every rollout chunk. [FL G30]
- Symptom: one 240-latent rCM sample consumes about 70 GB. Cause: a full-length, non-windowed KV cache. Fix: keep batch one or redesign the cache. [FL G15]
- Symptom: the third 240-second video fails after earlier successes. Cause: GPU decode plus a second 17 to 18 GB fp32 copy fragmented memory. Fix: chunk, offload, and release every sample. [FL G17]
- Symptom: a separately created critic doubles activation use. Cause: it lacks inherited activation checkpointing. Fix: enable checkpointing explicitly and remeasure. [VSAO SMOKE.md:1136-1176]
- Symptom: rank zero reaches 38.49 of 39.49 GiB while peers fit. Cause: prompt and support models reside only on rank zero. Fix: pre-encode shared prompts and drop services before training. [VSAO SMOKE.md:2071-2088]
- Symptom: torch reports 32.8 GiB while `nvidia-smi` reports 78,752 MiB. Cause: allocator reservation and non-PyTorch allocations differ from live tensor accounting. Fix: record both rather than treating either as the whole truth. [VSAO vista/CALIBRATION.md:9,18-19]

## Pointers

- Related skills: `gpu-utilization-batching.md`, `multi-node-training.md`, `video-io.md`, `env-setup-aarch64.md`.
- Reference: `docs/HARDWARE-NOTES.md`.
- Compendium themes: 1, 2, 3, 7, and 11.
