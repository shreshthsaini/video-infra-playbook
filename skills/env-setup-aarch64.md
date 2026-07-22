---
name: env-setup-aarch64
description: Load when creating or repairing CUDA Python environments on aarch64 Grace systems, selecting wheels, attention backends, compilers, or cache locations.
---

# Environment Setup on aarch64

## When to use

- Create a reproducible UV environment for GH200 or GB200 nodes.
- Resolve missing CUDA operators, wheel ABI errors, or attention backend gaps.
- Build Triton, detectron2, or other native extensions on Grace.
- Isolate incompatible generation and evaluation dependency stacks.

## Procedure

1. Inspect the existing environment, architecture, driver, CUDA module, Python, and effective cache variables before changing anything.
2. Create environments only with UV at a stable work path that resolves to scratch:
   ```bash
   export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/$USER/vista/library/uv-cache}"
   ~/.local/bin/uv venv "/work/$USER/vista/library/uv-envs/my-env" --python 3.11
   ```
3. Build or install substantial packages on a compute node. Never mutate legacy Conda environments.
4. Install torch and torchvision from the same CUDA-enabled PyTorch index. Verify `torch.cuda.is_available()` and a CUDA torchvision operator before adding packages that can repin them.
5. Pin compatibility boundaries such as transformers and evaluator stacks. Put conflicting VBench or scorer dependencies in a separate UV environment.
6. Prefer parity-tested PyTorch SDPA when external flash-attn lacks a compatible aarch64 wheel. Freeze backend selection across comparison arms.
7. For Triton or native CUDA builds, set `CC=gcc`, `CXX=g++`, and where required `CUDAHOSTCXX=g++`. Unset `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, and `INCLUDE` if NVIDIA compiler headers leak into gcc.
8. Place `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` on scratch or node-local temporary storage, namespaced by host, run, role, and retry.
9. Make Hugging Face cache variables explicit in every wrapper. Disable Xet and cap download workers when login-node limits are relevant, then run jobs offline after prefetch.
10. Record Python, torch, CUDA index, package pins, backend, compiler, and cache roots in the run manifest.

## Rules (hard)

- Use UV only, because mixed environment managers make dependency provenance and recovery unreliable. [memory-facts]
- Keep environments and caches off home, because home has a small quota and scratch content is regenerable. [memory-facts]
- Install torch and torchvision from one CUDA index, because PyPI aarch64 builds can silently provide a CPU ABI. [VG1 report-videogen1.md, Environment pins]
- Keep attention backend fixed across arms, because backend switching confounds relative results. [FL D8; FL G4]
- Isolate generated compiler sources per active node and retry, because shared writes can corrupt them. [FL G27]
- Run heavy installs on compute nodes, because login nodes have an 8 GB virtual-memory limit. [memory-facts]

## Pitfalls seen in production

- Symptom: `torchvision::nms does not exist` or transformers cannot import UMT5. Cause: PyPI installed a CPU torchvision ABI. Fix: repin torch and torchvision from the matching CUDA index. [VG1 report-videogen1.md, Environment pins]
- Symptom: ImageReward works, then later generation imports fail. Cause: a transitive install replaced torchvision. Fix: repin torchvision last and record the lock. [memory-facts]
- Symptom: flash-attn import requires GLIBC_2.32 on a GLIBC_2.28 host. Cause: an incompatible x86 or newer-system wheel. Fix: use parity-tested SDPA or build a bounded aarch64 source package. [VSAO SMOKE.md:7,156-161]
- Symptom: Triton utility compilation invokes `nvc` and fails. Cause: the TACC module exported NVIDIA compilers. Fix: set gcc and g++ explicitly. [FL G7]
- Symptom: detectron2 source build sees incompatible headers. Cause: CPATH contains nvc-specific include roots. Fix: unset include variables and pin the CUDA architecture. [FL G10]
- Symptom: a model shard downloads twice and one copy corrupts. Cause: `HF_HUB_CACHE` overrides project `HF_HOME`. Fix: set the complete effective cache contract in every job. [VG1 report-videogen1.md, Storage layout]

## Pointers

- Related skills: `storage-and-caches.md`, `video-io.md`, `gpu-memory-hierarchy.md`.
- Compendium themes: 7, 9, 10, and 11.
