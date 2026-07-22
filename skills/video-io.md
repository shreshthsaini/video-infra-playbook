---
name: video-io
description: Load when decoding, encoding, inspecting HDR, converting video tensors, or stabilizing video evaluation across Vista and aarch64 environments.
---

# Video I/O

## When to use

- Read or write video in generation, scoring, or evaluation pipelines.
- Configure torchcodec and FFmpeg on Vista aarch64 systems.
- Detect HDR or 10-bit inputs before a lossy decode path.
- Investigate codec-dependent metric shifts or zero-byte outputs.

## Procedure

1. Import the portable helpers in `infra/python/video_io.py`.
2. Inspect `hdr_info(path)` before decode. Stop or select an HDR-capable path when color transfer or primaries indicate HDR.
3. Configure the complete vendored FFmpeg shared library path and the matching NPP wheel directory before importing torchcodec.
4. Install torchcodec and `nvidia-npp-cu12` from the same cu128 PyTorch index as torch when using that validated stack.
5. Use `read_video_np()` when downstream code needs a stable uint8 `[N,H,W,C]` representation independent of backend.
6. Convert float output with `to_uint8()`, which clips zero-to-one data, scales by 255, rounds, and casts.
7. Encode SDR comparison artifacts with one frozen codec and settings. Keep x264 threads bounded, then reject missing or zero-byte outputs.
8. Use a vendored full FFmpeg and x265 10-bit settings for HDR10 output. Do not use Vista's system FFmpeg.
9. Record decoder, encoder, version, pixel format, frame rate, quality settings, and HDR handling in manifests.
10. Re-run a small metric A/B before changing the encoder in an established study.

## Rules (hard)

- Use the vendored full FFmpeg on Vista, because the system build lacks required codecs and dependencies. [memory-facts]
- Bound encoder threads, because automatic x264 or x265 threading can exhaust the login process ceiling and write zero bytes. [memory-facts]
- Inspect HDR before torchcodec 0.11 decode, because it silently flattens HDR and 10-bit input to 8-bit. [VG1 report-videogen1.md, Video I/O]
- Scale zero-to-one floats before uint8 conversion, because direct casting produces nearly black frames and false quality scores. [VG1 video_io.py:94-98]
- Freeze codec settings across arms, because encoding alone moved VBench aesthetic quality by about three points. [FL G11]
- Keep VAE dtype model-specific, because Wan needs fp32 quality while CogVideoX requires bf16 compatibility. [VG1 gen.py:14-16; VG1 NOTES-e4-agent.md:43-46]

## Pitfalls seen in production

- Symptom: torchcodec import fails despite an installed package. Cause: the PyPI wheel links CUDA 13 nvrtc or NPP is missing. Fix: install torchcodec and NPP from the matching PyTorch CUDA index and set library paths. [VG1 report-videogen1.md, Video I/O]
- Symptom: decode succeeds but HDR range disappears. Cause: torchcodec 0.11 silently returns 8-bit uint8. Fix: warn and use torchcodec 0.14 or a verified HDR path. [memory-facts]
- Symptom: an MP4 is zero bytes. Cause: codec startup failed after unbounded encoder threading crossed a process limit. Fix: cap threads and assert nonzero size. [memory-facts]
- Symptom: ImageReward or displayed frames are nearly black. Cause: direct float-to-uint8 conversion. Fix: clip, scale, round, and cast. [VG1 README.md:98]
- Symptom: VBench aesthetic moves by 0.0297 with identical source videos. Cause: encoder settings changed. Fix: keep one encoder for all comparison arms. [FL G11]
- Symptom: official VBench total differs from a raw average. Cause: the benchmark uses per-dimension normalization and `(4*Quality+Semantic)/5`. Fix: use the official aggregation. [FL G12]

## Pointers

- Related skills: `env-setup-aarch64.md`, `gpu-memory-hierarchy.md`, `provenance-and-repro.md`.
- Utility: `infra/python/video_io.py`.
- Compendium themes: 7, 9, 10, and 11.
