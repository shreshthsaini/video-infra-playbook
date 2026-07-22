# Verified citation list for the TACC GPU-infrastructure blog
Verified 2026-07-22. Method: WebFetch/WebSearch plus `gh api` for GitHub repos and direct curl (from a TACC node) for tacc.utexas.edu pages that 403-block generic fetchers.

## A. GPU-architecture / performance explainers

1. **Inside NVIDIA GPUs: Anatomy of high performance matmul kernels** - Aleksa Gordić - https://www.aleksagordic.com/blog/matmul - Deep dive from GPU fundamentals through PTX/SASS to state-of-the-art Hopper (H100) matmul kernels (TMA, wgmma, swizzling, pipelines); this IS the "Inside NVIDIA GPUs" post, there is no separate companion. Gordić also runs the YouTube channel "The AI Epiphany". - VERIFIED
2. **Making Deep Learning Go Brrrr From First Principles** - Horace He - https://horace.io/brrr_intro.html - Frames DL performance as three regimes (compute-bound, memory-bandwidth-bound, overhead-bound) and how to tell which one you are in. - VERIFIED
3. **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog** - Simon Boehm - https://siboehm.com/articles/22/CUDA-MMM - Iterative worklog taking an SGEMM kernel from 309 to ~21,779 GFLOPS via coalescing, shared-memory tiling, warp tiling, vectorization. - VERIFIED
4. **GPU Glossary** - Modal - https://modal.com/gpu-glossary - Cross-linked reference on GPU device hardware, CUDA software stack, and host-side concepts. - VERIFIED
5. **The Ultra-Scale Playbook: Training LLMs on GPU Clusters** - Hugging Face (nanotron team) - https://huggingface.co/spaces/nanotron/ultrascale-playbook - Book-length guide to 5D parallelism (DP/TP/PP/CP/EP + ZeRO), memory anatomy, kernels, and compute/communication overlap, backed by 4000+ scaling experiments; discusses throughput/MFU-style efficiency metrics. - VERIFIED
6. Two canonical Hugging Face picks for GPU utilization/inference:
   - **GPU inference** (Transformers documentation) - Hugging Face - https://huggingface.co/docs/transformers/perf_infer_gpu_one - Official doc page on single-GPU inference optimization (FlashAttention-2, quantization, torch.compile). URL resolves; note the page lives in the "main" docs version and its on-page heading has been shortened to "GPU" in recent releases. - VERIFIED (with naming caveat)
   - **Visualize and understand GPU memory in PyTorch** - Quentin Gallouédec (Hugging Face blog) - https://huggingface.co/blog/train_memory - Profiling and estimating GPU memory during training (parameters, gradients, optimizer state, activations). - VERIFIED
7. **PaLM: Scaling Language Modeling with Pathways** - Aakanksha Chowdhery et al. (Google), 2022 - https://arxiv.org/abs/2204.02311 - Introduces Model FLOPs Utilization (MFU): observed tokens/sec over the theoretical peak-FLOPs throughput, counting only forward+backward FLOPs (no rematerialization); the formula is given in Appendix B. - VERIFIED
   - Canonical practical usage: **nanoGPT** - Andrej Karpathy - https://github.com/karpathy/nanoGPT - `model.py` line 289 `estimate_mfu()` explicitly implements MFU "see PaLM paper Appendix B as ref". - VERIFIED

## B. Open-source infrastructure (RL + video generation)

8. **TRL** - Hugging Face - https://github.com/huggingface/trl - "Train transformer language models with reinforcement learning" (SFT/DPO/GRPO/PPO trainers); ~18.9k stars. - VERIFIED
9. **diffusers** - Hugging Face - https://github.com/huggingface/diffusers - State-of-the-art diffusion models for image, video, and audio in PyTorch (~34k stars). **transformers** - https://github.com/huggingface/transformers - model-definition framework for SOTA ML models (~163k stars). **accelerate** - https://github.com/huggingface/accelerate - device-agnostic PyTorch launch/training with FSDP and DeepSpeed support (~9.8k stars). - all VERIFIED
10. **verl** - originally volcengine (ByteDance), now the verl-project org - https://github.com/volcengine/verl (redirects to https://github.com/verl-project/verl, the canonical URL) - "verl/HybridFlow: A Flexible and Efficient RL Post-Training Framework"; ~22.6k stars. Cite the verl-project URL. - VERIFIED
11. **OpenRLHF** - OpenRLHF - https://github.com/OpenRLHF/OpenRLHF - Easy-to-use, scalable, high-performance agentic RL framework on Ray + vLLM (PPO, DAPO, REINFORCE++); ~9.8k stars. - VERIFIED
12. **prime-rl** - Prime Intellect - https://github.com/PrimeIntellect-ai/prime-rl - "Agentic RL Training at Scale"; the flagship open RL/decentralized-training framework behind INTELLECT-2. Companion report: **INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning** - Prime Intellect Team (Jaghouar, Mattern, et al.) - https://arxiv.org/abs/2505.07291 - 32B reasoning model trained via fully asynchronous RL on a permissionless compute network. - VERIFIED
13. Video-generation serving/training infra (all four verified; established by stars in this order):
    - **Open-Sora** - hpcaitech (HPC-AI Tech) - https://github.com/hpcaitech/Open-Sora - "Democratizing Efficient Video Production for All"; open end-to-end video-gen training stack; ~29.2k stars. - VERIFIED
    - **FastVideo** - hao-ai-lab (UCSD Hao AI Lab) - https://github.com/hao-ai-lab/FastVideo - Unified inference and post-training framework for accelerated video generation; ~3.9k stars. - VERIFIED
    - **xDiT** - xdit-project - https://github.com/xdit-project/xDiT - Scalable inference engine for Diffusion Transformers with massive parallelism (USP, PipeFusion); ~2.7k stars. - VERIFIED
    - **VideoSys** - NUS-HPC-AI-Lab - https://github.com/NUS-HPC-AI-Lab/VideoSys - "An easy and efficient system for video generation" (DSP, PAB acceleration); ~2.0k stars. - VERIFIED
14. **DanceGRPO** - Zeyue Xue (ByteDance Seed collaboration; hosted under the author's personal GitHub, not a ByteDance org) - https://github.com/XueZeyue/DanceGRPO - "An official implementation of DanceGRPO: Unleashing GRPO on Visual Generation"; ~1.6k stars. - VERIFIED (note the non-org hosting when crediting)
15. **Wan2.1** - Wan-Video (Alibaba / Tongyi Wanxiang) - https://github.com/Wan-Video/Wan2.1 - "Wan: Open and Advanced Large-Scale Video Generative Models"; ~16.6k stars. - VERIFIED. **HunyuanVideo** - Tencent-Hunyuan - https://github.com/Tencent/HunyuanVideo (redirects to https://github.com/Tencent-Hunyuan/HunyuanVideo, the canonical URL) - "A Systematic Framework For Large Video Generation Model"; ~12.4k stars. - VERIFIED

## C. TACC / UT Austin acknowledgment material

16. **Citing TACC** - Texas Advanced Computing Center - https://tacc.utexas.edu/about/citing-tacc/ - Official page with the suggested citation/acknowledgment for publications. Verbatim suggested citation: "The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing computational resources that have contributed to the research results reported within this paper. URL: http://www.tacc.utexas.edu". Minimum citation: "Texas Advanced Computing Center (TACC), The University of Texas at Austin". Note: the page serves HTTP 403 to non-browser fetchers; it loads normally in a browser. - VERIFIED
17. **Vista** - TACC - system page https://tacc.utexas.edu/systems/vista/ (resolves, title "Vista") and user guide **Vista - TACC HPC Documentation** https://docs.tacc.utexas.edu/hpc/vista/ - NVIDIA Grace Hopper (GH200) system: GH nodes pair an H200 GPU with a 72-core Grace CPU, plus Grace-Grace CPU nodes; AI/ML bridge from Frontera to Horizon. - VERIFIED
18. **Lonestar6** - TACC - system page https://tacc.utexas.edu/systems/lonestar6/ (resolves, title "Lonestar6") and user guide **Lonestar6 - TACC HPC Documentation** https://docs.tacc.utexas.edu/hpc/lonestar6/ - Dell/AMD Milan cluster (with A100/H100 GPU nodes) for simulation, data analysis, visualization, and ML. - VERIFIED
    - Lonestar6 citation paper (PEARC '22, "Cazes et al."): [UNVERIFIED] - no dedicated Lonestar6 system paper found in the PEARC '22 proceedings (ACM DL 10.1145/3491418), John Cazes's dblp record, or targeted searches. Recommendation: do not cite a Lonestar6 paper; use TACC's suggested acknowledgment sentence (item 16) and the system/docs URLs instead.
19. **Institute for Foundations of Machine Learning (IFML)** - NSF AI Institute led by The University of Texas at Austin (est. 2020; director Adam Klivans) - https://www.ifml.institute/ - NSF AI Institute for Foundations of Machine Learning, part of the National AI Research Institutes program; partners include UW, Stanford, Caltech, and others. - VERIFIED
20. **Standard TACC acknowledgment sentence** - from the Citing TACC page (item 16) - VERIFIED verbatim, with one correction to the draft: TACC's official wording says "providing computational resources", not "providing HPC resources". Use: "The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing computational resources that have contributed to the research results reported within this paper. URL: http://www.tacc.utexas.edu"

## Reviewer notes for the revision (Fable)
- Use canonical URLs: verl -> https://github.com/verl-project/verl ;
  HunyuanVideo -> https://github.com/Tencent-Hunyuan/HunyuanVideo .
- TACC acknowledgment must use the verbatim official sentence with
  "computational resources".
- No Lonestar6 paper exists to cite; link the system page + user guide only.
- Do NOT relabel our project hardware based on TACC marketing pages: our
  measured devices remain GH200 96GB HBM3 (Vista gh), GB200 185GiB (Vista gb),
  3x A100-40 and 2x H100 nodes (LS6). Link the TACC pages as the system
  references without changing our numbers.
