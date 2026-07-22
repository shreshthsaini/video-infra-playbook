"""Portable video I/O helpers for video-generation experiments.

Usage: video_io.py [-h] [--probe VIDEO]
Environment: VIDEO_IO_FFMPEG_THREADS controls bounded encoder threads.

Reads use TorchCodec when available and fall back to imageio. TorchCodec 0.11
silently flattens HDR input to 8-bit, so read_video warns when metadata signals
HDR content. Writes use imageio with bounded x264 threads. A complete 10-bit
HDR encoder is intentionally not implemented here.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

try:
    from torchcodec.decoders import VideoDecoder
    _HAVE_TORCHCODEC = True
except Exception:
    _HAVE_TORCHCODEC = False


def have_torchcodec() -> bool:
    return _HAVE_TORCHCODEC


def read_video(path: str, start: int = 0, end: int | None = None):
    """Returns frames as torch.uint8 tensor [N, C, H, W] (torchcodec) or
    np.uint8 [N, H, W, C] (fallback). Use read_video_np for a uniform numpy view.
    WARNING: HDR sources are silently flattened to 8-bit by torchcodec 0.11, we
    warn (not raise) since SDR-ified frames are often still fine for metrics."""
    if _HAVE_TORCHCODEC:
        d = VideoDecoder(path)
        info = hdr_info(path)
        if info.get("looks_hdr"):
            import warnings
            warnings.warn(f"{path}: HDR content ({info.get('color_transfer')}), "
                          "torchcodec 0.11 silently flattens to 8-bit SDR!")
        n = d.metadata.num_frames
        return d[start:(end if end is not None else n)]
    return _read_imageio(path, start, end)


def read_video_np(path: str, start: int = 0, end: int | None = None) -> np.ndarray:
    """Frames as np.uint8 [N, H, W, C] regardless of backend."""
    v = read_video(path, start, end)
    if isinstance(v, np.ndarray):
        return v
    return v.permute(0, 2, 3, 1).contiguous().numpy()


def _read_imageio(path, start=0, end=None):
    import imageio
    rdr = imageio.get_reader(path)
    frames = []
    for i, f in enumerate(rdr):
        if i < start:
            continue
        if end is not None and i >= end:
            break
        frames.append(np.asarray(f))
    rdr.close()
    return np.stack(frames)


def hdr_info(path: str) -> dict:
    """Color metadata for HDR detection. Empty dict if torchcodec unavailable."""
    if not _HAVE_TORCHCODEC:
        return {}
    md = VideoDecoder(path).metadata
    out = {}
    for k in ("color_primaries", "color_space", "color_transfer", "pixel_format",
              "bit_rate", "codec"):
        v = getattr(md, k, None)
        if v is not None:
            out[k] = v
    out["looks_hdr"] = any("2020" in str(v) or "2084" in str(v) for v in out.values())
    return out


def to_uint8(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = (a.astype(np.float32).clip(0, 1) * 255).round().astype(np.uint8)
    return a


def write_video(video, path: str, fps: int = 16):
    """8-bit SDR mp4 (x264). For 10-bit HDR use write_video_hdr10.
    Encoder threads are bounded to protect shared-node process limits."""
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    threads = os.environ.get("VIDEO_IO_FFMPEG_THREADS", "4")
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                           ffmpeg_params=["-threads", str(threads)])
    for f in video:
        w.append_data(to_uint8(f))
    w.close()
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"ffmpeg wrote 0 bytes to {path}, encoder failed to open "
                           "(check thread/process limits; see docstring)")


def write_video_hdr10(frames_16bit, path: str, fps: int = 16):
    """Reserved for a complete 10-bit HDR10 x265 implementation."""
    raise NotImplementedError(
        "10-bit path: imageio_ffmpeg.get_ffmpeg_exe() + '-pix_fmt yuv420p10le "
        "-c:v libx265 -x265-params hdr10=1:colorprim=bt2020:transfer=smpte2084...'" )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", metavar="VIDEO", help="print HDR metadata as JSON")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(hdr_info(args.probe), indent=2, sort_keys=True, default=str))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
