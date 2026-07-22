#!/usr/bin/env python3
"""Estimate model FLOPs utilization from archived workload records.

Usage: estimate_mfu.py [--output FILE]
Environment: CACHEDSEARCH_RESULTS selects the JSONL archive and VSAO_ROOT
selects the calibration and trainer-record root.

ESTIMATED MFU is achieved model FLOPs per second divided by peak dense BF16
FLOPs per second, with inference counted as forward-only work and training
counted with the standard 2N forward plus 4N backward approximation.  A
forward call includes 2 * active parameters * tokens plus the attention term
4 * layers * sequence_length**2 * hidden_width.  Device peaks come from
docs/sources/gpu-arch-factsheet.md.  These values are post-hoc analytic
estimates, not profiler measurements.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_GEN_RESULTS = Path(
    os.environ.get("CACHEDSEARCH_RESULTS", "data/cachedsearch-results")  # per-record JSONL archive (private source repo)
)
VSAO_ROOT = Path(os.environ.get("VSAO_ROOT", "data/vsao"))  # calibration docs root (private source repo)

PEAK_FLOPS = {
    "A100": 312.0e12,
    "H100": 989.4e12,
    "GH200": 990.0e12,
    "GB200-class": 2.5e15,  # GB200 NVL72-class dense BF16 per GPU, computed from NVIDIA NVL72 totals (see docs/sources/gpu-arch-factsheet.md)
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    active_params: float
    layers: int
    hidden_width: int
    sequence_length: int
    call_batch: int = 1

    @property
    def tokens_per_call(self) -> int:
        return self.sequence_length * self.call_batch

    @property
    def flops_per_forward(self) -> float:
        dense = 2.0 * self.active_params * self.sequence_length * self.call_batch
        attention = (
            4.0
            * self.layers
            * self.sequence_length**2
            * self.hidden_width
            * self.call_batch
        )
        return dense + attention


@dataclass(frozen=True)
class Workload:
    project: str
    row_label: str
    mode: str
    hardware: str
    model: ModelSpec
    mean_seconds: float
    mean_forward_calls: float
    samples: int
    devices: int = 1

    @property
    def work_per_second(self) -> float:
        return self.model.flops_per_forward * self.mean_forward_calls / self.mean_seconds

    @property
    def peak(self) -> float:
        return PEAK_FLOPS[self.hardware] * self.devices

    @property
    def mfu(self) -> float:
        return self.work_per_second / self.peak


CACHEDSEARCH_SPECS = {
    "Wan2.1-1.3B": ModelSpec("Wan2.1-1.3B", 1.3e9, 30, 1536, 32760),
    "CogVideoX-5B": ModelSpec("CogVideoX-5B", 5.0e9, 42, 3072, 17550, 2),
    "LTX-Video-2B": ModelSpec("LTX-Video-2B", 2.0e9, 28, 2048, 6930, 2),
    "HunyuanVideo-13B": ModelSpec("HunyuanVideo-13B", 13.0e9, 60, 3072, 21600),
    "Wan2.2-5B": ModelSpec("Wan2.2-5B", 5.0e9, 30, 3072, 27280),
    "Wan2.1-14B": ModelSpec("Wan2.1-14B", 14.0e9, 40, 5120, 32760),
}

CACHEDSEARCH_DIRS = {
    "Wan2.1-1.3B": "b1_gate_v0",
    "CogVideoX-5B": "b1_gate_cog5b_v0",
    "LTX-Video-2B": "b1_gate_ltx_v0",
    "HunyuanVideo-13B": "b1_gate_hunyuan_v0",
    "Wan2.2-5B": "b1_gate_wan22_5b",
    "Wan2.1-14B": "b1_gate_wan14b",
}

CACHEDSEARCH_TOKEN_ACCOUNTING = (
    "Wan2.1-1.3B: 32,760 tokens/call = 21 x 30 x 52; 50 steps x 2 CFG calls.",
    "CogVideoX-5B: 35,100 tokens/call = 2 x (13 x 30 x 45); 50 steps x 1 batch-concatenated CFG call.",
    "LTX-Video-2B: 13,860 tokens/call = 2 x (21 x 15 x 22); 50 steps x 1 batch-concatenated CFG call.",
    "HunyuanVideo-13B: 21,600 tokens/call = 16 x 30 x 45; 50 steps x 1 guidance-distilled call.",
    "Wan2.2-5B: 27,280 tokens/call = 31 x 22 x 40; 50 steps x 2 CFG calls.",
    "Wan2.1-14B: 32,760 tokens/call = 21 x 30 x 52; 50 steps x 2 CFG calls.",
)

VSAO_TOKEN_ACCOUNTING = (
    "LTX 41f: 2,112 tokens/call = 6 x 16 x 22; 50 steps x 2 CFG calls.",
    "Wan2.1 33f: 14,040 tokens/call = 9 x 30 x 52; 50 steps x 2 CFG calls.",
    "CogVideoX 49f: 17,550 tokens/call = 13 x 30 x 45; 50 steps x 2 CFG calls.",
    "Hunyuan 33f: 14,040 tokens/call = 9 x 30 x 52; 50 steps x 1 guidance-distilled call.",
    "Wan2.1 81f: 32,760 tokens/call = 21 x 30 x 52; 50 steps x 2 CFG calls.",
)


def read_jsonl_records(directory: Path) -> list[dict]:
    """Read sorted JSONL files and keep the last stable-key record."""
    deduplicated: dict[tuple[str, int, str], dict] = {}
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    continue
                raise
            key = (record["prompt"], int(record["seed"]), record["variant"])
            deduplicated[key] = record
    return list(deduplicated.values())


def executed_calls(record: dict) -> int:
    branches = record.get("stats", {})
    if not branches:
        raise ValueError("CachedSearch record is missing per-branch execution stats")
    return sum(int(branch["computes"]) for branch in branches.values())


def cachedsearch_workloads() -> list[Workload]:
    workloads: list[Workload] = []
    for model_name, directory_name in CACHEDSEARCH_DIRS.items():
        records = read_jsonl_records(VIDEO_GEN_RESULTS / directory_name)
        for variant in ("full", "cached"):
            selected = [record for record in records if record["variant"] == variant]
            if not selected:
                raise ValueError(f"No {variant} records found for {model_name}")
            workloads.append(
                Workload(
                    project="CachedSearch",
                    row_label=model_name,
                    mode=variant,
                    hardware="GH200",
                    model=CACHEDSEARCH_SPECS[model_name],
                    mean_seconds=statistics.fmean(float(r["latency"]) for r in selected),
                    mean_forward_calls=statistics.fmean(executed_calls(r) for r in selected),
                    samples=len(selected),
                )
            )
    return workloads


def parse_markdown_table(path: Path, heading: str) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    rows = []
    lines = text[start:].splitlines()
    header: list[str] | None = None
    for line in lines[1:]:
        if not line.startswith("|"):
            if header is not None:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(dict(zip(header, cells)))
    if not rows:
        raise ValueError(f"No rows found below {heading!r} in {path}")
    return rows


def vsao_rollout_workloads() -> list[Workload]:
    frozen = parse_markdown_table(
        VSAO_ROOT / "docs/CALIBRATION.md", "## Frozen calibration table"
    )
    lookup = {(row["family"], row["hw"]): row for row in frozen}
    gb200 = parse_markdown_table(
        VSAO_ROOT / "vista/CALIBRATION.md",
        "# B200/GB200 calibration table",
    )[0]

    specs = [
        (
            "LTX 41f",
            "ltx",
            "h100",
            "H100",
            ModelSpec("LTX-Video-2B", 2.0e9, 28, 2048, 2112),
            100,
        ),
        (
            "LTX 41f",
            "ltx",
            "a100",
            "A100",
            ModelSpec("LTX-Video-2B", 2.0e9, 28, 2048, 2112),
            100,
        ),
        (
            "Wan2.1 33f",
            "wan21",
            "a100",
            "A100",
            ModelSpec("Wan2.1-1.3B", 1.3e9, 30, 1536, 14040),
            100,
        ),
        (
            "CogVideoX 49f",
            "cogvideox",
            "a100",
            "A100",
            ModelSpec("CogVideoX-2B", 2.0e9, 30, 1920, 17550),
            100,
        ),
        (
            "Hunyuan 33f",
            "hunyuan",
            "h100",
            "H100",
            ModelSpec("HunyuanVideo-13B", 13.0e9, 60, 3072, 14040),
            50,
        ),
    ]

    workloads = []
    for label, family, hw_key, hardware, model, calls in specs:
        seconds = float(lookup[(family, hw_key)]["s/rollout (mean)"])
        workloads.append(
            Workload(
                project="VSAO",
                row_label=label,
                mode="rollout",
                hardware=hardware,
                model=model,
                mean_seconds=seconds,
                mean_forward_calls=calls,
                samples=1,
            )
        )

    wall_match = re.search(r"([0-9.]+)\s*s", gb200["Rollout wall"])
    if wall_match is None:
        raise ValueError("Could not parse the GB200 rollout wall time")
    workloads.append(
        Workload(
            project="VSAO",
            row_label="Wan2.1 81f",
            mode="rollout",
            hardware="GB200-class",
            model=ModelSpec("Wan2.1-1.3B", 1.3e9, 30, 1536, 32760),
            mean_seconds=float(wall_match.group(1)),
            mean_forward_calls=100,
            samples=1,
        )
    )
    return workloads


def vsao_trainer_workload() -> Workload:
    path = VSAO_ROOT / "results/rl/wan_confirmations_step1_20260719.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    replicas = payload["replicas"]
    records = {int(replica["records"]) for replica in replicas}
    devices = {int(replica["data_parallel_world_size"]) for replica in replicas}
    if records != {64} or devices != {3}:
        raise ValueError("Unexpected VSAO trainer shape in confirmation records")

    config_text = (VSAO_ROOT / "configs/sao_wan_480p.yaml").read_text(
        encoding="utf-8"
    )

    def config_integer(name: str) -> int:
        match = re.search(rf"^\s*{re.escape(name)}:\s*([0-9]+)\b", config_text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Could not parse {name} from the VSAO trainer config")
        return int(match.group(1))

    transitions_per_record = config_integer("train_steps_per_traj")
    critic_updates = config_integer("k_critic_steps")
    policy_forward_backward = 2 * 3
    critic_forward_backward = critic_updates * 3
    value_forward = 1
    forward_equivalents_per_transition = (
        policy_forward_backward + critic_forward_backward + value_forward
    )
    return Workload(
        project="VSAO",
        row_label="Wan trainer update",
        mode="trainer",
        hardware="A100",
        model=ModelSpec("Wan2.1-1.3B", 1.3e9, 30, 1536, 14040),
        mean_seconds=statistics.fmean(
            float(replica["trainer_wall_seconds"]) for replica in replicas
        ),
        mean_forward_calls=(
            64 * transitions_per_record * forward_equivalents_per_transition
        ),
        samples=len(replicas),
        devices=3,
    )


def all_workloads() -> list[Workload]:
    return cachedsearch_workloads() + vsao_rollout_workloads() + [vsao_trainer_workload()]


def print_table(workloads: Iterable[Workload]) -> None:
    rows = list(workloads)
    headings = (
        "PROJECT",
        "WORKLOAD",
        "MODE",
        "FLOPs/forward",
        "tokens",
        "work/s",
        "peak used",
        "ESTIMATED MFU",
    )
    rendered = []
    for item in rows:
        rendered.append(
            (
                item.project,
                item.row_label,
                item.mode,
                f"{item.model.flops_per_forward / 1e12:.3f} TF",
                f"{item.model.tokens_per_call:,}",
                f"{item.work_per_second / 1e12:.2f} TF/s",
                f"{item.devices}x {item.hardware} {item.peak / 1e12:.1f} TF/s",
                f"{100.0 * item.mfu:.2f}%",
            )
        )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headings))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    print()
    print("Method decisions:")
    print("  ESTIMATED MFU is a post-hoc analytic estimate, not a profiler measurement.")
    print("  Attention FLOPs are included in every forward estimate.")
    print("  Inference is forward-only; the trainer uses standard forward/backward work.")
    print("  Text encoder, VAE, decode, and scoring FLOPs are excluded from work.")
    print("  Their elapsed time remains in end-to-end rollout and inference denominators.")
    print("  The trainer counts policy work, two critic updates, and one value pass.")
    print("  Forcing Laws is dropped because local records do not isolate step time.")
    print("  Optional WandB history is dropped because authenticated retrieval failed.")
    print()
    print("Token and call accounting from recorded model configurations:")
    for line in CACHEDSEARCH_TOKEN_ACCOUNTING:
        print(f"  CachedSearch {line}")
    for line in VSAO_TOKEN_ACCOUNTING:
        print(f"  VSAO {line}")


def svg_text(x: float, y: float, value: str, css_class: str, **attrs: str) -> str:
    extra = "".join(f' {key.replace("_", "-")}="{html.escape(val)}"' for key, val in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}"{extra}>'
        f"{html.escape(value)}</text>"
    )


def render_svg(workloads: list[Workload], output: Path) -> None:
    cached = [item for item in workloads if item.project == "CachedSearch"]
    vsao = [item for item in workloads if item.project == "VSAO"]
    cached_order = list(CACHEDSEARCH_SPECS)
    vsao_order = [
        "LTX 41f",
        "Wan2.1 33f",
        "CogVideoX 49f",
        "Hunyuan 33f",
        "Wan2.1 81f",
        "Wan trainer update",
    ]

    width = 1120
    height = 770
    plot_left = 300
    plot_right = 875
    plot_top = 150
    row_step = 43
    max_percent = max(item.mfu * 100.0 for item in workloads)
    axis_max = max(60, int((max_percent + 9.999) // 10) * 10)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="mfu_fig_title mfu_fig_desc" '
        f'preserveAspectRatio="xMidYMid meet" focusable="false" class="svg-figure telemetry-figure">',
        '<title id="mfu_fig_title">ESTIMATED MFU from archived workload records</title>',
        '<desc id="mfu_fig_desc">A horizontal dot plot compares analytic model FLOPs utilization estimates for CachedSearch inference, VSAO rollouts, and one VSAO trainer update.</desc>',
        "<style>",
        "text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: #0f172a; }",
        ".title { font-size: 25px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #64748b; }",
        ".group { font-size: 13px; font-weight: 700; letter-spacing: 0.08em; fill: #475569; }",
        ".label { font-size: 14px; font-weight: 600; }",
        ".hardware { font-size: 11px; fill: #64748b; }",
        ".tick { font-size: 12px; fill: #64748b; }",
        ".legend { font-size: 12px; fill: #475569; }",
        ".value { font-size: 11px; font-weight: 700; }",
        ".grid { stroke: #e2e8f0; stroke-width: 1; }",
        ".baseline { stroke: #cbd5e1; stroke-width: 1.2; }",
        "</style>",
        '<rect width="1120" height="770" rx="12" fill="#ffffff"/>',
        svg_text(32, 43, "ESTIMATED MFU from archived workload records", "title"),
        svg_text(32, 68, "Post-hoc analytic estimates, not profiler measurements", "subtitle"),
    ]

    legend_y = 101
    legend_items = [
        ("#2563eb", "full / rollout"),
        ("#f59e0b", "cached"),
        ("#0f172a", "trainer"),
    ]
    legend_x = 32
    for color, label in legend_items:
        elements.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color}"/>')
        elements.append(svg_text(legend_x + 11, legend_y + 4, label, "legend"))
        legend_x += 126

    for tick in range(0, axis_max + 1, 10):
        x = plot_left + (plot_right - plot_left) * tick / axis_max
        elements.append(f'<line x1="{x:.1f}" y1="125" x2="{x:.1f}" y2="721" class="grid"/>')
        elements.append(svg_text(x, 741, f"{tick}%", "tick", text_anchor="middle"))
    elements.append(f'<line x1="{plot_left}" y1="721" x2="{plot_right}" y2="721" class="baseline"/>')

    y = plot_top
    elements.append(svg_text(32, y - 18, "CACHEDSEARCH", "group"))
    by_cached = {(item.row_label, item.mode): item for item in cached}
    for label in cached_order:
        elements.append(svg_text(32, y + 5, label, "label"))
        elements.append(svg_text(890, y + 4, "GH200 · 990 TF/s", "hardware"))
        for mode, offset, color in (("full", -7, "#2563eb"), ("cached", 7, "#f59e0b")):
            item = by_cached[(label, mode)]
            percent = item.mfu * 100.0
            x = plot_left + (plot_right - plot_left) * percent / axis_max
            elements.append(f'<circle cx="{x:.1f}" cy="{y + offset:.1f}" r="5.5" fill="{color}"/>')
            elements.append(svg_text(x + 9, y + offset + 4, f"{percent:.1f}", "value"))
        y += row_step

    y += 20
    elements.append(svg_text(32, y - 18, "VSAO", "group"))
    for label in vsao_order:
        matches = [item for item in vsao if item.row_label == label]
        if label == "LTX 41f":
            display_label = "LTX 41f (H100 / A100)"
        else:
            display_label = label
        elements.append(svg_text(32, y + 5, display_label, "label"))
        hardware_label = " / ".join(
            (
                f"{item.devices}×{item.hardware} · {item.peak / 1e12:g} TF/s"
                if item.devices > 1
                else f"{item.hardware} · {item.peak / 1e12:g} TF/s"
            )
            for item in matches
        )
        elements.append(svg_text(890, y + 4, hardware_label, "hardware"))
        offsets = [0] if len(matches) == 1 else [-7, 7]
        for item, offset in zip(matches, offsets):
            percent = item.mfu * 100.0
            x = plot_left + (plot_right - plot_left) * percent / axis_max
            color = "#0f172a" if item.mode == "trainer" else "#2563eb"
            elements.append(f'<circle cx="{x:.1f}" cy="{y + offset:.1f}" r="5.5" fill="{color}"/>')
            value_label = f"{percent:.1f}"
            if len(matches) > 1:
                value_label += f" {item.hardware}"
            elements.append(svg_text(x + 9, y + offset + 4, value_label, "value"))
        y += row_step

    elements.extend(
        [
            svg_text((plot_left + plot_right) / 2, 761, "ESTIMATED MFU", "tick", text_anchor="middle"),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "out/estimated-mfu.svg",
        help="Output SVG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = all_workloads()
    print_table(workloads)
    render_svg(workloads, args.output)
    print()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
