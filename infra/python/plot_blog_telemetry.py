#!/usr/bin/env python3
"""Build privacy-safe figures from Fleetcraft's anonymized telemetry.

Usage: plot_blog_telemetry.py [--output-dir DIR]
Environment: MPLCONFIGDIR may select a writable Matplotlib cache directory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

cache_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "fleetcraft-mpl-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_ROOT = ROOT / "docs" / "sources" / "telemetry"
FORCING_ROOT = TELEMETRY_ROOT / "forcing-laws"
VSAO_TRACKER = TELEMETRY_ROOT / "util_tracker.log"
DEFAULT_OUTPUT = ROOT / "out" / "telemetry"

ACCENT = "#2563eb"
RAW_TRACE = "#94a3b8"
AMBER = "#f59e0b"
INK = "#0f172a"
SLATE = "#334155"
MUTED = "#64748b"
GRID = "#e2e8f0"
PAPER = "#ffffff"

FORBIDDEN_SVG_PATTERNS = {
    "allocation balance": re.compile(r"\b" + "bal" + r"=", re.IGNORECASE),
    "compute hostname": re.compile(r"\b[ci][0-9]{3}-[0-9]{3}\b", re.IGNORECASE),
    "scheduler job id": re.compile(r"\bjob[_ -]?[0-9]{5,}\b", re.IGNORECASE),
    "account id": re.compile(r"\bASC[0-9]+\b", re.IGNORECASE),
    "account username": re.compile(
        r"(?:user(?:name)?=|/users?/)[A-Za-z0-9_.-]+", re.IGNORECASE
    ),
    "absolute path": re.compile(r"/(?:work|scratch|home1)/"),
}

POINT_RE = re.compile(
    r"^\[(?P<timestamp>[0-9-]+ [0-9:]+)\] "
    r"util=(?P<util>[0-9]+)% "
    r"\((?P<working>[0-9]+)/(?P<total>[0-9]+) GPUs\) "
    r"free_idle=(?P<free_idle>[0-9]+) pending=(?P<pending>[0-9]+)$"
)
SUMMARY_RE = re.compile(
    r"^\[(?P<timestamp>[0-9-]+ [0-9:]+)\] rolling SUMMARY "
    r"window=(?P<window>[0-9]+)s hosts=[0-9]+ samples=(?P<samples>[0-9]+) "
    r"active_fraction=(?P<active>[0-9.]+) mean_util=(?P<mean_util>[0-9.]+) "
    r"mean_mem_fraction=(?P<mean_memory>[0-9.]+)$"
)


def configure_matplotlib() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": ["Inter", "DejaVu Sans", "sans-serif"],
            "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": 600,
            "axes.labelsize": 10,
            "axes.labelcolor": SLATE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.fontsize": 8.5,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",
            "svg.hashsalt": "fleetcraft-telemetry-v6",
            "path.simplify": True,
            "path.simplify_threshold": 0.15,
        }
    )


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.7)


def _aggregate_forcing_samples(
    samples: dict[tuple[str, datetime], list[float]],
    *,
    run_start: datetime,
    bin_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins_per_day = 24 * 60 // bin_minutes
    host_bins: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (anonymous_host, timestamp), duplicate_values in samples.items():
        bin_index = int((timestamp - run_start).total_seconds() // (bin_minutes * 60))
        host_bins[bin_index][anonymous_host].append(float(np.mean(duplicate_values)))

    last_bin = max(host_bins)
    days = np.arange(last_bin + 1, dtype=float) / bins_per_day
    mean_util = np.full(last_bin + 1, np.nan, dtype=float)
    hosts_reporting = np.zeros(last_bin + 1, dtype=float)
    for bin_index, private_streams in host_bins.items():
        per_host = [float(np.mean(values)) for values in private_streams.values()]
        mean_util[bin_index] = float(np.mean(per_host))
        hosts_reporting[bin_index] = len(per_host)
    return days, mean_util, hosts_reporting


def read_forcing_laws() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    samples: dict[tuple[str, datetime], list[float]] = defaultdict(list)
    for source in sorted(FORCING_ROOT.glob("*.csv")):
        anonymous_host = source.stem
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = {
                "timestamp",
                "mem_used_mib",
                "mem_total_mib",
                "util_pct",
                "power_w",
            }
            if set(reader.fieldnames or ()) != expected:
                raise ValueError("Unexpected Forcing Laws telemetry schema")
            for row in reader:
                if not row["timestamp"] or not row["util_pct"]:
                    continue
                timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                samples[(anonymous_host, timestamp)].append(float(row["util_pct"]))

    if not samples:
        raise ValueError("No Forcing Laws telemetry samples found")

    run_start = min(timestamp for _, timestamp in samples)
    plotted = _aggregate_forcing_samples(
        samples,
        run_start=run_start,
        bin_minutes=10,
    )
    evidence = _aggregate_forcing_samples(
        samples,
        run_start=run_start,
        bin_minutes=15,
    )
    return (*plotted, *evidence)


def _point_in_window(
    days: np.ndarray,
    values: np.ndarray,
    start_day: float,
    end_day: float,
    mode: str,
) -> tuple[float, float]:
    mask = (days >= start_day) & (days <= end_day) & np.isfinite(values)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("Annotation window contains no telemetry")
    if mode == "first":
        index = candidates[0]
    elif mode == "last":
        index = candidates[-1]
    elif mode == "min":
        index = candidates[int(np.nanargmin(values[candidates]))]
    elif mode == "max":
        index = candidates[int(np.nanargmax(values[candidates]))]
    else:
        raise ValueError(f"Unknown annotation mode: {mode}")
    return float(days[index]), float(values[index])


def _break_long_gaps(
    x_values: np.ndarray, y_values: np.ndarray, maximum_gap_days: float
) -> tuple[np.ndarray, np.ndarray]:
    plotted_x: list[float] = []
    plotted_y: list[float] = []
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
        if index and x_value - x_values[index - 1] > maximum_gap_days:
            plotted_x.append(float(x_value))
            plotted_y.append(float("nan"))
        plotted_x.append(float(x_value))
        plotted_y.append(float(y_value))
    return np.asarray(plotted_x), np.asarray(plotted_y)


def _centered_rolling_mean(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    window_hours: float,
    maximum_gap_days: float | None = None,
) -> np.ndarray:
    """Average measured points in a centered time window without crossing gaps."""
    rolled = np.full(y_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(y_values))
    if finite_indices.size == 0:
        return rolled

    half_window_days = window_hours / 48.0
    segment_start = 0
    segments: list[np.ndarray] = []
    for offset in range(1, finite_indices.size):
        previous = finite_indices[offset - 1]
        current = finite_indices[offset]
        missing_bin = current != previous + 1
        long_gap = (
            maximum_gap_days is not None
            and x_values[current] - x_values[previous] > maximum_gap_days
        )
        if missing_bin or long_gap:
            segments.append(finite_indices[segment_start:offset])
            segment_start = offset
    segments.append(finite_indices[segment_start:])

    for segment in segments:
        segment_x = x_values[segment]
        segment_y = y_values[segment]
        for local_index, source_index in enumerate(segment):
            centered = np.abs(segment_x - segment_x[local_index]) <= half_window_days
            rolled[source_index] = float(np.mean(segment_y[centered]))
    return rolled


def plot_forcing_laws(output: Path) -> dict[str, float]:
    (
        days,
        mean_util,
        hosts_reporting,
        evidence_days,
        evidence_mean_util,
        evidence_hosts_reporting,
    ) = read_forcing_laws()
    smooth_util = _centered_rolling_mean(days, mean_util, window_hours=2.0)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.10},
    )
    top, bottom = axes

    top.plot(
        days,
        mean_util,
        color=RAW_TRACE,
        alpha=0.35,
        linewidth=0.8,
        solid_capstyle="round",
        zorder=1,
    )
    top.plot(
        days,
        smooth_util,
        color=ACCENT,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=2,
    )
    top.set_title("Forcing Laws on Vista GH200: fleet activity over time", loc="left", color=INK)
    top.set_ylabel("Mean GPU util (%)")
    top.set_ylim(0, 104)
    top.yaxis.set_major_locator(MultipleLocator(25))
    style_axis(top)

    phases = [
        ("Calibration start", 0.0, 0.18, "min", (16, 20)),
        ("Batched migration", 0.25, 0.48, "max", (18, -27)),
        ("Training waves", 8.30, 8.75, "max", (-18, -29)),
        ("Quiet tail", 10.0, float(days[-1]), "last", (-55, 22)),
    ]
    for label, start_day, end_day, mode, offset in phases:
        x_value, y_value = _point_in_window(
            evidence_days,
            evidence_mean_util,
            start_day,
            end_day,
            mode,
        )
        top.annotate(
            label,
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            color=SLATE,
            fontsize=8.5,
            fontweight=600,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
            bbox={"boxstyle": "round,pad=0.25", "facecolor": PAPER, "edgecolor": GRID},
        )

    bottom.step(days, hosts_reporting, where="post", color=AMBER, linewidth=1.35)
    bottom.set_ylabel("Hosts reporting")
    bottom.set_xlabel("Days since project start")
    bottom.set_ylim(0, float(np.nanmax(hosts_reporting)) + 3)
    bottom.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    bottom.xaxis.set_major_locator(MultipleLocator(2))
    style_axis(bottom)
    figure.align_ylabels(axes)
    figure.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.12)
    save_private_svg(
        figure,
        output,
        title="Forcing Laws fleet utilization over relative project time",
        description=(
            "Two aligned panels show ten-minute fleet mean GPU utilization, its "
            "two-hour centered rolling mean, and the number of Vista GH200 hosts reporting."
        ),
    )

    early = evidence_mean_util[(evidence_days >= 0.0) & (evidence_days < 0.25)]
    migrated = evidence_mean_util[(evidence_days >= 0.25) & (evidence_days < 0.50)]
    training = evidence_mean_util[(evidence_days >= 8.50) & (evidence_days < 8.75)]
    tail = evidence_mean_util[
        (evidence_days >= 10.25) & np.isfinite(evidence_mean_util)
    ]
    return {
        "span_days": float(evidence_days[-1]),
        "calibration_mean": float(np.nanmean(early)),
        "migration_mean": float(np.nanmean(migrated)),
        "training_mean": float(np.nanmean(training)),
        "tail_mean": float(np.nanmean(tail)),
        "tail_hosts": float(
            evidence_hosts_reporting[np.flatnonzero(np.isfinite(evidence_mean_util))[-1]]
        ),
    }


def read_vsao_tracker() -> tuple[list[dict[str, float | datetime]], list[dict[str, float | datetime]]]:
    points: list[dict[str, float | datetime]] = []
    summaries: list[dict[str, float | datetime]] = []
    with VSAO_TRACKER.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "rolling SUMMARY" in line:
                match = SUMMARY_RE.fullmatch(line)
                if match is None:
                    continue
                summaries.append(
                    {
                        "timestamp": datetime.strptime(
                            match.group("timestamp"), "%Y-%m-%d %H:%M:%S"
                        ),
                        "mean_util": float(match.group("mean_util")),
                        "active_fraction": float(match.group("active")),
                    }
                )
                continue

            match = POINT_RE.fullmatch(line)
            if match is None:
                continue
            points.append(
                {
                    "timestamp": datetime.strptime(
                        match.group("timestamp"), "%Y-%m-%d %H:%M:%S"
                    ),
                    "util": float(match.group("util")),
                    "working": float(match.group("working")),
                    "total": float(match.group("total")),
                }
            )

    if not points or not summaries:
        raise ValueError("VSAO tracker did not contain both point and rolling records")
    return points, summaries


def plot_vsao(output: Path) -> dict[str, float]:
    points, summaries = read_vsao_tracker()
    tracker_start = min(item["timestamp"] for item in points)

    point_days = np.array(
        [
            (item["timestamp"] - tracker_start).total_seconds() / 86400
            for item in points
        ],
        dtype=float,
    )
    point_util = np.array([item["util"] for item in points], dtype=float)
    smooth_util = _centered_rolling_mean(
        point_days,
        point_util,
        window_hours=3.0,
        maximum_gap_days=0.05,
    )
    allocated = np.array([item["total"] for item in points], dtype=float)
    summary_days = np.array(
        [
            (item["timestamp"] - tracker_start).total_seconds() / 86400
            for item in summaries
        ],
        dtype=float,
    )
    summary_util = np.array([item["mean_util"] for item in summaries], dtype=float)
    summary_active = np.array([item["active_fraction"] for item in summaries], dtype=float)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.1),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.10},
    )
    top, bottom = axes

    plotted_days, plotted_util = _break_long_gaps(point_days, point_util, 0.05)
    smooth_days, plotted_smooth = _break_long_gaps(point_days, smooth_util, 0.05)
    allocated_days, plotted_allocated = _break_long_gaps(point_days, allocated, 0.05)

    top.plot(
        plotted_days,
        plotted_util,
        color=RAW_TRACE,
        alpha=0.35,
        linewidth=0.8,
        label="Raw working GPUs",
        solid_capstyle="round",
        zorder=1,
    )
    top.plot(
        smooth_days,
        plotted_smooth,
        color=ACCENT,
        linewidth=2.0,
        label="Three-hour rolling mean",
        solid_capstyle="round",
        zorder=2,
    )
    top.scatter(
        summary_days,
        summary_util,
        s=9,
        color=AMBER,
        alpha=0.82,
        linewidths=0,
        label="Five-minute mean GPU util",
        zorder=3,
    )
    top.set_title("VSAO on LS6: controller trace from a 47-GPU project", loc="left", color=INK)
    top.set_ylabel("Working GPUs (%)")
    top.set_ylim(0, 104)
    top.yaxis.set_major_locator(MultipleLocator(25))
    style_axis(top)
    top.legend(loc="upper left", frameon=False, ncol=3, handlelength=2.0, columnspacing=1.0)

    audit_candidates = np.flatnonzero((summary_days >= 3.35) & (summary_days <= 3.46))
    audit_index = audit_candidates[
        int(np.argmin(np.abs(summary_util[audit_candidates] - 23.45)))
    ]
    recovery_candidates = np.flatnonzero((summary_days > summary_days[audit_index]) & (summary_days < 3.56))
    recovery_index = recovery_candidates[
        int(np.argmin(np.abs(summary_active[recovery_candidates] - 0.815)))
    ]

    top.annotate(
        "Audit trough\n23.6% active samples",
        xy=(summary_days[audit_index], summary_util[audit_index]),
        xytext=(-66, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=SLATE,
        fontsize=8.5,
        fontweight=600,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PAPER, "edgecolor": GRID},
    )
    top.annotate(
        "Recovered evidence wave\n81.5% active samples",
        xy=(summary_days[recovery_index], summary_util[recovery_index]),
        xytext=(28, 27),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=SLATE,
        fontsize=8.5,
        fontweight=600,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PAPER, "edgecolor": GRID},
    )

    bottom.step(allocated_days, plotted_allocated, where="post", color=SLATE, linewidth=1.35)
    bottom.set_ylabel("Allocated GPUs")
    bottom.set_xlabel("Days since tracker start")
    bottom.set_ylim(0, max(50.0, float(np.max(allocated)) + 3))
    bottom.yaxis.set_major_locator(MultipleLocator(10))
    bottom.xaxis.set_major_locator(MultipleLocator(0.5))
    style_axis(bottom)
    figure.align_ylabels(axes)
    figure.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.12)
    save_private_svg(
        figure,
        output,
        title="VSAO working GPU percentage over relative tracker time",
        description=(
            "Two aligned panels show raw point-in-time working GPU percentage, its "
            "three-hour centered rolling mean, five-minute summary dots, and the "
            "controller denominator for the LS6 fleet."
        ),
    )

    return {
        "span_days": float(point_days[-1]),
        "audit_day": float(summary_days[audit_index]),
        "audit_mean_util": float(summary_util[audit_index]),
        "recovery_day": float(summary_days[recovery_index]),
        "recovery_mean_util": float(summary_util[recovery_index]),
        "recovery_active_fraction": float(summary_active[recovery_index]),
        "max_allocated_in_trace": float(np.max(allocated)),
    }


def save_private_svg(
    figure: plt.Figure, output: Path, *, title: str, description: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Fleetcraft"},
    )
    plt.close(figure)

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(output)
    svg_root = tree.getroot()
    svg_namespace = "http://www.w3.org/2000/svg"
    xlink_namespace = "http://www.w3.org/1999/xlink"
    for child in tuple(svg_root):
        if child.tag.rsplit("}", 1)[-1] == "metadata":
            svg_root.remove(child)
    for element in svg_root.iter():
        xlink_href = f"{{{xlink_namespace}}}href"
        if xlink_href in element.attrib:
            element.set("href", element.attrib.pop(xlink_href))
    identifier_prefix = output.stem.replace("-", "_")
    identifier_map: dict[str, str] = {}
    for element in svg_root.iter():
        old_identifier = element.get("id")
        if old_identifier:
            new_identifier = f"{identifier_prefix}_{old_identifier}"
            identifier_map[old_identifier] = new_identifier
            element.set("id", new_identifier)
    for element in svg_root.iter():
        for attribute, value in tuple(element.attrib.items()):
            for old_identifier, new_identifier in identifier_map.items():
                value = value.replace(f"url(#{old_identifier})", f"url(#{new_identifier})")
                if value == f"#{old_identifier}":
                    value = f"#{new_identifier}"
            element.set(attribute, value)

    title_id = f"{identifier_prefix}_title"
    description_id = f"{identifier_prefix}_description"
    title_element = ET.Element(f"{{{svg_namespace}}}title", {"id": title_id})
    title_element.text = title
    description_element = ET.Element(f"{{{svg_namespace}}}desc", {"id": description_id})
    description_element.text = description
    svg_root.insert(0, description_element)
    svg_root.insert(0, title_element)
    svg_root.set("class", "svg-figure telemetry-figure")
    svg_root.set("role", "img")
    svg_root.set("aria-labelledby", f"{title_id} {description_id}")
    svg_root.set("preserveAspectRatio", "xMidYMid meet")
    svg_root.set("focusable", "false")
    svg_root.attrib.pop("width", None)
    svg_root.attrib.pop("height", None)
    for element in svg_root.iter():
        if element.tag.rsplit("}", 1)[-1] == "text":
            element.set("font-family", "Inter")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    svg_text = output.read_text(encoding="utf-8")
    if "\u2014" in svg_text or "\u00a7" in svg_text:
        raise ValueError("Forbidden typography reached an SVG")
    for private_kind, pattern in FORBIDDEN_SVG_PATTERNS.items():
        if pattern.search(svg_text):
            raise ValueError(f"Private {private_kind} reached an SVG")
    if output.stat().st_size > 250_000:
        raise ValueError(f"SVG is too large: {output.name} has {output.stat().st_size} bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for deterministic telemetry SVG files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    forcing = plot_forcing_laws(args.output_dir / "forcing-laws-fleet-utilization.svg")
    vsao = plot_vsao(args.output_dir / "vsao-fleet-utilization.svg")
    print("Wrote forcing-laws-fleet-utilization.svg")
    print(
        "Forcing Laws evidence: "
        f"span={forcing['span_days']:.2f} days, "
        f"calibration={forcing['calibration_mean']:.1f}%, "
        f"batched={forcing['migration_mean']:.1f}%, "
        f"training={forcing['training_mean']:.1f}%, "
        f"tail={forcing['tail_mean']:.1f}% on {forcing['tail_hosts']:.0f} reporting hosts"
    )
    print("Wrote vsao-fleet-utilization.svg")
    print(
        "VSAO evidence: "
        f"span={vsao['span_days']:.2f} days, "
        f"audit_day={vsao['audit_day']:.2f}, "
        f"audit_mean_util={vsao['audit_mean_util']:.1f}%, "
        f"recovery_day={vsao['recovery_day']:.2f}, "
        f"recovery_mean_util={vsao['recovery_mean_util']:.1f}%, "
        f"recovery_active={100 * vsao['recovery_active_fraction']:.1f}%, "
        f"trace_peak={vsao['max_allocated_in_trace']:.0f} GPUs"
    )
    print("Privacy filter: allocation balances and private identities were not written")


if __name__ == "__main__":
    main()
