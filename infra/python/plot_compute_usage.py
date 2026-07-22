#!/usr/bin/env python3
"""Plot fleet telemetry and write time-averaged compute statistics.

Usage: plot_compute_usage.py [--telemetry-dir DIR] [--output-dir DIR]
       [--tag NAME] [--interval-seconds N]
Environment: TELEMETRY_DIR overrides the archived telemetry input and
MPLCONFIGDIR may select a writable Matplotlib cache directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fleetcraft-mpl-cache")
)


def default_telemetry_dir() -> Path:
    configured = os.environ.get("TELEMETRY_DIR")
    if configured:
        return Path(configured)
    return REPO_ROOT / "docs" / "sources" / "telemetry" / "forcing-laws"


def load(root: Path) -> dict[str, list[tuple[str, float, float, float, float]]]:
    series: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for path in sorted(root.glob("*.csv")):
        host = path.name.split("_job", 1)[0]
        rows: list[tuple[str, float, float, float, float]] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append(
                        (
                            row["timestamp"],
                            float(row["mem_used_mib"]),
                            float(row["mem_total_mib"]),
                            float(row["util_pct"]),
                            float(row["power_w"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        if rows:
            series.setdefault(host, []).extend(rows)
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=default_telemetry_dir())
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "compute")
    parser.add_argument("--tag", default="all")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    series = load(args.telemetry_dir)
    if not series:
        print("no telemetry found")
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stats: dict[str, dict[str, float | int]] = {}
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for host, rows in sorted(series.items()):
        memory = np.array([row[1] for row in rows]) / 1024.0
        total = rows[0][2] / 1024.0
        utilization = np.array([row[3] for row in rows])
        power = np.array([row[4] for row in rows])
        hours = len(rows) * args.interval_seconds / 3600.0
        stats[host] = {
            "samples": len(rows),
            "gpu_hours_observed": round(hours, 3),
            "util_mean_percent": round(float(utilization.mean()), 2),
            "memory_mean_gib": round(float(memory.mean()), 2),
            "memory_peak_gib": round(float(memory.max()), 2),
            "memory_total_gib": round(float(total), 2),
            "energy_kwh_estimate": round(float(power.mean()) * hours / 1000.0, 3),
        }
        x_values = np.arange(len(rows)) * args.interval_seconds / 3600.0
        axes[0].plot(x_values, utilization, linewidth=0.6, alpha=0.8, label=host)
        axes[1].plot(x_values, memory, linewidth=0.6, alpha=0.8)

    all_util = np.concatenate([[row[3] for row in rows] for rows in series.values()])
    all_memory = np.concatenate([[row[1] / 1024.0 for row in rows] for rows in series.values()])
    stats["_aggregate"] = {
        "util_mean_percent": round(float(all_util.mean()), 2),
        "memory_mean_gib": round(float(all_memory.mean()), 2),
        "hosts": len(series),
        "samples": int(len(all_util)),
    }
    axes[0].set_ylabel("GPU util (%)")
    axes[0].legend(fontsize=5, ncol=4)
    axes[1].set_ylabel("VRAM (GiB)")
    axes[1].set_xlabel("hours observed")
    figure.suptitle(
        f"Fleet compute usage ({args.tag}); mean util "
        f"{stats['_aggregate']['util_mean_percent']}%"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"compute_usage_{args.tag}.png"
    json_path = args.output_dir / f"compute_usage_{args.tag}.json"
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    json_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats["_aggregate"], sort_keys=True))
    print(f"WROTE {figure_path}")
    print(f"WROTE {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
