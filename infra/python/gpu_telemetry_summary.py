#!/usr/bin/env python3
"""Summarize a recent GPU telemetry window and optionally freeze JSON evidence.

Usage: gpu_telemetry_summary.py [--telemetry-dir DIR] [--window-seconds N]
       [--active-threshold PCT] [--project NAME] [--summary-only]
       [--output FILE]
Environment: PROJECT (default: my-project) and TELEMETRY_DIR configure the
default input location.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from statistics import mean


def default_telemetry_dir() -> Path:
    user = os.environ.get("USER", "unknown")
    project = os.environ.get("PROJECT", "my-project")
    configured = os.environ.get("TELEMETRY_DIR")
    return Path(configured) if configured else Path("/scratch") / user / project / "telemetry"


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")


def latest_file_per_host(root: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for path in root.glob("*.csv"):
        host = path.name.split("_job", 1)[0]
        current = latest.get(host)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest[host] = path
    return latest


def recent_rows(path: Path, cutoff: dt.datetime) -> list[tuple[dt.datetime, float, float]]:
    rows: list[tuple[dt.datetime, float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = parse_timestamp(row["timestamp"])
                if stamp < cutoff:
                    continue
                util = float(row["util_pct"])
                memory = float(row["mem_used_mib"]) / float(row["mem_total_mib"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            rows.append((stamp, util, memory))
    return rows


def gpu_count(rows: list[tuple[dt.datetime, float, float]]) -> int:
    """Infer GPU count from logger bursts separated by at least one second."""
    if not rows:
        return 0
    largest = current = 1
    previous = rows[0][0]
    for stamp, _, _ in rows[1:]:
        if (stamp - previous).total_seconds() >= 1.0:
            largest = max(largest, current)
            current = 1
        else:
            current += 1
        previous = stamp
    return max(largest, current)


def prefix_provenance(path: Path) -> dict[str, object]:
    """Hash the exact file prefix present when a live logger is sampled."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = size
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return {"path": str(path), "prefix_bytes": size, "prefix_sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, default=default_telemetry_dir())
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--active-threshold", type=float, default=5.0)
    parser.add_argument("--project", default=os.environ.get("PROJECT", "my-project"))
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path, help="write an immutable JSON snapshot")
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")

    now = dt.datetime.now()
    cutoff = now - dt.timedelta(seconds=args.window_seconds)
    by_host: dict[str, list[tuple[dt.datetime, float, float]]] = {}
    source_by_host: dict[str, Path] = {}
    for host, path in latest_file_per_host(args.telemetry_dir).items():
        rows = recent_rows(path, cutoff)
        if rows:
            by_host[host] = rows
            source_by_host[host] = path

    all_rows = [row for rows in by_host.values() for row in rows]
    if not all_rows:
        print(
            f"SUMMARY window={args.window_seconds}s hosts=0 samples=0 "
            "active_fraction=0.000 mean_util=0.00 mean_mem_fraction=0.000"
        )
        return 1

    active_fraction = mean(util >= args.active_threshold for _, util, _ in all_rows)
    mean_util = mean(util for _, util, _ in all_rows)
    mean_memory = mean(memory for _, _, memory in all_rows)
    if not args.summary_only:
        for host in sorted(by_host):
            rows = by_host[host]
            print(
                f"{host} samples={len(rows)} "
                f"active_fraction={mean(util >= args.active_threshold for _, util, _ in rows):.3f} "
                f"mean_util={mean(util for _, util, _ in rows):.2f} "
                f"mean_mem_fraction={mean(memory for _, _, memory in rows):.3f}"
            )
    print(
        f"SUMMARY window={args.window_seconds}s hosts={len(by_host)} samples={len(all_rows)} "
        f"active_fraction={active_fraction:.3f} mean_util={mean_util:.2f} "
        f"mean_mem_fraction={mean_memory:.3f}"
    )

    if args.output:
        hosts: dict[str, dict[str, object]] = {}
        for host in sorted(by_host):
            rows = by_host[host]
            hosts[host] = {
                "gpus_observed": gpu_count(rows),
                "samples": len(rows),
                "active_fraction": mean(util >= args.active_threshold for _, util, _ in rows),
                "mean_utilization_percent": mean(util for _, util, _ in rows),
                "mean_memory_fraction": mean(memory for _, _, memory in rows),
                "source": prefix_provenance(source_by_host[host]),
                "observations": [
                    {
                        "timestamp": stamp.isoformat(timespec="milliseconds"),
                        "utilization_percent": util,
                        "memory_fraction": memory,
                    }
                    for stamp, util, memory in rows
                ],
            }
        payload = {
            "schema_version": 2,
            "kind": "gpu_telemetry_window",
            "project": args.project,
            "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "window_seconds": args.window_seconds,
            "active_threshold_percent": args.active_threshold,
            "hosts": hosts,
            "summary": {
                "hosts": len(by_host),
                "gpus_observed": sum(int(row["gpus_observed"]) for row in hosts.values()),
                "samples": len(all_rows),
                "active_fraction": active_fraction,
                "mean_utilization_percent": mean_util,
                "mean_memory_fraction": mean_memory,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
