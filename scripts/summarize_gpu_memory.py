#!/usr/bin/env python3
"""Summarize trainer-side CUDA memory JSONL from train_trl_drgrpo.py."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


GB = 1024 ** 3


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rollout_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if path.is_dir():
        return sorted(path.glob("rollouts-*.jsonl"))
    return [path]


def load_rollouts(path: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in rollout_files(path):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return sorted(records, key=lambda row: float(row.get("ts") or 0.0))


def bytes_to_gb(value: int | float | None) -> float | None:
    return None if value is None else float(value) / GB


def numeric(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = record.get(key)
        if value is not None:
            out.append(float(value))
    return out


def scaled_threshold(total: float, fraction: float, absolute_cap: float) -> float:
    """Use real-byte caps on GPUs, but stay meaningful for tiny synthetic tests."""
    return min(absolute_cap, fraction * total)


def classify_memory(records: list[dict[str, Any]]) -> str:
    cuda_records = [record for record in records if record.get("cuda_available")]
    totals = numeric(cuda_records, "cuda_device_total") or numeric(cuda_records, "mem_get_info_total")
    total = max(totals) if totals else None
    if not total:
        return "inconclusive"

    max_alloc_frac = max(numeric(cuda_records, "max_memory_allocated_frac") or [0.0])
    if max_alloc_frac >= 0.90:
        return "allocated_ceiling"
    if len(cuda_records) < 2:
        return "inconclusive"

    current_alloc = numeric(cuda_records, "memory_allocated")
    current_reserved = numeric(cuda_records, "memory_reserved")
    free_fracs = numeric(cuda_records, "mem_get_info_free_frac")

    if len(current_alloc) < 2 or len(current_reserved) < 2:
        return "inconclusive"

    alloc_span = max(current_alloc) - min(current_alloc)
    reserved_span = max(current_reserved) - min(current_reserved)
    alloc_growth = current_alloc[-1] - current_alloc[0]
    low_free = min(free_fracs) if free_fracs else 1.0

    if alloc_growth > scaled_threshold(total, 0.05, GB):
        return "possible_leak"
    if reserved_span > scaled_threshold(total, 0.10, 2 * GB) and alloc_span < scaled_threshold(total, 0.05, GB):
        return "reserved_fragmentation"
    if low_free <= 0.02 and reserved_span > alloc_span:
        return "reserved_fragmentation"
    return "inconclusive"


def infer_task_groups(rollouts: list[dict[str, Any]], num_generations: int) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    if num_generations <= 0:
        return groups
    for index in range(0, len(rollouts), num_generations):
        step = index // num_generations + 1
        chunk = rollouts[index : index + num_generations]
        task_counts = Counter(str(row.get("task_id")) for row in chunk)
        success = sum(1 for row in chunk if (row.get("verdict") or {}).get("success") == 1.0)
        actions = [len(row.get("steps") or []) for row in chunk]
        groups[step] = {
            "task_ids": dict(task_counts),
            "episodes": len(chunk),
            "success": success,
            "mean_actions": (sum(actions) / len(actions)) if actions else 0.0,
        }
    return groups


def summarize(records: list[dict[str, Any]], rollouts: list[dict[str, Any]] | None = None, num_generations: int = 8) -> dict[str, Any]:
    rollouts = rollouts or []
    groups = infer_task_groups(rollouts, num_generations)
    records_by_step: list[dict[str, Any]] = []
    for record in records:
        step = int(record.get("global_step") or 0)
        metrics = record.get("metrics") or {}
        records_by_step.append(
            {
                "global_step": step,
                "task_group": groups.get(step),
                "memory_allocated_gb": bytes_to_gb(record.get("memory_allocated")),
                "memory_reserved_gb": bytes_to_gb(record.get("memory_reserved")),
                "max_memory_allocated_gb": bytes_to_gb(record.get("max_memory_allocated")),
                "max_memory_reserved_gb": bytes_to_gb(record.get("max_memory_reserved")),
                "mem_get_info_free_gb": bytes_to_gb(record.get("mem_get_info_free")),
                "memory_allocated_frac": record.get("memory_allocated_frac"),
                "memory_reserved_frac": record.get("memory_reserved_frac"),
                "max_memory_allocated_frac": record.get("max_memory_allocated_frac"),
                "max_memory_reserved_frac": record.get("max_memory_reserved_frac"),
                "mem_get_info_free_frac": record.get("mem_get_info_free_frac"),
                "cleanup_empty_cache_ran": bool(record.get("cleanup_empty_cache_ran")),
                "metrics": metrics,
            }
        )

    low_free_step = None
    for row in records_by_step:
        free_frac = row.get("mem_get_info_free_frac")
        if free_frac is not None and float(free_frac) <= 0.02:
            low_free_step = row["global_step"]
            break

    return {
        "schema_version": 1,
        "record_count": len(records),
        "classification": classify_memory(records),
        "max_memory_allocated_gb": bytes_to_gb(max((record.get("max_memory_allocated") or 0) for record in records)) if records else None,
        "max_memory_reserved_gb": bytes_to_gb(max((record.get("max_memory_reserved") or 0) for record in records)) if records else None,
        "min_mem_get_info_free_gb": bytes_to_gb(min((record.get("mem_get_info_free") for record in records if record.get("mem_get_info_free") is not None), default=None)),
        "first_low_free_step": low_free_step,
        "steps": records_by_step,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"records       : {report['record_count']}")
    print(f"classification: {report['classification']}")
    print(f"max allocated : {report['max_memory_allocated_gb']}")
    print(f"max reserved  : {report['max_memory_reserved_gb']}")
    print(f"min free      : {report['min_mem_get_info_free_gb']}")
    print(f"low-free step : {report['first_low_free_step']}")
    for step in report["steps"]:
        task_group = step.get("task_group") or {}
        tasks = ",".join(sorted((task_group.get("task_ids") or {}).keys()))
        print(
            f"step {step['global_step']:>3}: alloc={step['memory_allocated_gb']} "
            f"reserved={step['memory_reserved_gb']} peak_alloc={step['max_memory_allocated_gb']} "
            f"free={step['mem_get_info_free_gb']} tasks={tasks or '-'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("memory_log", type=Path)
    parser.add_argument("--rollouts", type=Path, default=None)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = summarize(load_jsonl(args.memory_log), load_rollouts(args.rollouts), args.num_generations)
    print_summary(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
