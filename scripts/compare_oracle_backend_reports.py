#!/usr/bin/env python3
"""Fail when supported oracle behavior differs between local and Docker.

Task success alone is not enough: backend-specific evidence or risk telemetry
changes the RL reward while leaving the oracle green. Compare the full reward
breakdown, accessed evidence, and trajectory risk for the same oracle replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["task_id"]: item for item in payload["tasks"]}


def _numbers_close(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _numbers_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numbers_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    return left == right


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_report", type=Path)
    parser.add_argument("docker_report", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    local = _load(args.local_report)
    docker = _load(args.docker_report)
    violations: list[str] = []
    if local.keys() != docker.keys():
        violations.append(
            f"task-id sets differ: local_only={sorted(local.keys() - docker.keys())}, "
            f"docker_only={sorted(docker.keys() - local.keys())}"
        )
    for task_id in sorted(local.keys() & docker.keys()):
        left, right = local[task_id], docker[task_id]
        for field in ("success", "reward", "accessed_evidence", "z_tau"):
            if not _numbers_close(left.get(field), right.get(field), args.tolerance):
                violations.append(
                    f"{task_id}: {field} differs: local={left.get(field)!r} "
                    f"docker={right.get(field)!r}"
                )

    result = {
        "task_count": len(local.keys() & docker.keys()),
        "tolerance": args.tolerance,
        "violations": violations,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if violations:
        for violation in violations:
            print(f"VIOLATION {violation}")
        print(f"oracle backend parity FAILED ({len(violations)} differences)")
        return 1
    print(f"oracle backend parity passed for {result['task_count']} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
