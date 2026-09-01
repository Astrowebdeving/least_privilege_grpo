#!/usr/bin/env python3
"""Partition-aware gate for the continuation400 parent-policy precheck.

analyze_reward_variance.py validates the rollout contract and trainability by
task family. This companion gate prevents one half of the mixed catalog from
hiding a zero-gradient other half: it classifies every task as corrective or
replay, verifies exact membership, and applies thresholds to both sources.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "continuation400_v2.json"
DEFAULT_CORRECTIVE = ROOT / "data" / "continuation_transient_v3.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_ids(rows: list[dict[str, Any]], label: str) -> list[str]:
    ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in ids):
        raise ValueError(f"{label}: empty task_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate task_id")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variance-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--corrective", type=Path, default=DEFAULT_CORRECTIVE)
    parser.add_argument("--min-corrective-variable-fraction", type=float, default=0.10)
    parser.add_argument("--min-replay-variable-fraction", type=float, default=0.25)
    parser.add_argument("--min-replay-success-rate", type=float, default=0.0)
    parser.add_argument("--variance-epsilon", type=float, default=1e-12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    for name, value in (
        ("--min-corrective-variable-fraction", args.min_corrective_variable_fraction),
        ("--min-replay-variable-fraction", args.min_replay_variable_fraction),
        ("--min-replay-success-rate", args.min_replay_success_rate),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be in [0, 1]")

    catalog = load_json(args.catalog)
    corrective = load_json(args.corrective)
    variance = load_json(args.variance_report)
    if not isinstance(catalog, list) or not isinstance(corrective, list):
        raise ValueError("catalogs must be JSON lists")
    catalog_ids = task_ids(catalog, "catalog")
    corrective_ids = set(task_ids(corrective, "corrective"))
    if len(catalog_ids) != 400:
        raise ValueError(f"catalog: expected 400 tasks, got {len(catalog_ids)}")
    if len(corrective_ids) != 200:
        raise ValueError(f"corrective: expected 200 tasks, got {len(corrective_ids)}")
    if not corrective_ids.issubset(set(catalog_ids)):
        missing = sorted(corrective_ids - set(catalog_ids))
        raise ValueError(f"corrective tasks missing from mixed catalog: {missing[:5]}")

    report_tasks = variance.get("tasks")
    if not isinstance(report_tasks, list):
        raise ValueError("variance report has no task list")
    report_by_id = {str(row.get("task_id", "")): row for row in report_tasks}
    if len(report_by_id) != len(report_tasks):
        raise ValueError("variance report has empty or duplicate task ids")
    if set(report_by_id) != set(catalog_ids):
        missing = sorted(set(catalog_ids) - set(report_by_id))
        extra = sorted(set(report_by_id) - set(catalog_ids))
        raise ValueError(
            f"variance report membership mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    violations: list[str] = []
    if variance.get("status") != "pass":
        violations.append("base reward-variance report did not pass")

    family_by_id = {str(task["task_id"]): str(task["family"]) for task in catalog}
    source_rows: dict[str, list[dict[str, Any]]] = {"corrective": [], "replay": []}
    family_counts: dict[str, dict[str, Counter[str]]] = {
        "corrective": defaultdict(Counter),
        "replay": defaultdict(Counter),
    }
    for task_id in catalog_ids:
        source = "corrective" if task_id in corrective_ids else "replay"
        row = report_by_id[task_id]
        std = float(row.get("reward_std") or 0.0)
        variable = std > args.variance_epsilon
        source_rows[source].append(row)
        counts = family_counts[source][family_by_id[task_id]]
        counts["tasks"] += 1
        counts["variable_tasks"] += int(variable)
        counts["std_gt_0_1"] += int(std > 0.1)

    thresholds = {
        "corrective": args.min_corrective_variable_fraction,
        "replay": args.min_replay_variable_fraction,
    }
    sources: dict[str, dict[str, Any]] = {}
    for source in ("corrective", "replay"):
        rows = source_rows[source]
        variable = sum(
            float(row.get("reward_std") or 0.0) > args.variance_epsilon for row in rows
        )
        std_gt = sum(float(row.get("reward_std") or 0.0) > 0.1 for row in rows)
        successful_episodes = sum(int(row.get("successful_episodes") or 0) for row in rows)
        episodes = sum(int(row.get("episodes") or 0) for row in rows)
        success_rate = successful_episodes / episodes if episodes else 0.0
        required = math.ceil(len(rows) * thresholds[source])
        if variable < required:
            violations.append(
                f"{source}: variable groups {variable}/{len(rows)}, require at least {required}"
            )
        if source == "replay" and success_rate < args.min_replay_success_rate:
            violations.append(
                f"replay: success rate {success_rate:.3f} < {args.min_replay_success_rate:.3f}"
            )
        families = {
            family: {
                "tasks": counts["tasks"],
                "variable_tasks": counts["variable_tasks"],
                "variable_fraction": counts["variable_tasks"] / counts["tasks"],
                "std_gt_0_1": counts["std_gt_0_1"],
            }
            for family, counts in sorted(family_counts[source].items())
        }
        sources[source] = {
            "tasks": len(rows),
            "variable_tasks": variable,
            "variable_fraction": variable / len(rows),
            "std_gt_0_1": std_gt,
            "successful_episodes": successful_episodes,
            "episodes": episodes,
            "success_rate": success_rate,
            "required_variable_tasks": required,
            "families": families,
        }

    result = {
        "schema_version": 1,
        "status": "pass" if not violations else "fail",
        "variance_report": str(args.variance_report),
        "catalog": str(args.catalog),
        "corrective": str(args.corrective),
        "thresholds": {
            "variance_epsilon": args.variance_epsilon,
            "min_corrective_variable_fraction": args.min_corrective_variable_fraction,
            "min_replay_success_rate": args.min_replay_success_rate,
            "min_replay_variable_fraction": args.min_replay_variable_fraction,
        },
        "sources": sources,
        "violations": violations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "sources": sources}, indent=2))
    if violations:
        print("continuation400 partition precheck failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("continuation400 partition precheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
