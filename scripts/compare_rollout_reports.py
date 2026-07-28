#!/usr/bin/env python3
"""Compare two rollout analyzer reports.

This is intentionally report-only: it reads ``summarize_rollouts.py`` JSON
outputs and writes a compact delta table for base-vs-checkpoint evals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def counter_value(counter: dict[str, Any], key: str) -> float:
    return float(counter.get(key, 0) or 0)


def success_rate(counter: dict[str, Any]) -> float:
    episodes = counter_value(counter, "episodes")
    return counter_value(counter, "success") / episodes if episodes else 0.0


def totals(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report.get("episodes", [])
    total = len(episodes)
    success = sum(1 for episode in episodes if float(episode.get("verdict_success", 0.0) or 0.0) == 1.0)
    terminated = sum(1 for episode in episodes if episode.get("terminated"))
    return {
        "episodes": total,
        "success": success,
        "success_rate": success / total if total else 0.0,
        "terminated": terminated,
        "terminated_rate": terminated / total if total else 0.0,
        "non_terminated": total - terminated,
        "non_terminated_rate": (total - terminated) / total if total else 0.0,
        "flags": int(report.get("flag_count", 0) or 0),
        "flag_types": report.get("summary", {}).get("flag_types", {}),
    }


def compare_counters(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    names = sorted(set(base) | set(candidate))
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        lhs = base.get(name, {})
        rhs = candidate.get(name, {})
        rows[name] = {
            "base_episodes": int(counter_value(lhs, "episodes")),
            "candidate_episodes": int(counter_value(rhs, "episodes")),
            "base_success": int(counter_value(lhs, "success")),
            "candidate_success": int(counter_value(rhs, "success")),
            "delta_success": int(counter_value(rhs, "success") - counter_value(lhs, "success")),
            "base_success_rate": success_rate(lhs),
            "candidate_success_rate": success_rate(rhs),
            "delta_success_rate": success_rate(rhs) - success_rate(lhs),
            "base_non_terminated": int(counter_value(lhs, "non_terminated")),
            "candidate_non_terminated": int(counter_value(rhs, "non_terminated")),
            "delta_non_terminated": int(counter_value(rhs, "non_terminated") - counter_value(lhs, "non_terminated")),
            "base_flags": sum(int(value) for key, value in lhs.items() if key.startswith("flag:")),
            "candidate_flags": sum(int(value) for key, value in rhs.items() if key.startswith("flag:")),
        }
        rows[name]["delta_flags"] = rows[name]["candidate_flags"] - rows[name]["base_flags"]
    return rows


def build_comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_summary = base.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    return {
        "base": totals(base),
        "candidate": totals(candidate),
        "delta": {
            "success": totals(candidate)["success"] - totals(base)["success"],
            "success_rate": totals(candidate)["success_rate"] - totals(base)["success_rate"],
            "terminated": totals(candidate)["terminated"] - totals(base)["terminated"],
            "non_terminated": totals(candidate)["non_terminated"] - totals(base)["non_terminated"],
            "flags": totals(candidate)["flags"] - totals(base)["flags"],
        },
        "by_task": compare_counters(base_summary.get("by_task", {}), candidate_summary.get("by_task", {})),
        "by_family": compare_counters(base_summary.get("by_family", {}), candidate_summary.get("by_family", {})),
        "flag_types": compare_counters(
            {key: {"episodes": 1, "success": value} for key, value in totals(base)["flag_types"].items()},
            {key: {"episodes": 1, "success": value} for key, value in totals(candidate)["flag_types"].items()},
        ),
    }


def print_table(comparison: dict[str, Any]) -> None:
    base = comparison["base"]
    candidate = comparison["candidate"]
    delta = comparison["delta"]
    print(
        "overall: "
        f"base={base['success']}/{base['episodes']} ({base['success_rate']:.3f}) "
        f"candidate={candidate['success']}/{candidate['episodes']} ({candidate['success_rate']:.3f}) "
        f"delta_success={delta['success']:+d} delta_flags={delta['flags']:+d} "
        f"non_term={base['non_terminated']}->{candidate['non_terminated']} ({delta['non_terminated']:+d})"
    )
    print("tasks:")
    for task_id, row in comparison["by_task"].items():
        print(
            f"  {task_id:20s} "
            f"{row['base_success']:2d}/{row['base_episodes']:<2d} -> "
            f"{row['candidate_success']:2d}/{row['candidate_episodes']:<2d} "
            f"d_success={row['delta_success']:+d} d_flags={row['delta_flags']:+d} "
            f"d_nonterm={row['delta_non_terminated']:+d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    comparison = build_comparison(load_report(args.base_report), load_report(args.candidate_report))
    print_table(comparison)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
