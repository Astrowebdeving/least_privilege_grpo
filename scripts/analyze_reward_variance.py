#!/usr/bin/env python3
"""Fail-closed reward-variance audit for a frozen rollout precheck.

The continuation uses one eight-sample GRPO group per task. Oracle/negative
separation proves that a reward can distinguish policies, but a task only
supplies a policy gradient when the sampled group itself has non-zero reward
variance. This script validates exact rollout membership and reports that
trainability signal before any optimizer step is authorized.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"catalog must be a non-empty JSON list: {path}")
    ids = [str(row.get("task_id", "")) for row in data]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("catalog task_id values must be non-empty and unique")
    if any(not row.get("family") for row in data):
        raise ValueError("every catalog task must have a family")
    return data


def _load_rollouts(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no rollout JSONL files found under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {file}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"rollout row is not an object at {file}:{line_number}")
                row["_source"] = f"{file}:{line_number}"
                rows.append(row)
    return rows


def _reward(row: dict[str, Any]) -> float:
    reward = row.get("reward")
    value = reward.get("reward") if isinstance(reward, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"missing/non-finite reward at {row.get('_source')}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--expected-generations", type=int, default=8)
    parser.add_argument("--min-overall-variable-fraction", type=float, default=0.20)
    parser.add_argument("--min-family-variable-fraction", type=float, default=0.10)
    parser.add_argument("--min-family-variable-groups", type=int, default=2)
    parser.add_argument("--variance-epsilon", type=float, default=1e-12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.expected_generations < 2:
        parser.error("--expected-generations must be at least 2")
    if not 0 <= args.min_overall_variable_fraction <= 1:
        parser.error("--min-overall-variable-fraction must be in [0, 1]")
    if not 0 <= args.min_family_variable_fraction <= 1:
        parser.error("--min-family-variable-fraction must be in [0, 1]")
    if args.min_family_variable_groups < 1:
        parser.error("--min-family-variable-groups must be positive")

    catalog = _load_catalog(args.task_file)
    rows = _load_rollouts(args.rollouts)
    task_family = {str(row["task_id"]): str(row["family"]) for row in catalog}
    expected_ids = set(task_family)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    violations: list[str] = []

    episode_ids: list[str] = []
    missing_tool_stats = 0
    missing_tool_stats_with_behavior = 0
    malformed_tool_stats = 0
    nonzero_post_terminal = 0
    non_docker = 0
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in expected_ids:
            violations.append(f"unexpected task_id {task_id!r} at {row.get('_source')}")
            continue
        by_task[task_id].append(row)
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            violations.append(f"missing episode_id at {row.get('_source')}")
        episode_ids.append(episode_id)
        if row.get("sandbox") != "docker":
            non_docker += 1
        stats = row.get("trainer_tool_stats")
        if stats is None:
            missing_tool_stats += 1
            has_behavior = bool(row.get("steps")) or bool(row.get("terminated")) or row.get("final_answer") is not None or row.get("escalation") is not None
            if has_behavior:
                missing_tool_stats_with_behavior += 1
        elif not isinstance(stats, dict) or "post_terminal_calls" not in stats:
            malformed_tool_stats += 1
        elif int(stats.get("post_terminal_calls") or 0) != 0:
            nonzero_post_terminal += 1
        try:
            _reward(row)
        except ValueError as exc:
            violations.append(str(exc))

    duplicate_episode_ids = len(episode_ids) - len(set(episode_ids))
    if duplicate_episode_ids:
        violations.append(f"duplicate episode ids: {duplicate_episode_ids}")
    if non_docker:
        violations.append(f"non-Docker rollout rows: {non_docker}")
    if missing_tool_stats_with_behavior:
        violations.append(f"rows missing trainer_tool_stats despite behavior: {missing_tool_stats_with_behavior}")
    if malformed_tool_stats:
        violations.append(f"rows with malformed trainer_tool_stats: {malformed_tool_stats}")
    if len(rows) - missing_tool_stats - malformed_tool_stats <= 0:
        violations.append("no rollout rows carry valid terminal-stop trainer_tool_stats")
    if nonzero_post_terminal:
        violations.append(f"rows with post_terminal_calls != 0: {nonzero_post_terminal}")

    expected_total = len(catalog) * args.expected_generations
    if len(rows) != expected_total:
        violations.append(f"episode_count={len(rows)} expected={expected_total}")

    task_reports: list[dict[str, Any]] = []
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    variable_tasks = 0
    for task in catalog:
        task_id = str(task["task_id"])
        family = task_family[task_id]
        task_rows = by_task.get(task_id, [])
        count = len(task_rows)
        if count != args.expected_generations:
            violations.append(f"{task_id}: episodes={count} expected={args.expected_generations}")
        rewards = [_reward(row) for row in task_rows] if task_rows else []
        reward_std = statistics.pstdev(rewards) if len(rewards) >= 2 else 0.0
        variable = reward_std > args.variance_epsilon
        variable_tasks += int(variable)
        family_counts[family]["tasks"] += 1
        family_counts[family]["episodes"] += count
        family_counts[family]["variable_tasks"] += int(variable)
        family_counts[family]["successful_episodes"] += sum(
            1 for row in task_rows if float(row.get("reward", {}).get("success", 0.0)) >= 1.0
        )
        task_reports.append(
            {
                "task_id": task_id,
                "family": family,
                "episodes": count,
                "reward_mean": statistics.fmean(rewards) if rewards else None,
                "reward_std": reward_std,
                "reward_min": min(rewards) if rewards else None,
                "reward_max": max(rewards) if rewards else None,
                "variable": variable,
            }
        )

    overall_fraction = variable_tasks / len(catalog)
    if overall_fraction < args.min_overall_variable_fraction:
        violations.append(
            "overall variable-group fraction "
            f"{overall_fraction:.3f} < {args.min_overall_variable_fraction:.3f}"
        )

    family_reports: dict[str, dict[str, Any]] = {}
    for family, counts in sorted(family_counts.items()):
        tasks = counts["tasks"]
        variable = counts["variable_tasks"]
        required = max(args.min_family_variable_groups, math.ceil(tasks * args.min_family_variable_fraction))
        if variable < required:
            violations.append(f"{family}: variable groups {variable}/{tasks}, require at least {required}")
        family_reports[family] = {
            "tasks": tasks,
            "episodes": counts["episodes"],
            "variable_tasks": variable,
            "variable_fraction": variable / tasks,
            "required_variable_tasks": required,
            "successful_episodes": counts["successful_episodes"],
            "success_rate": counts["successful_episodes"] / counts["episodes"] if counts["episodes"] else None,
        }

    report = {
        "schema_version": 1,
        "status": "pass" if not violations else "fail",
        "task_file": str(args.task_file),
        "rollouts": str(args.rollouts),
        "expected_generations": args.expected_generations,
        "thresholds": {
            "variance_epsilon": args.variance_epsilon,
            "min_overall_variable_fraction": args.min_overall_variable_fraction,
            "min_family_variable_fraction": args.min_family_variable_fraction,
            "min_family_variable_groups": args.min_family_variable_groups,
        },
        "summary": {
            "task_count": len(catalog),
            "episode_count": len(rows),
            "expected_episode_count": expected_total,
            "variable_tasks": variable_tasks,
            "variable_fraction": overall_fraction,
            "zero_variance_tasks": len(catalog) - variable_tasks,
            "duplicate_episode_ids": duplicate_episode_ids,
            "missing_tool_stats": missing_tool_stats,
            "missing_tool_stats_with_behavior": missing_tool_stats_with_behavior,
            "malformed_tool_stats": malformed_tool_stats,
            "nonzero_post_terminal_calls": nonzero_post_terminal,
            "non_docker_rows": non_docker,
        },
        "families": family_reports,
        "violations": violations,
        "tasks": task_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"status": report["status"], **report["summary"], "families": family_reports}, indent=2))
    if violations:
        print("reward-variance precheck failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("reward-variance precheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
