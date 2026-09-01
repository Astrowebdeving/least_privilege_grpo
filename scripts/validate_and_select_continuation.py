#!/usr/bin/env python3
"""Validate frozen eval rollouts and select a continuation checkpoint.

Selection is internal-only. External benchmark outputs are not accepted as
inputs. The fixed lexicographic rule is: more safe success; higher worst-family
safe-success rate; fewer over-privileged successes; more verifier successes;
fewer non-terminations; then the earlier continuation checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_tasks(path: Path) -> tuple[list[str], dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("tasks") or raw.get("entries") or list(raw.values())
    ids = [str(row["task_id"]) for row in raw]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"task catalog has no tasks or duplicate IDs: {path}")
    return ids, {str(row["task_id"]): str(row["family"]) for row in raw}


def _load_rollouts(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("rollouts-*.jsonl"))
    if not files:
        raise ValueError(f"no rollout JSONL files in {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = f"{file}:{line_no}"
                    rows.append(row)
    return rows


def _behaviorful(row: dict[str, Any]) -> bool:
    return bool(
        row.get("steps")
        or row.get("terminated")
        or row.get("final_answer") is not None
        or row.get("escalation") is not None
    )


def _validate_policy(
    *,
    name: str,
    eval_root: Path,
    task_ids: list[str],
    families: dict[str, str],
    generations: int,
    safety: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    rollout_dir = eval_root / name / "rollouts"
    rows = _load_rollouts(rollout_dir)
    expected_counts = Counter({task_id: generations for task_id in task_ids})
    observed_counts = Counter(str(row.get("task_id")) for row in rows)
    expected_episodes = len(task_ids) * generations
    if len(rows) != expected_episodes:
        errors.append(f"episode_count={len(rows)} expected={expected_episodes}")
    if observed_counts != expected_counts:
        missing = sorted((expected_counts - observed_counts).elements())
        extra = sorted((observed_counts - expected_counts).elements())
        errors.append(f"membership mismatch missing={missing[:20]} extra={extra[:20]}")

    episode_ids = [str(row.get("episode_id") or "") for row in rows]
    blank_ids = sum(not value for value in episode_ids)
    duplicate_ids = sorted(
        value for value, count in Counter(episode_ids).items() if value and count > 1
    )
    if blank_ids:
        errors.append(f"blank episode IDs={blank_ids}")
    if duplicate_ids:
        errors.append(f"duplicate episode IDs={duplicate_ids[:10]}")

    non_docker = sum(str(row.get("sandbox")) != "docker" for row in rows)
    missing_stats = missing_with_behavior = malformed_stats = post_terminal = 0
    rows_with_stats = 0
    for row in rows:
        stats = row.get("trainer_tool_stats")
        if stats is None:
            missing_stats += 1
            if _behaviorful(row):
                missing_with_behavior += 1
            continue
        if not isinstance(stats, dict) or "post_terminal_calls" not in stats:
            malformed_stats += 1
            continue
        rows_with_stats += 1
        post_terminal += int(stats.get("post_terminal_calls") or 0)
    if non_docker:
        errors.append(f"non-Docker episodes={non_docker}")
    if missing_with_behavior:
        errors.append(f"missing trainer_tool_stats with behavior={missing_with_behavior}")
    if malformed_stats:
        errors.append(f"malformed trainer_tool_stats={malformed_stats}")
    if rows_with_stats == 0:
        errors.append("no valid trainer_tool_stats in any episode")
    if post_terminal:
        errors.append(f"post_terminal_calls={post_terminal}")

    raw_success = sum(
        float((row.get("verdict") or {}).get("success") or 0.0) == 1.0 for row in rows
    )
    non_terminated = sum(not bool(row.get("terminated")) for row in rows)
    policy = safety.get("policies", {}).get(name)
    if not isinstance(policy, dict):
        errors.append("policy missing from safety_metrics.json")
        overall: dict[str, Any] = {}
        by_task: dict[str, Any] = {}
    else:
        overall = policy.get("overall") or {}
        by_task = policy.get("by_task") or {}
        if int(overall.get("episodes", -1)) != expected_episodes:
            errors.append("safety report episode count mismatch")
        if int(overall.get("success", -1)) != raw_success:
            errors.append("safety report success count mismatch")
        if int(overall.get("safe_success", -1)) + int(
            overall.get("over_privileged_success", -1)
        ) != raw_success:
            errors.append("safe + over-privileged success does not equal success")

    family_totals: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        grouped[families[task_id]].append(task_id)
    for family, members in sorted(grouped.items()):
        episodes = sum(int((by_task.get(task_id) or {}).get("episodes", 0)) for task_id in members)
        safe_success = sum(
            int((by_task.get(task_id) or {}).get("safe_success", 0)) for task_id in members
        )
        success = sum(int((by_task.get(task_id) or {}).get("success", 0)) for task_id in members)
        family_totals[family] = {
            "tasks": len(members),
            "episodes": episodes,
            "success": success,
            "safe_success": safe_success,
            "safe_success_rate": safe_success / episodes if episodes else 0.0,
        }
        if episodes != len(members) * generations:
            errors.append(f"family safety membership mismatch: {family}")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "rollout_dir": str(rollout_dir),
        "tasks": len(task_ids),
        "episodes": len(rows),
        "unique_episode_ids": len({value for value in episode_ids if value}),
        "raw_success": raw_success,
        "non_terminated": non_terminated,
        "non_docker": non_docker,
        "missing_tool_stats": missing_stats,
        "missing_tool_stats_with_behavior": missing_with_behavior,
        "malformed_tool_stats": malformed_stats,
        "post_terminal_calls": post_terminal,
        "overall": overall,
        "by_family": family_totals,
    }


def _step(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    overall = row["overall"]
    worst_family = min(
        (float(item["safe_success_rate"]) for item in row["by_family"].values()),
        default=0.0,
    )
    return (
        float(overall["safe_success"]),
        worst_family,
        -float(overall["over_privileged_success"]),
        float(overall["success"]),
        -float(row["non_terminated"]),
        -float(_step(row["name"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--parent", default="")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selected-path-file", type=Path)
    args = parser.parse_args()

    if args.generations < 1:
        raise SystemExit("--generations must be positive")
    policy_names = _csv(args.policies)
    candidates = _csv(args.candidates)
    if not policy_names or not set(candidates).issubset(policy_names):
        raise SystemExit("policies are empty or candidates are outside policies")
    adapters: dict[str, str] = {}
    for spec in args.adapter:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in adapters:
            raise SystemExit(f"invalid/duplicate --adapter: {spec}")
        adapters[name] = path

    task_ids, families = _load_tasks(args.tasks)
    safety = json.loads((args.eval_root / "safety_metrics.json").read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name in policy_names:
        result = _validate_policy(
            name=name,
            eval_root=args.eval_root,
            task_ids=task_ids,
            families=families,
            generations=args.generations,
            safety=safety,
        )
        result["name"] = name
        results[name] = result
        errors.extend(f"{name}: {error}" for error in result["errors"])

    selected = None
    ranking: list[dict[str, Any]] = []
    if candidates and not errors:
        missing_adapters = [name for name in candidates if name not in adapters]
        if missing_adapters:
            errors.append(f"candidate adapter paths missing: {missing_adapters}")
        else:
            ranked = sorted((results[name] for name in candidates), key=_rank_key, reverse=True)
            selected = ranked[0]["name"]
            for rank, row in enumerate(ranked, 1):
                ranking.append(
                    {
                        "rank": rank,
                        "name": row["name"],
                        "rank_key": list(_rank_key(row)),
                        "safe_success": row["overall"]["safe_success"],
                        "success": row["overall"]["success"],
                        "over_privileged_success": row["overall"]["over_privileged_success"],
                        "non_terminated": row["non_terminated"],
                        "adapter_path": adapters[row["name"]],
                    }
                )

    parent_comparison = None
    if selected and args.parent:
        if args.parent not in results:
            errors.append(f"parent policy is absent: {args.parent}")
        else:
            chosen = results[selected]["overall"]
            parent = results[args.parent]["overall"]
            parent_comparison = {
                "parent": args.parent,
                "selected": selected,
                "safe_success_delta": int(chosen["safe_success"]) - int(parent["safe_success"]),
                "success_delta": int(chosen["success"]) - int(parent["success"]),
                "over_privileged_success_delta": int(chosen["over_privileged_success"])
                - int(parent["over_privileged_success"]),
                "selected_beats_parent_primary": int(chosen["safe_success"])
                > int(parent["safe_success"]),
            }

    report = {
        "schema": "lp-rl/continuation-eval-selection/1",
        "status": "pass" if not errors else "fail",
        "selection_inputs": "internal frozen task catalog and rollout telemetry only",
        "selection_rule": [
            "maximize safe_success",
            "maximize worst-family safe_success_rate",
            "minimize over_privileged_success",
            "maximize verifier success",
            "minimize non_termination",
            "prefer earlier continuation checkpoint",
        ],
        "task_file": str(args.tasks),
        "task_count": len(task_ids),
        "generations": args.generations,
        "policies": results,
        "candidates": candidates,
        "ranking": ranking,
        "selected": selected,
        "selected_adapter_path": adapters.get(selected or ""),
        "parent_comparison": parent_comparison,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if selected and args.selected_path_file:
        args.selected_path_file.write_text(adapters[selected] + "\n", encoding="utf-8")
    if selected:
        print(f"continuation selection PASS: selected={selected} path={adapters[selected]}")
    else:
        print(f"eval validation PASS: policies={','.join(policy_names)} episodes={len(task_ids) * args.generations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
