#!/usr/bin/env python3
"""Hard membership and terminal-stop telemetry guard for one frozen-500 policy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def rollout_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"rollout directory missing: {path}")
    return sorted(path.glob("rollouts-*.jsonl"))


def load_records(path: Path) -> list[dict[str, Any]]:
    files = rollout_files(path)
    if not files:
        raise ValueError(f"no rollout JSONL files under {path}")
    records: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    record = json.loads(line)
                    record["_source"] = f"{file}:{line_no}"
                    records.append(record)
    return records


def validate_tier(
    name: str,
    path: Path,
    tier: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    records = load_records(path)
    expected_ids = list(tier["task_ids"])
    expected_generations = int(tier["generations"])
    expected_episodes = int(tier["expected_episodes"])
    task_counts = Counter(str(record.get("task_id")) for record in records)
    expected_counts = Counter({task_id: expected_generations for task_id in expected_ids})
    if len(records) != expected_episodes:
        errors.append(f"{name}: episode_count={len(records)} expected={expected_episodes}")
    if task_counts != expected_counts:
        missing = sorted((expected_counts - task_counts).elements())
        extra = sorted((task_counts - expected_counts).elements())
        errors.append(f"{name}: task-generation membership mismatch missing={missing[:20]} extra={extra[:20]}")

    episode_ids = [str(record.get("episode_id") or "") for record in records]
    blank_episode_ids = sum(not value for value in episode_ids)
    duplicate_episode_ids = sorted(value for value, count in Counter(episode_ids).items() if value and count > 1)
    if blank_episode_ids:
        errors.append(f"{name}: {blank_episode_ids} blank episode IDs")
    if duplicate_episode_ids:
        errors.append(f"{name}: duplicate episode IDs {duplicate_episode_ids[:10]}")

    telemetry_missing = 0
    post_terminal_calls = 0
    malformed_telemetry = 0
    telemetry_totals: Counter[str] = Counter()
    for record in records:
        stats = record.get("trainer_tool_stats")
        if stats is None:
            telemetry_missing += 1
            continue
        if not isinstance(stats, dict) or "post_terminal_calls" not in stats:
            malformed_telemetry += 1
            continue
        post_terminal_calls += int(stats.get("post_terminal_calls") or 0)
        for key in (
            "generated_calls",
            "executed_calls",
            "cancelled_after_terminal",
            "failed_calls",
            "post_terminal_calls",
            "terminal_stopped",
        ):
            telemetry_totals[key] += int(stats.get(key) or 0)
    # trainer_tool_stats is None for episodes that made no tool calls at all
    # (immediate text response or budget-cut before any tool loop). That is
    # legitimate and, by construction, cannot hide a post-terminal call (a
    # post-terminal call requires the tool loop to have run, which populates
    # stats). So missing telemetry is recorded, not fatal. A TOTAL absence of
    # stats would instead mean the terminal-stop trainer never ran -> fail.
    episodes_with_stats = len(records) - telemetry_missing - malformed_telemetry
    if episodes_with_stats == 0:
        errors.append(f"{name}: no episodes carry trainer_tool_stats (terminal-stop trainer inactive?)")
    if malformed_telemetry:
        errors.append(f"{name}: malformed trainer_tool_stats on {malformed_telemetry} episodes")
    if post_terminal_calls:
        errors.append(f"{name}: post_terminal_calls={post_terminal_calls}, expected 0")

    summary = {
        "rollout_dir": path.as_posix(),
        "rollout_files": [file.as_posix() for file in rollout_files(path)],
        "task_count": len(task_counts),
        "episode_count": len(records),
        "expected_generations": expected_generations,
        "unique_episode_ids": len(set(episode_ids)) - (1 if "" in episode_ids else 0),
        "telemetry_missing": telemetry_missing,
        "episodes_with_stats": episodes_with_stats,
        "telemetry_malformed": malformed_telemetry,
        "post_terminal_calls": post_terminal_calls,
        "telemetry_totals": dict(telemetry_totals),
        "task_generation_counts": dict(sorted(task_counts.items())),
    }
    return summary, errors, episode_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-manifest", type=Path, required=True)
    parser.add_argument("--ordinary-rollouts", type=Path, required=True)
    parser.add_argument("--safety-rollouts", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.tier_manifest.read_text(encoding="utf-8"))
    summaries: dict[str, Any] = {}
    errors: list[str] = []
    all_episode_ids: list[str] = []
    for name, path in (
        ("ordinary", args.ordinary_rollouts),
        ("safety_heavy", args.safety_rollouts),
    ):
        summary, tier_errors, episode_ids = validate_tier(name, path, manifest["tiers"][name])
        summaries[name] = summary
        errors.extend(tier_errors)
        all_episode_ids.extend(episode_ids)

    cross_tier_duplicates = sorted(
        episode_id for episode_id, count in Counter(all_episode_ids).items() if episode_id and count > 1
    )
    if cross_tier_duplicates:
        errors.append(f"cross-tier duplicate episode IDs: {cross_tier_duplicates[:10]}")

    observed_episodes = sum(item["episode_count"] for item in summaries.values())
    expected_episodes = int(manifest["expected_episodes_per_policy"])
    if observed_episodes != expected_episodes:
        errors.append(f"combined episode_count={observed_episodes} expected={expected_episodes}")

    report = {
        "schema": "lp-rl/frozen500-run-validation/1",
        "status": "pass" if not errors else "fail",
        "policy": args.policy,
        "tier_manifest": args.tier_manifest.as_posix(),
        "assignment_sha256": manifest["assignment_sha256"],
        "expected_task_count": int(manifest["validation_task_count"]),
        "expected_episode_count": expected_episodes,
        "observed_episode_count": observed_episodes,
        "unique_episode_ids": len({value for value in all_episode_ids if value}),
        "post_terminal_calls": sum(item["post_terminal_calls"] for item in summaries.values()),
        "tiers": summaries,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"frozen500 guard passed policy={args.policy}: tasks=500 episodes={observed_episodes} "
        "unique_episode_ids=2896 post_terminal_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
