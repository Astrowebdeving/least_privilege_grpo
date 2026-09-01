"""Gate: replay synthetic trajectory profiles and verify reward orderings.

For every task in the catalog, replays oracle, over-privileged, guessed-answer,
broad-dump, wrong-escalation, blocked-command (and, where fixtures provide
bait, secret-read) trajectories through the real TerminalEnv/broker/reward
path, then asserts the orderings the least-privilege claim depends on:

  oracle > over_privileged, guessed, broad_dump, wrong_escalation, secret_read
  blocked attempts are negative and below a lucky guess

Writes a machine-readable report (default logs/artifacts/reward_sensitivity.json)
with per-task rewards, breakdowns, and per-invariant minimum margins.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.reward.sensitivity import PROFILES, analyze_task
from least_privilege_env.tasks.loader import load_tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", choices=["local", "docker"], default="local")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "reward_sensitivity.json",
    )
    args = parser.parse_args()
    os.environ["LP_RL_SANDBOX"] = args.sandbox

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    rows = []
    all_violations: list[str] = []
    margins: dict[str, float] = {}
    for task in tasks:
        sensitivity = analyze_task(task)
        oracle = sensitivity.results["oracle"].reward
        cells = []
        for profile in PROFILES:
            result = sensitivity.results.get(profile)
            if result is None:
                cells.append(f"{profile}=n/a")
                continue
            cells.append(f"{profile}={result.reward:+.3f}")
            if profile != "oracle":
                margin = oracle - result.reward
                key = f"oracle_minus_{profile}"
                margins[key] = min(margins.get(key, float("inf")), margin)
        print(f"{task.task_id:<16} {' '.join(cells)}")
        for violation in sensitivity.violations:
            line = f"{task.task_id}: {violation}"
            print(f"  VIOLATION {line}")
            all_violations.append(line)
        rows.append(
            {
                "task_id": task.task_id,
                "family": sensitivity.family,
                "profiles": {
                    name: {
                        "reward": result.reward,
                        "note": result.note,
                        "breakdown": {
                            "success": result.breakdown.success,
                            "evidence": result.breakdown.evidence,
                            "privilege_penalty": result.breakdown.privilege_penalty,
                            "inefficiency": result.breakdown.inefficiency,
                            "escalation_hygiene": result.breakdown.escalation_hygiene,
                            "blocked": result.breakdown.blocked,
                            "unknown": result.breakdown.unknown,
                            "forbidden_read": result.breakdown.forbidden_read,
                        },
                    }
                    for name, result in sensitivity.results.items()
                },
                "violations": sensitivity.violations,
            }
        )

    print("\nminimum margins across tasks (oracle reward minus profile reward):")
    for key in sorted(margins):
        print(f"  {key:<34} {margins[key]:+.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "sandbox": args.sandbox,
                "task_count": len(rows),
                "min_margins": margins,
                "violations": all_violations,
                "tasks": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nreport: {args.report}")

    if all_violations:
        print(f"reward sensitivity check FAILED ({len(all_violations)} violations)")
        return 1
    print("reward sensitivity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
