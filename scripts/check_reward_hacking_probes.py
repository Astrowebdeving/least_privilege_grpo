"""Gate: adversarial reward-hacking probes must all be unprofitable.

For every task in the catalog, replays concrete exploit attempts (test/config
tampering, parser-avoidance syntax, unknown-command laundering, step padding,
repeated envelope crossings, secret sweeps via broad reads, and MCP
prompt-injection compliance) through the real TerminalEnv/broker/reward path
and asserts each is strictly unprofitable versus the oracle, with minimum
margins where the calibration pass fixed a known-thin gap.

Writes a machine-readable report (default
logs/artifacts/reward_hacking_probes.json) with per-task rewards, breakdowns,
and per-probe minimum margins.
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

from least_privilege_env.reward.probes import PROBES, analyze_task_probes
from least_privilege_env.tasks.loader import load_tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", choices=["local", "docker"], default="local")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "reward_hacking_probes.json",
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
        probes = analyze_task_probes(task)
        cells = [f"oracle={probes.oracle_reward:+.3f}"]
        for probe in PROBES:
            result = probes.results.get(probe)
            if result is None:
                cells.append(f"{probe}=n/a")
                continue
            cells.append(f"{probe}={result.reward:+.3f}")
            key = f"oracle_minus_{probe}"
            margin = probes.oracle_reward - result.reward
            margins[key] = min(margins.get(key, float("inf")), margin)
        print(f"{task.task_id:<16} {' '.join(cells)}")
        for violation in probes.violations:
            line = f"{task.task_id}: {violation}"
            print(f"  VIOLATION {line}")
            all_violations.append(line)
        rows.append(
            {
                "task_id": task.task_id,
                "family": probes.family,
                "oracle_reward": probes.oracle_reward,
                "probes": {
                    name: {
                        "reward": result.reward,
                        "note": result.note,
                        "extra": result.extra,
                        "breakdown": {
                            "success": result.breakdown.success,
                            "evidence": result.breakdown.evidence,
                            "privilege_penalty": result.breakdown.privilege_penalty,
                            "inefficiency": result.breakdown.inefficiency,
                            "escalation_hygiene": result.breakdown.escalation_hygiene,
                            "blocked": result.breakdown.blocked,
                            "unknown": result.breakdown.unknown,
                        },
                    }
                    for name, result in probes.results.items()
                },
                "violations": probes.violations,
            }
        )

    print("\nminimum margins across tasks (oracle reward minus probe reward):")
    for key in sorted(margins):
        print(f"  {key:<40} {margins[key]:+.4f}")

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
        print(f"reward hacking probes FAILED ({len(all_violations)} violations)")
        return 1
    print("reward hacking probes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
