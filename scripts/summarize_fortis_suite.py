#!/usr/bin/env python3
"""Aggregate a base-plus-three-seed FORTIS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


ARMS = ("base_default", "seed0_default", "seed1_default", "seed2_default")
TASKS = ("task1", "task2")


def _load_rows(path: Path, task: str) -> dict[str, dict]:
    rows = {}
    for result_path in sorted((path / task).glob("*/*.json")):
        row = json.loads(result_path.read_text(encoding="utf-8"))
        key = f"{row['domain']}:{row['id']}"
        if key in rows:
            raise ValueError(f"duplicate result key: {key}")
        rows[key] = row
    return rows


def _paired_bootstrap(
    left: dict[str, dict],
    right: dict[str, dict],
    field: str,
    *,
    samples: int = 50_000,
    seed: int = 20260716,
) -> tuple[float, float, float]:
    ids = sorted(left)
    if set(ids) != set(right):
        raise ValueError("paired result membership differs")
    diffs = [
        int(left[sid]["classification"] == field)
        - int(right[sid]["classification"] == field)
        for sid in ids
    ]
    observed = statistics.fmean(diffs)
    rng = random.Random(f"{seed}:{field}")
    estimates = [
        statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(samples)
    ]
    estimates.sort()
    return observed, estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    summaries = {
        arm: json.loads((args.suite / arm / "summary.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    comparisons = {}
    for task in TASKS:
        task_rows = {arm: _load_rows(args.suite / arm, task) for arm in ARMS}
        base = task_rows["base_default"]
        comparisons[task] = {}
        for arm in ARMS[1:]:
            other = task_rows[arm]
            item = {}
            for field in ("over_privilege", "exact_match", "no_action"):
                effect, low, high = _paired_bootstrap(base, other, field)
                item[field] = {
                    "base_minus_arm_rate": effect,
                    "paired_bootstrap_95": [low, high],
                }
            item["base_over_arm_not"] = sum(
                base[sid]["classification"] == "over_privilege"
                and other[sid]["classification"] != "over_privilege"
                for sid in base
            )
            item["base_not_over_arm_over"] = sum(
                base[sid]["classification"] != "over_privilege"
                and other[sid]["classification"] == "over_privilege"
                for sid in base
            )
            comparisons[task][arm] = item

    report = {"arms": summaries, "paired_vs_base_default": comparisons}
    (args.suite / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# FORTIS comparison",
        "",
        "| Task | Arm | Exact | Under | Over | No action | Safe rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for arm in ARMS:
            metrics = summaries[arm]["by_task"][task]
            counts = metrics["counts"]
            lines.append(
                f"| {task} | {arm} | {counts['exact_match']}/{metrics['total']} "
                f"({metrics['success_rate']:.1%}) | {counts['under_privilege']} | "
                f"{counts['over_privilege']} ({metrics['over_privilege_rate']:.1%}) | "
                f"{counts['no_action']} | {metrics['safe_rate']:.1%} |"
            )
    lines.extend(["", "## Paired comparisons versus base", ""])
    for task in TASKS:
        lines.append(f"### {task}")
        lines.append("")
        for arm, item in comparisons[task].items():
            over = item["over_privilege"]
            exact = item["exact_match"]
            lines.append(
                f"- `{arm}`: base-minus-arm over-privilege {over['base_minus_arm_rate']:.1%} "
                f"(95% [{over['paired_bootstrap_95'][0]:.1%}, "
                f"{over['paired_bootstrap_95'][1]:.1%}]); base-minus-arm exact "
                f"{exact['base_minus_arm_rate']:.1%} "
                f"(95% [{exact['paired_bootstrap_95'][0]:.1%}, "
                f"{exact['paired_bootstrap_95'][1]:.1%}])."
            )
        lines.append("")
    lines.append("Bootstrap intervals are post-run descriptive analyses.")
    lines.append("")
    (args.suite / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
