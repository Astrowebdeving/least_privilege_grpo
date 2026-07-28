#!/usr/bin/env python3
"""Aggregate and pairwise-check a completed ToolPrivBench comparison suite."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics


ARMS = (
    "base_default",
    "base_prompt_only",
    "seed0_default",
    "seed1_default",
    "seed2_default",
)


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for file_path in sorted(path.glob("*.json")):
        if file_path.name == "summary.json":
            continue
        row = json.loads(file_path.read_text(encoding="utf-8"))
        rows[row["scenario_id"]] = row
    return rows


def _paired_bootstrap(
    left: dict[str, dict],
    right: dict[str, dict],
    *,
    samples: int = 50_000,
    seed: int = 20260716,
) -> tuple[float, float, float]:
    ids = sorted(set(left) & set(right))
    diffs = [
        int(bool(left[sid].get("over_privileged_selection")))
        - int(bool(right[sid].get("over_privileged_selection")))
        for sid in ids
    ]
    observed = statistics.fmean(diffs)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs))
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
    rows = {arm: _load_rows(args.suite / arm) for arm in ARMS}
    ids = set(rows[ARMS[0]])
    if any(set(rows[arm]) != ids for arm in ARMS[1:]):
        raise SystemExit("scenario-id sets differ across arms")

    comparisons = {}
    base = rows["base_default"]
    for arm in ARMS[1:]:
        other = rows[arm]
        base_only = sum(
            bool(base[sid].get("over_privileged_selection"))
            and not bool(other[sid].get("over_privileged_selection"))
            for sid in ids
        )
        other_only = sum(
            bool(other[sid].get("over_privileged_selection"))
            and not bool(base[sid].get("over_privileged_selection"))
            for sid in ids
        )
        effect, low, high = _paired_bootstrap(base, other)
        comparisons[arm] = {
            "base_overpriv_other_safe": base_only,
            "base_safe_other_overpriv": other_only,
            "base_minus_arm_overpriv_rate": effect,
            "paired_bootstrap_95": [low, high],
        }

    report = {
        "scenario_count": len(ids),
        "arms": summaries,
        "paired_vs_base_default": comparisons,
    }
    (args.suite / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ToolPrivBench comparison",
        "",
        "| Arm | OPUR@1 | OPUR@3 | OPUR@5 | Over-priv@5 | PED mean | Success proxy | Mean completion tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = summaries[arm]
        lines.append(
            f"| {arm} | {item['opur']['1']:.1%} | {item['opur']['3']:.1%} | "
            f"{item['opur']['5']:.1%} | {item['over_privilege_rate']['5']:.1%} | "
            f"{item['ped']['mean']:.2f} | {item['success_proxy']}/{item['total']} | "
            f"{item['completion_tokens_mean']:.1f} |"
        )
    lines.extend(["", "## Paired over-privilege comparisons versus base/default", ""])
    for arm, item in comparisons.items():
        low, high = item["paired_bootstrap_95"]
        lines.append(
            f"- `{arm}`: base-only violations {item['base_overpriv_other_safe']}; "
            f"arm-only violations {item['base_safe_other_overpriv']}; "
            f"base minus arm {item['base_minus_arm_overpriv_rate']:.1%} "
            f"(descriptive paired bootstrap 95% [{low:.1%}, {high:.1%}])."
        )
    lines.extend(
        [
            "",
            "The bootstrap intervals are post-run descriptive analyses, not preregistered tests.",
            "The success proxy means that at least one simulated tool returned a success output;",
            "it is not a substitute for the benchmark's OPUR/PED metrics or an external judge.",
            "",
        ]
    )
    (args.suite / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
