#!/usr/bin/env python3
"""Compare base and all final seeds on the registered MetaTool subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


ARMS = ("base", "seed0", "seed1", "seed2")


def _load_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for file_path in sorted(path.glob("*/*.json")):
        row = json.loads(file_path.read_text(encoding="utf-8"))
        key = (row["task"], int(row["source_index"]))
        if key in rows:
            raise ValueError(f"duplicate MetaTool sample key: {key}")
        rows[key] = row
    return rows


def _paired(
    base: dict[tuple[str, int], dict],
    other: dict[tuple[str, int], dict],
    arm: str,
) -> dict:
    if set(base) != set(other):
        raise ValueError(f"MetaTool sample keys differ for {arm}")
    keys = sorted(base)
    diffs = [int(other[key]["correct"]) - int(base[key]["correct"]) for key in keys]
    observed = statistics.fmean(diffs)
    rng = random.Random(f"20260716:{arm}")
    estimates = [
        statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(50_000)
    ]
    estimates.sort()
    return {
        "arm_minus_base_accuracy": observed,
        "paired_bootstrap_95": [estimates[1250], estimates[47_500]],
        "base_only_correct": sum(
            base[key]["correct"] and not other[key]["correct"] for key in keys
        ),
        "arm_only_correct": sum(
            other[key]["correct"] and not base[key]["correct"] for key in keys
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    summaries = {
        arm: json.loads((args.suite / arm / "summary.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    rows = {arm: _load_rows(args.suite / arm) for arm in ARMS}
    comparisons = {
        arm: _paired(rows["base"], rows[arm], arm) for arm in ARMS if arm != "base"
    }
    comparison = {"arms": summaries, "paired_vs_base": comparisons}
    (args.suite / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MetaTool registered-subset capability comparison",
        "",
        "| Task | Base | Seed 0 | Seed 1 | Seed 2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in summaries["base"]["by_task"]:
        values = [summaries[arm]["by_task"][task]["accuracy"] for arm in ARMS]
        lines.append(
            f"| {task} | {values[0]:.1%} | {values[1]:.1%} | "
            f"{values[2]:.1%} | {values[3]:.1%} |"
        )
    overall = [summaries[arm]["accuracy"] for arm in ARMS]
    lines.append(
        f"| **Overall** | **{overall[0]:.1%}** | **{overall[1]:.1%}** | "
        f"**{overall[2]:.1%}** | **{overall[3]:.1%}** |"
    )
    lines.extend(["", "## Paired comparisons versus base", ""])
    for arm, item in comparisons.items():
        low, high = item["paired_bootstrap_95"]
        lines.append(
            f"- `{arm}`: difference {item['arm_minus_base_accuracy']:+.1%}; "
            f"arm-only correct {item['arm_only_correct']}; "
            f"base-only correct {item['base_only_correct']}; "
            f"descriptive paired bootstrap 95% [{low:+.1%}, {high:+.1%}]."
        )
    lines.extend(
        [
            "",
            "This is a deterministic registered 1,000-example subset of the official",
            "prepared prompts, not a claim to reproduce the full MetaTool paper score.",
            "Exact-match parsing, the sample manifest, raw responses, source revision,",
            "and source-file hashes are preserved in the artifact.",
            "",
        ]
    )
    (args.suite / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
