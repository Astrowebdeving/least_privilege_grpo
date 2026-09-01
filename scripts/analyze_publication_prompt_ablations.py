#!/usr/bin/env python3
"""Analyze the registered publication prompt/norm ablations.

Task-cluster paired bootstrap is the primary uncertainty analysis. An exact
McNemar diagnostic pairs common-random-number generation ordinals within each
task; it is secondary because eight generations from a task are a cluster.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from safety_metrics_report import classify_episode, load_tasks


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_rows(report_path: Path, tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    report = _json(report_path)
    ordinal: dict[str, int] = defaultdict(int)
    rows = []
    for episode in report.get("episodes", []):
        classified = classify_episode(episode, tasks)
        task_id = classified["task_id"]
        rows.append({
            "task_id": task_id,
            "ordinal": ordinal[task_id],
            "safe": bool(classified["safe_success"]),
            "success": bool(classified["success"]),
            "overpriv": bool(classified["over_privileged_success"]),
            "excess": float(episode.get("episode_excess_score") or 0.0),
        })
        ordinal[task_id] += 1
    counts = set(ordinal.values())
    if counts != {8}:
        raise ValueError(f"{report_path}: expected 8 episodes/task, got {sorted(counts)}")
    return rows


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "episodes": n,
        "success": sum(row["success"] for row in rows),
        "safe_success": sum(row["safe"] for row in rows),
        "overprivileged_success": sum(row["overpriv"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / n,
        "safe_success_rate": sum(row["safe"] for row in rows) / n,
        "overprivileged_success_rate": sum(row["overpriv"] for row in rows) / n,
        "excess_mean": statistics.fmean(row["excess"] for row in rows),
    }


def _task_means(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(float(row[field]))
    return {task_id: statistics.fmean(values) for task_id, values in grouped.items()}


def _paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    field: str,
    *,
    samples: int = 50_000,
    seed: int = 20260724,
) -> dict[str, Any]:
    lmean = _task_means(left, field)
    rmean = _task_means(right, field)
    if set(lmean) != set(rmean):
        raise ValueError("paired task membership differs")
    ids = sorted(lmean)
    diffs = [rmean[task_id] - lmean[task_id] for task_id in ids]
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(samples)
    ]
    estimates.sort()
    return {
        "direction": "right_minus_left",
        "task_count": len(ids),
        "estimate": statistics.fmean(diffs),
        "paired_task_bootstrap_95": [
            estimates[int(0.025 * samples)],
            estimates[int(0.975 * samples)],
        ],
    }


def _mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    lmap = {(row["task_id"], row["ordinal"]): row["safe"] for row in left}
    rmap = {(row["task_id"], row["ordinal"]): row["safe"] for row in right}
    if set(lmap) != set(rmap):
        raise ValueError("episode pairing differs")
    left_only = sum(lmap[key] and not rmap[key] for key in lmap)
    right_only = sum(rmap[key] and not lmap[key] for key in lmap)
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(left_only, right_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "left_safe_right_not": left_only,
        "left_not_right_safe": right_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
        "note": "secondary common-random-number diagnostic; task-cluster bootstrap is primary",
    }


def _toolpriv_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in sorted(path.glob("*.json")):
        if item.name == "summary.json":
            continue
        row = _json(item)
        rows[row["scenario_id"]] = row
    return rows


def _toolpriv_analysis(run_root: Path, original_root: Path) -> dict[str, Any]:
    paths = {
        "base_default": original_root / "base_default",
        "base_privilege_aware": original_root / "base_prompt_only",
        "seed1_default": original_root / "seed1_default",
        "seed1_privilege_aware": run_root / "toolpriv" / "seed1_privilege_aware",
    }
    rows = {name: _toolpriv_rows(path) for name, path in paths.items()}
    ids = set(rows["base_default"])
    if len(ids) != 544 or any(set(value) != ids for value in rows.values()):
        raise ValueError("ToolPriv 2x2 scenario membership differs")
    summaries = {name: _json(path / "summary.json") for name, path in paths.items()}
    rates = {
        name: statistics.fmean(
            bool(row.get("over_privileged_selection")) for row in members.values()
        )
        for name, members in rows.items()
    }
    per_scenario = []
    for scenario_id in sorted(ids):
        bd = int(bool(rows["base_default"][scenario_id].get("over_privileged_selection")))
        bp = int(bool(rows["base_privilege_aware"][scenario_id].get("over_privileged_selection")))
        sd = int(bool(rows["seed1_default"][scenario_id].get("over_privileged_selection")))
        sp = int(bool(rows["seed1_privilege_aware"][scenario_id].get("over_privileged_selection")))
        per_scenario.append((sp - sd) - (bp - bd))
    rng = random.Random(20260724)
    estimates = [
        statistics.fmean(
            per_scenario[rng.randrange(len(per_scenario))]
            for _ in per_scenario
        )
        for _ in range(50_000)
    ]
    estimates.sort()
    return {
        "scenario_count": len(ids),
        "overprivileged_selection_rate": rates,
        "prompt_effect_base": rates["base_privilege_aware"] - rates["base_default"],
        "prompt_effect_seed1": rates["seed1_privilege_aware"] - rates["seed1_default"],
        "training_effect_default": rates["seed1_default"] - rates["base_default"],
        "training_effect_privilege_aware": (
            rates["seed1_privilege_aware"] - rates["base_privilege_aware"]
        ),
        "interaction_difference_in_differences": statistics.fmean(per_scenario),
        "interaction_paired_bootstrap_95": [
            estimates[1250],
            estimates[47_500],
        ],
        "summaries": summaries,
    }


def _external_pmin(
    run_root: Path,
    fortis_original: Path,
    metatool_original: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fortis_root = run_root / "external_p_min" / "fortis"
    if (fortis_root / "base" / "summary.json").is_file():
        cells = {
            "base_default": _json(fortis_original / "base_default" / "summary.json"),
            "seed1_default": _json(fortis_original / "seed1_default" / "summary.json"),
            "base_p_min": _json(fortis_root / "base" / "summary.json"),
            "seed1_p_min": _json(fortis_root / "seed1" / "summary.json"),
        }
        effects = {}
        for task in ("task1", "task2"):
            for metric in ("success_rate", "over_privilege_rate"):
                values = {name: cell["by_task"][task][metric] for name, cell in cells.items()}
                effects[f"{task}_{metric}"] = {
                    "values": values,
                    "prompt_effect_base": values["base_p_min"] - values["base_default"],
                    "prompt_effect_seed1": values["seed1_p_min"] - values["seed1_default"],
                    "training_effect_default": values["seed1_default"] - values["base_default"],
                    "training_effect_p_min": values["seed1_p_min"] - values["base_p_min"],
                }
        result["fortis"] = {"cells": cells, "effects": effects}
    metatool_root = run_root / "external_p_min" / "metatool"
    if (metatool_root / "base" / "summary.json").is_file():
        cells = {
            "base_default": _json(metatool_original / "base" / "summary.json"),
            "seed1_default": _json(metatool_original / "seed1" / "summary.json"),
            "base_p_min": _json(metatool_root / "base" / "summary.json"),
            "seed1_p_min": _json(metatool_root / "seed1" / "summary.json"),
        }
        values = {name: cell["accuracy"] for name, cell in cells.items()}
        result["metatool"] = {
            "cells": cells,
            "effects": {
                "accuracy": {
                    "values": values,
                    "prompt_effect_base": values["base_p_min"] - values["base_default"],
                    "prompt_effect_seed1": values["seed1_p_min"] - values["seed1_default"],
                    "training_effect_default": values["seed1_default"] - values["base_default"],
                    "training_effect_p_min": values["seed1_p_min"] - values["base_p_min"],
                }
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--toolpriv-root", type=Path, required=True)
    parser.add_argument(
        "--fortis-root",
        type=Path,
        default=Path("logs/artifacts/fortis_full_2026-07-16"),
    )
    parser.add_argument(
        "--metatool-root",
        type=Path,
        default=Path("logs/artifacts/metatool_subset_full_2026-07-16"),
    )
    parser.add_argument("--tasks", type=Path, default=Path("data/expansion_v2_microslice.json"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    reports: dict[str, dict[str, list[dict[str, Any]]]] = {
        "full": {
            "base": _episode_rows(args.full_root / "base" / "rollout_review.json", tasks),
            "seed-1": _episode_rows(args.full_root / "seed-1" / "rollout_review.json", tasks),
        }
    }
    for variant in ("none", "para", "min"):
        root = args.run_root / "internal" / f"p_{variant}"
        if (root / "safety_metrics.json").is_file():
            reports[variant] = {
                policy: _episode_rows(root / policy / "rollout_review.json", tasks)
                for policy in ("base", "seed-1")
            }

    internal: dict[str, Any] = {
        "variants": {
            variant: {policy: _stats(rows) for policy, rows in policies.items()}
            for variant, policies in reports.items()
        },
        "comparisons_vs_full": {},
    }
    for variant, policies in reports.items():
        if variant == "full":
            continue
        internal["comparisons_vs_full"][variant] = {}
        for policy, rows in policies.items():
            full = reports["full"][policy]
            internal["comparisons_vs_full"][variant][policy] = {
                "safe_success": _paired_bootstrap(full, rows, "safe"),
                "excess": _paired_bootstrap(full, rows, "excess"),
                "mcnemar_safe_success": _mcnemar(full, rows),
            }

    report = {
        "method": {
            "primary_unit": "task",
            "primary_uncertainty": "paired task-cluster bootstrap, 50000 resamples",
            "secondary_test": "exact McNemar over common-random-number task/generation ordinals",
            "full_comparator_caveat": (
                "preserved P_full used the pre-repair outer generation loop; first-terminal "
                "success/safety outcomes remain auditable, but tail-efficiency metrics are not compared"
            ),
        },
        "internal": internal,
        "toolpriv_2x2": _toolpriv_analysis(args.run_root, args.toolpriv_root),
        "external_p_min": _external_pmin(
            args.run_root,
            args.fortis_root,
            args.metatool_root,
        ),
    }
    (args.run_root / "ANALYSIS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Publication prompt/norm ablations",
        "",
        "Primary uncertainty uses the task as the paired cluster. McNemar is a secondary",
        "common-random-number diagnostic because eight generations from one task are dependent.",
        "",
        "## Internal 206-task norm ablation",
        "",
        "| Prompt | Policy | Safe success | Success | Over-privileged success | Mean excess |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant, policies in internal["variants"].items():
        for policy, item in policies.items():
            lines.append(
                f"| P_{variant} | {policy} | {item['safe_success']}/{item['episodes']} "
                f"({item['safe_success_rate']:.1%}) | {item['success']}/{item['episodes']} "
                f"({item['success_rate']:.1%}) | {item['overprivileged_success']}/{item['episodes']} "
                f"({item['overprivileged_success_rate']:.1%}) | {item['excess_mean']:.4f} |"
            )
    lines.extend(["", "## ToolPrivBench 2x2", ""])
    t = report["toolpriv_2x2"]
    for name, rate in t["overprivileged_selection_rate"].items():
        lines.append(f"- {name}: over-privileged selection {rate:.1%}")
    lines.extend([
        f"- Prompt effect on base: {t['prompt_effect_base']:+.1%}",
        f"- Prompt effect on seed-1: {t['prompt_effect_seed1']:+.1%}",
        f"- Training effect under default prompt: {t['training_effect_default']:+.1%}",
        f"- Training effect under privilege-aware prompt: {t['training_effect_privilege_aware']:+.1%}",
        f"- Interaction: {t['interaction_difference_in_differences']:+.1%} "
        f"(paired bootstrap 95% {t['interaction_paired_bootstrap_95'][0]:+.1%} to "
        f"{t['interaction_paired_bootstrap_95'][1]:+.1%})",
        "",
        "Official-prompt FORTIS/MetaTool results remain primary. Their P_min cells are",
        "secondary prompt-interaction controls and are preserved in ANALYSIS.json.",
        "",
    ])
    (args.run_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
