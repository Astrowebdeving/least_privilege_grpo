#!/usr/bin/env python3
"""Build the repaired 400-step corrective + variance-aware replay catalog.

The historical v1 mixed catalog remains untouched. V2 uses:
  * all 200 versioned corrective-v3 tasks;
  * 200 exact rows from the original 1,500-task train split;
  * the same damage-weighted family allocation as v1;
  * deterministic priority for rows with demonstrated seed-1 reward variance;
  * family-balanced ordering inside each source, then strict C/R alternation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORRECTIVE = ROOT / "data" / "continuation_transient_v3.json"
CATALOG = ROOT / "data" / "expansion_v2_microslice.json"
HELDOUT = ROOT / "data" / "continuation_transient_heldout_v3.json"
CONFIRMATORY = ROOT / "data" / "continuation_confirmatory_v1.json"
VARIANCE = ROOT / "data" / "continuation_replay_variance_seed1.json"
FROZEN = ROOT / "data" / "validation_frozen_2000_2026-06-30.json"
OUT = ROOT / "data" / "continuation400_v2.json"
MANIFEST = ROOT / "data" / "continuation400_v2_manifest.json"

SEED = "2026-07-31-continuation400-v2"

REPLAY_WEIGHTS = {
    "run_the_right_test": 30,
    "escalation_required": 30,
    "noisy_error_recovery": 25,
    "patch_proposal_only": 25,
    "scoped_cleanup": 20,
    "honeypot_restraint": 15,
    "multi_tool_workflow": 15,
    "mcp_stub": 12,
    "prompt_injection": 10,
    "scoped_write": 10,
    "structured_output": 8,
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def balanced_family_order(tasks: list[dict]) -> list[dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        pools[task["family"]].append(task)
    for family in pools:
        pools[family].sort(key=lambda task: stable_rank(task["task_id"]))

    totals = {family: len(pool) for family, pool in pools.items()}
    used = Counter()
    out: list[dict] = []
    total = len(tasks)
    for position in range(1, total + 1):
        available = [family for family, pool in pools.items() if pool]
        family = max(
            available,
            key=lambda name: (
                position * totals[name] / total - used[name],
                stable_rank(f"{position}:{name}"),
            ),
        )
        out.append(pools[family].pop())
        used[family] += 1
    return out


def interleave(corrective: list[dict], replay: list[dict]) -> list[dict]:
    if len(corrective) != len(replay):
        raise AssertionError("v2 requires equal corrective/replay halves")
    out: list[dict] = []
    for left, right in zip(corrective, replay):
        out.extend((left, right))
    return out


def build() -> tuple[list[dict], dict]:
    corrective = load(CORRECTIVE)
    catalog = load(CATALOG)
    heldout = load(HELDOUT)
    confirmatory = load(CONFIRMATORY)
    variance = load(VARIANCE)
    frozen = load(FROZEN)

    if len(corrective) != 200:
        raise SystemExit(f"expected 200 corrective tasks, got {len(corrective)}")
    if variance.get("tasks") != 1500 or variance.get("generations_per_task") != 8:
        raise SystemExit("unexpected replay-variance artifact dimensions")

    excluded_ids = set(frozen["validation_task_ids"])
    excluded_ids.update(task["task_id"] for task in heldout)
    excluded_ids.update(task["task_id"] for task in confirmatory)
    excluded_prompts = {task["prompt"] for task in heldout + confirmatory}
    excluded_groups = {
        task.get("template_group_id")
        for task in heldout + confirmatory
        if task.get("template_group_id")
    }
    corrective_prompts = {task["prompt"] for task in corrective}

    score_map = {item["task_id"]: item for item in variance["scores"]}
    train = [
        task
        for task in catalog
        if task.get("split") == "train"
        and task["task_id"] not in excluded_ids
        and task["prompt"] not in excluded_prompts
        and task["prompt"] not in corrective_prompts
        and task.get("template_group_id") not in excluded_groups
    ]
    by_family: dict[str, list[dict]] = defaultdict(list)
    for task in train:
        if task["task_id"] not in score_map:
            raise SystemExit(f"missing replay variance score: {task['task_id']}")
        by_family[task["family"]].append(task)

    replay: list[dict] = []
    selected_stats: dict[str, dict] = {}
    for family, want in sorted(REPLAY_WEIGHTS.items()):
        pool = by_family.get(family, [])
        if len(pool) < want:
            raise SystemExit(f"{family}: need {want} tasks, only {len(pool)} available")
        pool.sort(
            key=lambda task: (
                -score_map[task["task_id"]]["reward_std"],
                -(score_map[task["task_id"]]["reward_max"] - score_map[task["task_id"]]["reward_min"]),
                stable_rank(task["task_id"]),
            )
        )
        chosen = pool[:want]
        replay.extend(chosen)
        selected_stats[family] = {
            "slots": want,
            "historical_nonzero_variance": sum(
                score_map[task["task_id"]]["nonzero_variance"] for task in chosen
            ),
            "historical_std_gt_0_1": sum(
                score_map[task["task_id"]]["std_gt_0_1"] for task in chosen
            ),
            "reward_std_min": min(score_map[task["task_id"]]["reward_std"] for task in chosen),
            "reward_std_max": max(score_map[task["task_id"]]["reward_std"] for task in chosen),
        }

    corrective_order = balanced_family_order(corrective)
    replay_order = balanced_family_order(replay)
    tasks = interleave(corrective_order, replay_order)

    manifest = {
        "schema": "continuation400/v2",
        "seed": SEED,
        "composition": {"corrective": len(corrective), "replay": len(replay)},
        "sources": {
            "corrective": str(CORRECTIVE.relative_to(ROOT)),
            "catalog": str(CATALOG.relative_to(ROOT)),
            "heldout_development": str(HELDOUT.relative_to(ROOT)),
            "confirmatory_lockbox": str(CONFIRMATORY.relative_to(ROOT)),
            "replay_variance": str(VARIANCE.relative_to(ROOT)),
        },
        "replay_weights": REPLAY_WEIGHTS,
        "selected_replay_signal": selected_stats,
        "selected_replay_totals": {
            "tasks": len(replay),
            "historical_nonzero_variance": sum(
                score_map[task["task_id"]]["nonzero_variance"] for task in replay
            ),
            "historical_std_gt_0_1": sum(
                score_map[task["task_id"]]["std_gt_0_1"] for task in replay
            ),
        },
        "selection_rule": (
            "within each damage-weighted family, rank original train rows by "
            "historical seed-1 reward std, then reward range, then pinned hash"
        ),
        "endpoint": {
            "training_steps": 400,
            "checkpoint_selection": "none; step 400 is the fixed endpoint",
            "intermediate_checkpoints": "diagnostic and safety-stop only",
        },
    }
    text = json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
    manifest["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return tasks, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    tasks, manifest = build()
    text = json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("continuation400_v2 differs from a fresh build")
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != manifest_text:
            raise SystemExit("continuation400_v2 manifest differs from a fresh build")
        print(
            "continuation400 v2 current:",
            len(tasks),
            manifest["output_sha256"],
            manifest["selected_replay_totals"],
        )
        return 0

    OUT.write_text(text, encoding="utf-8")
    MANIFEST.write_text(manifest_text, encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)} tasks={len(tasks)} "
        f"sha={manifest['output_sha256']}"
    )
    print("replay signal:", manifest["selected_replay_totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
