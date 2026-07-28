#!/usr/bin/env python3
"""Build/check the registered generation tiers for the frozen v2 validation set."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

CATALOG = Path("data/expansion_v2_microslice.json")
IDS_FILE = Path("data/validation_task_ids_v2.txt")
FREEZE = Path("data/validation_frozen_2000_2026-06-30.json")
OUTPUT = Path("data/validation_generation_tiers_v2.json")

# Registered publication-eval cost rule: rare safety/interface behaviors receive
# eight samples; ordinary held-out tasks receive four. Keep this list explicit so
# future family additions cannot silently inherit a tier.
SAFETY_HEAVY_FAMILIES = frozenset(
    {
        "cross_surface_lure",
        "data_minimization",
        "honeypot_restraint",
        "mcp_stub",
        "multi_hop_injection",
        "multi_tool_workflow",
        "plausible_lure",
        "prompt_injection",
        "secret_lure",
        "unseen_mcp_server",
    }
)
EXPECTED_COUNTS = {"ordinary": 276, "safety_heavy": 224}
GENERATIONS = {"ordinary": 4, "safety_heavy": 8}
EXPECTED_EPISODES = 2896


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    catalog_path: Path = CATALOG,
    ids_path: Path = IDS_FILE,
    freeze_path: Path = FREEZE,
) -> dict[str, Any]:
    freeze = load_json(freeze_path)
    catalog = load_json(catalog_path)
    if not isinstance(catalog, list):
        raise ValueError("v2 catalog must be a JSON list")

    expected_digests = freeze.get("file_digests", {})
    for path in (catalog_path, ids_path):
        key = path.as_posix()
        expected = expected_digests.get(key)
        actual = sha256(path)
        if expected is None:
            raise ValueError(f"freeze has no digest for {key}")
        if actual != expected:
            raise ValueError(f"frozen file drift: {key} sha256={actual} expected={expected}")

    ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frozen_ids = freeze.get("validation_task_ids") or []
    if ids != frozen_ids:
        raise ValueError("validation ID file does not exactly match frozen ordered membership")
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError(f"expected 500 unique validation IDs, got {len(ids)} / {len(set(ids))} unique")

    tasks_by_id = {task["task_id"]: task for task in catalog}
    if len(tasks_by_id) != len(catalog):
        raise ValueError("catalog has duplicate task IDs")
    missing = [task_id for task_id in ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"validation IDs missing from catalog: {missing[:10]}")

    known_families = {str(task.get("family")) for task in catalog}
    unknown_registered = sorted(SAFETY_HEAVY_FAMILIES - known_families)
    if unknown_registered:
        raise ValueError(f"registered safety-heavy families absent from catalog: {unknown_registered}")

    tiers: dict[str, list[str]] = {"ordinary": [], "safety_heavy": []}
    tier_family_counts: dict[str, Counter[str]] = {
        "ordinary": Counter(),
        "safety_heavy": Counter(),
    }
    for task_id in ids:
        family = str(tasks_by_id[task_id].get("family"))
        tier = "safety_heavy" if family in SAFETY_HEAVY_FAMILIES else "ordinary"
        tiers[tier].append(task_id)
        tier_family_counts[tier][family] += 1

    actual_counts = {tier: len(values) for tier, values in tiers.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"generation-tier drift: {actual_counts} != {EXPECTED_COUNTS}")

    expected_episodes = sum(len(tiers[tier]) * GENERATIONS[tier] for tier in tiers)
    if expected_episodes != EXPECTED_EPISODES:
        raise ValueError(f"episode-count drift: {expected_episodes} != {EXPECTED_EPISODES}")

    assignment_lines = [
        f"{task_id}\t{'safety_heavy' if tasks_by_id[task_id]['family'] in SAFETY_HEAVY_FAMILIES else 'ordinary'}\t"
        f"{GENERATIONS['safety_heavy'] if tasks_by_id[task_id]['family'] in SAFETY_HEAVY_FAMILIES else GENERATIONS['ordinary']}"
        for task_id in ids
    ]
    assignment_sha = hashlib.sha256(("\n".join(assignment_lines) + "\n").encode("utf-8")).hexdigest()

    return {
        "schema": "lp-rl/validation-generation-tiers/1",
        "version": "v2.0",
        "description": "Registered 4/8-generation cost tiers for the frozen 500-task publication evaluation.",
        "source": {
            "catalog": catalog_path.as_posix(),
            "catalog_sha256": sha256(catalog_path),
            "task_ids": ids_path.as_posix(),
            "task_ids_sha256": sha256(ids_path),
            "freeze": freeze_path.as_posix(),
            "freeze_sha256": sha256(freeze_path),
            "validation_combined_sha256": freeze.get("validation_combined_sha256"),
        },
        "rules": {
            "ordinary_generations": GENERATIONS["ordinary"],
            "safety_heavy_generations": GENERATIONS["safety_heavy"],
            "safety_heavy_families": sorted(SAFETY_HEAVY_FAMILIES),
        },
        "validation_task_count": len(ids),
        "expected_episodes_per_policy": expected_episodes,
        "assignment_sha256": assignment_sha,
        "tiers": {
            tier: {
                "generations": GENERATIONS[tier],
                "task_count": len(tiers[tier]),
                "expected_episodes": len(tiers[tier]) * GENERATIONS[tier],
                "family_counts": dict(sorted(tier_family_counts[tier].items())),
                "task_ids": tiers[tier],
            }
            for tier in ("ordinary", "safety_heavy")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-csv", choices=("ordinary", "safety_heavy"))
    args = parser.parse_args()

    manifest = build_manifest()
    if args.print_csv:
        print(",".join(manifest["tiers"][args.print_csv]["task_ids"]))
        return 0
    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"missing tier manifest: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"tier manifest drift: rerun {Path(__file__).as_posix()}")
        print(
            f"validation tiers check passed: 500 tasks, ordinary={EXPECTED_COUNTS['ordinary']}x4, "
            f"safety_heavy={EXPECTED_COUNTS['safety_heavy']}x8, episodes={EXPECTED_EPISODES}"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}: {EXPECTED_EPISODES} episodes/policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
