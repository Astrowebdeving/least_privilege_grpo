#!/usr/bin/env python3
"""Build the frozen 150-new + 150-original replay continuation catalog."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OLD_SOURCE = ROOT / "data" / "expansion_v2_microslice.json"
NEW_SOURCE = ROOT / "data" / "continuation_transient_v1.json"
HELDOUT_SOURCE = ROOT / "data" / "continuation_transient_heldout_v1.json"
OUTPUT = ROOT / "data" / "continuation_replay300_v1.json"
MANIFEST = ROOT / "data" / "continuation_replay300_manifest.json"
SEED = "2026-07-24-replay300-v1"


def _bytes_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(*parts: str) -> str:
    return hashlib.sha256("|".join((SEED, *parts)).encode()).hexdigest()


def _quotas(rows: list[dict[str, Any]], target: int) -> dict[str, int]:
    counts = Counter(row["family"] for row in rows)
    total = len(rows)
    raw = {family: count * target / total for family, count in counts.items()}
    quotas = {family: int(value) for family, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda family: (-(raw[family] - quotas[family]), family),
    )
    for family in order[:remaining]:
        quotas[family] += 1
    assert sum(quotas.values()) == target
    assert all(0 < quotas[family] <= counts[family] for family in counts)
    return dict(sorted(quotas.items()))


def _sample_family(
    rows: list[dict[str, Any]],
    family: str,
    count: int,
    source_label: str,
) -> list[dict[str, Any]]:
    members = [row for row in rows if row["family"] == family]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        group = str(row.get("template_group_id") or row["task_id"])
        buckets[group].append(row)
    groups = sorted(
        buckets,
        key=lambda group: _stable_key(source_label, family, "group", group),
    )
    for group in groups:
        buckets[group].sort(
            key=lambda row: _stable_key(source_label, family, "task", row["task_id"])
        )
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        made_progress = False
        for group in groups:
            if depth < len(buckets[group]):
                selected.append(buckets[group][depth])
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise AssertionError((family, count, len(selected)))
        depth += 1
    return selected


def _sample(
    rows: list[dict[str, Any]],
    target: int,
    source_label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = _quotas(rows, target)
    selected = []
    for family, count in quotas.items():
        selected.extend(_sample_family(rows, family, count, source_label))
    selected.sort(key=lambda row: _stable_key(source_label, "output", row["task_id"]))
    return selected, quotas


def build() -> tuple[bytes, bytes]:
    old_all = json.loads(OLD_SOURCE.read_text(encoding="utf-8"))
    old = [row for row in old_all if row.get("split") == "train"]
    new = json.loads(NEW_SOURCE.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_SOURCE.read_text(encoding="utf-8"))
    assert len(old) == 1500
    assert len(new) == 200 and all(row.get("split") == "train" for row in new)
    assert len(heldout) == 50

    old_selected, old_quotas = _sample(old, 150, "original")
    new_selected, new_quotas = _sample(new, 150, "corrective")

    combined = []
    for index in range(150):
        combined.extend((new_selected[index], old_selected[index]))

    ids = [row["task_id"] for row in combined]
    prompts = [row["prompt"] for row in combined]
    heldout_ids = {row["task_id"] for row in heldout}
    assert len(combined) == len(ids) == len(set(ids)) == 300
    assert len(prompts) == len(set(prompts))
    assert not (set(ids) & heldout_ids)
    assert all(row.get("split") == "train" for row in combined)

    old_by_id = {row["task_id"]: row for row in old}
    new_by_id = {row["task_id"]: row for row in new}
    for row in old_selected:
        assert row == old_by_id[row["task_id"]]
    for row in new_selected:
        assert row == new_by_id[row["task_id"]]

    catalog_bytes = (json.dumps(combined, indent=2, ensure_ascii=False) + "\n").encode()
    manifest = {
        "schema": "continuation-replay300-v1",
        "seed": SEED,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "sources": {
            "original": {
                "path": str(OLD_SOURCE.relative_to(ROOT)),
                "sha256": _bytes_sha(OLD_SOURCE),
                "eligible": len(old),
                "selected": len(old_selected),
                "family_quotas": old_quotas,
                "task_ids": [row["task_id"] for row in old_selected],
                "template_groups": len(
                    {row.get("template_group_id") for row in old_selected}
                ),
            },
            "corrective": {
                "path": str(NEW_SOURCE.relative_to(ROOT)),
                "sha256": _bytes_sha(NEW_SOURCE),
                "eligible": len(new),
                "selected": len(new_selected),
                "family_quotas": new_quotas,
                "task_ids": [row["task_id"] for row in new_selected],
                "template_groups": len(
                    {row.get("template_group_id") for row in new_selected}
                ),
            },
            "heldout_exclusion": {
                "path": str(HELDOUT_SOURCE.relative_to(ROOT)),
                "sha256": _bytes_sha(HELDOUT_SOURCE),
                "task_count": len(heldout),
                "overlap": 0,
            },
        },
        "method": (
            "largest-remainder proportional family quotas; deterministic "
            "template-group round robin; rows copied unchanged; sources interleaved"
        ),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return catalog_bytes, manifest_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog, manifest = build()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != catalog:
            raise SystemExit(f"drift: {OUTPUT}")
        if not MANIFEST.is_file() or MANIFEST.read_bytes() != manifest:
            raise SystemExit(f"drift: {MANIFEST}")
        print("replay300 catalog and manifest are reproducible")
        return
    OUTPUT.write_bytes(catalog)
    MANIFEST.write_bytes(manifest)
    print(f"wrote {OUTPUT}: 300 tasks")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
