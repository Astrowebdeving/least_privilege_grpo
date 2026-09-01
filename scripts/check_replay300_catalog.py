#!/usr/bin/env python3
"""Fail-closed validation for the 300-task corrective/replay continuation.

This is intentionally a subset gate. The global catalog gate also requires
the complete 2,000-task split/family structure, which a training-only replay
slice cannot and should not satisfy.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.tasks.loader import load_tasks
from least_privilege_env.tasks.validate import validate_task
from scripts.build_continuation_replay300 import (
    HELDOUT_SOURCE,
    MANIFEST,
    NEW_SOURCE,
    OLD_SOURCE,
    OUTPUT,
    build,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_catalog() -> dict[str, object]:
    expected_catalog, expected_manifest = build()
    assert OUTPUT.read_bytes() == expected_catalog, f"catalog drift: {OUTPUT}"
    assert MANIFEST.read_bytes() == expected_manifest, f"manifest drift: {MANIFEST}"

    rows = json.loads(expected_catalog)
    manifest = json.loads(expected_manifest)
    old = {
        row["task_id"]: row
        for row in json.loads(OLD_SOURCE.read_text(encoding="utf-8"))
        if row.get("split") == "train"
    }
    new = {
        row["task_id"]: row
        for row in json.loads(NEW_SOURCE.read_text(encoding="utf-8"))
    }
    heldout = {
        row["task_id"]
        for row in json.loads(HELDOUT_SOURCE.read_text(encoding="utf-8"))
    }

    ids = [row["task_id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    assert len(rows) == len(ids) == len(set(ids)) == 300
    assert len(prompts) == len(set(prompts)) == 300
    assert set(ids).isdisjoint(heldout)
    assert all(row.get("split") == "train" for row in rows)
    assert all(row == (new.get(row["task_id"]) or old[row["task_id"]]) for row in rows)

    # Keep sources interleaved even if the trainer consumes catalog order.
    assert all(rows[i]["task_id"] in new for i in range(0, 300, 2))
    assert all(rows[i]["task_id"] in old for i in range(1, 300, 2))

    old_ids = [task_id for task_id in ids if task_id in old]
    new_ids = [task_id for task_id in ids if task_id in new]
    assert len(old_ids) == len(new_ids) == 150
    old_counts = Counter(old[task_id]["family"] for task_id in old_ids)
    new_counts = Counter(new[task_id]["family"] for task_id in new_ids)
    assert dict(sorted(old_counts.items())) == manifest["sources"]["original"]["family_quotas"]
    assert dict(sorted(new_counts.items())) == manifest["sources"]["corrective"]["family_quotas"]
    assert set(old_counts) == {row["family"] for row in old.values()}
    assert set(new_counts) == {row["family"] for row in new.values()}

    assert manifest["output_sha256"] == hashlib.sha256(expected_catalog).hexdigest()
    for label, path in (("original", OLD_SOURCE), ("corrective", NEW_SOURCE)):
        assert manifest["sources"][label]["sha256"] == _sha256(path)
    assert manifest["sources"]["heldout_exclusion"]["sha256"] == _sha256(HELDOUT_SOURCE)
    assert manifest["sources"]["heldout_exclusion"]["overlap"] == 0

    errors: list[str] = []
    warnings: list[str] = []
    tasks = load_tasks(OUTPUT)
    assert len(tasks) == 300
    for task in tasks:
        report = validate_task(task)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
    assert not errors, "\n".join(errors)

    return {
        "status": "pass",
        "tasks": 300,
        "corrective": 150,
        "original_replay": 150,
        "heldout_overlap": 0,
        "schema_or_envelope_errors": 0,
        "warnings": len(warnings),
        "corrective_families": dict(sorted(new_counts.items())),
        "original_families": dict(sorted(old_counts.items())),
        "catalog_sha256": _sha256(OUTPUT),
        "manifest_sha256": _sha256(MANIFEST),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check_catalog()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
