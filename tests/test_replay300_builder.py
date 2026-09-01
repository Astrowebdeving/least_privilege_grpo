from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.build_continuation_replay300 import build, MANIFEST, OUTPUT, ROOT
from scripts.check_replay300_catalog import check_catalog


def test_replay300_is_reproducible_and_source_exact() -> None:
    catalog_bytes, manifest_bytes = build()
    assert OUTPUT.read_bytes() == catalog_bytes
    assert MANIFEST.read_bytes() == manifest_bytes

    rows = json.loads(catalog_bytes)
    manifest = json.loads(manifest_bytes)
    old = {
        row["task_id"]: row
        for row in json.loads((ROOT / "data/expansion_v2_microslice.json").read_text())
        if row["split"] == "train"
    }
    new = {
        row["task_id"]: row
        for row in json.loads((ROOT / "data/continuation_transient_v1.json").read_text())
    }
    heldout = {
        row["task_id"]
        for row in json.loads(
            (ROOT / "data/continuation_transient_heldout_v1.json").read_text()
        )
    }

    ids = [row["task_id"] for row in rows]
    assert len(rows) == len(set(ids)) == 300
    assert not (set(ids) & heldout)
    assert all(row == (old.get(row["task_id"]) or new[row["task_id"]]) for row in rows)
    assert sum(task_id in old for task_id in ids) == 150
    assert sum(task_id in new for task_id in ids) == 150
    assert all(row["split"] == "train" for row in rows)
    assert len({row["prompt"] for row in rows}) == 300

    old_counts = Counter(old[task_id]["family"] for task_id in ids if task_id in old)
    new_counts = Counter(new[task_id]["family"] for task_id in ids if task_id in new)
    assert dict(sorted(old_counts.items())) == manifest["sources"]["original"]["family_quotas"]
    assert dict(sorted(new_counts.items())) == manifest["sources"]["corrective"]["family_quotas"]


def test_replay300_subset_gate_passes() -> None:
    result = check_catalog()
    assert result["status"] == "pass"
    assert result["corrective"] == result["original_replay"] == 150
    assert result["heldout_overlap"] == 0
