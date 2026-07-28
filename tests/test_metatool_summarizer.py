from __future__ import annotations

import json

from scripts.summarize_metatool_subset import _load_rows, _paired


def test_metatool_loader_rejects_no_registered_fields(tmp_path) -> None:
    task = tmp_path / "awareness"
    task.mkdir()
    (task / "000001.json").write_text(
        json.dumps({"task": "awareness", "source_index": 1, "correct": True}),
        encoding="utf-8",
    )
    assert list(_load_rows(tmp_path)) == [("awareness", 1)]


def test_metatool_pairing_reports_direction() -> None:
    base = {
        ("awareness", 1): {"correct": True},
        ("awareness", 2): {"correct": False},
    }
    other = {
        ("awareness", 1): {"correct": False},
        ("awareness", 2): {"correct": False},
    }
    result = _paired(base, other, "seed0")
    assert result["arm_minus_base_accuracy"] == -0.5
    assert result["base_only_correct"] == 1
    assert result["arm_only_correct"] == 0
