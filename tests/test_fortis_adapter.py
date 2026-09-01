from __future__ import annotations

import json

from scripts.evaluate_fortis_api import _metrics, _parse_csv
from scripts.summarize_fortis_suite import _load_rows


def test_fortis_selection_parser_rejects_unknown_values() -> None:
    assert _parse_csv("task1,task2", ("task1", "task2")) == ("task1", "task2")


def test_fortis_metrics_preserve_four_way_counts() -> None:
    rows = [
        {"classification": name, "parsed": name != "no_action", "usage": {}}
        for name in ("exact_match", "under_privilege", "over_privilege", "no_action")
    ]
    metrics = _metrics(rows)
    assert metrics["total"] == 4
    assert metrics["success_rate"] == 0.25
    assert metrics["safe_rate"] == 0.5
    assert metrics["over_privilege_rate"] == 0.25
    assert metrics["fail_rate"] == 0.5


def test_fortis_loader_keys_same_ids_by_domain(tmp_path) -> None:
    for domain in ("email", "filesystem"):
        target = tmp_path / "task1" / domain
        target.mkdir(parents=True)
        (target / "same_id.json").write_text(
            json.dumps({"domain": domain, "id": "same_id"}),
            encoding="utf-8",
        )
    assert set(_load_rows(tmp_path, "task1")) == {
        "email:same_id",
        "filesystem:same_id",
    }
