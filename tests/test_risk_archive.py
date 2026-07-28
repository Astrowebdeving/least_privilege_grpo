from __future__ import annotations

import copy
import json
from pathlib import Path

from least_privilege_env.tasks.risk_archive import (
    canonical_task_hash,
    load_archive,
    load_raw_tasks,
    reward_context,
    validate_archive,
    ARCHIVE_VERSION,
)
from scripts.build_risk_archive import build_entry


def test_archive_matches_catalog() -> None:
    report = validate_archive()
    assert report.ok, "\n".join(report.errors)


def test_every_task_has_entry_and_no_orphans() -> None:
    archive = load_archive()
    catalog_ids = {raw["task_id"] for raw in load_raw_tasks()}
    archive_ids = {entry["task_id"] for entry in archive["entries"]}
    assert archive_ids == catalog_ids


def test_required_risk_divergence_fails() -> None:
    archive = copy.deepcopy(load_archive())
    entry = archive["entries"][0]
    entry["required_risk"]["write"] = 0.9
    report = validate_archive(archive)
    assert not report.ok
    assert any("required_risk diverges" in error for error in report.errors)


def test_label_confidence_divergence_fails() -> None:
    archive = copy.deepcopy(load_archive())
    archive["entries"][0]["label_confidence"] = 0.123
    report = validate_archive(archive)
    assert not report.ok
    assert any("label_confidence diverges" in error for error in report.errors)


def test_task_edit_invalidates_hash() -> None:
    raw_tasks = copy.deepcopy(load_raw_tasks())
    raw_tasks[0]["prompt"] += " (edited)"
    report = validate_archive(load_archive(), raw_tasks)
    assert not report.ok
    assert any("task_hash stale" in error for error in report.errors)


def test_missing_entry_fails() -> None:
    archive = copy.deepcopy(load_archive())
    removed = archive["entries"].pop(0)
    report = validate_archive(archive)
    assert not report.ok
    assert any(f"{removed['task_id']}: missing archive entry" in error for error in report.errors)


def test_reward_context_drift_fails() -> None:
    archive = copy.deepcopy(load_archive())
    archive["reward_context"]["reward_weights"]["blocked"] = -0.5
    report = validate_archive(archive)
    assert not report.ok
    assert any("reward_context drift" in error for error in report.errors)


def test_rationale_must_cover_exactly_nonzero_dims() -> None:
    archive = copy.deepcopy(load_archive())
    entry = archive["entries"][0]
    entry["risk_rationale"]["external"] = "stale text for a zero dimension"
    report = validate_archive(archive)
    assert not report.ok
    assert any("zero dimension 'external'" in error for error in report.errors)


def test_oracle_classifier_drift_fails() -> None:
    archive = copy.deepcopy(load_archive())
    entry = next(e for e in archive["entries"] if e["oracle_command_risks"])
    entry["oracle_command_risks"][0]["z_pre"]["scope"] = 0.99
    report = validate_archive(archive)
    assert not report.ok
    assert any("oracle_command_risks drift" in error for error in report.errors)


def test_task_hash_is_order_insensitive() -> None:
    raw = load_raw_tasks()[0]
    reordered = dict(reversed(list(raw.items())))
    assert canonical_task_hash(raw) == canonical_task_hash(reordered)


def test_reward_context_pins_live_constants() -> None:
    context = reward_context()
    assert context["reward_weights"]["blocked"] == -0.75
    assert context["risk_weights"] == [0.22, 0.20, 0.18, 0.16, 0.14, 0.10]
    assert context["lambda_step"] == 0.15
    assert context["secret_excess_lambda"] == 1.0
    assert context["mcp_contracts_version"] == 2
    assert context["evidence_empty_requirement"].startswith("success_gated")


def test_oracle_mcp_risk_drift_fails() -> None:
    archive = copy.deepcopy(load_archive())
    entry = next(e for e in archive["entries"] if e.get("oracle_mcp_risks"))
    entry["oracle_mcp_risks"][0]["z_pre"]["external"] = 0.01
    report = validate_archive(archive)
    assert not report.ok
    assert any("oracle_mcp_risks drift" in error for error in report.errors)



def test_v2_family_templates_cover_working_catalog() -> None:
    raw_tasks = load_raw_tasks(Path("data/expansion_v2_microslice.json"))
    v2_tasks = [raw for raw in raw_tasks if "expansion_v2" in raw.get("command_pattern_tags", ())]
    assert len(v2_tasks) == 1740
    entries = [build_entry(raw) for raw in v2_tasks]
    assert len(entries) == 1740


def test_v2_archive_can_merge_additively(tmp_path, monkeypatch) -> None:
    # Isolate from an ambient campaign env (the run guide exports both); this test
    # controls its own archive/catalog via the explicit paths + setenv below.
    monkeypatch.delenv("LP_RL_RISK_ARCHIVE", raising=False)
    monkeypatch.delenv("LP_RL_TASK_FILE", raising=False)
    v2_raw = next(
        raw
        for raw in load_raw_tasks(Path("data/expansion_v2_microslice.json"))
        if "expansion_v2" in raw.get("command_pattern_tags", ())
    )
    extra = {
        "archive_version": ARCHIVE_VERSION,
        "generated": "test",
        "catalog_path": "data/expansion_v2_microslice.json",
        "catalog_filter": "expansion_v2",
        "purpose": "test additive archive",
        "reward_context": reward_context(),
        "entries": [build_entry(v2_raw)],
    }
    extra_path = tmp_path / "risk_archive_v2.json"
    extra_path.write_text(json.dumps(extra), encoding="utf-8")

    monkeypatch.setenv("LP_RL_RISK_ARCHIVE", str(extra_path))
    merged = load_archive()
    assert any(entry["task_id"] == v2_raw["task_id"] for entry in merged["entries"])

    report = validate_archive(merged, load_raw_tasks() + [v2_raw])
    assert report.ok, "\n".join(report.errors)
