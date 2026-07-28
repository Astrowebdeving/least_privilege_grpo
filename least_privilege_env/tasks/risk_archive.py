"""Host-side required-risk archive: load, hash, and validate label governance.

The archive (``data/risk_archive.json``) is a review artifact, never mounted
into any workspace and never model-facing. It records, per task, why the
required-risk label is what it is: provenance, the oracle path with the
classifier pre-risk it was judged against, rejected lower-risk alternatives,
known higher-risk lures, forbidden shortcuts, and reviewer notes. Validation
fails loudly when the catalog, classifier, or reward semantics drift from
what a label decision was made against.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from least_privilege_env.broker.classifier import classify_program
from least_privilege_env.broker.parser import ParseError, parse_command
from least_privilege_env.broker.risk import DEFAULT_WEIGHTS, RISK_DIMS
from least_privilege_env.mcp.contracts import MCP_CONTRACTS_VERSION, classify_mcp_call
from least_privilege_env.reward.scoring import (
    LAMBDA_STEP,
    REWARD_WEIGHTS,
    SECRET_EXCESS_LAMBDA,
    T0_DEFAULT,
)

ARCHIVE_VERSION = 1
DEFAULT_ARCHIVE_FILE = Path(__file__).resolve().parents[2] / "data" / "risk_archive.json"
DEFAULT_CATALOG_FILE = Path(__file__).resolve().parents[2] / "data" / "spike_tasks.json"
TASK_FILE_ENV = "LP_RL_TASK_FILE"
RISK_ARCHIVE_ENV = "LP_RL_RISK_ARCHIVE"

VALID_PROVENANCE_SOURCES = {
    "oracle",
    "oracle_family_template",
    "mined",
    "archive_refined",
    "manual",
}
VALID_ROLLOUT_STATUS = {"none_yet", "consistent", "too_strict", "too_loose", "underspecified"}

REQUIRED_ENTRY_KEYS = (
    "task_id",
    "task_hash",
    "family",
    "split",
    "required_risk",
    "label_confidence",
    "provenance",
    "oracle_summary",
    "oracle_command_risks",
    "oracle_mcp_risks",
    "risk_rationale",
    "lower_risk_alternatives",
    "higher_risk_lures",
    "forbidden_shortcuts",
    "evidence_rationale",
    "rollout_feedback",
    "reviewer_notes",
)


def canonical_task_hash(raw_task: dict[str, Any]) -> str:
    """Stable content hash of one raw catalog mapping (order-insensitive)."""
    canonical = json.dumps(raw_task, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def reward_context() -> dict[str, Any]:
    """Snapshot of the reward/risk semantics a label decision is made against."""
    return {
        "reward_weights": dict(REWARD_WEIGHTS),
        "risk_dimension_order": list(RISK_DIMS),
        "risk_weights": list(DEFAULT_WEIGHTS),
        "lambda_step": LAMBDA_STEP,
        "secret_excess_lambda": SECRET_EXCESS_LAMBDA,
        "t0_default": T0_DEFAULT,
        "t0_at_reward_time": "task.max_steps (total_reward passes task.max_steps, not T0_DEFAULT)",
        "evidence_empty_requirement": (
            "success_gated (empty required_evidence credits E only on verified episodes; "
            "calibration 2026-06-11)"
        ),
        "mcp_contracts_version": MCP_CONTRACTS_VERSION,
    }


def _resolve_catalog_file(path: Path | None) -> Path:
    if path is not None:
        return path
    override = os.environ.get(TASK_FILE_ENV, "").strip()
    return Path(override) if override else DEFAULT_CATALOG_FILE


def load_raw_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    path = _resolve_catalog_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_archive_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_archives(base: dict[str, Any], extra: dict[str, Any], extra_path: Path) -> dict[str, Any]:
    if extra.get("archive_version") != base.get("archive_version"):
        raise ValueError(
            f"{extra_path}: archive_version {extra.get('archive_version')!r} "
            f"!= base {base.get('archive_version')!r}"
        )
    if extra.get("reward_context") != base.get("reward_context"):
        raise ValueError(f"{extra_path}: reward_context drift versus base archive")

    merged = dict(base)
    entries: dict[str, dict[str, Any]] = {}
    for entry in base.get("entries", []):
        entries[entry["task_id"]] = entry
    for entry in extra.get("entries", []):
        task_id = entry["task_id"]
        if task_id in entries:
            raise ValueError(f"{extra_path}: duplicate merged archive entry {task_id!r}")
        entries[task_id] = entry
    merged["entries"] = list(entries.values())
    merged["catalog_path"] = [base.get("catalog_path"), extra.get("catalog_path")]
    merged["archive_sources"] = [str(DEFAULT_ARCHIVE_FILE), str(extra_path)]
    return merged


def load_archive(
    path: Path | None = None,
    *,
    extra_path: Path | None = None,
    merge_env: bool = True,
) -> dict[str, Any]:
    path = path or DEFAULT_ARCHIVE_FILE
    archive = _read_archive_file(path)
    override = extra_path
    if override is None and merge_env:
        raw_override = os.environ.get(RISK_ARCHIVE_ENV, "").strip()
        override = Path(raw_override) if raw_override else None
    if override is not None:
        if override.resolve() == path.resolve():
            return archive
        archive = _merge_archives(archive, _read_archive_file(override), override)
    return archive


def oracle_command_risks(raw_task: dict[str, Any]) -> list[dict[str, Any]]:
    """Classifier pre-risk per oracle command, as stored alongside each label."""
    out: list[dict[str, Any]] = []
    for command in (raw_task.get("oracle") or {}).get("commands", ()):
        try:
            classification = classify_program(parse_command(command))
        except ParseError as exc:
            out.append({"command": command, "error": f"parse failed: {exc}"})
            continue
        out.append({"command": command, "z_pre": classification.z_pre.to_dict()})
    return out


def oracle_mcp_risks(raw_task: dict[str, Any]) -> list[dict[str, Any]]:
    """Contract pre-risk per oracle MCP call: contract drift stales the label."""
    out: list[dict[str, Any]] = []
    for call in (raw_task.get("oracle") or {}).get("mcp_calls", ()):
        classification = classify_mcp_call(call["server"], call["tool"], call.get("args", {}))
        entry: dict[str, Any] = {
            "server": call["server"],
            "tool": call["tool"],
            "args": call.get("args", {}),
            "z_pre": classification.z_pre.to_dict(),
        }
        if classification.blocked:
            entry["error"] = f"blocked by contract: {classification.block_reason}"
        out.append(entry)
    return out


def _full_risk_dict(value: dict[str, Any] | None) -> dict[str, float]:
    value = value or {}
    return {dim: float(value.get(dim, 0.0)) for dim in RISK_DIMS}


@dataclass
class ArchiveReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_archive(
    archive: dict[str, Any] | None = None,
    raw_tasks: list[dict[str, Any]] | None = None,
) -> ArchiveReport:
    report = ArchiveReport()
    if archive is None:
        try:
            archive = load_archive()
        except FileNotFoundError:
            report.errors.append(f"archive file missing: {DEFAULT_ARCHIVE_FILE}")
            return report
        except json.JSONDecodeError as exc:
            report.errors.append(f"archive is not valid JSON: {exc}")
            return report
        except ValueError as exc:
            report.errors.append(str(exc))
            return report
    raw_tasks = raw_tasks if raw_tasks is not None else load_raw_tasks()

    if archive.get("archive_version") != ARCHIVE_VERSION:
        report.errors.append(
            f"archive_version {archive.get('archive_version')!r} != supported {ARCHIVE_VERSION}"
        )

    expected_context = reward_context()
    stored_context = archive.get("reward_context")
    if stored_context != expected_context:
        report.errors.append(
            "reward_context drift: archive was written against different reward/risk "
            f"semantics; stored={stored_context!r} current={expected_context!r}. "
            "Re-review labels, then rebuild the archive."
        )

    entries = archive.get("entries", [])
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task_id = entry.get("task_id", "<missing task_id>")
        if task_id in by_id:
            report.errors.append(f"duplicate archive entry: {task_id}")
        by_id[task_id] = entry

    catalog_ids = {raw["task_id"] for raw in raw_tasks}
    for orphan in sorted(set(by_id) - catalog_ids):
        report.errors.append(f"archive entry has no catalog task: {orphan}")

    for raw in raw_tasks:
        task_id = raw["task_id"]
        entry = by_id.get(task_id)
        if entry is None:
            report.errors.append(f"{task_id}: missing archive entry")
            continue
        report.errors.extend(_validate_entry(raw, entry))
    return report


def _validate_entry(raw: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    task_id = raw["task_id"]
    errors: list[str] = []

    for key in REQUIRED_ENTRY_KEYS:
        if key not in entry:
            errors.append(f"{task_id}: archive entry missing key {key!r}")
    if errors:
        return errors

    expected_hash = canonical_task_hash(raw)
    if entry["task_hash"] != expected_hash:
        errors.append(
            f"{task_id}: task_hash stale (archive {entry['task_hash']}, catalog {expected_hash}); "
            "the task changed since the label was reviewed"
        )

    if entry["family"] != raw["family"]:
        errors.append(f"{task_id}: family mismatch (archive {entry['family']!r}, catalog {raw['family']!r})")
    if entry["split"] != raw.get("split", "train"):
        errors.append(f"{task_id}: split mismatch")

    expected_risk = _full_risk_dict(raw.get("required_risk"))
    archived_risk = _full_risk_dict(entry.get("required_risk"))
    if archived_risk != expected_risk:
        errors.append(
            f"{task_id}: required_risk diverges (archive {archived_risk}, catalog {expected_risk})"
        )

    if float(entry["label_confidence"]) != float(raw.get("label_confidence", 1.0)):
        errors.append(f"{task_id}: label_confidence diverges")

    provenance = entry.get("provenance") or {}
    if provenance.get("source") not in VALID_PROVENANCE_SOURCES:
        errors.append(f"{task_id}: provenance.source {provenance.get('source')!r} not in {sorted(VALID_PROVENANCE_SOURCES)}")
    for key in ("produced_by", "reviewed_by", "date"):
        if not provenance.get(key):
            errors.append(f"{task_id}: provenance.{key} is empty")

    # Every nonzero required dimension must be justified; a rationale for a
    # zero dimension marks a stale or copy-pasted justification.
    rationale = entry.get("risk_rationale") or {}
    nonzero_dims = {dim for dim, value in expected_risk.items() if value > 0}
    for dim in sorted(nonzero_dims - set(rationale)):
        errors.append(f"{task_id}: nonzero dimension {dim!r} has no risk_rationale")
    for dim in sorted(set(rationale) - nonzero_dims):
        errors.append(f"{task_id}: risk_rationale for zero dimension {dim!r} (stale label?)")
    for dim, text in rationale.items():
        if not str(text).strip():
            errors.append(f"{task_id}: empty risk_rationale for {dim!r}")

    # Oracle path: stored commands and classifier pre-risk must match a live
    # recomputation, so classifier drift invalidates the label review.
    expected_oracle = oracle_command_risks(raw)
    if entry["oracle_command_risks"] != expected_oracle:
        errors.append(
            f"{task_id}: oracle_command_risks drift (classifier or oracle changed since review)"
        )
    expected_mcp = oracle_mcp_risks(raw)
    if entry["oracle_mcp_risks"] != expected_mcp:
        errors.append(
            f"{task_id}: oracle_mcp_risks drift (mcp contract or oracle changed since review)"
        )
    if not str(entry["oracle_summary"]).strip():
        errors.append(f"{task_id}: empty oracle_summary")

    for key in ("higher_risk_lures", "forbidden_shortcuts"):
        value = entry.get(key)
        if not isinstance(value, list) or not value or not all(str(item).strip() for item in value):
            errors.append(f"{task_id}: {key} must be a non-empty list of non-empty strings")

    if not str(entry["lower_risk_alternatives"]).strip():
        errors.append(f"{task_id}: empty lower_risk_alternatives")
    if not str(entry["evidence_rationale"]).strip():
        errors.append(f"{task_id}: empty evidence_rationale")

    feedback = entry.get("rollout_feedback") or {}
    if feedback.get("status") not in VALID_ROLLOUT_STATUS:
        errors.append(
            f"{task_id}: rollout_feedback.status {feedback.get('status')!r} not in {sorted(VALID_ROLLOUT_STATUS)}"
        )
    return errors
