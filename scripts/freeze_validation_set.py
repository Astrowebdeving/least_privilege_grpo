#!/usr/bin/env python3
"""Freeze the corrected 60-task validation set.

Writes (or verifies) a content-addressed manifest that pins the held-out
benchmark so future drift in the catalog, verifiers, or registry is
detectable. This is a benchmark *integrity* artifact: it does not run any
model and does not mutate the catalog.

The manifest records, per validation task, a sha256 over the task's canonical
(sorted-key) JSON definition from ``data/spike_tasks.json`` (which carries the
verifier config, required_risk, oracle, evidence, and fixtures), plus
file-level digests of the catalog source, the validation id registry, and the
report-only risk archive, plus catalog split counts.

Usage:
    python scripts/freeze_validation_set.py            # write manifest (refuse overwrite)
    python scripts/freeze_validation_set.py --force    # overwrite existing manifest
    python scripts/freeze_validation_set.py --check     # verify catalog matches a frozen manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "spike_tasks.json"
VAL_IDS = ROOT / "data" / "validation_task_ids.txt"
RISK_ARCHIVE = ROOT / "data" / "risk_archive.json"
DEFAULT_MANIFEST = ROOT / "data" / "validation_frozen_2026-06-23.json"

REBASELINE_ARTIFACT = (
    "logs/artifacts/validation_rebaseline_60task_corrected_2026-06-23"
)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def load_validation_ids() -> list[str]:
    raw = VAL_IDS.read_text(encoding="utf-8").strip()
    ids = [t.strip() for t in raw.split(",") if t.strip()]
    if len(set(ids)) != len(ids):
        raise SystemExit("validation_task_ids.txt contains duplicate ids")
    return ids


def load_catalog() -> list[dict]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def build_manifest() -> dict:
    val_ids = load_validation_ids()
    catalog = load_catalog()
    by_id = {t["task_id"]: t for t in catalog}

    missing = [tid for tid in val_ids if tid not in by_id]
    if missing:
        raise SystemExit(f"validation ids not present in catalog: {missing}")

    split_counts: dict[str, int] = {}
    for t in catalog:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1

    # held-out tasks must not be in the train split
    train_leak = [tid for tid in val_ids if by_id[tid]["split"] == "train"]
    if train_leak:
        raise SystemExit(f"validation ids leak into train split: {train_leak}")

    task_hashes = {}
    family_counts: dict[str, int] = {}
    for tid in val_ids:
        spec = by_id[tid]
        task_hashes[tid] = {
            "sha256": _sha256_bytes(_canonical(spec)),
            "family": spec["family"],
            "split": spec["split"],
        }
        family_counts[spec["family"]] = family_counts.get(spec["family"], 0) + 1

    # a single combined digest over the ordered (id, spec-hash) pairs
    combined = hashlib.sha256()
    for tid in val_ids:
        combined.update(tid.encode())
        combined.update(b":")
        combined.update(task_hashes[tid]["sha256"].encode())
        combined.update(b"\n")

    manifest = {
        "schema": "lp-rl/validation-freeze/1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Frozen corrected 60-task held-out validation benchmark for the "
            "least-privilege RL project. Pins task definitions, registry, and "
            "catalog source so the benchmark cannot drift silently."
        ),
        "rebaseline_artifact": REBASELINE_ARTIFACT,
        "catalog_total": len(catalog),
        "split_counts": split_counts,
        "validation_count": len(val_ids),
        "validation_family_counts": family_counts,
        "validation_task_ids": val_ids,
        "validation_combined_sha256": combined.hexdigest(),
        "file_digests": {
            "data/spike_tasks.json": _sha256_file(CATALOG),
            "data/validation_task_ids.txt": _sha256_file(VAL_IDS),
            "data/risk_archive.json": _sha256_file(RISK_ARCHIVE)
            if RISK_ARCHIVE.exists()
            else None,
        },
        "task_hashes": task_hashes,
    }
    return manifest


def check(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"FROZEN MANIFEST MISSING: {manifest_path}", file=sys.stderr)
        return 1
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_manifest()

    problems: list[str] = []
    if frozen.get("validation_task_ids") != current["validation_task_ids"]:
        problems.append("validation id set/order changed")
    if frozen.get("validation_combined_sha256") != current["validation_combined_sha256"]:
        problems.append("validation task definitions changed (combined digest mismatch)")
    # per-task detail for actionable output
    fz = frozen.get("task_hashes", {})
    cu = current["task_hashes"]
    for tid in current["validation_task_ids"]:
        if tid in fz and fz[tid]["sha256"] != cu[tid]["sha256"]:
            problems.append(f"task changed: {tid}")
    for key, digest in current["file_digests"].items():
        if key in frozen.get("file_digests", {}) and frozen["file_digests"][key] != digest:
            problems.append(f"file digest changed: {key}")
    if frozen.get("split_counts") != current["split_counts"]:
        problems.append(
            f"split counts changed {frozen.get('split_counts')} -> {current['split_counts']}"
        )

    if problems:
        print("VALIDATION FREEZE CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(
        f"VALIDATION FREEZE OK: {current['validation_count']} tasks, "
        f"combined={current['validation_combined_sha256'][:16]}..., "
        f"catalog={current['catalog_total']} {current['split_counts']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ap.add_argument("--check", action="store_true", help="verify catalog against the frozen manifest")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)

    if args.check:
        return check(manifest_path)

    if manifest_path.exists() and not args.force:
        print(
            f"refusing to overwrite existing manifest {manifest_path}; pass --force "
            f"or use --check to verify",
            file=sys.stderr,
        )
        return 2

    manifest = build_manifest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"FROZEN {manifest['validation_count']} validation tasks -> {manifest_path}\n"
        f"  combined_sha256 = {manifest['validation_combined_sha256']}\n"
        f"  catalog_total   = {manifest['catalog_total']} {manifest['split_counts']}\n"
        f"  families        = {manifest['validation_family_counts']}\n"
        f"  spike_tasks.json sha256 = {manifest['file_digests']['data/spike_tasks.json'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
