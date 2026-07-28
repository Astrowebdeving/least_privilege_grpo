#!/usr/bin/env python3
"""Freeze the 2000-task expansion-v2 benchmark (separate from the 260 freeze).

Versioned integrity artifact for the 2000-task working catalog. Pins, by content
hash, the catalog, the 500-task held-out validation set, the registered routine
eval subset, and BOTH risk archives (the frozen 260 archive + the v2 archive), so
the benchmark cannot drift silently. Does NOT run any model and does NOT touch the
260 freeze (`data/validation_frozen_2026-06-23.json`) or `data/risk_archive.json`.

Documents the freeze-quality deviations explicitly (reserved_new_behavior
placeholder; <30% hand-authored held-out floor) rather than silently meeting them.

Usage:
    python scripts/freeze_expansion_v2.py            # write manifest (refuse overwrite)
    python scripts/freeze_expansion_v2.py --force    # overwrite
    python scripts/freeze_expansion_v2.py --check     # verify catalog matches the freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "expansion_v2_microslice.json"
VAL_IDS = ROOT / "data" / "validation_task_ids_v2.txt"
ROUTINE = ROOT / "data" / "routine_eval_subset_v2.txt"
BASE_ARCHIVE = ROOT / "data" / "risk_archive.json"
V2_ARCHIVE = ROOT / "data" / "risk_archive_v2.json"
DEFAULT_MANIFEST = ROOT / "data" / "validation_frozen_2000_2026-06-30.json"

PARITY_REFERENCE = {
    "artifact": "logs/artifacts/parity_2000_ram_2026-06-30/parity.json",
    "result": "local 2000/0 + docker 2000/0; oracle backend parity 2000/2000, violations=[], tol 1e-9 (2026-06-30)",
}
DEVIATIONS = [
    "reserved_new_behavior (35 whole_family_heldout) is a non-degenerate label-read "
    "placeholder, NOT a strong novel-behavior generalization probe. The v2 risk archive "
    "makes its report metrics defined, but novel-behavior generalization must not be "
    "claimed from it until it is hand-authored/rewritten (deferred to a v2.1 freeze).",
    "Hand-authored held-out fraction is ~18% (proving-slice scale), below the plan §4 "
    ">=30%-per-held-out-family floor. This v2.0 freeze documents the deviation and defers "
    "the floor (and reserved_new_behavior hand-authoring) to a v2.1 freeze.",
]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _read_ids(path: Path) -> list[str]:
    ids = [x.strip() for x in path.read_text(encoding="utf-8").replace(",", "\n").split() if x.strip()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path.name} contains duplicate ids")
    return ids


def build_manifest() -> dict:
    val_ids = _read_ids(VAL_IDS)
    routine = _read_ids(ROUTINE)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {t["task_id"]: t for t in catalog}

    missing = [tid for tid in val_ids if tid not in by_id]
    if missing:
        raise SystemExit(f"validation ids not in catalog: {missing[:5]}...")
    train_leak = [tid for tid in val_ids if by_id[tid]["split"] == "train"]
    if train_leak:
        raise SystemExit(f"validation ids leak into train: {train_leak[:5]}...")
    routine_outside = [tid for tid in routine if tid not in set(val_ids)]
    if routine_outside:
        raise SystemExit(f"routine subset has ids outside the held-out set: {routine_outside[:5]}...")

    split_counts: dict[str, int] = {}
    for t in catalog:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1

    task_hashes = {}
    family_counts: dict[str, int] = {}
    for tid in sorted(val_ids):
        spec = by_id[tid]
        task_hashes[tid] = {
            "sha256": _sha256_bytes(_canonical(spec)),
            "family": spec["family"],
            "split": spec["split"],
        }
        family_counts[spec["family"]] = family_counts.get(spec["family"], 0) + 1

    combined = hashlib.sha256()
    for tid in sorted(val_ids):
        combined.update(tid.encode()); combined.update(b":")
        combined.update(task_hashes[tid]["sha256"].encode()); combined.update(b"\n")

    v2_entries = len(json.loads(V2_ARCHIVE.read_text(encoding="utf-8")).get("entries", []))

    return {
        "schema": "lp-rl/validation-freeze/2",
        "version": "v2.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Frozen 2000-task expansion-v2 benchmark for the least-privilege RL project. "
            "Pins the catalog, the 500-task held-out validation set, the routine eval "
            "subset, and both risk archives. Separate from and preserves the 260 freeze."
        ),
        "preserves_260_freeze": "data/validation_frozen_2026-06-23.json",
        "catalog_total": len(catalog),
        "split_counts": split_counts,
        "validation_count": len(val_ids),
        "validation_family_counts": family_counts,
        "validation_task_ids": sorted(val_ids),
        "validation_combined_sha256": combined.hexdigest(),
        "routine_subset_count": len(routine),
        "routine_subset_ids": sorted(routine),
        "v2_archive_entries": v2_entries,
        "file_digests": {
            "data/expansion_v2_microslice.json": _sha256_file(CATALOG),
            "data/validation_task_ids_v2.txt": _sha256_file(VAL_IDS),
            "data/routine_eval_subset_v2.txt": _sha256_file(ROUTINE),
            "data/risk_archive.json": _sha256_file(BASE_ARCHIVE),
            "data/risk_archive_v2.json": _sha256_file(V2_ARCHIVE),
        },
        "oracle_backend_parity": PARITY_REFERENCE,
        "documented_deviations": DEVIATIONS,
        "task_hashes": task_hashes,
    }


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
    if frozen.get("routine_subset_ids") != current["routine_subset_ids"]:
        problems.append("routine subset changed")
    fz = frozen.get("task_hashes", {})
    for tid in current["validation_task_ids"]:
        if tid in fz and fz[tid]["sha256"] != current["task_hashes"][tid]["sha256"]:
            problems.append(f"task changed: {tid}")
    for key, digest in current["file_digests"].items():
        if key in frozen.get("file_digests", {}) and frozen["file_digests"][key] != digest:
            problems.append(f"file digest changed: {key}")
    if frozen.get("split_counts") != current["split_counts"]:
        problems.append(f"split counts changed {frozen.get('split_counts')} -> {current['split_counts']}")
    if problems:
        print("EXPANSION-V2 FREEZE CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(
        f"EXPANSION-V2 FREEZE OK: {current['validation_count']} held-out "
        f"(routine {current['routine_subset_count']}), combined="
        f"{current['validation_combined_sha256'][:16]}..., catalog={current['catalog_total']} "
        f"{current['split_counts']}, v2_archive_entries={current['v2_archive_entries']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)

    if args.check:
        return check(manifest_path)
    if manifest_path.exists() and not args.force:
        print(f"refusing to overwrite {manifest_path}; pass --force or --check", file=sys.stderr)
        return 2
    manifest = build_manifest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"FROZEN {manifest['validation_count']} held-out tasks (routine "
        f"{manifest['routine_subset_count']}) -> {manifest_path}\n"
        f"  combined_sha256 = {manifest['validation_combined_sha256']}\n"
        f"  catalog_total   = {manifest['catalog_total']} {manifest['split_counts']}\n"
        f"  catalog sha256  = {manifest['file_digests']['data/expansion_v2_microslice.json'][:16]}...\n"
        f"  v2 archive      = {manifest['v2_archive_entries']} entries, "
        f"sha {manifest['file_digests']['data/risk_archive_v2.json'][:16]}...\n"
        f"  deviations documented: {len(manifest['documented_deviations'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
