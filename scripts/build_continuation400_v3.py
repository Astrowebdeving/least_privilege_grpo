#!/usr/bin/env python3
"""Build continuation400 v3 from repaired corrective-v4 plus unchanged replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import build_continuation400_v2 as v2

CORRECTIVE = ROOT / "data" / "continuation_transient_v4.json"
HELDOUT = ROOT / "data" / "continuation_transient_heldout_v4.json"
CONFIRMATORY = ROOT / "data" / "continuation_confirmatory_v2.json"
OUT = ROOT / "data" / "continuation400_v3.json"
MANIFEST = ROOT / "data" / "continuation400_v3_manifest.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> tuple[list[dict], dict]:
    old_tasks, old_manifest = v2.build()
    corrective = load(CORRECTIVE)
    heldout = load(HELDOUT)
    confirmatory = load(CONFIRMATORY)
    old_corrective_ids = {
        row["task_id"] for row in load(v2.CORRECTIVE)
    }
    corrective_by_id = {row["task_id"]: row for row in corrective}

    if set(corrective_by_id) != old_corrective_ids:
        raise SystemExit("v4 corrective ids differ from v3")
    tasks: list[dict] = []
    for index, row in enumerate(old_tasks):
        if index % 2 == 0:
            tasks.append(copy.deepcopy(corrective_by_id[row["task_id"]]))
        else:
            tasks.append(copy.deepcopy(row))

    if tasks[1::2] != old_tasks[1::2]:
        raise AssertionError("replay rows changed")
    if len(heldout) != 50 or len(confirmatory) != 50:
        raise AssertionError("expected 50-task development and lockbox catalogs")

    manifest = copy.deepcopy(old_manifest)
    manifest["schema"] = "continuation400/v3"
    manifest["sources"]["corrective"] = str(CORRECTIVE.relative_to(ROOT))
    manifest["sources"]["heldout_development"] = str(HELDOUT.relative_to(ROOT))
    manifest["sources"]["confirmatory_lockbox"] = str(CONFIRMATORY.relative_to(ROOT))
    manifest["repair"] = {
        "sentinel_semantics": (
            "literal unavailability markers are disclosed in every EAS prompt"
        ),
        "ordering": (
            "the launcher disables TRL dataset shuffling so the existing C/R "
            "catalog alternation is the executed curriculum"
        ),
        "precheck": (
            "overall and corrective variance remain mandatory; replay requires "
            "some current variance plus high parent success, while a low-variance "
            "family may pass only when parent success demonstrates saturation"
        ),
    }
    text = json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
    manifest["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return tasks, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tasks, manifest = build()
    text = json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            raise SystemExit("continuation400_v3 differs from a fresh build")
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != manifest_text:
            raise SystemExit("continuation400_v3 manifest differs from a fresh build")
        print("continuation400 v3 current:", len(tasks), manifest["output_sha256"])
        return 0
    OUT.write_text(text, encoding="utf-8")
    MANIFEST.write_text(manifest_text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} tasks={len(tasks)} sha={manifest['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
