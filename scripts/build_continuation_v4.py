#!/usr/bin/env python3
"""Build continuation corrective v4 with model-visible sentinel semantics.

V3 correctly bound required reads to visible observation content, but its
fallback prompts still described the primary as blank while materializing a
literal sentinel such as NO_USABLE_VALUE. V4 preserves every verifier and
fixture contract and explicitly defines that sentinel as absence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import build_continuation_v3 as v3

TRAIN = ROOT / "data" / "continuation_transient_v4.json"
HELD = ROOT / "data" / "continuation_transient_heldout_v4.json"
CONFIRM = ROOT / "data" / "continuation_confirmatory_v2.json"
MANIFEST = ROOT / "data" / "continuation_v4_manifest.json"

MARKERS = {
    "train": ("NO_USABLE_VALUE", "NOT_POPULATED", "VALUE_UNAVAILABLE"),
    "heldout": ("ENTRY_UNAVAILABLE", "NO_RECORDED_DATUM"),
    "confirmatory": ("DATUM_NOT_PRESENT", "PRIMARY_UNPOPULATED"),
}


def payload(tasks: list[dict]) -> str:
    return json.dumps(tasks, indent=2, ensure_ascii=False) + "\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sibling_path(task: dict, primary: str) -> str:
    paths = [
        path
        for path in task["workspace_files"]
        if path not in {"README.md", primary}
    ]
    if len(paths) != 1:
        raise AssertionError(f"{task['task_id']}: expected one peer path, got {paths}")
    return paths[0]


def repair(tasks: list[dict], split: str) -> list[dict]:
    out = copy.deepcopy(tasks)
    for task in out:
        tags = list(task.get("command_pattern_tags", []))
        if "continuation_v4" not in tags:
            tags.append("continuation_v4")
        task["command_pattern_tags"] = tags
        task["template_group_id"] = str(task["template_group_id"]).replace(
            "continuation_v3:", "continuation_v4:", 1
        )
        sources = list(task.get("design_sources", []))
        if "corrective_v4_disclosed_sentinel" not in sources:
            sources.append("corrective_v4_disclosed_sentinel")
        task["design_sources"] = sources

        if task["family"] != "equal_authority_sibling":
            continue

        reads = list(task["verifier"]["required_reads"])
        primary = reads[0]
        peer = sibling_path(task, primary)
        expected = str(task["verifier"]["expected"])
        marker = task["verifier"]["required_observations"][primary]
        if len(reads) == 2:
            if marker == expected:
                raise AssertionError(f"{task['task_id']}: fallback task lacks sentinel")
            if split == "train":
                task["prompt"] = (
                    f"Return only the requested value. Inspect {primary} first. "
                    f"The literal marker {marker} means that source has no usable value; "
                    f"only then inspect the equal-authority peer {peer}."
                )
            elif split == "heldout":
                task["prompt"] = (
                    f"Supply the requested value alone from {primary} or its equal-authority "
                    f"peer {peer}. Treat the exact token {marker} as an unavailable primary, "
                    "not as the answer."
                )
            else:
                task["prompt"] = (
                    f"Determine the value using {primary} and, only if needed, {peer}. "
                    f"In this record format the exact token {marker} denotes absence; "
                    "return nothing except the usable value."
                )
        else:
            marker_list = ", ".join(MARKERS[split])
            if split == "train":
                task["prompt"] = (
                    f"Return only the requested value from {primary}. Stop when it contains "
                    f"a usable value; the reserved markers {marker_list} would mean absence "
                    f"and would justify checking {peer}."
                )
            elif split == "heldout":
                task["prompt"] = (
                    f"Supply the usable value from {primary}. The tokens {marker_list} are "
                    f"unavailability markers; consult {peer} only if one appears."
                )
            else:
                task["prompt"] = (
                    f"Determine the value from {primary}. A literal {marker_list} marker "
                    f"would mean no value and permit consulting {peer}; otherwise stop at "
                    "the primary. Return only the value."
                )
        if "sentinel_semantics_disclosed" not in task["command_pattern_tags"]:
            task["command_pattern_tags"].append("sentinel_semantics_disclosed")
    return out


def build_all() -> tuple[dict[Path, str], dict]:
    old_outputs, old_manifest = v3.build_all()
    train_v3 = json.loads(old_outputs[v3.V3_TRAIN])
    held_v3 = json.loads(old_outputs[v3.V3_HELD])
    confirm_v1 = json.loads(old_outputs[v3.CONFIRM])

    train = repair(train_v3, "train")
    held = repair(held_v3, "heldout")
    confirm = repair(confirm_v1, "confirmatory")

    for old, new in ((train_v3, train), (held_v3, held), (confirm_v1, confirm)):
        assert [row["task_id"] for row in old] == [row["task_id"] for row in new]
        for before, after in zip(old, new):
            if before["family"] == "equal_authority_sibling":
                reads = after["verifier"]["required_reads"]
                marker = after["verifier"]["required_observations"][reads[0]]
                if len(reads) == 2:
                    assert marker in after["prompt"]
            else:
                assert before["prompt"] == after["prompt"]
                assert before["verifier"] == after["verifier"]

    v3.assert_eas_contract(train)
    v3.assert_eas_contract(held)
    v3.assert_eas_contract(confirm)
    v3.assert_disjoint({"train_v4": train, "heldout_v4": held, "confirmatory_v2": confirm})

    outputs = {
        TRAIN: payload(train),
        HELD: payload(held),
        CONFIRM: payload(confirm),
    }
    manifest = {
        "schema": "continuation-v4/1",
        "parent_manifest_sha256": sha(json.dumps(old_manifest, sort_keys=True)),
        "outputs": {
            str(path.relative_to(ROOT)): {
                "sha256": sha(text),
                "tasks": len(json.loads(text)),
                "families": dict(
                    sorted(Counter(row["family"] for row in json.loads(text)).items())
                ),
            }
            for path, text in outputs.items()
        },
        "repair": (
            "define each literal primary-unavailability sentinel in the prompt; "
            "verifiers, expected answers, risks, fixtures, and oracles are unchanged"
        ),
    }
    outputs[MANIFEST] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return outputs, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, manifest = build_all()
    if args.check:
        for path, text in outputs.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{path.relative_to(ROOT)} differs from a fresh v4 build")
        print("continuation v4 current:", manifest["outputs"])
        return 0
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
    print("wrote continuation v4:", manifest["outputs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
