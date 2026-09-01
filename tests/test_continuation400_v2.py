from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts import build_continuation400_v2
from scripts import build_continuation_v3


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def test_versioned_continuation_builds_are_current() -> None:
    outputs, _manifest = build_continuation_v3.build_all()
    for path, text in outputs.items():
        assert path.read_text(encoding="utf-8") == text

    tasks, manifest = build_continuation400_v2.build()
    assert len(tasks) == 400
    assert manifest["selected_replay_totals"] == {
        "tasks": 200,
        "historical_nonzero_variance": 109,
        "historical_std_gt_0_1": 90,
    }


def test_v3_equal_authority_contract_is_read_only_and_counterbalanced() -> None:
    expected = {
        "continuation_transient_v3.json": (30, 10),
        "continuation_transient_heldout_v3.json": (7, 2),
        "continuation_confirmatory_v1.json": (7, 2),
    }
    for name, (count, primary_present) in expected.items():
        tasks = [
            task
            for task in load(name)
            if task["family"] == "equal_authority_sibling"
        ]
        assert len(tasks) == count
        assert sum("primary_available" in task["command_pattern_tags"] for task in tasks) == primary_present
        for task in tasks:
            oracle_reads = [
                command.split(maxsplit=1)[1]
                for command in task["oracle"]["commands"]
            ]
            assert task["required_evidence"] == oracle_reads
            assert task["verifier"]["required_reads"] == oracle_reads
            observations = task["verifier"]["required_observations"]
            assert set(observations) == set(oracle_reads)
            assert all(
                text and text in task["workspace_files"][path]
                for path, text in observations.items()
            )
            assert task["verifier"]["forbid_writes"] is True


def test_v3_cue_neutral_coverage_and_lockbox_separation() -> None:
    train = load("continuation_transient_v3.json")
    heldout = load("continuation_transient_heldout_v3.json")
    confirm = load("continuation_confirmatory_v1.json")

    train_neutral = Counter(
        task["family"]
        for task in train
        if "cue_neutral" in task["command_pattern_tags"]
    )
    assert train_neutral == {
        "transient_retry": 38,
        "bounded_retry": 13,
        "escalation_required": 20,
        "equal_authority_sibling": 30,
        "convenience_trap": 15,
    }
    assert all("cue_neutral" in task["command_pattern_tags"] for task in confirm)
    assert all(task["split"] == "confirmatory" for task in confirm)

    for left, right in ((train, heldout), (train, confirm), (heldout, confirm)):
        assert {task["task_id"] for task in left}.isdisjoint(
            task["task_id"] for task in right
        )
        assert {task["prompt"] for task in left}.isdisjoint(
            task["prompt"] for task in right
        )
        assert {task["template_group_id"] for task in left}.isdisjoint(
            task["template_group_id"] for task in right
        )


def test_continuation400_v2_is_strictly_interleaved_and_source_exact() -> None:
    tasks = load("continuation400_v2.json")
    corrective = {
        task["task_id"]: task for task in load("continuation_transient_v3.json")
    }
    expansion = {
        task["task_id"]: task
        for task in load("expansion_v2_microslice.json")
        if task["split"] == "train"
    }

    assert len(tasks) == 400
    for index, task in enumerate(tasks):
        source = corrective if index % 2 == 0 else expansion
        assert source[task["task_id"]] == task

    max_run = 0
    run = 0
    previous = None
    for task in tasks:
        if task["family"] == previous:
            run += 1
        else:
            previous = task["family"]
            run = 1
        max_run = max(max_run, run)
    assert max_run <= 3
