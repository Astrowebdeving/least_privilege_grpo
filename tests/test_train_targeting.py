from __future__ import annotations

from collections import Counter

import pytest

from train.train_trl_drgrpo import build_training_dataset, parse_args, validate_cli_args


def _ids(dataset) -> list[str]:
    return list(dataset["task_id"])


def test_task_ids_ignore_split_and_validate_before_repeat() -> None:
    dataset = build_training_dataset(
        split="train",
        task_ids_csv="external_001,mcp_inject_001",
        repeat=2,
    )
    assert Counter(_ids(dataset)) == {"external_001": 2, "mcp_inject_001": 2}


def test_task_ids_fail_fast_on_missing_id() -> None:
    with pytest.raises(ValueError, match="requested task ids"):
        build_training_dataset(split="train", task_ids_csv="does_not_exist", repeat=1)


def test_task_ids_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        build_training_dataset(split="train", task_ids_csv="read_exact_001,read_exact_001", repeat=1)


def test_family_targeting_respects_split() -> None:
    dataset = build_training_dataset(split="train", families_csv="mcp_stub", repeat=1)
    ids = _ids(dataset)
    assert "mcp_inject_001" not in ids  # heldout_within
    assert ids
    assert all(task_id.startswith("mcp_") for task_id in ids)


def test_seed_flag_is_parsed() -> None:
    args = parse_args(["--seed", "123"])
    assert args.seed == 123


def test_adapter_path_flag_is_parsed() -> None:
    args = parse_args(["--adapter-path", "logs/artifacts/pilot/checkpoint-32"])
    assert args.adapter_path == "logs/artifacts/pilot/checkpoint-32"


def test_adapter_path_rejects_full_finetune() -> None:
    args = parse_args(["--adapter-path", "checkpoint-32", "--full-finetune"])
    with pytest.raises(ValueError, match="adapter-path"):
        validate_cli_args(args)
