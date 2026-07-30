from __future__ import annotations

import json
from pathlib import Path

from scripts import build_validation_tiers_v2 as tiers
from scripts import validate_frozen500_run as guard


def test_registered_validation_tiers_match_frozen_v2() -> None:
    manifest = tiers.build_manifest()
    ordinary = manifest["tiers"]["ordinary"]
    safety = manifest["tiers"]["safety_heavy"]

    assert manifest["validation_task_count"] == 500
    assert manifest["expected_episodes_per_policy"] == 2896
    assert ordinary["task_count"] == 276
    assert ordinary["generations"] == 4
    assert ordinary["expected_episodes"] == 1104
    assert safety["task_count"] == 224
    assert safety["generations"] == 8
    assert safety["expected_episodes"] == 1792
    assert set(ordinary["task_ids"]).isdisjoint(safety["task_ids"])
    assert len(set(ordinary["task_ids"] + safety["task_ids"])) == 500
    assert safety["family_counts"] == {
        "cross_surface_lure": 17,
        "data_minimization": 15,
        "honeypot_restraint": 21,
        "mcp_stub": 16,
        "multi_hop_injection": 16,
        "multi_tool_workflow": 10,
        "plausible_lure": 19,
        "prompt_injection": 30,
        "secret_lure": 25,
        "unseen_mcp_server": 55,
    }


def test_eval_generation_setting_reaches_trainer_primitive() -> None:
    primitive = Path("scripts/run_eval_capped.sh").read_text(encoding="utf-8")
    suite = Path("scripts/run_eval_suite.sh").read_text(encoding="utf-8")
    assert 'NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-8}"' in primitive
    assert '--num-generations "$NUM_GENERATIONS"' in primitive
    assert 'EVAL_NUM_GENERATIONS="$GENERATIONS"' in suite


def _record(task_id: str, episode_id: str, post_terminal: int = 0) -> dict:
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "trainer_tool_stats": {
            "generated_calls": 2,
            "executed_calls": 2,
            "cancelled_after_terminal": 0,
            "failed_calls": 0,
            "post_terminal_calls": post_terminal,
            "terminal_stopped": 1,
        },
    }


def test_frozen500_tier_guard_accepts_exact_membership_and_zero_tail(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    records = [_record("a", "ep-a"), _record("b", "ep-b")]
    (rollouts / "rollouts-1.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary, errors, episode_ids = guard.validate_tier(
        "ordinary",
        rollouts,
        {"task_ids": ["a", "b"], "generations": 1, "expected_episodes": 2},
    )
    assert errors == []
    assert episode_ids == ["ep-a", "ep-b"]
    assert summary["post_terminal_calls"] == 0
    assert summary["episode_count"] == 2


def test_frozen500_tier_guard_rejects_post_terminal_tail(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    (rollouts / "rollouts-1.jsonl").write_text(
        json.dumps(_record("a", "ep-a", post_terminal=1)) + "\n", encoding="utf-8"
    )
    summary, errors, _ = guard.validate_tier(
        "safety_heavy",
        rollouts,
        {"task_ids": ["a"], "generations": 1, "expected_episodes": 1},
    )
    assert summary["post_terminal_calls"] == 1
    assert any("post_terminal_calls=1" in error for error in errors)


def test_frozen500_orchestrator_has_hard_publication_guards() -> None:
    script = Path("scripts/run_frozen500_confirmatory.sh").read_text(encoding="utf-8")
    for required in (
        "build_validation_tiers_v2.py --check",
        "check_rootless_docker.py",
        # The storage policy stays fail-closed, but the expected locations are
        # overridable so the guard is not pinned to our hardware.
        'DOCKER_ROOT" = "$EXPECTED_DOCKER_ROOT"',
        'EXPECTED_DOCKER_ROOT:-/mnt/nvme-docker/lp-docker',
        'TMPDIR" = "$EXPECTED_TMPDIR"',
        'EXPECTED_TMPDIR:-/dev/shm',
        "validate_frozen500_run.py",
        "safety_metrics_v2.json",
        "COMPLETION_GUARD.json",
        "EVAL_NUM_GENERATIONS",
    ):
        assert required in script


def test_frozen500_orchestrator_merges_base_and_v2_risk_archives() -> None:
    script = Path("scripts/run_frozen500_confirmatory.sh").read_text(encoding="utf-8")
    assert 'BASE_RISK_ARCHIVE="data/risk_archive.json"' in script
    assert 'EXTRA_RISK_ARCHIVE="data/risk_archive_v2.json"' in script
    assert 'export LP_RL_RISK_ARCHIVE="$EXTRA_RISK_ARCHIVE"' in script
    assert '--archive "$BASE_RISK_ARCHIVE"' in script
    assert '--archive "$EXTRA_RISK_ARCHIVE"' not in script
