from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.analyze_reward_variance import main


def _run(monkeypatch, tmp_path: Path, second: dict) -> tuple[int, dict]:
    catalog = tmp_path / "catalog.json"
    rollouts = tmp_path / "rollouts.jsonl"
    report = tmp_path / "report.json"
    catalog.write_text(json.dumps([{"task_id": "t1", "family": "f"}]))
    first = {
        "episode_id": "e1",
        "task_id": "t1",
        "family": "f",
        "sandbox": "docker",
        "steps": [],
        "terminated": False,
        "trainer_tool_stats": {"post_terminal_calls": 0},
        "reward": {"reward": 0.1, "success": 0.0},
    }
    rollouts.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_reward_variance.py",
            "--rollouts",
            str(rollouts),
            "--task-file",
            str(catalog),
            "--expected-generations",
            "2",
            "--min-overall-variable-fraction",
            "0",
            "--min-family-variable-fraction",
            "0",
            "--min-family-variable-groups",
            "1",
            "--report",
            str(report),
        ],
    )
    rc = main()
    return rc, json.loads(report.read_text())


def test_missing_stats_without_behavior_is_report_only(monkeypatch, tmp_path: Path) -> None:
    second = {
        "episode_id": "e2",
        "task_id": "t1",
        "family": "f",
        "sandbox": "docker",
        "steps": [],
        "terminated": False,
        "final_answer": None,
        "escalation": None,
        "reward": {"reward": 0.9, "success": 1.0},
    }
    rc, report = _run(monkeypatch, tmp_path, second)
    assert rc == 0
    assert report["status"] == "pass"
    assert report["summary"]["missing_tool_stats"] == 1
    assert report["summary"]["missing_tool_stats_with_behavior"] == 0


def test_missing_stats_with_behavior_fails(monkeypatch, tmp_path: Path) -> None:
    second = {
        "episode_id": "e2",
        "task_id": "t1",
        "family": "f",
        "sandbox": "docker",
        "steps": [{"action_type": "shell", "status": "executed"}],
        "terminated": False,
        "reward": {"reward": 0.9, "success": 1.0},
    }
    rc, report = _run(monkeypatch, tmp_path, second)
    assert rc == 1
    assert report["status"] == "fail"
    assert report["summary"]["missing_tool_stats_with_behavior"] == 1
    assert any("despite behavior" in message for message in report["violations"])
