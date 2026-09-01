from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.analyze_reward_variance import main


def _run(monkeypatch, tmp_path: Path, success: float) -> tuple[int, dict]:
    catalog = tmp_path / "catalog.json"
    rollouts = tmp_path / "rollouts.jsonl"
    report = tmp_path / "report.json"
    catalog.write_text(
        json.dumps([{"task_id": "t1", "family": "saturated_family"}]) + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "episode_id": f"e{index}",
            "task_id": "t1",
            "family": "saturated_family",
            "sandbox": "docker",
            "steps": [],
            "terminated": False,
            "trainer_tool_stats": {"post_terminal_calls": 0},
            "reward": {"reward": 0.8, "success": success},
        }
        for index in (1, 2)
    ]
    rollouts.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
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
            "0.10",
            "--min-family-variable-groups",
            "1",
            "--min-saturated-family-success-rate",
            "0.90",
            "--report",
            str(report),
        ],
    )
    rc = main()
    return rc, json.loads(report.read_text(encoding="utf-8"))


def test_uniform_high_success_family_is_saturated(monkeypatch, tmp_path: Path) -> None:
    rc, report = _run(monkeypatch, tmp_path, 1.0)
    assert rc == 0
    family = report["families"]["saturated_family"]
    assert family["saturated"] is True
    assert family["variance_gate_passed"] is False


def test_uniform_failure_family_remains_dead(monkeypatch, tmp_path: Path) -> None:
    rc, report = _run(monkeypatch, tmp_path, 0.0)
    assert rc == 1
    family = report["families"]["saturated_family"]
    assert family["saturated"] is False
    assert any("saturated_family" in item for item in report["violations"])
