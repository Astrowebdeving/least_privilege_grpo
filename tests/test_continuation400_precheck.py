from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_continuation400_precheck.py"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_inputs(tmp_path: Path, corrective_variable: int, replay_variable: int):
    corrective = [
        {"task_id": f"c{i:03d}", "family": "corrective_family"}
        for i in range(200)
    ]
    replay = [
        {"task_id": f"r{i:03d}", "family": "replay_family"}
        for i in range(200)
    ]
    catalog = [item for pair in zip(corrective, replay) for item in pair]
    report_tasks = []
    for index, task in enumerate(corrective):
        report_tasks.append(
            {
                "task_id": task["task_id"],
                "reward_std": 0.2 if index < corrective_variable else 0.0,
            }
        )
    for index, task in enumerate(replay):
        report_tasks.append(
            {
                "task_id": task["task_id"],
                "reward_std": 0.2 if index < replay_variable else 0.0,
            }
        )
    catalog_path = tmp_path / "catalog.json"
    corrective_path = tmp_path / "corrective.json"
    variance_path = tmp_path / "variance.json"
    output_path = tmp_path / "partition.json"
    write_json(catalog_path, catalog)
    write_json(corrective_path, corrective)
    write_json(variance_path, {"status": "pass", "tasks": report_tasks})
    return catalog_path, corrective_path, variance_path, output_path


def run_gate(paths, extra_args=()):
    catalog, corrective, variance, output = paths
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--catalog",
            str(catalog),
            "--corrective",
            str(corrective),
            "--variance-report",
            str(variance),
            *extra_args,
            "--report",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_partition_gate_passes_when_both_sources_have_signal(tmp_path: Path):
    result, report = run_gate(make_inputs(tmp_path, 20, 50))
    assert result.returncode == 0, result.stderr
    assert report["status"] == "pass"
    assert report["sources"]["corrective"]["variable_tasks"] == 20
    assert report["sources"]["replay"]["variable_tasks"] == 50


def test_partition_gate_rejects_dead_corrective_half(tmp_path: Path):
    result, report = run_gate(make_inputs(tmp_path, 19, 100))
    assert result.returncode == 1
    assert report["status"] == "fail"
    assert any("corrective" in item for item in report["violations"])


def test_partition_gate_rejects_dead_replay_half(tmp_path: Path):
    result, report = run_gate(make_inputs(tmp_path, 100, 49))
    assert result.returncode == 1
    assert report["status"] == "fail"
    assert any("replay" in item for item in report["violations"])


def _set_replay_success(paths, successful_per_task: int) -> None:
    variance = paths[2]
    report = json.loads(variance.read_text(encoding="utf-8"))
    for row in report["tasks"]:
        row["episodes"] = 8
        row["successful_episodes"] = (
            successful_per_task if row["task_id"].startswith("r") else 0
        )
    write_json(variance, report)


def test_partition_gate_accepts_saturated_replay_with_some_signal(tmp_path: Path):
    paths = make_inputs(tmp_path, 20, 10)
    _set_replay_success(paths, 8)
    result, report = run_gate(
        paths,
        (
            "--min-replay-variable-fraction",
            "0.05",
            "--min-replay-success-rate",
            "0.90",
        ),
    )
    assert result.returncode == 0, result.stderr
    assert report["sources"]["replay"]["variable_fraction"] == 0.05
    assert report["sources"]["replay"]["success_rate"] == 1.0


def test_partition_gate_rejects_low_success_replay(tmp_path: Path):
    paths = make_inputs(tmp_path, 20, 10)
    _set_replay_success(paths, 4)
    result, report = run_gate(
        paths,
        (
            "--min-replay-variable-fraction",
            "0.05",
            "--min-replay-success-rate",
            "0.90",
        ),
    )
    assert result.returncode == 1
    assert report["sources"]["replay"]["success_rate"] == 0.5
    assert any("success rate" in item for item in report["violations"])
