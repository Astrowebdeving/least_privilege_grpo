from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from least_privilege_env.env.terminal_env import TerminalEnv, least_privilege_reward


def _run_episode_with_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    monkeypatch.setenv("LP_RL_ROLLOUT_LOG_DIR", str(tmp_path))
    env = TerminalEnv()
    env.reset(task_id="read_exact_001")
    env.terminal_run("cat docs/release.txt")
    with pytest.raises(ValueError):
        env.terminal_run("sudo id")
    env.final_answer("cedar")
    # Reward fetched twice (TRL calls once; smokes may call again): one record.
    least_privilege_reward([env])
    env._reward()
    records = []
    for path in tmp_path.glob("rollouts-*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    return records


def test_one_record_per_episode_with_full_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _run_episode_with_log(tmp_path, monkeypatch)
    assert len(records) == 1
    record = records[0]
    assert record["task_id"] == "read_exact_001"
    assert record["family"] == "read_only_exact"
    assert record["split"] == "train"
    assert record["terminated"] is True
    assert record["final_answer"] == "cedar"
    assert record["verdict"]["success"] == 1.0
    assert record["reward"]["reward"] == pytest.approx(record["reward"]["reward"])
    assert record["pid"] == os.getpid()

    executed, blocked = record["steps"]
    assert executed["action_type"] == "terminal"
    assert executed["status"] == "executed"
    assert "cedar" in executed["observation_excerpt"]
    assert executed["read_paths"] == ["docs/release.txt"]
    assert blocked["action_type"] == "terminal"
    assert blocked["status"] == "blocked"
    assert blocked["observation_excerpt"] == "command blocked by environment policy"
    assert blocked["blocked_severity"] == 1.0
    assert record["z_tau"]["scope"] == 1.0  # blocked attempt dominates trajectory risk


def test_unterminated_episode_still_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LP_RL_ROLLOUT_LOG_DIR", str(tmp_path))
    env = TerminalEnv()
    env.reset(task_id="read_exact_001")
    env.terminal_run("cat docs/release.txt")
    # No terminal action: simulates a token-budget cut. Reward time logs it.
    least_privilege_reward([env])
    records = [
        json.loads(line)
        for path in tmp_path.glob("rollouts-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["terminated"] is False
    assert records[0]["verdict"] is None
    assert records[0]["reward"]["success"] == 0.0


def test_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LP_RL_ROLLOUT_LOG_DIR", raising=False)
    env = TerminalEnv()
    env.reset(task_id="read_exact_001")
    env.terminal_run("cat docs/release.txt")
    env.final_answer("cedar")
    env._reward()
    assert not list(tmp_path.glob("*.jsonl"))


def test_observation_excerpt_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LP_RL_ROLLOUT_LOG_DIR", str(tmp_path))
    env = TerminalEnv()
    env.reset(task_id="localize_001")
    env.terminal_run("grep ERROR_417 src")
    env.final_answer("src/handlers/cache.py")
    env._reward()
    records = [
        json.loads(line)
        for path in tmp_path.glob("rollouts-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(len(step["observation_excerpt"]) <= 1500 for step in records[0]["steps"])


def test_mcp_action_type_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LP_RL_ROLLOUT_LOG_DIR", str(tmp_path))
    env = TerminalEnv()
    env.reset(task_id="mcp_read_001")
    env.mcp_call("github_stub", "get_issue", {"issue_id": "7"})
    env.final_answer("Safari")
    env._reward()
    records = [
        json.loads(line)
        for path in tmp_path.glob("rollouts-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["steps"][0]["action_type"] == "mcp"
    assert records[0]["steps"][0]["raw"].startswith("mcp:github_stub.get_issue")
