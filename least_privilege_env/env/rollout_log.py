"""Durable per-episode rollout logging for training-run review.

Writes one JSONL record per episode when ``LP_RL_ROLLOUT_LOG_DIR`` is set:
task identity and split, every tool call with its (already F0-shaped,
additionally truncated) observation excerpt, host-only audit detail (risk
vectors, reasons, paths, blocked/unknown scores, quarantine summaries),
the verifier verdict, and the full reward breakdown.

This is the review artifact the handoff calls out for diagnosing reward
hacking after real runs. It is host-side only: keep the log dir outside any
workspace mount and never expose records to the model. Logging must never
break training — failures degrade to a stderr warning.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

ENV_VAR = "LP_RL_ROLLOUT_LOG_DIR"
OBSERVATION_EXCERPT_LIMIT = 1500
_process_episode_counter = 0


def rollout_log_dir() -> Path | None:
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def episode_record(env: Any) -> dict[str, Any]:
    """Build the JSONL record from a finished (or budget-cut) episode."""
    global _process_episode_counter
    _process_episode_counter += 1
    task = env.task
    audit = env.audit
    breakdown = env.last_reward
    verdict = env.verdict
    return {
        "schema_version": 1,
        "episode_id": uuid.uuid4().hex,
        "ts": time.time(),
        "pid": os.getpid(),
        "episode_index_in_process": _process_episode_counter,
        "task_id": task.task_id if task else None,
        "family": task.family if task else None,
        "split": task.split if task else None,
        "sandbox": os.environ.get("LP_RL_SANDBOX", "local"),
        "terminated": audit.frozen,
        # Trainer-side tool-loop telemetry (terminal-stop override): generated /
        # executed / cancelled_after_terminal / failed / post_terminal_calls /
        # terminal_stopped. None for episodes with no tool calls or when running
        # under the pinned TRL loop.
        "trainer_tool_stats": getattr(env, "trainer_tool_stats", None),
        "steps": [
            {
                "action_type": item.action_type,
                "raw": item.raw,
                "status": item.status,
                "z_pre": item.z_pre.to_dict(),
                "z_post": item.z_post.to_dict(),
                "z_final": item.z_final.to_dict(),
                "reasons": list(item.reasons),
                "read_paths": sorted(item.read_paths),
                "written_paths": sorted(item.written_paths),
                "blocked_severity": item.blocked_severity,
                "unknown_score": item.unknown_score,
                "exit_code": item.exit_code,
                "observation_excerpt": item.observation[:OBSERVATION_EXCERPT_LIMIT],
            }
            for item in audit.actions
        ],
        "escalation": (
            {"code": audit.escalation.code, "detail": audit.escalation.detail} if audit.escalation else None
        ),
        "final_answer": (audit.final_answer or "")[:500] if audit.final_answer is not None else None,
        "z_tau": audit.z_tau().to_dict(),
        "verdict": {"success": verdict.success, "reason": verdict.reason} if verdict else None,
        "reward": (
            {
                "reward": breakdown.reward,
                "success": breakdown.success,
                "evidence": breakdown.evidence,
                "privilege_penalty": breakdown.privilege_penalty,
                "inefficiency": breakdown.inefficiency,
                "escalation_hygiene": breakdown.escalation_hygiene,
                "blocked": breakdown.blocked,
                "unknown": breakdown.unknown,
            }
            if breakdown
            else None
        ),
    }


def write_episode(env: Any) -> None:
    """Append one episode record; never raises into the training loop."""
    directory = rollout_log_dir()
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # One file per process: concurrent trainer/env processes never
        # interleave partial lines within a file.
        path = directory / f"rollouts-{os.getpid()}.jsonl"
        record = episode_record(env)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        import sys

        print(f"[lp-rl rollout-log] WARNING: failed to write episode record: {exc}", file=sys.stderr)
