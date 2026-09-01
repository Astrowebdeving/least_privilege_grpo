from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from least_privilege_env.tasks.schema import TaskSpec


DEFAULT_TASK_FILE = Path(__file__).resolve().parents[2] / "data" / "spike_tasks.json"
# Opt-in override so an alternate catalog (e.g. an expansion_v2 micro-slice) can
# be driven through the full oracle / quality / reward gate stack WITHOUT
# mutating the frozen production catalog. When LP_RL_TASK_FILE is set, every
# consumer that calls load_tasks()/task_by_id() with no explicit path reads it
# instead of spike_tasks.json; when unset, behavior is unchanged. An explicit
# path argument always wins over the env var.
TASK_FILE_ENV = "LP_RL_TASK_FILE"


def _resolve_task_file(path: Path | None) -> Path:
    if path is not None:
        return path
    override = os.environ.get(TASK_FILE_ENV, "").strip()
    return Path(override) if override else DEFAULT_TASK_FILE


def load_tasks(path: Path | None = None) -> list[TaskSpec]:
    path = _resolve_task_file(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TaskSpec.from_mapping(item) for item in raw]


def task_by_id(task_id: str, path: Path | None = None) -> TaskSpec:
    for task in load_tasks(path):
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown task_id: {task_id}")


def iter_task_ids(path: Path | None = None) -> Iterable[str]:
    for task in load_tasks(path):
        yield task.task_id
