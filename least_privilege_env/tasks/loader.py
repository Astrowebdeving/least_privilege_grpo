from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from least_privilege_env.tasks.schema import TaskSpec


# The DEFAULT is the 260-task development catalog, NOT the benchmark.
#
# The benchmark is data/expansion_v2_microslice.json (2,000 tasks), of which
# these 260 are a strict subset and the base it is built by appending to. The
# small catalog is the default because the gate stack executes every task's
# oracle in a container: 260 tasks take ~8 minutes, 2,000 take about an hour.
#
# Select the benchmark with `source scripts/benchmark_env.sh`, which sets
# LP_RL_TASK_FILE together with the matching LP_RL_RISK_ARCHIVE. Setting the
# task file alone is valid but leaves the risk archive covering only 260 tasks,
# which silently voids the safety columns in rollout summaries.
DEFAULT_TASK_FILE = Path(__file__).resolve().parents[2] / "data" / "spike_tasks.json"
# When LP_RL_TASK_FILE is set, every consumer that calls load_tasks() or
# task_by_id() with no explicit path reads it instead of the default. An
# explicit path argument always wins over the env var.
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
