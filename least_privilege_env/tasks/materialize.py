from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from least_privilege_env.sandbox.safe_fs import resolve_inside, write_file_no_follow
from least_privilege_env.tasks.schema import TaskSpec


class MaterializedTask:
    def __init__(self, task: TaskSpec, tmp: TemporaryDirectory[str]):
        self.task = task
        self._tmp = tmp
        self.root = Path(tmp.name)
        self.workspace = self.root / "workspace"
        self.hidden = self.root / "hidden_host_only"

    def cleanup(self) -> None:
        self._tmp.cleanup()


def materialize_task(task: TaskSpec) -> MaterializedTask:
    tmp = TemporaryDirectory(prefix=f"lp-{task.task_id}-")
    mat = MaterializedTask(task, tmp)
    mat.workspace.mkdir(parents=True)
    mat.hidden.mkdir(parents=True)
    for rel, content in task.workspace_files.items():
        target = resolve_inside(mat.workspace, mat.workspace, rel)
        write_file_no_follow(target, content.encode("utf-8"))
    for rel, content in task.hidden_files.items():
        target = resolve_inside(mat.hidden, mat.hidden, rel)
        write_file_no_follow(target, content.encode("utf-8"))
    return mat
