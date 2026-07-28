from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from least_privilege_env.broker.parser import CommandNode, PipelineNode, Program, is_devnull
from least_privilege_env.sandbox.base import (
    CommandResult,
    Sandbox,
    execution_read_paths,
    input_redirect,
    is_pytest_invocation,
    pytest_env_overrides,
    route_output_streams,
)
from least_privilege_env.sandbox.safe_fs import (
    SafeFSError,
    diff_snapshots,
    read_file_no_follow,
    resolve_inside,
    snapshot_tree,
    write_file_no_follow,
)


class LocalSandbox(Sandbox):
    """Broker-owned local execution sandbox for tests and smoke runs.

    DockerSandbox is the intended training backend. This local backend keeps the
    same parser-derived execution semantics so unit tests can run without Docker.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.cwd = self.workspace
        self._tmp: TemporaryDirectory[str] | None = None

    @classmethod
    def from_files(cls, files: dict[str, str]) -> "LocalSandbox":
        tmp = TemporaryDirectory(prefix="lp-episode-")
        workspace = Path(tmp.name) / "workspace"
        workspace.mkdir()
        for rel, content in files.items():
            target = resolve_inside(workspace, workspace, rel)
            write_file_no_follow(target, content.encode("utf-8"))
        sandbox = cls(workspace)
        sandbox._tmp = tmp
        return sandbox

    def snapshot(self):
        return snapshot_tree(self.workspace)

    def execute(self, program: Program, timeout_s: int = 10) -> CommandResult:
        before = self.snapshot()
        stdout = ""
        stderr = ""
        exit_code = 0
        read_paths: set[str] = set()
        written_paths: set[str] = set()

        for chain_item in program.chain:
            if chain_item.op_before == "&&" and exit_code != 0:
                continue
            if chain_item.op_before == "||" and exit_code == 0:
                continue
            result = self._execute_pipeline(chain_item.pipeline, timeout_s)
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code
            read_paths.update(result.read_paths)
            written_paths.update(result.written_paths)

        after = self.snapshot()
        added, changed, deleted = diff_snapshots(before, after)
        written_paths.update(added | changed | deleted)
        return CommandResult(stdout=stdout, stderr=stderr, exit_code=exit_code, read_paths=read_paths, written_paths=written_paths)

    def _execute_pipeline(self, pipeline: PipelineNode, timeout_s: int) -> CommandResult:
        input_bytes: bytes | None = None
        aggregate_stderr: list[str] = []
        read_paths: set[str] = set()
        written_paths: set[str] = set()
        exit_code = 0
        stdout_bytes = b""
        for command in pipeline.commands:
            result = self._execute_command(command, input_bytes=input_bytes, timeout_s=timeout_s)
            input_bytes = result.stdout.encode("utf-8", errors="replace")
            stdout_bytes = input_bytes
            aggregate_stderr.append(result.stderr)
            exit_code = result.exit_code
            read_paths.update(result.read_paths)
            written_paths.update(result.written_paths)
            if exit_code != 0:
                break
        return CommandResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr="".join(aggregate_stderr),
            exit_code=exit_code,
            read_paths=read_paths,
            written_paths=written_paths,
        )

    def _execute_command(self, command: CommandNode, input_bytes: bytes | None, timeout_s: int) -> CommandResult:
        stdin = input_bytes
        read_paths: set[str] = set()
        inp = input_redirect(command.redirects)
        if inp is not None:
            if is_devnull(inp.target):
                stdin = b""
            else:
                target = resolve_inside(self.workspace, self.cwd, inp.target)
                stdin = read_file_no_follow(target)
                read_paths.add(target.relative_to(self.workspace).as_posix())

        name = command.name
        if name == "cd":
            return self._builtin_cd(command.argv)
        if name in {"pwd", "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "mkdir", "touch", "cp", "mv"}:
            result = self._execute_builtin(command.argv, stdin)
        else:
            result = self._execute_subprocess(command.argv, stdin, timeout_s)
            result.read_paths.update(execution_read_paths(self.workspace, self.cwd, command))

        def resolve_target(token: str) -> tuple[Path, str]:
            path = resolve_inside(self.workspace, self.cwd, token)
            return path, path.relative_to(self.workspace).as_posix()

        result.stdout, result.stderr, written = route_output_streams(
            result.stdout,
            result.stderr,
            command.redirects,
            resolve_target,
            lambda path, data, append: write_file_no_follow(path, data, append=append),
        )
        result.read_paths.update(read_paths)
        result.written_paths.update(written)
        return result

    def _builtin_cd(self, argv: tuple[str, ...]) -> CommandResult:
        if len(argv) > 2:
            return CommandResult(stderr="cd: too many arguments\n", exit_code=2)
        target = argv[1] if len(argv) == 2 else "."
        try:
            resolved = resolve_inside(self.workspace, self.cwd, target)
            if not resolved.exists() or not resolved.is_dir():
                return CommandResult(stderr=f"cd: no such directory: {target}\n", exit_code=1)
            self.cwd = resolved
            return CommandResult()
        except SafeFSError as exc:
            return CommandResult(stderr=f"{exc}\n", exit_code=1)

    def _execute_builtin(self, argv: tuple[str, ...], stdin: bytes | None) -> CommandResult:
        try:
            name = argv[0]
            if name == "pwd":
                return CommandResult(stdout="/" + self.cwd.relative_to(self.workspace).as_posix().strip(".") + "\n")
            if name == "ls":
                target = resolve_inside(self.workspace, self.cwd, next((a for a in argv[1:] if not a.startswith("-")), "."))
                if target.is_dir():
                    return CommandResult(stdout="\n".join(sorted(p.name for p in target.iterdir())) + "\n")
                return CommandResult(stdout=target.name + "\n", read_paths={target.relative_to(self.workspace).as_posix()})
            if name == "cat":
                return self._read_many(argv[1:])
            if name in {"head", "tail"}:
                paths = [a for a in argv[1:] if not a.startswith("-")]
                count = 10
                if "-n" in argv:
                    idx = argv.index("-n")
                    if idx + 1 < len(argv):
                        count = int(argv[idx + 1])
                data = self._read_many(paths)
                lines = data.stdout.splitlines()
                selected = lines[:count] if name == "head" else lines[-count:]
                return CommandResult(stdout="\n".join(selected) + ("\n" if selected else ""), read_paths=data.read_paths, exit_code=data.exit_code, stderr=data.stderr)
            if name == "wc":
                read_paths: set[str] = set()
                if stdin is not None:
                    data = stdin.decode("utf-8", errors="replace")
                else:
                    # Reading file operands exposes their content for counting:
                    # credit them so the secret scan / forbidden-read gate see it
                    # (Docker already credits wc reads; this matches it).
                    file_result = self._read_many(argv[1:])
                    data = file_result.stdout
                    read_paths = file_result.read_paths
                return CommandResult(
                    stdout=f"{len(data.splitlines())} {len(data.split())} {len(data)}\n",
                    read_paths=read_paths,
                )
            if name in {"grep", "rg"}:
                return self._grep(argv, stdin)
            if name == "find":
                target = resolve_inside(self.workspace, self.cwd, argv[1] if len(argv) > 1 and not argv[1].startswith("-") else ".")
                lines = [p.relative_to(self.cwd).as_posix() for p in target.rglob("*") if not p.is_symlink()]
                return CommandResult(stdout="\n".join(sorted(lines)) + "\n")
            if name == "mkdir":
                for arg in argv[1:]:
                    if arg == "-p":
                        continue
                    target = resolve_inside(self.workspace, self.cwd, arg)
                    target.mkdir(parents="-p" in argv, exist_ok="-p" in argv)
                return CommandResult()
            if name == "touch":
                written = set()
                for arg in argv[1:]:
                    target = resolve_inside(self.workspace, self.cwd, arg)
                    existing = read_file_no_follow(target) if target.exists() else b""
                    write_file_no_follow(target, existing, append=False)
                    written.add(target.relative_to(self.workspace).as_posix())
                return CommandResult(written_paths=written)
            if name == "cp":
                src = resolve_inside(self.workspace, self.cwd, argv[-2])
                dst = resolve_inside(self.workspace, self.cwd, argv[-1])
                if src.is_symlink() or dst.is_symlink():
                    raise SafeFSError("symlink copy rejected")
                shutil.copy2(src, dst)
                return CommandResult(read_paths={src.relative_to(self.workspace).as_posix()}, written_paths={dst.relative_to(self.workspace).as_posix()})
            if name == "mv":
                src = resolve_inside(self.workspace, self.cwd, argv[-2])
                dst = resolve_inside(self.workspace, self.cwd, argv[-1])
                if src.is_symlink() or dst.is_symlink():
                    raise SafeFSError("symlink move rejected")
                shutil.move(src, dst)
                return CommandResult(written_paths={src.relative_to(self.workspace).as_posix(), dst.relative_to(self.workspace).as_posix()})
        except Exception as exc:
            return CommandResult(stderr=f"{exc}\n", exit_code=1)
        return CommandResult(stderr=f"unsupported builtin: {argv[0]}\n", exit_code=127)

    def _read_many(self, paths: tuple[str, ...] | list[str]) -> CommandResult:
        if not paths:
            return CommandResult()
        chunks: list[str] = []
        read: set[str] = set()
        for item in paths:
            if item.startswith("-"):
                continue
            path = resolve_inside(self.workspace, self.cwd, item)
            data = read_file_no_follow(path).decode("utf-8", errors="replace")
            chunks.append(data)
            read.add(path.relative_to(self.workspace).as_posix())
        return CommandResult(stdout="".join(chunks), read_paths=read)

    def _grep(self, argv: tuple[str, ...], stdin: bytes | None) -> CommandResult:
        args = [a for a in argv[1:] if not a.startswith("-")]
        if not args:
            return CommandResult(stderr="missing pattern\n", exit_code=2)
        pattern = args[0]
        paths = args[1:]
        read: set[str] = set()
        matches: list[str] = []
        if stdin is not None and not paths:
            for line in stdin.decode("utf-8", errors="replace").splitlines():
                if pattern in line:
                    matches.append(line)
        elif paths:
            for raw_path in paths:
                path = resolve_inside(self.workspace, self.cwd, raw_path)
                candidates = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()]
                for file_path in candidates:
                    text = read_file_no_follow(file_path).decode("utf-8", errors="replace")
                    rel = file_path.relative_to(self.workspace).as_posix()
                    read.add(rel)
                    for line in text.splitlines():
                        if pattern in line or fnmatch.fnmatch(line, pattern):
                            matches.append(f"{rel}:{line}")
        return CommandResult(stdout="\n".join(matches) + ("\n" if matches else ""), exit_code=0 if matches else 1, read_paths=read)

    def _execute_subprocess(self, argv: tuple[str, ...], stdin: bytes | None, timeout_s: int) -> CommandResult:
        exec_argv = list(argv)
        if exec_argv and exec_argv[0] == "pytest":
            exec_argv = [sys.executable, "-m", "pytest", *exec_argv[1:]]
        elif exec_argv and exec_argv[0] == "python":
            exec_argv[0] = sys.executable
        try:
            env = None
            if is_pytest_invocation(exec_argv):
                env = os.environ.copy()
                env.update(pytest_env_overrides(env.get("PYTEST_ADDOPTS")))
            proc = subprocess.run(
                exec_argv,
                input=stdin,
                cwd=self.cwd,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
            return CommandResult(
                stdout=proc.stdout.decode("utf-8", errors="replace"),
                stderr=proc.stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return CommandResult(stderr=f"{argv[0]}: command not found\n", exit_code=127)
        except subprocess.TimeoutExpired:
            return CommandResult(stderr="command timed out\n", exit_code=124)
