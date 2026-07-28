from __future__ import annotations

from pathlib import Path
import os
import subprocess
import uuid
import weakref
from urllib.parse import urlparse

from least_privilege_env.broker.classifier import (
    EXECUTION_COMMANDS,
    PRINT_COMMANDS,
    READ_COMMANDS,
    SCOPED_MUTATION,
    TRANSFORM_COMMANDS,
)
from least_privilege_env.broker.parser import CommandNode, Program, is_devnull
from least_privilege_env.sandbox.base import (
    CommandResult,
    Sandbox,
    execution_read_paths,
    input_redirect,
    is_pytest_invocation,
    pytest_env_overrides,
    route_output_streams,
)
from least_privilege_env.sandbox.safe_fs import diff_snapshots, read_file_no_follow, resolve_inside, snapshot_tree, write_file_no_follow
from least_privilege_env.sandbox.strace import (
    STRACE_SENTINEL,
    QuarantineEvent,
    split_quarantine_stderr,
)


# Commands the classifier maps to a known risk family. Anything else reaching
# the sandbox is an "unknown command" and is executed under strace quarantine.
QUARANTINE_EXEMPT_COMMANDS = (
    READ_COMMANDS | TRANSFORM_COMMANDS | PRINT_COMMANDS | SCOPED_MUTATION | EXECUTION_COMMANDS | {"cd", "curl", "wget"}
)

PATH_ARG_COMMANDS = {
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "ls",
    "find",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "python",
    "python3",
    "pytest",
}

GREP_OPTIONS_WITH_VALUE = {
    "-A",
    "-B",
    "-C",
    "-D",
    "-d",
    "-e",
    "-f",
    "-m",
    "--after-context",
    "--before-context",
    "--binary-files",
    "--context",
    "--devices",
    "--directories",
    "--exclude",
    "--exclude-dir",
    "--exclude-from",
    "--include",
    "--label",
    "--max-count",
    "--regexp",
}


class DockerUnavailable(RuntimeError):
    pass


class DockerSandbox(Sandbox):
    """Docker-backed broker-owned execution.

    Commands are executed from validated argv nodes, never from raw shell text.
    Under rootless Docker, container-root maps to the unprivileged host user, so
    this sandbox intentionally does not pass `--user`.
    """

    def __init__(self, workspace: Path, image: str = "lp-rl-worker:latest", allowed_services: tuple[str, ...] = ()):
        self.workspace = workspace.resolve()
        self.cwd = self.workspace
        self.image = image
        self.allowed_services = set(allowed_services)
        self.name = f"lp-rl-{uuid.uuid4().hex[:12]}"
        self._started = False
        self._proxy_started = False
        self.proxy_requests: list[str] = []

    @staticmethod
    def docker_host_hint() -> str:
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        return os.environ.get("DOCKER_HOST", f"unix:///run/user/{uid}/docker.sock")

    @staticmethod
    def check_available(require_rootless: bool = False) -> tuple[bool, str]:
        try:
            proc = subprocess.run(["docker", "info", "--format", "{{json .SecurityOptions}}"], capture_output=True, text=True, timeout=10, check=False)
        except FileNotFoundError:
            return False, "docker CLI not found"
        except subprocess.TimeoutExpired:
            return False, "docker info timed out"
        if proc.returncode != 0:
            return False, proc.stderr.strip() or proc.stdout.strip()
        security = proc.stdout.strip()
        if require_rootless and "rootless" not in security:
            return False, f"docker daemon is available but not rootless; security={security}"
        return True, security

    def start(self) -> None:
        ok, detail = self.check_available(require_rootless=os.environ.get("LP_RL_REQUIRE_ROOTLESS", "1") != "0")
        if not ok:
            raise DockerUnavailable(f"{detail}; set DOCKER_HOST={self.docker_host_hint()} after rootless setup")
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--network=none",
            "--cpus=2",
            "--memory=4g",
            "--pids-limit=256",
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "-v",
            f"{self.workspace}:/workspace:rw",
            self.image,
            "sleep",
            "infinity",
        ]
        subprocess.run(args, check=True, capture_output=True, text=True)
        self._started = True
        # GC backstop: TRL never resets the last env instance of a rollout, so
        # without this every episode that ends off the close() path leaks one
        # running sleep-infinity container.
        self._finalizer = weakref.finalize(self, DockerSandbox._force_remove, self.name)
        if "loopback_proxy" in self.allowed_services:
            self._start_loopback_proxy()

    @staticmethod
    def _force_remove(name: str) -> None:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    def snapshot(self):
        return snapshot_tree(self.workspace)

    def execute(self, program: Program, timeout_s: int = 10) -> CommandResult:
        if not self._started:
            self.start()
        before = self.snapshot()
        stdin: bytes | None = None
        stdout = ""
        stderr = ""
        exit_code = 0
        read_paths: set[str] = set()
        written_paths: set[str] = set()
        proxy_requests: list[str] = []
        quarantine_events: list[QuarantineEvent] = []
        resource_limited = False
        for chain_item in program.chain:
            if chain_item.op_before == "&&" and exit_code != 0:
                continue
            if chain_item.op_before == "||" and exit_code == 0:
                continue
            for command in chain_item.pipeline.commands:
                proxy_requests.extend(self._record_proxy_requests(command))
                result = self._exec_argv(command, stdin, timeout_s)
                stdin = result.stdout.encode("utf-8", errors="replace")
                stdout, stderr, exit_code = result.stdout, stderr + result.stderr, result.exit_code
                read_paths.update(result.read_paths)
                written_paths.update(result.written_paths)
                quarantine_events.extend(result.quarantine_events)
                resource_limited = resource_limited or result.resource_limited
                if exit_code != 0:
                    break
        after = self.snapshot()
        added, changed, deleted = diff_snapshots(before, after)
        written_paths.update(added | changed | deleted)
        rootfs_anomalies = self._docker_diff_anomalies()
        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            read_paths=read_paths,
            written_paths=written_paths,
            rootfs_anomalies=rootfs_anomalies,
            proxy_requests=proxy_requests,
            resource_limited=resource_limited,
            quarantine_events=tuple(quarantine_events),
        )

    def _exec_argv(self, command: CommandNode, stdin: bytes | None, timeout_s: int) -> CommandResult:
        if command.name == "cd":
            target = command.argv[1] if len(command.argv) > 1 else "."
            resolved = resolve_inside(self.workspace, self.cwd, target)
            if not resolved.is_dir():
                return CommandResult(stderr=f"cd: no such directory: {target}\n", exit_code=1)
            self.cwd = resolved
            return CommandResult()

        read_paths: set[str] = self._read_evidence_paths(
            command, has_stdin=stdin is not None
        )
        written_paths: set[str] = self._write_evidence_paths(command)
        inp = input_redirect(command.redirects)
        if inp is not None:
            if is_devnull(inp.target):
                stdin = b""
            else:
                target = resolve_inside(self.workspace, self.cwd, inp.target)
                stdin = read_file_no_follow(target)
                read_paths.add(target.relative_to(self.workspace).as_posix())

        argv = self._container_argv(command)
        rel_cwd = self.cwd.relative_to(self.workspace).as_posix()
        workdir = "/workspace" if rel_cwd == "." else f"/workspace/{rel_cwd}"
        quarantine_events: tuple[QuarantineEvent, ...] = ()
        if self._needs_quarantine(command):
            raw_stdout, stderr, returncode, quarantine_events = self._run_quarantined(argv, stdin, workdir, timeout_s)
        else:
            args = ["docker", "exec", "-i", "-w", workdir]
            if is_pytest_invocation(argv):
                for key, value in pytest_env_overrides().items():
                    args.extend(["-e", f"{key}={value}"])
            args.extend([self.name, *argv])
            try:
                proc = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout_s, check=False)
            except subprocess.TimeoutExpired:
                return CommandResult(stderr="command timed out\n", exit_code=124, resource_limited=True)
            raw_stdout = proc.stdout
            stderr = proc.stderr.decode("utf-8", errors="replace")
            returncode = proc.returncode
        stdout = raw_stdout.decode("utf-8", errors="replace")
        resource_limited = returncode in {124, 137, 143}

        def resolve_target(token: str) -> tuple[Path, str]:
            path = resolve_inside(self.workspace, self.cwd, token)
            return path, path.relative_to(self.workspace).as_posix()

        stdout, stderr, written = route_output_streams(
            stdout,
            stderr,
            command.redirects,
            resolve_target,
            lambda path, data, append: write_file_no_follow(path, data, append=append),
        )
        written_paths |= written
        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            read_paths=read_paths,
            written_paths=written_paths,
            resource_limited=resource_limited,
            quarantine_events=quarantine_events,
        )

    def _container_argv(self, command: CommandNode) -> list[str]:
        if command.name in {"curl", "wget"}:
            return list(command.argv)
        if command.name not in PATH_ARG_COMMANDS:
            return list(command.argv)
        path_indices = self._path_arg_indices(command)
        out: list[str] = [command.argv[0]]
        skip_next = False
        for index, arg in enumerate(command.argv[1:], start=1):
            if skip_next:
                out.append(arg)
                skip_next = False
                continue
            if command.name in {"python", "python3"} and arg in {"-c", "-m"}:
                out.append(arg)
                skip_next = True
                continue
            if arg.startswith("-"):
                out.append(arg)
                continue
            if "://" in arg:
                out.append(arg)
                continue
            if index not in path_indices:
                out.append(arg)
                continue
            resolved = resolve_inside(self.workspace, self.cwd, arg)
            rel = resolved.relative_to(self.workspace).as_posix()
            out.append(f"/workspace/{rel}")
        return out

    @staticmethod
    def _path_arg_indices(command: CommandNode) -> set[int]:
        """Return argv indices that are workspace paths for supported tools.

        This is deliberately command-aware. Treating every non-option token as
        a path corrupts grep patterns after flags (``grep -r PATTERN .``) and
        find expression operands (``find . -type f``).
        """
        argv = command.argv
        if command.name in {"grep", "rg"}:
            paths: set[int] = set()
            pattern_seen = False
            pattern_from_option = False
            index = 1
            while index < len(argv):
                arg = argv[index]
                if arg == "--":
                    index += 1
                    while index < len(argv):
                        if not pattern_seen and not pattern_from_option:
                            pattern_seen = True
                        else:
                            paths.add(index)
                        index += 1
                    break
                if arg in GREP_OPTIONS_WITH_VALUE:
                    if arg in {"-e", "--regexp"}:
                        pattern_from_option = True
                    index += 2
                    continue
                if arg.startswith("-") and arg != "-":
                    if arg.startswith(("-e", "--regexp=")):
                        pattern_from_option = True
                    index += 1
                    continue
                if not pattern_seen and not pattern_from_option:
                    pattern_seen = True
                else:
                    paths.add(index)
                index += 1
            return paths
        if command.name == "find":
            paths: set[int] = set()
            for index, arg in enumerate(argv[1:], start=1):
                if arg.startswith("-") or arg in {"!", "(", ")"}:
                    break
                paths.add(index)
            return paths
        if command.name in {"python", "python3"}:
            if len(argv) >= 3 and argv[1] == "-m":
                return set(range(3, len(argv))) if argv[2] == "pytest" else set()
            if len(argv) >= 2 and argv[1] == "-c":
                return set(range(3, len(argv)))
            return {1} if len(argv) >= 2 and not argv[1].startswith("-") else set()
        if command.name == "pytest":
            return {index for index, arg in enumerate(argv[1:], start=1) if not arg.startswith("-")}
        return {index for index, arg in enumerate(argv[1:], start=1) if not arg.startswith("-")}

    def _read_evidence_paths(
        self, command: CommandNode, *, has_stdin: bool = False
    ) -> set[str]:
        if command.name in {"pwd", "mkdir", "touch"}:
            return set()
        if command.name in {"cp", "mv"} and len(command.argv) >= 3:
            return self._expand_existing_path(command.argv[-2])
        if command.name in {"grep", "rg"}:
            targets = [command.argv[index] for index in sorted(self._path_arg_indices(command))]
            # A pathless grep with piped stdin reads the stream, not cwd.
            return self._expand_many(targets)
        if command.name in {"cat", "head", "tail", "wc"}:
            targets = [arg for arg in command.argv[1:] if not arg.startswith("-")]
            return self._expand_many(targets)
        # `ls`/`find` enumerate names; they do not read file CONTENT. Expanding
        # a directory into per-file reads here made plain `ls` register every
        # file (incl. .env) as accessed, firing the secret post-scan and
        # inflating scope. The local sandbox never did this; match it.
        return execution_read_paths(self.workspace, self.cwd, command)

    def _write_evidence_paths(self, command: CommandNode) -> set[str]:
        if command.name in {"cp", "mv"} and len(command.argv) >= 3:
            return self._expand_target_path(command.argv[-1])
        if command.name in {"touch", "mkdir"}:
            return self._expand_many([arg for arg in command.argv[1:] if not arg.startswith("-")])
        return set()

    def _expand_many(self, targets: list[str]) -> set[str]:
        paths: set[str] = set()
        for target in targets:
            paths.update(self._expand_existing_path(target))
        return paths

    def _expand_existing_path(self, target: str) -> set[str]:
        try:
            resolved = resolve_inside(self.workspace, self.cwd, target)
        except Exception:
            return set()
        if not resolved.exists():
            return set()
        if resolved.is_file():
            return {resolved.relative_to(self.workspace).as_posix()}
        if resolved.is_dir():
            out: set[str] = set()
            for path in resolved.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    out.add(path.relative_to(self.workspace).as_posix())
            return out
        return set()

    def _expand_target_path(self, target: str) -> set[str]:
        try:
            resolved = resolve_inside(self.workspace, self.cwd, target)
        except Exception:
            return set()
        return {resolved.relative_to(self.workspace).as_posix()}

    def _start_loopback_proxy(self) -> None:
        if self._proxy_started:
            return
        subprocess.run(
            ["docker", "exec", "-d", self.name, "python", "/opt/lp-rl/fake_proxy.py", "--port", "8765"],
            capture_output=True,
            text=True,
            check=True,
        )
        self._proxy_started = True

    def _record_proxy_requests(self, command: CommandNode) -> list[str]:
        if command.name not in {"curl", "wget"}:
            return []
        requests: list[str] = []
        for arg in command.argv[1:]:
            if arg.startswith(("http://", "https://")):
                parsed = urlparse(arg)
                if parsed.hostname in {"127.0.0.1", "localhost"}:
                    request = parsed.path or "/"
                    self.proxy_requests.append(request)
                    requests.append(request)
        return requests

    def _docker_diff_anomalies(self) -> set[str]:
        proc = subprocess.run(["docker", "diff", self.name], capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode != 0:
            return {f"docker-diff-error:{proc.stderr.strip()[:120]}"}
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    # ---- strace quarantine for unknown commands -------------------------
    #
    # Unknown commands (anything the classifier cannot map to a known risk
    # family) run in a dedicated one-shot container under
    # `strace -f -e trace=%file,%process,%network`. The long-lived episode
    # container keeps --cap-drop=ALL; only the throwaway quarantine container
    # adds CAP_SYS_PTRACE, and it shares nothing but the /workspace mount
    # (its rootfs is --read-only with a private /tmp, so rootfs writes are
    # discarded; host no-follow snapshots stay authoritative for /workspace).

    @staticmethod
    def quarantine_enabled() -> bool:
        return os.environ.get("LP_RL_STRACE_QUARANTINE", "1") != "0"

    def _needs_quarantine(self, command: CommandNode) -> bool:
        return self.quarantine_enabled() and command.name not in QUARANTINE_EXEMPT_COMMANDS

    def _run_quarantined(
        self, argv: list[str], stdin: bytes | None, workdir: str, timeout_s: int
    ) -> tuple[bytes, str, int, tuple[QuarantineEvent, ...]]:
        qname = f"{self.name}-q{uuid.uuid4().hex[:8]}"
        wrapper = (
            "strace -qq -f -s 256 -e trace=%file,%process,%network -o /tmp/.lp_strace -- \"$@\"; rc=$?; "
            f"printf '%s\\n' {STRACE_SENTINEL} >&2; cat /tmp/.lp_strace >&2 2>/dev/null; exit $rc"
        )
        args = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            qname,
            "--network=none",
            "--cpus=2",
            "--memory=4g",
            "--pids-limit=256",
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "--cap-add=SYS_PTRACE",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "-v",
            f"{self.workspace}:/workspace:rw",
            "-w",
            workdir,
            self.image,
            "bash",
            "-c",
            wrapper,
            "lp-quarantine",
            *argv,
        ]
        try:
            # Extra headroom over the exec timeout for one-shot container startup.
            proc = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout_s + 20, check=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", qname], capture_output=True, check=False)
            return b"", "command timed out\n", 124, (QuarantineEvent("meta", "quarantine-timeout", ""),)
        stderr, events = split_quarantine_stderr(proc.stderr.decode("utf-8", errors="replace"))
        return proc.stdout, stderr, proc.returncode, events

    def close(self) -> None:
        if self._started:
            finalizer = getattr(self, "_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, check=False)
            self._started = False
