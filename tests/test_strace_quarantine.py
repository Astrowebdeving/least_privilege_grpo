from __future__ import annotations

from pathlib import Path

import pytest

from least_privilege_env.broker.parser import CommandNode
from least_privilege_env.sandbox.docker import DockerSandbox, QUARANTINE_EXEMPT_COMMANDS
from least_privilege_env.sandbox.strace import (
    STRACE_SENTINEL,
    QuarantineEvent,
    parse_strace_events,
    split_quarantine_stderr,
)


SAMPLE_TRACE = """\
12345 execve("/usr/bin/uname", ["uname", "-a"], 0x7ffd environ) = 0
12345 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
12345 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=12345, ...}, AT_EMPTY_PATH) = 0
12345 socket(AF_INET, SOCK_DGRAM, IPPROTO_IP) = 4
12345 connect(4, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.1")}, 16) = -1 ENETUNREACH
12346 clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID) = 12347
12345 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
+++ exited with 0 +++
--- SIGCHLD {si_signo=SIGCHLD} ---
12345 <... wait4 resumed>[{WIFEXITED(s)}], 0, NULL) = 12347
"""


def test_parse_strace_events_categorizes_and_dedupes() -> None:
    events = parse_strace_events(SAMPLE_TRACE)
    rendered = [event.render() for event in events]
    assert "process:execve:/usr/bin/uname" in rendered
    assert "file:openat:/etc/ld.so.cache" in rendered
    assert "network:socket:AF_INET" in rendered
    assert "network:connect:10.0.0.1" in rendered
    assert "process:clone:child_stack=NULL" in rendered
    assert "process:wait4:" in rendered[-1] or any(e.syscall == "wait4" for e in events)
    # duplicate openat collapsed; +++/--- lines skipped
    assert sum(1 for e in events if e.syscall == "openat") == 1


def test_parse_strace_events_caps_output() -> None:
    lines = "\n".join(f'1 openat(AT_FDCWD, "/tmp/f{i}", O_RDONLY) = 3' for i in range(300))
    events = parse_strace_events(lines, limit=10)
    assert len(events) == 11
    assert events[-1] == QuarantineEvent("meta", "events-truncated", ">10")


def test_split_quarantine_stderr_extracts_trace() -> None:
    stderr = f"warning: something\n{STRACE_SENTINEL}\n1 execve(\"/usr/bin/id\", [\"id\"], env) = 0\n"
    program_stderr, events = split_quarantine_stderr(stderr)
    assert program_stderr == "warning: something\n"
    assert events == (QuarantineEvent("process", "execve", "/usr/bin/id"),)


def test_split_quarantine_stderr_flags_missing_trace() -> None:
    program_stderr, events = split_quarantine_stderr("bash: strace: command not found\n")
    assert "strace" in program_stderr
    assert len(events) == 1
    assert events[0].category == "meta"
    assert events[0].syscall == "strace-unavailable"


def test_unknown_commands_route_to_quarantine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sandbox = DockerSandbox(tmp_path)
    known = CommandNode(argv=("cat", "x.txt"))
    unknown = CommandNode(argv=("/bin/echo", "quarantine-probe"))
    monkeypatch.delenv("LP_RL_STRACE_QUARANTINE", raising=False)
    assert sandbox._needs_quarantine(known) is False
    assert sandbox._needs_quarantine(unknown) is True
    monkeypatch.setenv("LP_RL_STRACE_QUARANTINE", "0")
    assert sandbox._needs_quarantine(unknown) is False


def test_quarantine_exempt_covers_classifier_known_families() -> None:
    # Every command the classifier maps to a known family must bypass
    # quarantine; spot-check representatives of each set.
    for name in ("cat", "jq", "mkdir", "python", "curl", "cd"):
        assert name in QUARANTINE_EXEMPT_COMMANDS
