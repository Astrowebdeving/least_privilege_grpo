"""Strace quarantine event model and trace parsing.

Unknown commands are executed under ``strace -f -e trace=%file,%process,
%network`` inside a dedicated one-shot container that adds only
CAP_SYS_PTRACE (see ``DockerSandbox``). The raw trace is parsed here into a
small structured event model so the broker audit can reason about what an
unknown command actually touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


# Marker emitted by the quarantine wrapper between the program's own stderr
# and the strace trace dump. rpartition keeps the wrapper's (last) marker
# authoritative even if the traced program prints the same string.
STRACE_SENTINEL = "__LP_STRACE_EVENTS__"

PROCESS_SYSCALLS = {
    "execve",
    "execveat",
    "fork",
    "vfork",
    "clone",
    "clone3",
    "wait4",
    "waitid",
    "kill",
    "tkill",
    "tgkill",
    "exit",
    "exit_group",
    "ptrace",
}
NETWORK_SYSCALLS = {
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "sendmmsg",
    "recvmmsg",
    "getsockname",
    "getpeername",
    "getsockopt",
    "setsockopt",
    "shutdown",
}

_PID_PREFIX_RE = re.compile(r"^\d+\s+")
_SYSCALL_RE = re.compile(r"^(\w+)\((.*)$")
_RESUMED_RE = re.compile(r"^<\.\.\.\s+(\w+)\s+resumed>(.*)$")
_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class QuarantineEvent:
    category: str  # "file" | "process" | "network" | "meta"
    syscall: str
    target: str

    def render(self) -> str:
        return f"{self.category}:{self.syscall}:{self.target}"


def _categorize(syscall: str) -> str:
    if syscall in PROCESS_SYSCALLS:
        return "process"
    if syscall in NETWORK_SYSCALLS:
        return "network"
    return "file"


def _extract_target(args: str) -> str:
    quoted = _QUOTED_RE.search(args)
    if quoted:
        return quoted.group(1)
    head = args.rsplit(")", 1)[0] if ")" in args else args
    return head.split(",", 1)[0].strip(") ").strip()


def parse_strace_events(text: str, limit: int = 256) -> tuple[QuarantineEvent, ...]:
    """Parse strace -f output into deduplicated structured events."""
    events: list[QuarantineEvent] = []
    seen: set[QuarantineEvent] = set()
    truncated = False
    for line in text.splitlines():
        line = _PID_PREFIX_RE.sub("", line.strip())
        if not line or line.startswith(("+++", "---")):
            continue
        resumed = _RESUMED_RE.match(line)
        if resumed:
            syscall, args = resumed.group(1), resumed.group(2)
        else:
            match = _SYSCALL_RE.match(line)
            if not match:
                continue
            syscall, args = match.group(1), match.group(2)
        event = QuarantineEvent(category=_categorize(syscall), syscall=syscall, target=_extract_target(args)[:200])
        if event in seen:
            continue
        if len(events) >= limit:
            truncated = True
            break
        seen.add(event)
        events.append(event)
    if truncated:
        events.append(QuarantineEvent(category="meta", syscall="events-truncated", target=f">{limit}"))
    return tuple(events)


def split_quarantine_stderr(stderr_text: str) -> tuple[str, tuple[QuarantineEvent, ...]]:
    """Split wrapper stderr into (program stderr, parsed strace events).

    When the sentinel is missing the trace never ran (e.g. strace not in the
    image); a meta event records the degraded telemetry instead of silently
    dropping the audit signal.
    """
    before, sep, after = stderr_text.rpartition(STRACE_SENTINEL + "\n")
    if not sep:
        return stderr_text, (
            QuarantineEvent(category="meta", syscall="strace-unavailable", target=stderr_text.strip()[:200]),
        )
    return before, parse_strace_events(after)
