#!/usr/bin/env python3
"""Docker gate for the strace quarantine path.

Runs one known command (must NOT be quarantined) and one unknown command
(must run under strace in a one-shot CAP_SYS_PTRACE container) against a real
rootless Docker daemon, and checks the parsed event model plus container
cleanup.

Run:
    PYTHONPATH=. .venv/bin/python scripts/check_strace_quarantine.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.broker.parser import parse_command
from least_privilege_env.sandbox.docker import DockerSandbox


def leftover_containers() -> list[str]:
    proc = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "name=lp-rl-"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    ok, detail = DockerSandbox.check_available(require_rootless=os.environ.get("LP_RL_REQUIRE_ROOTLESS", "1") != "0")
    if not ok:
        print(f"docker unavailable: {detail}")
        return 1

    with tempfile.TemporaryDirectory(prefix="lp-quarantine-smoke-") as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "x.txt").write_text("hello\n")
        sandbox = DockerSandbox(workspace)
        try:
            known = sandbox.execute(parse_command("cat x.txt"))
            assert known.exit_code == 0 and "hello" in known.stdout, known
            assert known.quarantine_events == (), known.quarantine_events
            print("known command: executed in episode container, no quarantine events")

            unknown = sandbox.execute(parse_command("/bin/echo quarantine-probe"))
            assert unknown.exit_code == 0 and "quarantine-probe" in unknown.stdout, unknown
            events = unknown.quarantine_events
            assert events, "unknown command produced no quarantine events"
            categories = {event.category for event in events}
            syscalls = {event.syscall for event in events}
            assert "meta" not in categories, events
            assert "execve" in syscalls, sorted(syscalls)
            assert "file" in categories, sorted(categories)
            print(f"unknown command: quarantined, {len(events)} events, categories={sorted(categories)}")
            sample = [event.render() for event in events[:5]]
            for line in sample:
                print(f"  {line}")
        finally:
            sandbox.close()

    leftovers = leftover_containers()
    if leftovers:
        print(f"leftover containers: {leftovers}")
        return 1
    print("no leftover containers")
    print("strace quarantine check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
