#!/usr/bin/env python3
"""Differential gate: tree-sitter AST lowering vs the conservative subset
lowering.

For every command in the supported v1 subset (a curated battery plus every
oracle command in the task catalog), the direct tree-sitter AST lowering must
produce exactly the same Program chain as the legacy shlex subset lowering.
This is the parser/executor one-to-one consistency check that gates expanding
the shell subset.

Run:
    PYTHONPATH=. .venv/bin/python scripts/check_parser_consistency.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.broker.parser import (
    ParseError,
    TS_BACKEND,
    _subset_lower,
    parse_command,
)

# Curated battery spanning the supported v1 grammar surface.
BATTERY = [
    "pwd",
    "ls",
    "ls -la docs",
    "cat docs/release.txt",
    "cat README.md | grep ready && pwd",
    "ls | wc -l | head -n 2",
    "cd docs && cat release.txt || ls",
    "mkdir -p a/b ; touch a/b/c.txt",
    "cp input.txt output.txt",
    "mv a.txt b.txt",
    "grep needle src",
    "rg needle src",
    'grep "hello world" file.txt',
    "grep 'a b' file.txt",
    "head -n 5 file.txt",
    "tail -n 5 file.txt",
    "wc -l file.txt",
    "cat a.txt > out.txt",
    "cat a.txt >> out.txt",
    "wc -l < in.txt",
    "cat < in.txt > out.txt",
    "cat foo > bar | wc -l",
    "find . -name needle.txt 2>/dev/null",
    "grep needle src 2>/dev/null",
    "cat a.txt 2>&1",
    "python3 script.py > out.txt 2>&1",
    "ls -la 1>listing.txt",
    "cat a.txt 2>>errors.log",
    "wc -l < in.txt 2>/dev/null",
    "cat a.txt &>/dev/null",
    "echo hello",
    "printf '%s' value",
    "uname -a",
    "tr a b < in.txt",
    "sort names.txt | uniq",
    "find . -name needle.txt",
    "python -c 'print(1)'",
    "python3 script.py",
    "pytest -q",
    "git status",
    "curl http://127.0.0.1:8765/data",
    "jq .key data.json",
    "sed -n 1p file.txt",
    "awk NF file.txt",
    "touch one.txt two.txt",
    "cat one.txt two.txt | wc -l > counts.txt",
]


def oracle_commands() -> list[str]:
    catalog = json.loads((ROOT / "data" / "spike_tasks.json").read_text())
    commands: list[str] = []
    for task in catalog:
        commands.extend(task.get("oracle", {}).get("commands", []))
    return commands


def main() -> int:
    cases = BATTERY + oracle_commands()
    mismatches: list[str] = []
    checked = 0
    for raw in cases:
        try:
            program = parse_command(raw)
        except ParseError as exc:
            mismatches.append(f"AST rejected supported command {raw!r}: {exc}")
            continue
        if program.parser_backend != TS_BACKEND:
            print(f"tree-sitter unavailable ({program.parser_backend}); nothing to compare")
            return 0
        try:
            subset_chain = _subset_lower(raw)
        except ParseError as exc:
            mismatches.append(f"subset rejected supported command {raw!r}: {exc}")
            continue
        checked += 1
        if program.chain != subset_chain:
            mismatches.append(
                f"lowering mismatch for {raw!r}:\n  ast    = {program.chain}\n  subset = {subset_chain}"
            )

    print(f"compared {checked} commands ({len(BATTERY)} battery + {len(cases) - len(BATTERY)} oracle)")
    if mismatches:
        for item in mismatches:
            print(f"MISMATCH: {item}")
        print("parser consistency check FAILED")
        return 1
    print("parser consistency check passed: AST lowering == subset lowering on the v1 surface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
