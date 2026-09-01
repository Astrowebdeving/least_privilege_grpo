#!/usr/bin/env python3
"""Disjointness firewall against locally pinned external benchmark corpora.

The vocabulary axis rejects distinctive external tool identifiers in every MCP
surface (oracle, negative oracle, transient faults, services, and prompt API
mentions). In --strict mode, the checker also scans the full local prompt/error
sources for equal-length n-gram overlap. Strict mode fails closed when those
pinned sources are unavailable; it never substitutes a small sample and calls
that proof of disjointness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "external_benchmark_vocab.json"
SHELL_ALLOWLIST = {
    "ls", "cat", "grep", "rm", "cp", "mv", "stat", "find", "sed", "awk", "head",
    "tail", "echo", "python", "python3", "pytest", "git", "wc", "sort", "uniq",
    "chmod", "touch", "mkdir", "rmdir", "diff", "patch", "tar", "curl", "test",
    "read", "write", "list", "search", "update", "delete", "create", "get", "set",
    "none", "null", "true", "false",
}
EXTERNAL_FILES = [
    ROOT / "external/toolprivbench/evaluation/data/goodcase_final/benchmark.jsonl",
    ROOT / "external/fortis/shared/all_skills.json",
    ROOT / "external/fortis/shared/prompts.py",
    ROOT / "external/metatool/dataset/plugin_info.json",
    ROOT / "external/metatool/dataset/tmp_dataset/Task1.json",
    ROOT / "external/metatool/dataset/tmp_dataset/Task2-Subtask1.json",
    ROOT / "external/metatool/dataset/tmp_dataset/Task2-Subtask2.json",
    ROOT / "external/metatool/dataset/tmp_dataset/Task2-Subtask3.json",
    ROOT / "external/metatool/dataset/tmp_dataset/Task2-Subtask4.json",
]
ERROR_WORDS = re.compile(r"\b(error|failed|failure|timeout|timed out|unavailable|offline|reset|busy)\b", re.I)


def distinctive(token: str) -> bool:
    token = token.strip().casefold()
    if not token or token in SHELL_ALLOWLIST:
        return False
    return "_" in token or "-" in token or len(token) >= 12


def strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def word_ngrams(text: str, n: int) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def load_catalog(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = raw.get("tasks", []) if isinstance(raw, dict) else raw
    tool_tokens: set[str] = set()
    prompts: list[str] = []
    errors: list[str] = []
    for task in tasks:
        prompts.append(str(task.get("prompt", "")))
        tool_tokens.update(str(x).casefold() for x in task.get("allowed_services", []))
        for section in ("oracle", "negative_oracle"):
            for call in (task.get(section) or {}).get("mcp_calls", []):
                for key in ("server", "tool"):
                    if call.get(key):
                        tool_tokens.add(str(call[key]).casefold())
        for fault in task.get("transient_faults", []):
            for key in ("server", "tool"):
                if fault.get(key):
                    tool_tokens.add(str(fault[key]).casefold())
            if fault.get("error"):
                errors.append(str(fault["error"]))
        # Catch an API identifier named only in the prompt, without treating
        # ordinary prose words as tool names.
        tool_tokens.update(re.findall(r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+\b", prompts[-1].casefold()))
    return tasks, tool_tokens, prompts, errors


def external_strings(path: Path) -> Iterable[tuple[str, str]]:
    rel = str(path.relative_to(ROOT))
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    for text in strings(json.loads(line)):
                        yield f"{rel}:{line_no}", text
    elif path.suffix == ".json":
        for text in strings(json.loads(path.read_text(encoding="utf-8"))):
            yield rel, text
    else:
        yield rel, path.read_text(encoding="utf-8")


def scan_external(candidate_grams: set[str], n: int, error_only: bool = False):
    hits: list[dict[str, str]] = []
    for path in EXTERNAL_FILES:
        for source, text in external_strings(path):
            if error_only and not ERROR_WORDS.search(text):
                continue
            shared = candidate_grams & word_ngrams(text, n)
            for span in sorted(shared):
                hits.append({"source": source, "span": span})
                if len(hits) >= 20:
                    return hits
    return hits


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    external_tools = {str(x).casefold() for x in vocab.get("union", []) if distinctive(str(x))}
    tasks, candidate_tools, prompts, errors = load_catalog(args.catalog)
    candidate_distinctive = {x for x in candidate_tools if distinctive(x)}
    tool_hits = sorted(candidate_distinctive & external_tools)

    missing = [str(path.relative_to(ROOT)) for path in EXTERNAL_FILES if not path.is_file()]
    prompt_hits: list[dict[str, str]] = []
    error_hits: list[dict[str, str]] = []
    if args.strict and not missing:
        prompt_grams = set().union(*(word_ngrams(prompt, args.ngram) for prompt in prompts))
        error_grams = set().union(*(word_ngrams(error, 6) for error in errors)) if errors else set()
        prompt_hits = scan_external(prompt_grams, args.ngram)
        error_hits = scan_external(error_grams, 6, error_only=True) if error_grams else []

    failures: list[str] = []
    if tool_hits:
        failures.append(f"{len(tool_hits)} distinctive external tool identifiers: {tool_hits[:15]}")
    if args.strict and missing:
        failures.append(f"strict corpus unavailable; missing {missing}")
    if args.strict and prompt_hits:
        failures.append(f"{len(prompt_hits)} prompt spans overlap pinned external text: {prompt_hits[:5]}")
    if args.strict and error_hits:
        failures.append(f"{len(error_hits)} failure-message spans overlap pinned external errors: {error_hits[:5]}")

    report = {
        "catalog": str(args.catalog), "n_tasks": len(tasks), "strict": args.strict,
        "candidate_distinctive_tools": sorted(candidate_distinctive),
        "external_distinctive_tool_count": len(external_tools), "tool_name_overlap": tool_hits,
        "prompt_span_overlap": prompt_hits, "error_span_overlap": error_hits,
        "missing_external_sources": missing,
        "external_source_fingerprints": [file_fingerprint(path) for path in EXTERNAL_FILES if path.is_file()],
        "status": "fail" if failures else "pass", "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"disjointness: catalog={args.catalog.name} tasks={len(tasks)} strict={args.strict}")
    print(f"  distinctive tool overlap={len(tool_hits)}")
    if args.strict:
        print(f"  full-corpus prompt overlap={len(prompt_hits)} error overlap={len(error_hits)} missing_sources={len(missing)}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    claim = "distinctive-tool vocabulary"
    if args.strict:
        claim += " and equal-length full-corpus prompt/error spans"
    print(f"PASS: disjoint on {claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())