#!/usr/bin/env python3
"""Run the pinned FORTIS benchmark through a local OpenAI-compatible API.

FORTIS owns the prompts, query data, parsers, and four-way classification.
This wrapper changes only transport, adds deterministic Qwen decoding,
resumable per-query artifacts, and explicit source provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORTIS_ROOT = PROJECT_ROOT / "external" / "fortis"
DOMAINS = ("email", "ecommerce", "filesystem")
TASKS = ("task1", "task2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value == "all":
        return allowed
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"invalid selection {value!r}; unknown={unknown}")
    return selected


def _request(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    attempts: int = 3,
) -> tuple[Any, int]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response, attempt
        except Exception as exc:  # transport retry, preserved in artifact
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise last_error


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    total = len(rows)
    completion_tokens = [
        int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in rows
    ]
    return {
        "total": total,
        "counts": {
            name: counts[name]
            for name in ("exact_match", "under_privilege", "over_privilege", "no_action")
        },
        "success_rate": counts["exact_match"] / total if total else 0.0,
        "safe_rate": (
            (counts["exact_match"] + counts["under_privilege"]) / total if total else 0.0
        ),
        "over_privilege_rate": counts["over_privilege"] / total if total else 0.0,
        "fail_rate": (
            (counts["over_privilege"] + counts["no_action"]) / total if total else 0.0
        ),
        "parse_rate": sum(bool(row.get("parsed")) for row in rows) / total if total else 0.0,
        "finish_reasons": dict(Counter(row.get("finish_reason", "missing") for row in rows)),
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else 0.0
        ),
        "completion_tokens_max": max(completion_tokens, default=0),
        "transport_retries": sum(max(0, int(row.get("request_attempts", 1)) - 1) for row in rows),
    }


def _grouped(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: _metrics(members) for name, members in sorted(groups.items())}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortis-root", type=Path, default=DEFAULT_FORTIS_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--task1-max-tokens", type=int, default=128)
    parser.add_argument("--task2-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--norm-prompt-file",
        type=Path,
        help="optional pure norm prefix; official benchmark prompt remains otherwise unchanged",
    )
    parser.add_argument("--prompt-label", default="official")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fortis_root = args.fortis_root.resolve()
    norm_prompt_file = args.norm_prompt_file.resolve() if args.norm_prompt_file else None
    norm_prompt = (
        norm_prompt_file.read_text(encoding="utf-8").strip()
        if norm_prompt_file
        else ""
    )
    norm_prompt_sha256 = _sha256(norm_prompt_file) if norm_prompt_file else None
    selected_tasks = _parse_csv(args.tasks, TASKS)
    selected_domains = _parse_csv(args.domains, DOMAINS)
    if args.limit_per_domain is not None and args.limit_per_domain <= 0:
        raise SystemExit("--limit-per-domain must be positive")
    sys.path.insert(0, str(fortis_root))

    from runners import gpt_task1_batch as task1
    from runners import gpt_task2_batch as task2

    args.output.mkdir(parents=True, exist_ok=True)
    prompt_config = {
        "prompt_label": args.prompt_label,
        "norm_prompt_file": str(norm_prompt_file) if norm_prompt_file else None,
        "norm_prompt_sha256": norm_prompt_sha256,
    }
    prompt_config_path = args.output / "PROMPT_CONFIG.json"
    if prompt_config_path.is_file():
        if json.loads(prompt_config_path.read_text(encoding="utf-8")) != prompt_config:
            raise SystemExit("refusing to mix prompt conditions in one FORTIS output")
    else:
        prompt_config_path.write_text(
            json.dumps(prompt_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    manifest: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for task_name in selected_tasks:
        for domain in selected_domains:
            if task_name == "task1":
                queries = task1.load_task1_queries(domain)
                skills = task1.load_skills(domain)
                system_prompt = task1.build_system_prompt(skills)
                max_tokens = args.task1_max_tokens
                query_path = (
                    fortis_root
                    / "tasks/task1_skill_selection/scenarios"
                    / domain
                    / "queries.json"
                )
                source_hashes[str(query_path.relative_to(fortis_root))] = _sha256(query_path)
                skills_path = fortis_root / "shared/scenarios" / domain / "skills.json"
                source_hashes[str(skills_path.relative_to(fortis_root))] = _sha256(skills_path)
            else:
                queries = task2.load_task2_queries(domain)
                tools = task2.load_tools(domain)
                max_tokens = args.task2_max_tokens
                query_path = (
                    fortis_root
                    / "tasks/task2_tool_selection/scenarios"
                    / domain
                    / "queries.json"
                )
                source_hashes[str(query_path.relative_to(fortis_root))] = _sha256(query_path)
                tools_path = fortis_root / "shared/scenarios" / domain / "tools.py"
                source_hashes[str(tools_path.relative_to(fortis_root))] = _sha256(tools_path)
            if args.limit_per_domain is not None:
                queries = queries[: args.limit_per_domain]
            manifest[task_name][domain] = [query["id"] for query in queries]
            task_dir = args.output / task_name / domain
            task_dir.mkdir(parents=True, exist_ok=True)

            for position, query in enumerate(queries, 1):
                result_path = task_dir / f"{query['id']}.json"
                if result_path.is_file():
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if cached.get("prompt_label") != args.prompt_label:
                        raise SystemExit(f"{result_path}: cached prompt label differs")
                    all_rows.append(cached)
                    continue
                if task_name == "task1":
                    active_prompt = system_prompt
                else:
                    skill_name = query["given_skill"]
                    skill_path = fortis_root / "shared/skills" / domain / skill_name / "SKILL.md"
                    source_hashes[str(skill_path.relative_to(fortis_root))] = _sha256(skill_path)
                    active_prompt = task2.build_system_prompt(
                        tools,
                        skill_name,
                        task2.load_skill_content(domain, skill_name),
                    )
                if norm_prompt:
                    active_prompt = f"{norm_prompt}\n\n{active_prompt}"
                user_message = f"User request: {query['query']}"
                print(
                    f"[{task_name}/{domain} {position}/{len(queries)}] "
                    f"{args.model} {query['id']}",
                    flush=True,
                )
                response, attempts = _request(
                    client,
                    model=args.model,
                    system_prompt=active_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason or "missing"
                if task_name == "task1":
                    prediction = task1.parse_skill_response(text)
                    classification = task1.classify_skill_result(
                        query["gt_skill"], prediction, query["gt_level"], skills
                    )
                    row = {
                        "gt_skill": query["gt_skill"],
                        "gt_level": query["gt_level"],
                        "prediction": prediction,
                        "parsed": bool(prediction),
                    }
                else:
                    calls = task2.parse_tool_calls(text)
                    prediction = task2.extract_tool_names(calls)
                    classification = task2.classify_result(query["gt_tools"], prediction)
                    row = {
                        "given_skill": query["given_skill"],
                        "gt_tools": query["gt_tools"],
                        "prediction": prediction,
                        "parsed_calls": calls,
                        "parsed": bool(prediction),
                    }
                result = {
                    "task": task_name,
                    "domain": domain,
                    "id": query["id"],
                    "category": query.get("category", "unknown"),
                    "query": query["query"],
                    "model": args.model,
                    "prompt_label": args.prompt_label,
                    "response": text,
                    "classification": classification,
                    "finish_reason": finish_reason,
                    "request_attempts": attempts,
                    "usage": response.usage.model_dump() if response.usage else {},
                    **row,
                }
                result_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                all_rows.append(result)

    by_task = {
        task_name: _metrics([row for row in all_rows if row["task"] == task_name])
        for task_name in selected_tasks
    }
    summary = {
        "model": args.model,
        "prompt": {
            "label": args.prompt_label,
            "norm_prompt_file": str(norm_prompt_file) if norm_prompt_file else None,
            "norm_prompt_sha256": norm_prompt_sha256,
        },
        "official_revision": subprocess.check_output(
            ["git", "-C", str(fortis_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": dict(sorted(source_hashes.items())),
        "manifest": manifest,
        "tasks": list(selected_tasks),
        "domains": list(selected_domains),
        "limit_per_domain": args.limit_per_domain,
        "task1_max_tokens": args.task1_max_tokens,
        "task2_max_tokens": args.task2_max_tokens,
        "thinking_disabled": True,
        "total": len(all_rows),
        "by_task": by_task,
        "by_task_domain": {
            task_name: _grouped(
                [row for row in all_rows if row["task"] == task_name], "domain"
            )
            for task_name in selected_tasks
        },
        "by_task_category": {
            task_name: _grouped(
                [row for row in all_rows if row["task"] == task_name], "category"
            )
            for task_name in selected_tasks
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
