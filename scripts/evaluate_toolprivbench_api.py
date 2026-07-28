#!/usr/bin/env python3
"""Run the pinned official ToolPrivBench simulator against a local vLLM API.

The official benchmark owns scenario construction, transient-error injection,
turn handling, and OPUR/PED definitions. This wrapper supplies only a local
OpenAI-compatible client, deterministic per-scenario tool order, resumable
artifacts, and explicit provenance for the base/LoRA comparisons.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLPRIV_ROOT = PROJECT_ROOT / "external" / "toolprivbench"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _success_proxy(row: dict[str, Any]) -> bool:
    return any(
        turn.get("action") == "tool_call" and not turn.get("is_error", False)
        for turn in row.get("turns", [])
    )


def _group_rates(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    output: dict[str, dict[str, float | int]] = {}
    for name, members in sorted(groups.items()):
        over = sum(bool(row.get("over_privileged_selection")) for row in members)
        output[name] = {
            "total": len(members),
            "over_privileged": over,
            "over_privileged_rate": over / len(members),
            "success_proxy": sum(_success_proxy(row) for row in members),
        }
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolpriv-root", type=Path, default=DEFAULT_TOOLPRIV_ROOT)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--prompt-label", default="official_default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--error-strategy", default="first_attempt_fail")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    toolpriv_root = args.toolpriv_root.resolve()
    sys.path.insert(0, str(toolpriv_root))

    from src.evaluation import simulator
    from src.metrics.opur import compute_opur_curve, compute_over_privilege_curve
    from src.metrics.ped import compute_ped

    benchmark = (
        args.benchmark.resolve()
        if args.benchmark
        else toolpriv_root / "evaluation" / "data" / "goodcase_final" / "benchmark.jsonl"
    )
    prompt = (
        args.system_prompt.resolve()
        if args.system_prompt
        else toolpriv_root / "prompts" / "system_prompts" / "agent_default.txt"
    )
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark not found: {benchmark}")
    if not prompt.is_file():
        raise FileNotFoundError(f"system prompt not found: {prompt}")

    api_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    def local_chat_completion(
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        client: OpenAI | None = None,
        **_: Any,
    ) -> Any:
        active_client = client or api_client
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": args.max_new_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return active_client.chat.completions.create(**kwargs)

    simulator.chat_completion = local_chat_completion

    scenarios = [
        json.loads(line)
        for line in benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        scenarios = scenarios[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    system_prompt = prompt.read_text(encoding="utf-8").strip()

    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        scenario_id = scenario["scenario_id"]
        result_path = args.output / f"{scenario_id}.json"
        if result_path.is_file():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        random.seed(f"{args.seed}:{scenario_id}")
        print(f"[{index}/{len(scenarios)}] {args.model} {scenario_id}", flush=True)
        row = simulator.simulate(
            scenario=scenario,
            model=args.model,
            system_prompt=system_prompt,
            max_turns=args.max_turns,
            error_strategy=args.error_strategy,
            randomize_tools=True,
            client=api_client,
        )
        row["prompt_label"] = args.prompt_label
        row["tool_order_seed"] = args.seed
        result_path.write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows.append(row)

    completion_tokens = [
        int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in rows
    ]
    terminations = Counter(row.get("termination_reason", "unknown") for row in rows)
    official_revision = subprocess.check_output(
        ["git", "-C", str(toolpriv_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    summary = {
        "model": args.model,
        "prompt_label": args.prompt_label,
        "system_prompt": str(prompt),
        "base_url": args.base_url,
        "official_revision": official_revision,
        "benchmark_path": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "error_strategy": args.error_strategy,
        "tool_order_seed": args.seed,
        "total": len(rows),
        "opur": compute_opur_curve(rows, [1, 3, 5]),
        "over_privilege_rate": compute_over_privilege_curve(rows, [1, 3, 5]),
        "ped": compute_ped(rows),
        "over_privileged": sum(bool(row.get("over_privileged_selection")) for row in rows),
        "risk_called": sum(row.get("first_risk_turn") is not None for row in rows),
        "success_proxy": sum(_success_proxy(row) for row in rows),
        "termination_reasons": dict(terminations),
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else 0.0
        ),
        "completion_tokens_max": max(completion_tokens, default=0),
        "by_domain": _group_rates(rows, "domain"),
        "by_type": _group_rates(rows, "type"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
