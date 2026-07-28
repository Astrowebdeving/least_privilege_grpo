#!/usr/bin/env python3
"""Evaluate a deterministic MetaTool capability subset through local vLLM.

This uses the official prepared prompts and labels unchanged. It is explicitly
a registered capability diagnostic rather than a claim to reproduce every
number in the original 21k-example paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METATOOL_ROOT = PROJECT_ROOT / "external" / "metatool"
TASKS = {
    "awareness": ("Task1.json", "thought_prompt", "awareness"),
    "similar": ("Task2-Subtask1.json", "action_prompt", "single"),
    "scenario": ("Task2-Subtask2.json", "action_prompt", "single"),
    # The official reliability/hallucination construction deliberately excludes
    # the source row's original tool from the offered candidate list. Its target
    # is therefore None, not row["tool"].
    "reliability": ("Task2-Subtask3.json", "action_prompt", "reliability"),
    "multi": ("Task2-Subtask4.json", "action_prompt", "multi"),
}
SCORING_VERSION = "2026-07-16-reliability-none-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _candidate_names(prompt: str) -> list[str]:
    names = re.findall(r"tool name:\s*([^,\n]+)", prompt, flags=re.IGNORECASE)
    output = []
    for name in names:
        clean = name.strip().strip("'\"")
        if clean and clean not in output:
            output.append(clean)
    return output


def _ordered_mentions(text: str, candidates: list[str]) -> list[str]:
    lower = text.lower()
    found = []
    for candidate in candidates:
        position = lower.find(candidate.lower())
        if position >= 0:
            found.append((position, candidate))
    found.sort()
    return [candidate for _, candidate in found]


def _score(row: dict[str, Any], prompt: str, response: str, kind: str) -> dict[str, Any]:
    cleaned = _clean_response(response)
    if kind == "awareness":
        expected = "yes" if str(row.get("label", "")).lower() == "positive" else "no"
        match = re.search(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE)
        predicted = match.group(1).lower() if match else None
        return {"expected": expected, "predicted": predicted, "correct": predicted == expected}

    candidates = _candidate_names(prompt)
    mentions = _ordered_mentions(cleaned, candidates)
    if kind in {"single", "reliability"}:
        expected = "None" if kind == "reliability" else str(row.get("tool", "None"))
        none_match = re.search(r"\bnone\b", cleaned, flags=re.IGNORECASE)
        first_candidate_position = (
            cleaned.lower().find(mentions[0].lower()) if mentions else 10**9
        )
        if none_match and none_match.start() < first_candidate_position:
            predicted = "None"
        else:
            predicted = mentions[0] if mentions else None
        return {
            "expected": expected,
            "predicted": predicted,
            "correct": predicted is not None and predicted.lower() == expected.lower(),
            "candidates": candidates,
        }

    expected_list = [str(item) for item in row.get("tool", [])]
    predicted_list = mentions[:2]
    return {
        "expected": expected_list,
        "predicted": predicted_list,
        "correct": (
            len(predicted_list) == len(expected_list)
            and {item.lower() for item in predicted_list}
            == {item.lower() for item in expected_list}
        ),
        "candidates": candidates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metatool-root", type=Path, default=DEFAULT_METATOOL_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="optional pure norm system message; official user prompts remain unchanged",
    )
    parser.add_argument("--prompt-label", default="official")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.metatool_root.resolve()
    system_prompt_file = (
        args.system_prompt_file.resolve() if args.system_prompt_file else None
    )
    system_prompt = (
        system_prompt_file.read_text(encoding="utf-8").strip()
        if system_prompt_file
        else ""
    )
    system_prompt_sha256 = _sha256(system_prompt_file) if system_prompt_file else None
    data_root = root / "dataset" / "tmp_dataset"
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_config = {
        "prompt_label": args.prompt_label,
        "system_prompt_file": str(system_prompt_file) if system_prompt_file else None,
        "system_prompt_sha256": system_prompt_sha256,
    }
    prompt_config_path = args.output / "PROMPT_CONFIG.json"
    if prompt_config_path.is_file():
        if json.loads(prompt_config_path.read_text(encoding="utf-8")) != prompt_config:
            raise SystemExit("refusing to mix prompt conditions in one MetaTool output")
    else:
        prompt_config_path.write_text(
            json.dumps(prompt_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    all_results = []
    source_digests = {}
    sample_manifest = {}
    for task_name, (filename, prompt_field, kind) in TASKS.items():
        source = data_root / filename
        rows = json.loads(source.read_text(encoding="utf-8"))
        source_digests[task_name] = _sha256(source)
        rng = random.Random(f"{args.seed}:{task_name}")
        count = min(args.sample_per_task, len(rows))
        indices = sorted(rng.sample(range(len(rows)), count))
        sample_manifest[task_name] = indices
        task_dir = args.output / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for position, source_index in enumerate(indices, 1):
            output_path = task_dir / f"{source_index:06d}.json"
            row = rows[source_index]
            prompt = row[prompt_field]
            if output_path.is_file():
                result = json.loads(output_path.read_text(encoding="utf-8"))
                if result.get("prompt_label") != args.prompt_label:
                    raise SystemExit(f"{output_path}: cached prompt label differs")
                rescored = _score(row, prompt, result.get("response", ""), kind)
                changed = (
                    result.get("scoring_version") != SCORING_VERSION
                    or any(result.get(key) != value for key, value in rescored.items())
                )
                if changed:
                    result.update(rescored)
                    result["scoring_version"] = SCORING_VERSION
                    output_path.write_text(
                        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                all_results.append(result)
                continue
            print(
                f"[{task_name} {position}/{len(indices)}] {args.model} source={source_index}",
                flush=True,
            )
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.0,
                max_tokens=args.max_new_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = response.choices[0].message.content or ""
            scored = _score(row, prompt, text, kind)
            result = {
                "task": task_name,
                "source_index": source_index,
                "query": row.get("query"),
                "model": args.model,
                "prompt_label": args.prompt_label,
                "scoring_version": SCORING_VERSION,
                "response": text,
                "usage": response.usage.model_dump() if response.usage else {},
                **scored,
            }
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            all_results.append(result)

    by_task = {}
    for task_name in TASKS:
        members = [row for row in all_results if row["task"] == task_name]
        correct = sum(bool(row["correct"]) for row in members)
        parsed = sum(row.get("predicted") not in (None, []) for row in members)
        by_task[task_name] = {
            "total": len(members),
            "correct": correct,
            "accuracy": correct / len(members),
            "parsed": parsed,
            "parse_rate": parsed / len(members),
        }
    tokens = [
        int(row.get("usage", {}).get("completion_tokens", 0) or 0)
        for row in all_results
    ]
    correct = sum(bool(row["correct"]) for row in all_results)
    summary = {
        "model": args.model,
        "prompt": {
            "label": args.prompt_label,
            "system_prompt_file": str(system_prompt_file) if system_prompt_file else None,
            "system_prompt_sha256": system_prompt_sha256,
        },
        "scoring_version": SCORING_VERSION,
        "official_revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": source_digests,
        "sample_manifest": sample_manifest,
        "sample_per_task": args.sample_per_task,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "total": len(all_results),
        "correct": correct,
        "accuracy": correct / len(all_results),
        "by_task": by_task,
        "completion_tokens_mean": statistics.fmean(tokens) if tokens else 0.0,
        "completion_tokens_max": max(tokens, default=0),
        "prediction_types": dict(
            Counter(type(row.get("predicted")).__name__ for row in all_results)
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
