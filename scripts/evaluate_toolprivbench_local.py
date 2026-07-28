#!/usr/bin/env python3
"""Evaluate the base model or an LP-RL LoRA adapter on ToolPrivBench.

This is a deliberately thin adapter around the pinned official evaluator in
``external/toolprivbench/scripts/evaluate_local_qwen.py``. It preserves the
official scenario simulator, first-attempt-failure protocol, Qwen3.5 tool-call
parser, and over-privilege definition while adding:

- PEFT adapter loading without materializing merged model copies;
- deterministic tool-order randomization;
- explicit model/prompt/revision provenance in every result and summary; and
- resumable per-scenario output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLPRIV_ROOT = PROJECT_ROOT / "external" / "toolprivbench"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_module(toolpriv_root: Path) -> ModuleType:
    source = toolpriv_root / "scripts" / "evaluate_local_qwen.py"
    if not source.is_file():
        raise FileNotFoundError(f"official evaluator not found: {source}")
    spec = importlib.util.spec_from_file_location("toolprivbench_official_local", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalQwenWithAdapter:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None,
        model_label: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_label = model_label
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        import torch

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolpriv-root", type=Path, default=DEFAULT_TOOLPRIV_ROOT)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--prompt-label", default="official_default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--error-strategy", default="first_attempt_fail")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    toolpriv_root = args.toolpriv_root.resolve()
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
    if args.adapter_path and not (Path(args.adapter_path) / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter missing: {args.adapter_path}")

    random.seed(args.seed)
    official = _load_official_module(toolpriv_root)
    original_simulate = official.simulate

    def simulate_with_provenance(*sim_args: Any, **sim_kwargs: Any) -> dict[str, Any]:
        result = original_simulate(*sim_args, **sim_kwargs)
        model = sim_kwargs.get("model")
        if model is None and len(sim_args) >= 2:
            model = sim_args[1]
        result["model"] = getattr(model, "model_label", args.model_label)
        result["prompt_label"] = args.prompt_label
        result["tool_order_seed"] = args.seed
        return result

    official.simulate = simulate_with_provenance
    model = LocalQwenWithAdapter(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        model_label=args.model_label,
        max_new_tokens=args.max_new_tokens,
    )
    results = official.run_batch(
        benchmark_path=benchmark,
        model=model,
        system_prompt=prompt.read_text(encoding="utf-8").strip(),
        output_dir=args.output,
        max_turns=args.max_turns,
        error_strategy=args.error_strategy,
        limit=args.limit,
    )
    official.print_summary(results)

    summary = {
        "model_label": args.model_label,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "prompt_label": args.prompt_label,
        "system_prompt": str(prompt),
        "tool_order_seed": args.seed,
        "benchmark_path": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
        "official_revision": (
            __import__("subprocess")
            .check_output(["git", "-C", str(toolpriv_root), "rev-parse", "HEAD"], text=True)
            .strip()
        ),
        "max_turns": args.max_turns,
        "error_strategy": args.error_strategy,
        "max_new_tokens": args.max_new_tokens,
        "total": len(results),
        "over_privileged": sum(bool(row.get("over_privileged_selection")) for row in results),
        "risk_called": sum(row.get("first_risk_turn") is not None for row in results),
        "termination_reasons": {},
    }
    for row in results:
        reason = row.get("termination_reason", "unknown")
        summary["termination_reasons"][reason] = summary["termination_reasons"].get(reason, 0) + 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
