from __future__ import annotations

from typing import Iterable

from datasets import Dataset

from least_privilege_env.env.prompts import system_prompt_for_variant
from least_privilege_env.tasks.loader import load_tasks


def build_chat_dataset(
    split: str | None = "train",
    task_ids: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    repeat: int = 1,
    system_prompt_variant: str = "full",
) -> Dataset:
    """Build a TRL GRPO dataset.

    Each row carries ONLY ``task_id`` (so ``TerminalEnv.reset`` can load the
    full task, including hidden verifier data, host-side from disk) and a
    conversational ``prompt`` (system + task). Hidden files and verifier
    config are deliberately NOT serialized into the dataset.
    """
    system_prompt = system_prompt_for_variant(system_prompt_variant)
    wanted_ids = set(task_ids) if task_ids is not None else None
    wanted_families = set(families) if families is not None else None
    rows = []
    for task in load_tasks():
        if split is not None and task.split != split:
            continue
        if wanted_ids is not None and task.task_id not in wanted_ids:
            continue
        if wanted_families is not None and task.family not in wanted_families:
            continue
        rows.append(
            {
                "task_id": task.task_id,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.prompt},
                ],
            }
        )
    rows = rows * max(1, repeat)
    return Dataset.from_list(rows)


def lora_config(r: int = 32, alpha: int = 64, dropout: float = 0.0):
    """LoRA over all linear layers (the 'LoRA without regret' recipe).

    RL gradients carry little information per sample, so an all-linear LoRA at
    modest rank matches full fine-tune for this task while keeping the trainer
    GPU footprint far under one 32 GB card and protecting the base policy from
    drift.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
