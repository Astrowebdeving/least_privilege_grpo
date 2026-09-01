from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from least_privilege_env.env.terminal_env import TerminalEnv, least_privilege_reward
from least_privilege_env.env.trl_helpers import build_chat_dataset, lora_config


# Recommended real-RL profile for the current dual RTX 5090 WSL machine.
#
# `max_completion_length` budgets the whole completion side of the multi-turn
# episode and doubles as the Dr. GRPO L0 normalizer. Probe data: 8192 spills
# (38.9 GB allocated / 42.3 GB reserved), 5120 is narrow (27.7 / 30.1 GB),
# 4096 is safe (24.0 / 26.2 GB). The 2026-06-12 controlled rollout hit
# 32.0 / 32.6 GB reserved at 5120 with real multi-turn episodes (the probes
# were synthetic single steps), triggering the runbook's documented fallback
# condition — 4096 is now the default. Keep per-device batch size at 1 and
# use accumulation; num_generations does NOT affect the trainer-GPU peak
# (one completion per micro-batch), only wall-clock and group quality.
DEFAULT_MAX_COMPLETION_LENGTH = 4096
DEFAULT_NUM_GENERATIONS = 8
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_VLLM_IMPORTANCE_SAMPLING_MODE = "token_truncate"
# Hard cap on generate->tool->generate turns per episode (TRL default is
# unlimited). The 2026-06-12 controlled rollout measured ~49.5 tool calls per
# episode at ~92% failure while the audit showed only ~3.2 real actions and
# 55/56 clean terminations: after final_answer froze the audit, every further
# tool call raised "episode over" / "step budget exhausted", TRL fed the
# error back, and the model kept going — burning tokens that are invisible to
# reward. Legitimate ceiling is task.max_steps (<=10) + the terminal action
# = 11 turns; 20 adds generous slack for the base model's rejected-syntax
# retries (blocked/parse-rejected turns also consume iterations) so episodes
# that WOULD terminate are not cut into unterminated S=0 groups, while still
# bounding the post-terminal garbage tail. An extra turn costs only ~50-80
# tokens; the 4096 completion budget remains the hard memory wall.
DEFAULT_MAX_TOOL_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Long-run monitor: automated alert/abort thresholds for unattended runs.
#
# Measured healthy values on the one-step real gate (5120/1/8/8, token
# truncate): ISR mean 0.9901, clipped ratio 0, GPU reserved 30.1/32 GB
# (~0.94). The collapse failure mode observed at lr=0 with sequence_mask was
# ISR mean 0.08-0.33, so 0.5 separates healthy from collapsed cleanly.
# ---------------------------------------------------------------------------

ISR_KEY = "sampling/importance_sampling_ratio/mean"
CLIPPED_KEY = "completions/clipped_ratio"
TOOL_FAILURE_KEY = "tools/failure_frequency"
MEMORY_LOG_METRIC_KEYS = (
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "sampling/importance_sampling_ratio/mean",
    "rewards/least_privilege_reward/mean",
    "rewards/least_privilege_reward/std",
    "reward",
    "reward_std",
    "tools/call_frequency",
    "tools/failure_frequency",
    "step_time",
    "epoch",
)


@dataclass(frozen=True)
class MonitorThresholds:
    isr_mean_min: float = 0.5
    clipped_ratio_max: float = 0.5
    gpu_reserved_frac_max: float = 0.98
    gpu_free_frac_alert_min: float = 0.02  # alert-only until calibrated by Diag-A/Diag-B
    max_containers: int = 32
    tool_failure_max: float = 0.9  # alert-only, never aborts
    patience: int = 3
    abort: bool = True


class LongRunMonitor:
    """Pure threshold logic, kept free of torch/transformers for unit tests.

    A signal must breach its threshold for ``patience`` consecutive logged
    steps before the monitor requests an abort; a healthy reading resets the
    streak so transient spikes only alert.
    """

    def __init__(self, thresholds: MonitorThresholds):
        self.thresholds = thresholds
        self._streaks: dict[str, int] = {}

    def observe(
        self,
        logs: dict,
        gpu_reserved_frac: float | None = None,
        gpu_free_frac: float | None = None,
        container_count: int | None = None,
    ) -> tuple[list[str], bool]:
        t = self.thresholds
        checks: list[tuple[str, bool, str]] = []
        if ISR_KEY in logs:
            value = float(logs[ISR_KEY])
            checks.append(("isr_collapse", value < t.isr_mean_min, f"ISR mean {value:.4f} < {t.isr_mean_min}"))
        if CLIPPED_KEY in logs:
            value = float(logs[CLIPPED_KEY] or 0.0)
            checks.append(("completion_clipping", value > t.clipped_ratio_max, f"clipped ratio {value:.3f} > {t.clipped_ratio_max}"))
        if gpu_reserved_frac is not None:
            checks.append(
                (
                    "gpu_memory",
                    gpu_reserved_frac > t.gpu_reserved_frac_max,
                    f"GPU reserved fraction {gpu_reserved_frac:.3f} > {t.gpu_reserved_frac_max}",
                )
            )
        if container_count is not None:
            checks.append(
                (
                    "leftover_containers",
                    container_count > t.max_containers,
                    f"{container_count} lp-rl containers > {t.max_containers}",
                )
            )

        alerts: list[str] = []
        should_abort = False
        for name, breached, detail in checks:
            if breached:
                streak = self._streaks.get(name, 0) + 1
                self._streaks[name] = streak
                alerts.append(f"[lp-rl monitor] {name}: {detail} (consecutive {streak}/{t.patience})")
                if streak >= t.patience:
                    should_abort = True
            else:
                self._streaks[name] = 0

        if TOOL_FAILURE_KEY in logs:
            value = float(logs[TOOL_FAILURE_KEY] or 0.0)
            if value > t.tool_failure_max:
                alerts.append(f"[lp-rl monitor] tool_failure (alert only): {value:.3f} > {t.tool_failure_max}")
        if gpu_free_frac is not None and gpu_free_frac < t.gpu_free_frac_alert_min:
            alerts.append(
                f"[lp-rl monitor] gpu_free (alert only): free fraction {gpu_free_frac:.4f} < {t.gpu_free_frac_alert_min}"
            )
        return alerts, should_abort


def lp_container_count() -> int | None:
    try:
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=lp-rl-"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def _ratio(value: int | float | None, total: int | float | None) -> float | None:
    if value is None or not total:
        return None
    return float(value) / float(total)


def _null_cuda_snapshot(error: str | None = None) -> dict[str, Any]:
    return {
        "cuda_available": False,
        "cuda_error": error,
        "cuda_device": None,
        "cuda_device_name": None,
        "cuda_device_total": None,
        "memory_allocated": None,
        "memory_reserved": None,
        "max_memory_allocated": None,
        "max_memory_reserved": None,
        "mem_get_info_free": None,
        "mem_get_info_total": None,
        "memory_allocated_frac": None,
        "memory_reserved_frac": None,
        "max_memory_allocated_frac": None,
        "max_memory_reserved_frac": None,
        "mem_get_info_free_frac": None,
    }


def cuda_memory_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Current and peak CUDA memory stats, with a no-CUDA safe null record."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - defensive
            return _null_cuda_snapshot(type(exc).__name__)
    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is None or not cuda.is_available():
            return _null_cuda_snapshot()
        device = cuda.current_device()
        props = cuda.get_device_properties(device)
        device_total = int(getattr(props, "total_memory", 0) or 0)
        try:
            free, info_total = cuda.mem_get_info(device)
        except TypeError:
            free, info_total = cuda.mem_get_info()
        info_total = int(info_total or 0)
        total = device_total or info_total or None
        allocated = int(cuda.memory_allocated(device))
        reserved = int(cuda.memory_reserved(device))
        max_allocated = int(cuda.max_memory_allocated(device))
        max_reserved = int(cuda.max_memory_reserved(device))
        free = int(free)
        name = cuda.get_device_name(device) if hasattr(cuda, "get_device_name") else getattr(props, "name", None)
        return {
            "cuda_available": True,
            "cuda_error": None,
            "cuda_device": int(device),
            "cuda_device_name": name,
            "cuda_device_total": total,
            "memory_allocated": allocated,
            "memory_reserved": reserved,
            "max_memory_allocated": max_allocated,
            "max_memory_reserved": max_reserved,
            "mem_get_info_free": free,
            "mem_get_info_total": info_total or total,
            "memory_allocated_frac": _ratio(allocated, total),
            "memory_reserved_frac": _ratio(reserved, total),
            "max_memory_allocated_frac": _ratio(max_allocated, total),
            "max_memory_reserved_frac": _ratio(max_reserved, total),
            "mem_get_info_free_frac": _ratio(free, info_total or total),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return _null_cuda_snapshot(type(exc).__name__)


def reset_cuda_peak_memory_stats(torch_module: Any | None = None) -> None:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return
    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            cuda.reset_peak_memory_stats()
    except Exception:
        return


def cuda_cleanup(torch_module: Any | None = None) -> dict[str, bool]:
    """Best-effort GC + CUDA cache release, with separate telemetry flags."""
    gc.collect()
    result = {"gc_collect_ran": True, "empty_cache_ran": False}
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return result
    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            result["empty_cache_ran"] = True
    except Exception:
        return result
    return result


def clear_cuda_cache(torch_module: Any | None = None) -> bool:
    """Compatibility helper; returns whether torch.cuda.empty_cache ran."""
    return cuda_cleanup(torch_module)["empty_cache_ran"]


class GpuMemoryJsonlLogger:
    def __init__(
        self,
        path: str | None,
        empty_cache_steps: int = 0,
        check_containers: bool = False,
        torch_module: Any | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.empty_cache_steps = max(0, int(empty_cache_steps))
        self.check_containers = check_containers
        self.torch_module = torch_module
        self.last_record: dict[str, Any] | None = None

    def reset_peak(self) -> None:
        reset_cuda_peak_memory_stats(self.torch_module)

    def should_empty_cache(self, global_step: int) -> bool:
        return self.empty_cache_steps > 0 and global_step > 0 and global_step % self.empty_cache_steps == 0

    def maybe_cleanup(self, global_step: int, force: bool = False) -> dict[str, bool]:
        if force or self.should_empty_cache(global_step):
            return cuda_cleanup(self.torch_module)
        return {"gc_collect_ran": False, "empty_cache_ran": False}

    def record(self, global_step: int, logs: dict[str, Any] | None = None) -> dict[str, Any]:
        logs = logs or {}
        before = cuda_memory_snapshot(self.torch_module)
        cleanup = self.maybe_cleanup(global_step)
        after = cuda_memory_snapshot(self.torch_module) if cleanup["gc_collect_ran"] else None
        record = {
            "schema_version": 1,
            "ts": time.time(),
            "global_step": int(global_step),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "cleanup_gc_collect_ran": cleanup["gc_collect_ran"],
            "cleanup_empty_cache_ran": cleanup["empty_cache_ran"],
            "container_count": lp_container_count() if self.check_containers else None,
            "metrics": {key: logs.get(key) for key in MEMORY_LOG_METRIC_KEYS if key in logs},
            **before,
        }
        if after is not None:
            record["post_cleanup"] = after
        self.last_record = record
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[lp-rl gpu-memory] WARNING: failed to write memory record: {exc}", flush=True)
        return record


def build_monitor_callback(thresholds: MonitorThresholds, output_dir: str, memory_logger: GpuMemoryJsonlLogger):
    """Wrap LongRunMonitor in a transformers TrainerCallback (lazy imports)."""
    from transformers import TrainerCallback

    monitor = LongRunMonitor(thresholds)
    check_containers = os.environ.get("LP_RL_SANDBOX", "local") == "docker"

    class LongRunMonitorCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            memory_logger.reset_peak()
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            memory_logger.reset_peak()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            memory_record = memory_logger.record(state.global_step, logs)
            alerts, should_abort = monitor.observe(
                logs,
                gpu_reserved_frac=memory_record.get("memory_reserved_frac"),
                gpu_free_frac=memory_record.get("mem_get_info_free_frac"),
                container_count=lp_container_count() if check_containers else None,
            )
            for alert in alerts:
                print(alert, flush=True)
            if should_abort and thresholds.abort:
                cleanup = memory_logger.maybe_cleanup(state.global_step, force=memory_logger.empty_cache_steps > 0)
                report = {
                    "global_step": state.global_step,
                    "alerts": alerts,
                    "logs": {key: logs.get(key) for key in (ISR_KEY, CLIPPED_KEY, TOOL_FAILURE_KEY) if key in logs},
                    "memory": memory_record,
                    "abort_cleanup": cleanup,
                    "abort_cleanup_ran": cleanup["gc_collect_ran"] or cleanup["empty_cache_ran"],
                    "save_requested": True,
                }
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / "monitor_abort.json").write_text(json.dumps(report, indent=2))
                print(f"[lp-rl monitor] ABORTING RUN at step {state.global_step}; report: {out / 'monitor_abort.json'}", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    return LongRunMonitorCallback()


def visible_gpu_count() -> int:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))


def choose_vllm_mode(value: str) -> str:
    if value != "auto":
        return value
    return "server" if visible_gpu_count() >= 2 else "colocate"


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def dataset_task_ids(dataset) -> list[str]:
    if "task_id" not in getattr(dataset, "column_names", ()):
        return []
    return list(dataset["task_id"])


def build_training_dataset(
    split: str,
    task_ids_csv: str = "",
    families_csv: str = "",
    repeat: int = 1,
    system_prompt_variant: str = "full",
):
    if repeat < 1:
        raise ValueError("--repeat must be >= 1")
    task_ids = parse_csv_list(task_ids_csv)
    families = parse_csv_list(families_csv)
    if len(set(task_ids)) != len(task_ids):
        duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
        raise ValueError(f"--task-ids contains duplicates: {', '.join(duplicates)}")

    # Explicit task ids are an exact-targeting mode: do not let --split filter
    # held-out ids away before validation.
    dataset_split = None if task_ids else split
    filters = {
        "task_ids": task_ids or None,
        "families": families or None,
    }
    base = build_chat_dataset(
        dataset_split,
        repeat=1,
        system_prompt_variant=system_prompt_variant,
        **filters,
    )
    base_ids = dataset_task_ids(base)
    if task_ids:
        counts = Counter(base_ids)
        expected = Counter(task_ids)
        if counts != expected:
            missing = sorted(set(expected) - set(counts))
            extra = sorted(set(counts) - set(expected))
            wrong_counts = sorted(
                task_id for task_id in set(expected) & set(counts) if counts[task_id] != expected[task_id]
            )
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"extra={extra}")
            if wrong_counts:
                detail.append(f"wrong_counts={wrong_counts}")
            raise ValueError(f"requested task ids did not match filtered dataset exactly ({'; '.join(detail)})")

    dataset = build_chat_dataset(
        dataset_split,
        repeat=repeat,
        system_prompt_variant=system_prompt_variant,
        **filters,
    )
    if task_ids:
        counts = Counter(dataset_task_ids(dataset))
        expected = Counter({task_id: repeat for task_id in task_ids})
        if counts != expected:
            raise ValueError(f"repeated dataset ids {dict(counts)} != expected {dict(expected)}")
    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-ids", default="", help="comma-separated exact task ids; ignores --split when set")
    parser.add_argument("--families", default="", help="comma-separated task families; still respects --split")
    parser.add_argument("--repeat", type=int, default=1, help="repeat filtered dataset rows this many times")
    parser.add_argument(
        "--shuffle-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "shuffle training rows (historical default); disable only for a "
            "registered ordered curriculum such as continuation400"
        ),
    )
    parser.add_argument(
        "--system-prompt-variant",
        default="full",
        choices=("full", "para", "min", "none"),
        help="registered least-privilege norm treatment; interface protocol is preserved",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--per-device-batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max-completion-length", type=int, default=DEFAULT_MAX_COMPLETION_LENGTH)
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=DEFAULT_MAX_TOOL_ITERATIONS,
        help="cap on generate->tool->generate turns per episode; 0 means unlimited (TRL default)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=-1, help="training steps; -1 uses epoch-based Trainer default")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--vllm-mode", choices=["auto", "colocate", "server"], default="auto")
    parser.add_argument(
        "--vllm-importance-sampling-mode",
        default=DEFAULT_VLLM_IMPORTANCE_SAMPLING_MODE,
        choices=["sequence", "sequence_mask", "token_truncate"],
    )
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--full-finetune", action="store_true", help="disable LoRA (needs >1 card / 8-bit optim)")
    parser.add_argument(
        "--adapter-path",
        default="",
        help="optional PEFT/LoRA adapter checkpoint to load on top of --model for frozen eval or resume-like probes",
    )
    parser.add_argument("--output-dir", default="logs/artifacts/dr_grpo_run")
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help=(
            "HF Trainer checkpoint dir to resume from (restores optimizer, "
            "scheduler, RNG, and global_step). Use the same --seed/--max-steps/"
            "--output-dir as the original run; empty string starts fresh."
        ),
    )
    parser.add_argument(
        "--rollout-log-dir",
        default=None,
        help="JSONL rollout log dir (host-side review artifact); defaults to <output-dir>/rollouts, pass '' to disable",
    )
    parser.add_argument(
        "--gpu-memory-log",
        default=None,
        help="JSONL CUDA memory log; defaults to <output-dir>/gpu_memory.jsonl, pass '' to disable",
    )
    parser.add_argument(
        "--cuda-empty-cache-steps",
        type=int,
        default=0,
        help="run gc.collect() + torch.cuda.empty_cache() after logging every N steps; 0 disables",
    )
    defaults = MonitorThresholds()
    parser.add_argument("--monitor-isr-min", type=float, default=defaults.isr_mean_min, help="abort if ISR mean stays below this")
    parser.add_argument("--monitor-clipped-max", type=float, default=defaults.clipped_ratio_max, help="abort if completion clipped ratio stays above this")
    parser.add_argument("--monitor-gpu-frac-max", type=float, default=defaults.gpu_reserved_frac_max, help="abort if GPU reserved fraction stays above this")
    parser.add_argument("--monitor-gpu-free-frac-min", type=float, default=defaults.gpu_free_frac_alert_min, help="alert if CUDA free fraction drops below this; alert-only")
    parser.add_argument("--monitor-max-containers", type=int, default=defaults.max_containers, help="abort if leftover lp-rl container count stays above this")
    parser.add_argument("--monitor-patience", type=int, default=defaults.patience, help="consecutive breached log steps before abort")
    parser.add_argument("--no-monitor-abort", action="store_true", help="alert only; never stop the run")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def validate_cli_args(cli: argparse.Namespace) -> None:
    if cli.adapter_path and cli.full_finetune:
        raise ValueError("--adapter-path cannot be combined with --full-finetune")


def main() -> None:
    from trl import GRPOConfig, GRPOTrainer

    # Terminal-stop override (2026-07-15): stop each sample at its first
    # successful terminal action instead of letting TRL regenerate an
    # audit-invisible post-terminal tail. LP_RL_NO_TERMINAL_STOP=1 falls back
    # to the pinned TRL loop for A/B diagnosis only.
    if os.environ.get("LP_RL_NO_TERMINAL_STOP") == "1":
        trainer_cls = GRPOTrainer
    else:
        try:
            from lp_grpo_trainer import LPTerminalStopGRPOTrainer
        except ImportError:  # imported as a package module instead of a script
            from train.lp_grpo_trainer import LPTerminalStopGRPOTrainer

        trainer_cls = LPTerminalStopGRPOTrainer
    from transformers import set_seed

    cli = parse_args()
    validate_cli_args(cli)
    set_seed(cli.seed)
    use_vllm = not cli.no_vllm
    vllm_mode = choose_vllm_mode(cli.vllm_mode)
    # TerminalEnv instances live in this process, so the env var is the
    # simplest reliable channel for the rollout-log destination.
    rollout_log_dir = cli.rollout_log_dir if cli.rollout_log_dir is not None else f"{cli.output_dir}/rollouts"
    if rollout_log_dir:
        os.environ["LP_RL_ROLLOUT_LOG_DIR"] = rollout_log_dir
    else:
        os.environ.pop("LP_RL_ROLLOUT_LOG_DIR", None)
    gpu_memory_log = cli.gpu_memory_log if cli.gpu_memory_log is not None else f"{cli.output_dir}/gpu_memory.jsonl"
    args = GRPOConfig(
        output_dir=cli.output_dir,
        loss_type="dr_grpo",
        scale_rewards=False,
        # Multi-turn tool use creates a few turn-boundary logp outliers between
        # vLLM sampling and trainer recomputation. TRL's default sequence_mask
        # can zero whole trajectories; token_truncate clamps only outlier tokens.
        # ISR health is watched at runtime by LongRunMonitorCallback.
        vllm_importance_sampling_mode=cli.vllm_importance_sampling_mode,
        num_generations=cli.num_generations,
        per_device_train_batch_size=cli.per_device_batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        max_completion_length=cli.max_completion_length,
        max_tool_calling_iterations=cli.max_tool_iterations or None,
        max_steps=cli.max_steps,
        learning_rate=cli.learning_rate,
        seed=cli.seed,
        data_seed=cli.seed,
        shuffle_dataset=cli.shuffle_dataset,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs=None if cli.adapter_path else {"dtype": "bfloat16"},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_host=cli.vllm_server_host,
        vllm_server_port=cli.vllm_server_port,
        log_completions=cli.log_completions,
        logging_steps=cli.logging_steps,
        save_strategy=cli.save_strategy,
        save_steps=cli.save_steps,
        report_to=cli.report_to,
    )
    model = cli.model
    peft_cfg = None if cli.full_finetune or cli.adapter_path else lora_config()
    if cli.adapter_path:
        from peft import PeftModel
        from trl.trainer.utils import create_model_from_path

        base_model = create_model_from_path(cli.model, dtype="bfloat16")
        model = PeftModel.from_pretrained(base_model, cli.adapter_path, is_trainable=True)
    thresholds = MonitorThresholds(
        isr_mean_min=cli.monitor_isr_min,
        clipped_ratio_max=cli.monitor_clipped_max,
        gpu_reserved_frac_max=cli.monitor_gpu_frac_max,
        gpu_free_frac_alert_min=cli.monitor_gpu_free_frac_min,
        max_containers=cli.monitor_max_containers,
        patience=cli.monitor_patience,
        abort=not cli.no_monitor_abort,
    )
    memory_logger = GpuMemoryJsonlLogger(
        gpu_memory_log,
        empty_cache_steps=cli.cuda_empty_cache_steps,
        check_containers=os.environ.get("LP_RL_SANDBOX", "local") == "docker",
    )
    print(f"model={cli.model} split={cli.split}")
    dataset = build_training_dataset(
        cli.split,
        cli.task_ids,
        cli.families,
        cli.repeat,
        cli.system_prompt_variant,
    )
    print(
        f"dataset_rows={len(dataset)} task_ids={parse_csv_list(cli.task_ids) or 'all'} "
        f"families={parse_csv_list(cli.families) or 'all'} repeat={cli.repeat} "
        f"system_prompt_variant={cli.system_prompt_variant}"
    )
    print(
        f"max_completion_length(L0)={cli.max_completion_length} num_generations={cli.num_generations} "
        f"max_tool_iterations={cli.max_tool_iterations or 'unlimited'}"
    )
    print(
        f"use_vllm={use_vllm} vllm_mode={vllm_mode} "
        f"lora={peft_cfg is not None or bool(cli.adapter_path)} adapter_path={cli.adapter_path or 'none'}"
    )
    print(f"rollout_log_dir={rollout_log_dir or 'disabled'}")
    print(
        f"gpu_memory_log={gpu_memory_log or 'disabled'} "
        f"cuda_empty_cache_steps={cli.cuda_empty_cache_steps}"
    )
    print(
        f"monitor: isr_min={thresholds.isr_mean_min} clipped_max={thresholds.clipped_ratio_max} "
        f"gpu_frac_max={thresholds.gpu_reserved_frac_max} gpu_free_alert_min={thresholds.gpu_free_frac_alert_min} "
        f"max_containers={thresholds.max_containers} "
        f"patience={thresholds.patience} abort={thresholds.abort}"
    )
    print(f"trainer_class={trainer_cls.__name__}")
    trainer = trainer_cls(
        model=model,
        args=args,
        reward_funcs=least_privilege_reward,
        environment_factory=TerminalEnv,
        train_dataset=dataset,
        peft_config=peft_cfg,
        callbacks=[build_monitor_callback(thresholds, cli.output_dir, memory_logger)],
    )
    resume = cli.resume_from_checkpoint or None
    if resume:
        print(f"[train] resuming from checkpoint: {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
