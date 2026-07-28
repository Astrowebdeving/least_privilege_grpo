from __future__ import annotations

import inspect
import importlib.metadata as metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_trl_drgrpo import (
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_PER_DEVICE_BATCH_SIZE,
    DEFAULT_VLLM_IMPORTANCE_SAMPLING_MODE,
    LongRunMonitor,
    MonitorThresholds,
)


def main() -> int:
    from trl import GRPOConfig, GRPOTrainer

    versions = {}
    for name in ("trl", "transformers", "vllm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    print("versions:", versions)

    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "environment_factory" not in trainer_sig.parameters:
        raise SystemExit("GRPOTrainer.__init__ does not expose environment_factory")

    cfg = GRPOConfig(
        loss_type="dr_grpo",
        scale_rewards=False,
        use_vllm=False,
        max_completion_length=128,
        chat_template_kwargs={"enable_thinking": False},
    )
    for attr in ("loss_type", "scale_rewards", "use_vllm", "vllm_mode", "max_completion_length", "chat_template_kwargs"):
        if not hasattr(cfg, attr):
            raise SystemExit(f"GRPOConfig missing {attr}")
    if "max_tool_calling_iterations" not in inspect.signature(GRPOConfig.__init__).parameters and not hasattr(
        cfg, "max_tool_calling_iterations"
    ):
        raise SystemExit("GRPOConfig missing max_tool_calling_iterations (tool-loop cap)")
    expected = {
        "DEFAULT_MAX_COMPLETION_LENGTH": (DEFAULT_MAX_COMPLETION_LENGTH, 4096),
        "DEFAULT_PER_DEVICE_BATCH_SIZE": (DEFAULT_PER_DEVICE_BATCH_SIZE, 1),
        "DEFAULT_GRADIENT_ACCUMULATION_STEPS": (DEFAULT_GRADIENT_ACCUMULATION_STEPS, 8),
        "DEFAULT_NUM_GENERATIONS": (DEFAULT_NUM_GENERATIONS, 8),
        "DEFAULT_MAX_TOOL_ITERATIONS": (DEFAULT_MAX_TOOL_ITERATIONS, 20),
        "DEFAULT_VLLM_IMPORTANCE_SAMPLING_MODE": (DEFAULT_VLLM_IMPORTANCE_SAMPLING_MODE, "token_truncate"),
    }
    for name, (actual, want) in expected.items():
        if actual != want:
            raise SystemExit(f"{name}={actual}, expected {want}")
    print(
        "real-RL defaults: max_completion_length=4096 pdbs=1 "
        "grad_accum=8 num_generations=8 max_tool_iterations=20 isr=token_truncate"
    )

    thresholds = MonitorThresholds()
    monitor = LongRunMonitor(MonitorThresholds(patience=1))
    _, abort = monitor.observe({"sampling/importance_sampling_ratio/mean": 0.0})
    if not abort:
        raise SystemExit("LongRunMonitor failed to flag an ISR collapse")
    if not thresholds.abort or thresholds.patience < 1:
        raise SystemExit(f"unexpected monitor defaults: {thresholds}")
    print(
        f"long-run monitor defaults: isr_min={thresholds.isr_mean_min} "
        f"clipped_max={thresholds.clipped_ratio_max} gpu_frac_max={thresholds.gpu_reserved_frac_max} "
        f"gpu_free_alert_min={thresholds.gpu_free_frac_alert_min} "
        f"max_containers={thresholds.max_containers} patience={thresholds.patience}"
    )
    parser = __import__("train.train_trl_drgrpo", fromlist=["build_arg_parser"]).build_arg_parser()
    option_names = {action.dest for action in parser._actions}
    for required in ("gpu_memory_log", "cuda_empty_cache_steps", "monitor_gpu_free_frac_min"):
        if required not in option_names:
            raise SystemExit(f"missing training CLI option: {required}")
    print("training stack check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
