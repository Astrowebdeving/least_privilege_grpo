from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import summarize_gpu_memory as sgm
from train.train_trl_drgrpo import GpuMemoryJsonlLogger, cuda_memory_snapshot


class FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.empty_cache_calls = 0
        self.reset_peak_calls = 0
        self.total = 1000
        self.allocated = 200
        self.reserved = 300
        self.max_allocated = 400
        self.max_reserved = 500
        self.free = 700

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, device: int):
        return SimpleNamespace(total_memory=self.total, name="Fake GPU")

    def get_device_name(self, device: int) -> str:
        return "Fake GPU"

    def mem_get_info(self, device: int | None = None):
        return self.free, self.total

    def memory_allocated(self, device: int) -> int:
        return self.allocated

    def memory_reserved(self, device: int) -> int:
        return self.reserved

    def max_memory_allocated(self, device: int) -> int:
        return self.max_allocated

    def max_memory_reserved(self, device: int) -> int:
        return self.max_reserved

    def reset_peak_memory_stats(self) -> None:
        self.reset_peak_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1
        self.reserved = self.allocated
        self.free = self.total - self.reserved


def test_cuda_unavailable_memory_logging_writes_null_fields(tmp_path: Path) -> None:
    fake = SimpleNamespace(cuda=FakeCuda(available=False))
    path = tmp_path / "gpu_memory.jsonl"
    logger = GpuMemoryJsonlLogger(str(path), empty_cache_steps=1, torch_module=fake)

    record = logger.record(1, {"reward": 0.5})

    assert record["cuda_available"] is False
    assert record["memory_allocated"] is None
    assert record["metrics"]["reward"] == 0.5
    assert path.read_text(encoding="utf-8").strip()


def test_fake_cuda_snapshot_serializes_memory_fields() -> None:
    fake_cuda = FakeCuda()
    snapshot = cuda_memory_snapshot(SimpleNamespace(cuda=fake_cuda))

    assert snapshot["cuda_available"] is True
    assert snapshot["cuda_device_name"] == "Fake GPU"
    assert snapshot["memory_allocated"] == 200
    assert snapshot["memory_reserved"] == 300
    assert snapshot["max_memory_allocated"] == 400
    assert snapshot["max_memory_reserved"] == 500
    assert snapshot["mem_get_info_free"] == 700
    assert snapshot["memory_reserved_frac"] == 0.3


def test_empty_cache_steps_zero_never_clears_cache(tmp_path: Path) -> None:
    fake_cuda = FakeCuda()
    logger = GpuMemoryJsonlLogger(str(tmp_path / "m.jsonl"), empty_cache_steps=0, torch_module=SimpleNamespace(cuda=fake_cuda))

    logger.record(1, {})

    assert fake_cuda.empty_cache_calls == 0
    assert logger.last_record is not None
    assert logger.last_record["cleanup_empty_cache_ran"] is False


def test_empty_cache_steps_one_clears_after_logging(tmp_path: Path) -> None:
    fake_cuda = FakeCuda()
    logger = GpuMemoryJsonlLogger(str(tmp_path / "m.jsonl"), empty_cache_steps=1, torch_module=SimpleNamespace(cuda=fake_cuda))

    record = logger.record(1, {})

    assert fake_cuda.empty_cache_calls == 1
    assert record["cleanup_empty_cache_ran"] is True
    assert record["memory_reserved"] == 300
    assert record["post_cleanup"]["memory_reserved"] == 200


def _record(step: int, allocated: int, reserved: int, max_allocated: int, free: int, total: int = 1000) -> dict:
    return {
        "global_step": step,
        "cuda_available": True,
        "cuda_device_total": total,
        "memory_allocated": allocated,
        "memory_reserved": reserved,
        "max_memory_allocated": max_allocated,
        "max_memory_reserved": reserved,
        "mem_get_info_free": free,
        "mem_get_info_total": total,
        "memory_allocated_frac": allocated / total,
        "memory_reserved_frac": reserved / total,
        "max_memory_allocated_frac": max_allocated / total,
        "max_memory_reserved_frac": reserved / total,
        "mem_get_info_free_frac": free / total,
    }


def test_memory_summary_classifies_reserved_fragmentation() -> None:
    records = [
        _record(1, allocated=200, reserved=300, max_allocated=300, free=700),
        _record(2, allocated=210, reserved=650, max_allocated=320, free=350),
        _record(3, allocated=220, reserved=850, max_allocated=330, free=10),
    ]
    assert sgm.summarize(records)["classification"] == "reserved_fragmentation"


def test_memory_summary_classifies_possible_leak() -> None:
    records = [
        _record(1, allocated=100, reserved=300, max_allocated=200, free=700),
        _record(2, allocated=300, reserved=500, max_allocated=400, free=500),
    ]
    assert sgm.summarize(records)["classification"] == "possible_leak"


def test_memory_summary_classifies_allocated_ceiling() -> None:
    records = [_record(1, allocated=600, reserved=900, max_allocated=920, free=80)]
    assert sgm.summarize(records)["classification"] == "allocated_ceiling"


def test_memory_summary_classifies_missing_cuda_as_inconclusive() -> None:
    records = [{"global_step": 1, "cuda_available": False, "memory_allocated": None}]
    assert sgm.summarize(records)["classification"] == "inconclusive"


def test_eval_wrapper_preserves_memory_diag_modes() -> None:
    # The memory-diagnostic observe/mitigate modes were folded into the eval
    # primitive as EVAL_MODE=mem-observe|mem-mitigate (2026-06-24 launcher
    # consolidation), so the same allocator/monitor knobs must survive there.
    script = Path("scripts/run_eval_capped.sh").read_text(encoding="utf-8")
    assert "--vllm-mode server" in script
    assert "EVAL_MODE" in script
    assert "mem-observe" in script
    assert "mem-mitigate" in script
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in script   # observe: no mitigation
    assert "expandable_segments:True" in script        # mitigate
    assert "--no-monitor-abort" in script              # observe: alert-only
    assert "summarize_gpu_memory.py" in script         # mem-* writes a memory summary
    assert '--cuda-empty-cache-steps "$CUDA_EMPTY_CACHE_STEPS"' in script
