from __future__ import annotations

from train.train_trl_drgrpo import (
    CLIPPED_KEY,
    ISR_KEY,
    TOOL_FAILURE_KEY,
    LongRunMonitor,
    MonitorThresholds,
)


def test_isr_collapse_aborts_after_patience() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=3))
    for step in range(2):
        alerts, abort = monitor.observe({ISR_KEY: 0.1})
        assert alerts and abort is False, step
    alerts, abort = monitor.observe({ISR_KEY: 0.1})
    assert abort is True
    assert "isr_collapse" in alerts[0]


def test_recovery_resets_breach_streak() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=2))
    monitor.observe({ISR_KEY: 0.1})
    alerts, abort = monitor.observe({ISR_KEY: 0.99})
    assert alerts == [] and abort is False
    _, abort = monitor.observe({ISR_KEY: 0.1})
    assert abort is False  # streak restarted


def test_healthy_logs_produce_no_alerts() -> None:
    monitor = LongRunMonitor(MonitorThresholds())
    alerts, abort = monitor.observe(
        {ISR_KEY: 0.9901, CLIPPED_KEY: 0.0, TOOL_FAILURE_KEY: 0.0},
        gpu_reserved_frac=0.94,
        container_count=2,
    )
    assert alerts == [] and abort is False


def test_clipping_gpu_and_containers_breach() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=1))
    alerts, abort = monitor.observe(
        {CLIPPED_KEY: 0.8},
        gpu_reserved_frac=1.05,
        container_count=200,
    )
    assert abort is True
    joined = "\n".join(alerts)
    assert "completion_clipping" in joined
    assert "gpu_memory" in joined
    assert "leftover_containers" in joined


def test_tool_failure_is_alert_only() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=1))
    for _ in range(5):
        alerts, abort = monitor.observe({TOOL_FAILURE_KEY: 1.0})
        assert abort is False
        assert alerts and "alert only" in alerts[0]


def test_low_gpu_free_fraction_is_alert_only() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=1, gpu_free_frac_alert_min=0.02))
    alerts, abort = monitor.observe({}, gpu_free_frac=0.001)
    assert abort is False
    assert alerts and "gpu_free" in alerts[0] and "alert only" in alerts[0]


def test_missing_signals_are_ignored() -> None:
    monitor = LongRunMonitor(MonitorThresholds(patience=1))
    alerts, abort = monitor.observe({"loss": 0.5})
    assert alerts == [] and abort is False
