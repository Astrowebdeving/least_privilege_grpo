"""Focused regressions for the terminal-stop tool-loop override.

Covers the repair contract of train/lp_grpo_trainer.py:
- first successful terminal action stops the sample immediately;
- calls bundled after the terminal call in the same assistant message are
  cancelled, never executed;
- ordinary pre-terminal multi-turn behavior is byte-for-byte the pinned flow
  (tool message appended, suffix tokens masked 0, post-tool tokens masked 1);
- batching: a terminated sample drops out of regeneration while others
  continue;
- a terminal call that raises is an ordinary failure, not a termination;
- post-terminal calls against a frozen env are counted (must be 0 under the
  override, but the counter is the proof);
- the version+source pin guard trips on drift;
- per-episode telemetry lands on the env and in the rollout record.

The tests drive the copied ``_tool_call_loop`` directly with a stub trainer
(no GPU, no model): the method only touches documented ``self`` attributes.
Requires trl (the .venv-train environment); skipped elsewhere.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import lp_grpo_trainer  # noqa: E402
from lp_grpo_trainer import (  # noqa: E402
    LPTerminalStopGRPOTrainer,
    TERMINAL_TOOL_NAMES,
    assert_pinned_tool_loop,
)


class FakeEnv:
    """Mimics TerminalEnv's tool surface + freeze semantics."""

    def __init__(self):
        self.calls = []
        self.frozen = False
        self.trainer_tool_stats = None

    def terminal_run(self, cmd):
        if self.frozen:
            raise ValueError("episode over")
        self.calls.append(("terminal_run", cmd))
        return "stdout:\nok"

    def final_answer(self, text):
        if self.frozen:
            raise ValueError("episode over")
        if text is None:
            raise ValueError("final answer text must be a string or scalar")
        self.calls.append(("final_answer", text))
        self.frozen = True
        return "answer recorded"

    def human_escalation(self, code, detail):
        if self.frozen:
            raise ValueError("episode over")
        self.calls.append(("human_escalation", code))
        self.frozen = True
        return "escalation recorded"


class StubTrainer:
    """Bare object carrying only what the copied loop reads from ``self``."""

    _tool_call_loop = LPTerminalStopGRPOTrainer._tool_call_loop
    _lp_env_for = LPTerminalStopGRPOTrainer._lp_env_for

    def __init__(self, envs, regen_script=None, max_iters=8):
        self.environments = envs
        self._sync_tool_dicts = [
            {
                "terminal_run": env.terminal_run,
                "final_answer": env.final_answer,
                "human_escalation": env.human_escalation,
            }
            for env in envs
        ]
        self._async_tool_dicts = [{} for _ in envs]
        self.max_tool_calling_iterations = max_iters
        self.use_vllm = False
        self.vllm_mode = None
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=100_000), training=True
        )
        self._is_vlm = False
        self._tokenizer = object()
        self.async_loop = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.max_completion_length = 4096
        # each _generate_single_turn call pops the next entry: (ids_per_sample)
        self._regen_script = list(regen_script or [])
        self.generate_calls = []

        def _generate_single_turn(ids, images, fields):
            self.generate_calls.append([list(x) for x in ids])
            if not self._regen_script:
                raise AssertionError("unexpected _generate_single_turn call")
            per_sample = self._regen_script.pop(0)
            assert len(per_sample) == len(ids), "regen script/sample count mismatch"
            return [list(x) for x in per_sample], None

        self._generate_single_turn = _generate_single_turn
        self._get_tool_suffix_ids = lambda msgs: [900] * len(msgs)


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def assistant(*tool_calls):
    msg = {"role": "assistant", "content": ""}
    if tool_calls:
        msg["tool_calls"] = list(tool_calls)
    return msg


def test_terminal_names_match_env_surface():
    assert TERMINAL_TOOL_NAMES == {"final_answer", "human_escalation"}


def _run(trainer, completions):
    n = len(completions)
    prompts = [[{"role": "user", "content": f"task-{i}"}] for i in range(n)]
    prompt_ids = [[11, 12] for _ in range(n)]
    completion_ids = [[1, 2, 3] for _ in range(n)]
    out = trainer._tool_call_loop(prompts, prompt_ids, completion_ids, completions, None, None, {})
    tool_mask, completions, completion_ids, logprobs, call_count, failure_count, tool_images = out
    return SimpleNamespace(
        tool_mask=tool_mask,
        completions=completions,
        completion_ids=completion_ids,
        logprobs=logprobs,
        call_count=call_count,
        failure_count=failure_count,
        prompts=prompts,
    )


def test_same_message_cancellation(monkeypatch):
    env = FakeEnv()
    trainer = StubTrainer([env])
    completions = [[assistant(call("final_answer", text="42"), call("terminal_run", cmd="rm -rf /"))]]
    r = _run(trainer, completions)
    # terminal executed, bundled follow-up cancelled — never reached the env
    assert env.calls == [("final_answer", "42")]
    assert env.frozen is True
    # no regeneration happened for a sample that terminated in its first message
    assert trainer.generate_calls == []
    # completion untouched: ends at the assistant message with the terminal call
    assert r.completions[0] == completions[0]
    assert r.completion_ids[0] == [1, 2, 3]
    assert r.tool_mask[0] == [1, 1, 1]
    # prompt copy rolled back (overlong-path bookkeeping)
    assert r.prompts[0] == [{"role": "user", "content": "task-0"}]
    # telemetry
    st = env.trainer_tool_stats
    assert st["generated_calls"] == 2
    assert st["executed_calls"] == 1
    assert st["cancelled_after_terminal"] == 1
    assert st["failed_calls"] == 0
    assert st["post_terminal_calls"] == 0
    assert st["terminal_stopped"] == 1
    # only the attempted call is counted
    assert r.call_count == 1 and r.failure_count == 0
    assert trainer._metrics["train"]["tools/terminal_stopped_frac"] == [1.0]
    assert trainer._metrics["train"]["tools/cancelled_after_terminal_per_episode"] == [1.0]
    assert trainer._metrics["train"]["tools/post_terminal_calls_per_episode"] == [0.0]


def test_human_escalation_is_terminal():
    env = FakeEnv()
    trainer = StubTrainer([env])
    completions = [[assistant(call("human_escalation", code="missing_auth", detail="x"))]]
    _run(trainer, completions)
    assert env.calls == [("human_escalation", "missing_auth")]
    assert env.frozen is True
    assert trainer.generate_calls == []
    assert env.trainer_tool_stats["terminal_stopped"] == 1


def test_pre_terminal_multi_turn_flow_preserved(monkeypatch):
    env = FakeEnv()
    # iteration 0 regen produces an assistant message that calls final_answer
    trainer = StubTrainer([env], regen_script=[[[50, 51]]])
    monkeypatch.setattr(
        lp_grpo_trainer,
        "parse_response",
        lambda tok, ids: assistant(call("final_answer", text="done")),
    )
    completions = [[assistant(call("terminal_run", cmd="ls"))]]
    r = _run(trainer, completions)
    # both calls executed in order; episode frozen by the second turn
    assert env.calls == [("terminal_run", "ls"), ("final_answer", "done")]
    assert env.frozen is True
    # exactly one regeneration, fed prompt+completion+tool-suffix ids
    assert trainer.generate_calls == [[[11, 12, 1, 2, 3, 900]]]
    # pinned bookkeeping: suffix masked 0, post-tool tokens masked 1
    assert r.completion_ids[0] == [1, 2, 3, 900, 50, 51]
    assert r.tool_mask[0] == [1, 1, 1, 0, 1, 1]
    # transcript: assistant0, tool result, assistant1 (terminal turn)
    roles = [(m["role"], m.get("name")) for m in r.completions[0]]
    assert roles == [("assistant", None), ("tool", "terminal_run"), ("assistant", None)]
    st = env.trainer_tool_stats
    assert st["generated_calls"] == 2
    assert st["executed_calls"] == 2
    assert st["cancelled_after_terminal"] == 0
    assert st["terminal_stopped"] == 1


def test_batching_terminated_sample_drops_out(monkeypatch):
    env0, env1 = FakeEnv(), FakeEnv()
    # only sample 1 regenerates; its post-tool message has no further calls
    trainer = StubTrainer([env0, env1], regen_script=[[[60]]])
    monkeypatch.setattr(lp_grpo_trainer, "parse_response", lambda tok, ids: assistant())
    completions = [
        [assistant(call("final_answer", text="a"))],
        [assistant(call("terminal_run", cmd="cat f"))],
    ]
    r = _run(trainer, completions)
    assert env0.frozen is True and env1.frozen is False
    # regeneration saw ONLY the continuing sample
    assert trainer.generate_calls == [[[11, 12, 1, 2, 3, 900]]]
    # terminated sample untouched
    assert r.completion_ids[0] == [1, 2, 3] and r.tool_mask[0] == [1, 1, 1]
    # continuing sample got the pinned treatment
    assert r.completion_ids[1] == [1, 2, 3, 900, 60]
    assert r.tool_mask[1] == [1, 1, 1, 0, 1]
    assert trainer._metrics["train"]["tools/terminal_stopped_frac"] == [0.5]


def test_failed_terminal_call_does_not_stop(monkeypatch):
    env = FakeEnv()
    trainer = StubTrainer([env], regen_script=[[[70]]])
    monkeypatch.setattr(lp_grpo_trainer, "parse_response", lambda tok, ids: assistant())
    # final_answer(None) raises -> ordinary failure; bundled terminal_run STILL runs
    completions = [[assistant(call("final_answer", text=None), call("terminal_run", cmd="ls"))]]
    r = _run(trainer, completions)
    assert env.frozen is False
    assert env.calls == [("terminal_run", "ls")]
    assert len(trainer.generate_calls) == 1  # loop continued normally
    st = env.trainer_tool_stats
    assert st["failed_calls"] == 1
    assert st["executed_calls"] == 1
    assert st["cancelled_after_terminal"] == 0
    assert st["terminal_stopped"] == 0
    assert st["post_terminal_calls"] == 0
    assert r.failure_count == 1


def test_post_terminal_call_counter(monkeypatch):
    env = FakeEnv()
    env.frozen = True  # simulate an already-over episode
    trainer = StubTrainer([env], regen_script=[[[80]]])
    monkeypatch.setattr(lp_grpo_trainer, "parse_response", lambda tok, ids: assistant())
    completions = [[assistant(call("terminal_run", cmd="ls"))]]
    _run(trainer, completions)
    st = env.trainer_tool_stats
    assert st["post_terminal_calls"] == 1
    assert st["failed_calls"] == 1
    assert trainer._metrics["train"]["tools/post_terminal_calls_per_episode"] == [1.0]


def test_no_tool_calls_no_stats():
    env = FakeEnv()
    trainer = StubTrainer([env])
    completions = [[assistant()]]  # no tool calls at all
    r = _run(trainer, completions)
    assert env.trainer_tool_stats is None
    assert r.tool_mask[0] == [1, 1, 1]


def test_pinned_guard_passes_and_trips(monkeypatch):
    # current environment matches the pin
    assert_pinned_tool_loop()
    # version drift trips
    monkeypatch.setattr(lp_grpo_trainer.trl, "__version__", "1.6.0")
    with pytest.raises(RuntimeError, match="1.6.0"):
        assert_pinned_tool_loop()
    # explicit override skips (manual-review escape hatch)
    monkeypatch.setenv("LP_RL_ALLOW_TRL_DRIFT", "1")
    assert_pinned_tool_loop()


def test_rollout_record_carries_trainer_tool_stats():
    from least_privilege_env.env.rollout_log import episode_record

    stats = {"generated_calls": 3, "executed_calls": 2, "cancelled_after_terminal": 1,
             "failed_calls": 0, "post_terminal_calls": 0, "terminal_stopped": 1}
    env = SimpleNamespace(
        task=None,
        audit=SimpleNamespace(
            frozen=True,
            actions=[],
            escalation=None,
            final_answer="ok",
            z_tau=lambda: SimpleNamespace(to_dict=lambda: {}),
        ),
        verdict=None,
        last_reward=None,
        trainer_tool_stats=stats,
    )
    record = episode_record(env)
    assert record["trainer_tool_stats"] == stats
