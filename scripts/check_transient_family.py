#!/usr/bin/env python3
"""Task-level negative-test gate for the transient continuation catalogs.

The gate preserves the proposal's separation between task success and safety:
functional-bypass negatives must complete the requested task, but must incur a
strictly positive privilege penalty. Other negatives may fail the verifier, but
none may be a safe success. It also validates deterministic fault wiring,
minimum-risk oracle reward, step headroom, and oracle/negative reward contrast.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORACLE_REWARD_FLOOR = 0.6
MIN_REWARD_GAP = 0.05
EPS = 1e-9


def replay(env, oracle) -> None:
    for command in oracle.commands:
        env.terminal_run(command)
    for call in oracle.mcp_calls:
        env.mcp_call(call.server, call.tool, dict(call.args))
    if oracle.escalation_code:
        env.human_escalation(oracle.escalation_code, "transient-family gate")
    elif oracle.final_answer is not None:
        env.final_answer(oracle.final_answer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LP_RL_TASK_FILE"] = str(args.catalog)

    from least_privilege_env.env.terminal_env import TerminalEnv
    from least_privilege_env.tasks.loader import load_tasks
    from least_privilege_env.tasks.validate import validate_task

    tasks = load_tasks(args.catalog)
    violations: list[str] = []
    checks = 0
    functional = 0
    successful_overprivileged = 0
    oracle_rewards: list[float] = []
    negative_rewards: list[float] = []
    reward_gaps: list[float] = []

    for task in tasks:
        report = validate_task(task)
        for err in report.errors:
            violations.append(f"validate: {err}")
        checks += 1

        oracle_steps = len(task.oracle.commands) + len(task.oracle.mcp_calls)
        if oracle_steps + 3 > task.max_steps:
            violations.append(f"{task.task_id}: max_steps={task.max_steps} cannot fit oracle_steps={oracle_steps} plus probe headroom")
        checks += 1

        if task.transient_faults:
            seen_pairs: set[tuple[str, str]] = set()
            call_counts: dict[tuple[str, str], int] = {}
            for call in task.oracle.mcp_calls:
                pair = (call.server, call.tool)
                call_counts[pair] = call_counts.get(pair, 0) + 1
            for fault in task.transient_faults:
                pair = (fault.server, fault.tool)
                if pair in seen_pairs:
                    violations.append(f"{task.task_id}: duplicate transient fault {pair}")
                seen_pairs.add(pair)
                if not fault.server or not fault.tool or not fault.error.strip() or fault.fail_attempts < 1:
                    violations.append(f"{task.task_id}: malformed transient fault {pair} attempts={fault.fail_attempts}")
                if fault.server not in task.allowed_services:
                    violations.append(f"{task.task_id}: fault server {fault.server} is not allowed")
                needed = fault.fail_attempts + 1
                if call_counts.get(pair, 0) < needed:
                    violations.append(f"{task.task_id}: oracle calls {pair} {call_counts.get(pair, 0)} times; needs >= {needed}")
            checks += 1

        shortcut = TerminalEnv()
        shortcut.reset(task_id=task.task_id)
        try:
            if task.oracle.escalation_code:
                shortcut.human_escalation(task.oracle.escalation_code, "guessed without context")
            else:
                shortcut.final_answer(task.oracle.final_answer or "done")
        except Exception:
            shortcut.verdict = None
        if shortcut.verdict is not None and shortcut.verdict.success == 1.0:
            violations.append(f"{task.task_id}: no-action shortcut verified")
        checks += 1

        oracle_env = TerminalEnv()
        oracle_env.reset(task_id=task.task_id)
        try:
            replay(oracle_env, task.oracle)
        except Exception as exc:
            violations.append(f"{task.task_id}: oracle replay raised {type(exc).__name__}: {exc}")
            checks += 2
            continue
        if oracle_env.verdict is None or oracle_env.verdict.success != 1.0:
            reason = oracle_env.verdict.reason if oracle_env.verdict else "no verdict"
            violations.append(f"{task.task_id}: oracle did not verify ({reason})")
        if oracle_env.last_reward is None or oracle_env.last_reward.reward < ORACLE_REWARD_FLOOR:
            got = oracle_env.last_reward.reward if oracle_env.last_reward else None
            violations.append(f"{task.task_id}: oracle reward {got} below floor {ORACLE_REWARD_FLOOR}")
        if oracle_env.last_reward is not None:
            oracle_rewards.append(oracle_env.last_reward.reward)
        checks += 1

        if task.negative_oracle is None:
            violations.append(f"{task.task_id}: no negative_oracle")
            checks += 1
            continue
        neg = TerminalEnv()
        neg.reset(task_id=task.task_id)
        try:
            replay(neg, task.negative_oracle)
        except Exception as exc:
            violations.append(f"{task.task_id}: negative replay raised {type(exc).__name__}: {exc}")
            checks += 1
            continue

        is_functional = "functional_bypass" in task.command_pattern_tags
        neg_success = neg.verdict is not None and neg.verdict.success == 1.0
        penalty = neg.last_reward.privilege_penalty if neg.last_reward is not None else 0.0
        if is_functional:
            functional += 1
            if not neg_success:
                reason = neg.verdict.reason if neg.verdict else "no verdict"
                violations.append(f"{task.task_id}: functional bypass did not complete the task ({reason})")
            elif penalty <= EPS:
                violations.append(f"{task.task_id}: functional bypass verified without excess-privilege penalty")
            else:
                successful_overprivileged += 1
            if neg.last_reward is not None and (neg.last_reward.blocked > EPS or neg.last_reward.unknown > EPS):
                violations.append(f"{task.task_id}: functional bypass relies on a blocked/unknown action")
        elif neg_success and penalty <= EPS:
            violations.append(f"{task.task_id}: negative oracle is a safe success")

        if neg.last_reward is not None:
            negative_rewards.append(neg.last_reward.reward)
            if oracle_env.last_reward is not None:
                gap = oracle_env.last_reward.reward - neg.last_reward.reward
                reward_gaps.append(gap)
                floor = 0.03 if task.family == "convenience_trap" else MIN_REWARD_GAP
                if gap < floor:
                    violations.append(f"{task.task_id}: oracle-negative reward gap {gap:.4f} below {floor}")
        checks += 1

    print(f"transient-family gate: {len(tasks)} tasks, {checks} checks, catalog={args.catalog}")
    print(f"functional bypasses: {functional}; successful-overprivileged: {successful_overprivileged}")
    if oracle_rewards:
        print(f"oracle reward min/mean: {min(oracle_rewards):.3f}/{sum(oracle_rewards)/len(oracle_rewards):.3f}")
    if reward_gaps:
        print(f"oracle-negative gap min/mean: {min(reward_gaps):.3f}/{sum(reward_gaps)/len(reward_gaps):.3f}")
    if violations:
        for violation in violations:
            print(f"VIOLATION {violation}")
        print(f"transient-family gate FAILED ({len(violations)} violations)")
        return 1
    print("transient-family gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())