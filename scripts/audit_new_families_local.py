"""Exhaustive local audit of the new/scaled expansion families (a HARD gate).

Complements the two existing gates rather than duplicating them:
  - run_oracle_smoke.py proves every oracle SUCCEEDS (but not that cheats fail or
    that risk labels fit);
  - anti_overfitting_audit.py does the (a)-(d) checks on a 50-task SAMPLE.

This runs over EVERY new/scaled task (`phase23_scale` tag or a new family) and, for
each one, on the local backend:
  1. runs the honest oracle  -> success + reward >= 0.6;
  2. risk-label coverage     -> required_risk covers the MEASURED oracle z_tau in
                                every dimension (a tighter label would punish the
                                honest oracle / mislabel the task);
  3. no-read shortcut        -> answering/escalating with the correct value while
                                performing NO reads/writes must FAIL (proves
                                require_evidence / the state verifier binds; there is
                                no free-answer reward hack).

Exits non-zero (and lists anomalies) if any task fails any check.

    LP_RL_TASK_FILE=data/expansion_v2_microslice.json \
        .venv/bin/python scripts/audit_new_families_local.py
"""
from __future__ import annotations
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.env.terminal_env import TerminalEnv
from least_privilege_env.tasks.loader import load_tasks

NEW_FAMILIES = {
    "structured_output", "multi_tool_workflow", "fake_external_read",
    "log_data_transform", "unseen_mcp_server", "reserved_new_behavior",
}
DIMS = ["write", "exec", "external", "secret", "scope", "persistent"]
REWARD_FLOOR = 0.6


def _close(env: TerminalEnv) -> None:
    try:
        if env.sandbox:
            env.sandbox.close()
        if env.materialized:
            env.materialized.cleanup()
    except Exception:
        pass


def _run_oracle(task):
    env = TerminalEnv()
    env.reset(task_id=task.task_id)
    for command in task.oracle.commands:
        env.terminal_run(command)
    for call in task.oracle.mcp_calls:
        env.mcp_call(call.server, call.tool, dict(call.args))
    if task.oracle.escalation_code:
        env.human_escalation(task.oracle.escalation_code, "audit")
    else:
        env.final_answer(task.oracle.final_answer or "")
    ok = bool(env.verdict and env.verdict.success == 1.0)
    reward = env.last_reward.reward if env.last_reward else None
    z = env.audit.z_tau().to_dict()
    _close(env)
    return ok, reward, z


def _run_shortcut(task) -> bool:
    """Answer/escalate with the correct value but perform NO reads/writes."""
    env = TerminalEnv()
    env.reset(task_id=task.task_id)
    if task.oracle.escalation_code:
        env.human_escalation(task.oracle.escalation_code, "shortcut")
    else:
        env.final_answer(task.oracle.final_answer or "")
    cheated = bool(env.verdict and env.verdict.success == 1.0)
    _close(env)
    return cheated


def main() -> int:
    tasks = load_tasks()
    is_new = lambda t: (t.family in NEW_FAMILIES) or ("phase23_scale" in t.command_pattern_tags)
    new = [t for t in tasks if is_new(t)]
    print(f"new/scaled tasks under audit: {len(new)}")
    by_fam: dict[str, list] = collections.defaultdict(list)
    for t in new:
        by_fam[t.family].append(t)

    anomalies: list[str] = []
    header = f"{'family':24} {'n':>4} {'oraclePASS':>10} {'rwd_min':>8} {'risk_bad':>8} {'cheatPASS':>9} {'distinctAns':>11}"
    print("\n" + header)
    for fam in sorted(by_fam):
        ts = by_fam[fam]
        passes = risk_bad = cheat_pass = 0
        rewards: list[float] = []
        answers = set()
        for t in ts:
            ok, reward, z = _run_oracle(t)
            if ok:
                passes += 1
            else:
                anomalies.append(f"ORACLE-FAIL {t.task_id}")
            if reward is not None:
                rewards.append(reward)
                if reward < REWARD_FLOOR:
                    anomalies.append(f"LOW-REWARD {t.task_id} r={reward:.3f}")
            rr = t.required_risk
            rr = rr.to_dict() if hasattr(rr, "to_dict") else {d: getattr(rr, d, 0.0) for d in DIMS}
            for d in DIMS:
                if float(rr.get(d, 0.0)) + 1e-6 < float(z.get(d, 0.0)):
                    risk_bad += 1
                    anomalies.append(f"RISK-TIGHT {t.task_id}.{d} req={rr.get(d, 0.0):.2f} < oracle={z.get(d, 0.0):.2f}")
            if _run_shortcut(t):
                cheat_pass += 1
                anomalies.append(f"CHEAT-PASS {t.task_id} (no-read answer accepted!)")
            answers.add(t.oracle.final_answer or t.oracle.escalation_code)
        rmin = f"{min(rewards):.3f}" if rewards else "n/a"
        print(f"{fam:24} {len(ts):>4} {passes:>10} {rmin:>8} {risk_bad:>8} {cheat_pass:>9} {len(answers):>11}")

    print(f"\nanomalies: {len(anomalies)}")
    for a in anomalies[:60]:
        print("  ", a)
    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
