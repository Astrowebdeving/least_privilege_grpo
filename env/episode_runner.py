from __future__ import annotations

from least_privilege_env.env.terminal_env import TerminalEnv


def run_oracle(task_id: str) -> float:
    from least_privilege_env.tasks.loader import task_by_id

    task = task_by_id(task_id)
    env = TerminalEnv()
    env.reset(task_id=task_id)
    for command in task.oracle.commands:
        env.terminal_run(command)
    if task.oracle.escalation_code:
        env.human_escalation(task.oracle.escalation_code, "oracle")
    else:
        env.final_answer(task.oracle.final_answer or "")
    return env._reward()
