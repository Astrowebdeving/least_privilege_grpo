from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.env.terminal_env import TerminalEnv
from least_privilege_env.tasks.loader import load_tasks


def _load_progress(path: Path) -> dict[str, dict]:
    """Last-wins map of task_id -> record from a JSONL progress file."""
    seen: dict[str, dict] = {}
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from a hard crash
        if "task_id" in rec:
            seen[rec["task_id"]] = rec
    return seen


def _run_one(task) -> dict:
    env = TerminalEnv()
    env.reset(task_id=task.task_id)
    try:
        for command in task.oracle.commands:
            env.terminal_run(command)
        for call in task.oracle.mcp_calls:
            env.mcp_call(call.server, call.tool, dict(call.args))
        if task.oracle.escalation_code:
            env.human_escalation(task.oracle.escalation_code, "oracle escalation")
        else:
            env.final_answer(task.oracle.final_answer or "")
        ok = bool(env.verdict and env.verdict.success == 1.0)
        print(f"{task.task_id}: {'PASS' if ok else 'FAIL'} reward={env._reward():.3f}", flush=True)
        return {
            "task_id": task.task_id,
            "success": ok,
            "verdict": asdict(env.verdict) if env.verdict else None,
            "reward": asdict(env.last_reward) if env.last_reward else None,
            "accessed_evidence": sorted(env.audit.accessed_evidence()),
            "z_tau": env.audit.z_tau().to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - record and continue
        print(f"{task.task_id}: ERROR {exc}", flush=True)
        return {
            "task_id": task.task_id,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env.sandbox:
            env.sandbox.close()
        if env.materialized:
            env.materialized.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--progress",
        type=Path,
        help="JSONL appended (flushed+fsynced) after every task; enables crash-safe resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks already completed in --progress (errored tasks are retried).",
    )
    parser.add_argument("--shards", type=int, default=1, help="Total shards for parallel runs.")
    parser.add_argument("--shard", type=int, default=0, help="This worker's shard index in [0, shards).")
    args = parser.parse_args()

    # Crash-safe resume: a task that previously ran to a verdict (success or fail)
    # is 'done' and skipped; a task recorded only with an 'error' (e.g. a transient
    # Docker hiccup) is retried. Final report dedupes last-wins.
    seen: dict[str, dict] = _load_progress(args.progress) if (args.progress and args.resume) else {}
    done = {tid for tid, rec in seen.items() if not rec.get("error")}

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    if args.shards > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.shards == args.shard]

    progress_fh = None
    if args.progress:
        args.progress.parent.mkdir(parents=True, exist_ok=True)
        progress_fh = args.progress.open("a", encoding="utf-8")

    skipped = 0
    try:
        for task in tasks:
            if task.task_id in done:
                skipped += 1
                continue
            record = _run_one(task)
            seen[task.task_id] = record
            if progress_fh:
                progress_fh.write(json.dumps(record) + "\n")
                progress_fh.flush()
                os.fsync(progress_fh.fileno())
    finally:
        if progress_fh:
            progress_fh.close()

    records = list(seen.values())
    failures = sum(0 if r.get("success") else 1 for r in records)
    if skipped:
        print(f"resumed: skipped {skipped} already-completed tasks", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "backend": os.environ.get("LP_RL_SANDBOX", "local"),
                    "task_count": len(records),
                    "failure_count": failures,
                    "tasks": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
