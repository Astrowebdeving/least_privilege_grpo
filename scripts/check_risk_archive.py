"""Gate: the required-risk archive must exist and match catalog/classifier/reward.

Fails when any task lacks an archive entry, when archived required_risk or
label_confidence diverge from data/spike_tasks.json, when a task changed since
its label review (hash drift), when the classifier's view of an oracle command
changed, or when reward/risk semantics drifted from the archived context.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.tasks.risk_archive import load_archive, load_raw_tasks, validate_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None, help="base archive path; defaults to data/risk_archive.json")
    parser.add_argument("--catalog", type=Path, default=None, help="catalog path; defaults to LP_RL_TASK_FILE or data/spike_tasks.json")
    parser.add_argument("--extra-archive", type=Path, default=None, help="additional archive to merge onto the base archive")
    args = parser.parse_args()

    try:
        archive = load_archive(args.archive, extra_path=args.extra_archive)
        raw_tasks = load_raw_tasks(args.catalog)
        report = validate_archive(archive, raw_tasks)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.ok:
        print("risk archive check passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
