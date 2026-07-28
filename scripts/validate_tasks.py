from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from least_privilege_env.tasks.validate import validate_all


def main() -> int:
    report = validate_all()
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.ok:
        print("task validation passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
