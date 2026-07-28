#!/usr/bin/env bash
# Host-side convenience: run every deterministic gate in order, stop on first failure.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run_gate() {
    local name="$1"; shift
    echo "=== GATE: $name ==="
    if "$@"; then
        echo "=== PASS: $name ==="
    else
        echo "=== FAIL: $name (exit $?) ==="
        exit 1
    fi
}

run_gate expanded_tasks  $PY scripts/build_expanded_tasks.py --check
run_gate build_archive   $PY scripts/build_risk_archive.py
run_gate validate_tasks  $PY scripts/validate_tasks.py
run_gate task_quality    $PY scripts/check_task_quality.py
run_gate risk_archive    $PY scripts/check_risk_archive.py
run_gate run_checks      $PY scripts/run_checks.py
run_gate parser          $PY scripts/check_parser_consistency.py
run_gate sensitivity     $PY scripts/check_reward_sensitivity.py
run_gate probes          $PY scripts/check_reward_hacking_probes.py
run_gate pytest          .venv-train/bin/python -m pytest tests/ -q
run_gate oracle_smoke    $PY scripts/run_oracle_smoke.py \
  --report logs/artifacts/oracle_local_latest.json
echo "ALL GATES PASSED"
