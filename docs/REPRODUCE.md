# Reproducing the experiments

This distils the operational runbook to what someone outside our hardware needs.
Machine-specific recovery procedures have been removed.

## Requirements

- Two GPUs. One serves generation (vLLM), one trains. The measured profile
  assumes ~32 GB each; a single GPU works only for evaluation, not training.
- Rootless Docker. Every episode spawns a container, so the Docker data-root
  should live on NVMe — on spinning disk, throughput collapses.
- Python 3.11+.
- Linux, or WSL2 with `systemd=true` in `/etc/wsl.conf`.

Two separate virtualenvs are used deliberately:

```bash
python -m venv .venv && .venv/bin/pip install -e .        # serving + analysis
bash scripts/create_train_venv.sh                          # .venv-train (TRL/vLLM)
```

Keep them apart. The training stack pins `trl==1.5.1`, `transformers==5.10.2`,
`vllm==0.22.1`; installing those into the serving environment will break it.

## 0. Select the benchmark

The library defaults to the 260-task development catalog. Every real run must
opt in to the 2,000-task benchmark:

```bash
source scripts/benchmark_env.sh
```

This exports `LP_RL_TASK_FILE` and `LP_RL_RISK_ARCHIVE` together. Exporting the
task file alone is the easiest mistake to make here: it runs the right tasks,
but the risk archive then covers only 260 of the 2,000, and every safety column
in the summaries (`safe_success`, `over_privileged_success`) becomes silently
meaningless rather than erroring.

The risk archives are report-only. They do not affect the training reward, which
comes from each task's own `required_risk` field.

## 1. Set up the sandbox

```bash
bash scripts/setup_rootless_docker_system.sh   # once, needs root
bash scripts/setup_rootless_docker_user.sh     # once, per user
bash scripts/build_agent_image.sh              # builds lp-rl-worker:latest
python scripts/check_rootless_docker.py        # fail-closed preflight
```

On WSL, verify cgroup v2 first — see `configure_wsl_cgroups.md`.

Containers run with `--network=none --cpus=2 --memory=4g --pids-limit=256
--security-opt=no-new-privileges --cap-drop=ALL --read-only --tmpfs /tmp`.
Commands the classifier does not recognise are re-run in a separate
`CAP_SYS_PTRACE` strace-quarantine container so their syscall behaviour can be
classified rather than trusted.

## 2. Verify the catalog

Every artifact is reproducible from its builder, and each builder supports
`--check` to assert the committed file matches what the builder would emit. The
chain runs in this order:

```bash
python scripts/build_expanded_tasks.py --check    # 260-task base
python scripts/build_expansion_v2.py --check      # 2,000-task benchmark
python scripts/freeze_expansion_v2.py --check     # 500-task held-out freeze
python scripts/build_risk_archive.py              # refresh host-only risk labels
python scripts/check_risk_archive.py              # gate: labels match catalog + reward semantics
python scripts/check_task_quality.py
python scripts/validate_tasks.py
```

Or run everything, in dependency order, stopping at the first failure:

```bash
bash scripts/_run_all_gates.sh                    # ~8 minutes
```

These gates are not decoration. They enforce that no held-out task shares a
prompt or `template_group_id` with a training task, that verifiers cannot be
satisfied by reading the workspace, and that reward is not trivially hackable
(`check_reward_hacking_probes.py`, `check_reward_sensitivity.py`).

Note the gate suite runs against the 260-task default on purpose — it executes
every task's oracle in a container, so the benchmark catalog would take about an
hour instead of eight minutes. To gate the full catalog, `source
scripts/benchmark_env.sh` first and expect that runtime.

## 3. Evaluate a baseline

```bash
source scripts/benchmark_env.sh
bash scripts/serve_train_vllm.sh                  # vLLM on GPU 1
export LP_RL_SANDBOX=docker
bash scripts/run_eval_capped.sh
python scripts/summarize_rollouts.py
python scripts/safety_metrics_report.py
```

## 4. Train

```bash
bash scripts/run_train_capped.sh          # single run; vLLM must already serve
bash scripts/run_multiseed.sh             # the multi-seed campaign
```

`run_train_capped.sh` is a primitive: it assumes vLLM is already serving on the
generation GPU and will die after its connect timeout otherwise. Callers must
start the server first.

Always run detached (`setsid nohup …`) so a closed terminal cannot kill a
multi-hour run.

On WSL, launch the detached Linux command from a Windows-owned `wsl.exe`
process with redirected stdin/stdout. Do not rely on `systemd-run --user` for a
multi-day campaign: a WSL teardown can end the user unit, and a clean unit may
not inherit the rootless-Docker socket environment. The launchers preserve
`DOCKER_HOST`, `XDG_RUNTIME_DIR`, GPU assignment, RAM scratch (`TMPDIR`), and
cleanup traps explicitly.

## 5. Confirmatory evaluation

```bash
bash scripts/run_frozen500_confirmatory.sh
python scripts/validate_frozen500_run.py
python scripts/compare_rollout_reports.py
```

Evaluate a **fixed, pre-registered** checkpoint. Selecting a checkpoint by its
score on the evaluation set turns a held-out measurement into a training signal.

## 6. Corrective continuation and reward arms

The post-hoc continuation and reward-arm experiments are reproducible from the
committed catalogs and launchers:

```bash
python scripts/build_continuation400_v3.py --check
python scripts/check_continuation400_v3.py
bash scripts/run_continuation400_v3_train_only.sh
bash scripts/run_continuation400_v3_eval_queue.sh

python scripts/build_ablation500.py --check
bash scripts/run_ablation_campaign.sh
bash scripts/run_cont400_arm_campaign.sh
```

The continuation uses 200 corrective tasks plus 200 source-exact replay tasks.
Its 50-task lockbox is disjoint and is the only continuation selection set.
Reward-arm evaluations always use the full `A0` scorer so a training arm cannot
change its own measuring instrument. Generated compact analyses are preserved in
`results/causal/`; see `docs/RESULTS.md` for the comparison contract.

## Statistical notes

The evaluation splits are a **paired** design — the same task set under two
policies. Use McNemar's test on the discordant pairs, not an unpaired comparison
of marginal rates; the unpaired version substantially understates significance
and will hide real effects.

For any metric conditioned on an event (for example, over-privilege rate computed
only over episodes that called a risky tool), decompose before interpreting:

```
rate = P(event) x P(outcome | event)
```

A policy that simply stops calling risky tools improves the conditional rate
without improving judgement. Reporting the marginal alone cannot distinguish
better decisions from fewer decisions.

## Gotchas

- **Zero-variance groups produce zero gradient.** With `loss_type="dr_grpo"`,
  `scale_rewards="none"` and `beta=0.0`, the advantage is `r - mean(r)` with no
  std normalisation and no KL term. A task whose generations all score
  identically contributes nothing at all. On a saturated catalog most sampled
  tasks are in exactly that state, so nominal step count can badly overstate how
  much learning signal a run actually received.
- **Reward spread is the gradient scale.** Because advantages are not normalised
  by std, a task spanning 0.698–0.703 contributes ~500x less than one spanning
  −0.95–0.70. Any "is this task informative" filter should threshold on
  magnitude, not on `std > 0`.
- **A saturated family can masquerade as a hard one.** If a family's reward is
  pinned below maximum by a capped component (an evidence term stuck at 0.5, for
  instance) while `success` fires on every generation, it looks mid-range but is
  in fact solved and gradient-free. Check reward *components*, not just totals.
- **Selecting training tasks by measured gradient magnitude does not work.** A
  task with zero measured variance at one checkpoint can still be holding a
  behaviour in place; filtering it out loses that behaviour. Measured advantage
  is also a high-variance statistic, so a filter fitted on one sample regresses
  hard toward the mean on the next.
