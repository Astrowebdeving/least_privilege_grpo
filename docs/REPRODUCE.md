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
`--check` to assert the committed file matches what the builder would emit:

```bash
python scripts/build_expansion_v2.py --check      # 2,000-task catalog
python scripts/build_risk_archive.py --check      # host-only risk labels
python scripts/freeze_expansion_v2.py --check     # the freeze manifest
python scripts/check_risk_archive.py
python scripts/check_task_quality.py
python scripts/validate_tasks.py
```

Or run everything:

```bash
bash scripts/_run_all_gates.sh
```

These gates are not decoration. They enforce that no held-out task shares a
prompt or `template_group_id` with a training task, that verifiers cannot be
satisfied by reading the workspace, and that reward is not trivially hackable
(`check_reward_hacking_probes.py`, `check_reward_sensitivity.py`).

## 3. Evaluate a baseline

```bash
bash scripts/serve_train_vllm.sh                  # vLLM on GPU 1
export LP_RL_SANDBOX=docker
export LP_RL_TASK_FILE=data/expansion_v2_microslice.json
export LP_RL_RISK_ARCHIVE=data/risk_archive_v2.json
bash scripts/run_eval_capped.sh
python scripts/summarize_rollouts.py
python scripts/safety_metrics_report.py
```

`LP_RL_RISK_ARCHIVE` is report-only — it makes `safe_success` and
`over_privileged_success` meaningful in summaries. It does not affect reward.
Omitting it does not change training; it silently makes the safety columns
meaningless, which is easy to miss.

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

## 5. Confirmatory evaluation

```bash
bash scripts/run_frozen500_confirmatory.sh
python scripts/validate_frozen500_run.py
python scripts/compare_rollout_reports.py
```

Evaluate a **fixed, pre-registered** checkpoint. Selecting a checkpoint by its
score on the evaluation set turns a held-out measurement into a training signal;
external benchmarks in particular must never be used for selection.

## Statistical notes

Both the internal splits and the external benchmarks are **paired** designs — the
same task set under two policies. Use McNemar's test on the discordant pairs, not
an unpaired comparison of marginal rates; the unpaired version substantially
understates significance and will hide real effects.

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
