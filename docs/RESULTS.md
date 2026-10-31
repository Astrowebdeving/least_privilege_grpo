# Experimental results

This document is the compact public index for the completed internal,
external-transfer, corrective-continuation, prompt-interaction, and reward-arm
experiments. The files under `results/` preserve the corresponding compact
analysis records, run configurations, and completion guards. Raw third-party
benchmark corpora and per-example generations are not redistributed.

## Comparison contract

- The base and trained policies use `Qwen/Qwen3.5-4B`; trained policies are LoRA
  adapters. The primary policy is seed 1 at checkpoint 1500.
- Internal evaluation uses the frozen 500-task set. Ordinary tasks receive four
  generations and safety-heavy tasks eight, for 2,896 episodes per policy.
- Deterministic external panels use identical item membership and prompts within
  each comparison. FORTIS and MetaTool use their official prompts. ToolPrivBench
  includes both default- and privilege-aware-prompt cells where stated.
- Internal uncertainty is clustered by task because generations from one task
  are dependent. Deterministic paired external outcomes use exact McNemar tests.
- External benchmarks were evaluation-only. Their distinctive tool vocabulary
  was checked against the training catalogs by
  `scripts/check_external_disjointness.py`.

## Primary checkpoint-1500 result

### Frozen internal evaluation

| Policy | Safe success | Success | Over-privileged success |
|---|---:|---:|---:|
| Base | 1,864/2,896 (64.36%) | 1,996/2,896 (68.92%) | 132/2,896 (4.56%) |
| Seed 1, checkpoint 1500 | **2,852/2,896 (98.48%)** | **2,875/2,896 (99.27%)** | **23/2,896 (0.79%)** |

The frozen panel is the strongest in-environment result: a 34.12-point aggregate
safe-success gain with 109 fewer over-privileged successes. These are the
primary frozen-500 run reported in the paper (`results/primary/frozen500_run_config.txt`).
The continuation analysis re-evaluated the parent in a separate run with a
different generation seed, so parent values under `results/continuation/` differ
slightly. Because safety-heavy
tasks receive more samples, aggregate episode rates are not task-equal rates;
the continuation analysis reports both forms where they are compared directly.

On the 206-task routine subset, the trajectory is already strong at checkpoint
500 and then reduces residual over-privilege: safe success is 96.91%, 97.57%,
and 97.63% at checkpoints 500, 1000, and 1500, while over-privileged success is
1.52%, 1.03%, and 0.73%. These are checkpoints from one trajectory, so they are
sample-efficiency evidence rather than independent training replicates.

### MetaTool registered subset

The deterministic 1,000-example registered subset improves from 819/1,000
(81.9%) for base to 836/1,000 (83.6%) for seed 1. The paired changes are 21
base-wrong/seed-right and four base-right/seed-wrong. Most of the gain is on
unavailable-tool restraint (162/200 to 174/200). This is a registered subset,
not a claim about the complete MetaTool leaderboard.

### FORTIS

On Task 1 (600 direct skill-selection items), seed 1 improves exact selection
from 225 to 237 and reduces over-privileged choices from 307 to 297. The
direction is consistent across all three trained seeds, but the per-seed
over-privilege result is borderline (16 beneficial versus six adverse
discordances; one-sided exact p about 0.052).

On Task 2 (1,543 tool-selection items), exact match improves from 253 to 308 and
unparseable/no-action outputs fall from 286 to 157. Unconditional
over-privilege rises from 546 to 571, while action-conditional over-privilege
falls from 43.4% to 41.2%. The increase is concentrated in convenience-sensitive
filesystem cases and reflects a real utility-safety tradeoff, not a uniform
safety gain. `results/primary/fortis.md` contains the paired transition matrices
and category/domain breakdowns.

### ToolPrivBench before correction

Under the official default prompt, checkpoint 1500 is nearly tied with base
overall: 245/544 versus 246/544 over-privileged selections. The aggregate masks
improvements in data exposure, scope expansion, and persistence alongside
regressions in authority escalation and safety bypass. This diagnosed a lack of
varied dynamic authority-recovery contrasts in the original training catalog;
it motivated a frozen post-hoc corrective continuation rather than retroactive
selection of the primary checkpoint.

## Prompt/norm ablation at the primary checkpoint

Removing the least-privilege norm while holding the internal tool-interface
scaffolding fixed does not remove the internal gain: seed 1 reaches 97.6% safe
success under both `P_full` and `P_none`, versus 61.7% and 60.9% for base. This
supports partial internalization rather than pure prompt compliance.

On ToolPrivBench, the privilege-aware prompt lowers over-privileged selection
from 45.2% to 41.2% for base and from 45.0% to 37.9% for checkpoint 1500. The
original interaction estimate is directionally favorable but its paired
bootstrap interval includes zero. See `results/primary/prompt_ablations.md`.

## Corrective continuation

The selected checkpoint-1500 policy was continued for 400 fixed endpoint steps
on 200 cue-neutral corrective tasks interleaved with 200 source-exact replay
tasks. The task mixture targets transient narrow-tool failures, bounded retry,
equal-authority alternatives, convenience traps, and genuine escalation while
replaying original behaviors to limit forgetting. Selection used a disjoint
50-task lockbox; external benchmarks remained evaluation-only.

| Panel | Parent checkpoint 1500 | Continuation 400 | Result |
|---|---:|---:|---|
| 50-task lockbox safe success | 271/400 (67.75%) | **362/400 (90.50%)** | +22.75 pp; task-bootstrap 95% CI +12.0 to +34.5; one-sided cluster-randomization p < 0.0002 |
| Frozen-500 safe success | 2,852/2,896 (98.48%) | **2,860/2,896 (98.76%)** | +0.28 pp; not significant |
| MetaTool accuracy | 836/1,000 (83.6%) | **835/1,000 (83.5%)** | -1; not significant overall |
| ToolPrivBench over-privileged selection | 245/544 (45.04%) | **207/544 (38.05%)** | -6.99 pp; exact one-sided p=3.7e-5 |

Against base, continuation 400 remains much stronger internally and reduces
ToolPrivBench over-privilege from 246/544 to 207/544 (66 beneficial versus 27
adverse discordances; p=3.23e-5). It also improves FORTIS Task-1 exact match
225 to 238 and over-privilege 307 to 293. FORTIS Task-2 exact match and safe
classification improve, but its unconditional over-privilege count remains
20 above base (one-sided regression p=0.0578).

The correction is not literally regression-free. MetaTool reliability falls
from 174/200 at the parent to 166/200, with all eight discordances adverse
(Holm-adjusted p=0.0195 across its five subsets), although continuation remains
above base overall. The publishable conclusion is targeted corrective
generalization with broad internal retention, not uniform dominance.

### Additivity with explicit policy guidance

Under ToolPrivBench's privilege-aware prompt, continuation 400 reaches 145/544
(26.65%) over-privileged selections, compared with 224/544 (41.18%) for base and
206/544 (37.87%) for the parent. Relative to its own default prompt, 78 cases
improve and 16 regress (exact one-sided p=2.89e-11), while the success proxy is
unchanged. The interaction versus base is -7.35 points with paired-bootstrap
95% CI -11.95 to -2.76, showing that learned correction and explicit guidance
are additive rather than redundant in this experiment.

## Reward-arm causal studies

The reward study changes one reward component at a time while keeping the
catalog, training budget, seeds, rollout sampling, parent policy, and control
evaluation scorer fixed. `A0` is the full reward, `B1` removes the privilege
penalty, `A2` replaces it with a flat count of actions that touch an unlicensed
write, execution, external, or secret dimension (no degrees or weights), and
`A4` keeps the penalty but removes the task envelope so all risk is penalized.

In the base-start 500-task campaign (three seeds per arm), held-out safe success
means are 0.9033 (`A0`), 0.8775 (`A2`), 0.9633 (`A4`), and 0.8700 (`B1`). The
apparent `A4` gain is driven by output-format families; after excluding them,
`A4` is 1.19 points below `A0` and its interval touches zero. `A2` and `B1` are
1.98 and 1.88 points below `A0`, respectively, on no-format safe success, with
task-bootstrap intervals excluding zero. Internal over-privilege is floor-limited
and should not be used as the main discriminator.

In the production-policy continuation campaign (five seeds per main arm), mean
ToolPrivBench over-privilege is 36.69% for `A0`, 35.55% for `A2`, and 42.21% for
`B1`, versus 46.32% for the same-run parent. `A0` and `A2` do not differ
significantly at the seed level (exact two-sided permutation p=0.611), while
`B1` is worse than `A0` (p=0.040) and `A2` (p=0.016). Thus removing the
privilege penalty loses a significant part of the transfer gain, while the flat
count and the graded penalty are statistically tied on this five-seed transfer
panel. Full per-seed outputs are under `results/causal/`.

## Reproducibility boundaries

The repository redistributes the environment, all first-party task catalogs,
builders, frozen manifests, launchers, analyzers, tests, and compact result
records. It intentionally excludes raw third-party benchmark corpora, raw model
generations, Docker state, optimizer states, and machine-specific recovery logs.
All released adapter endpoints and their SHA-256 manifest are published at
`tu101/least-privilege-model` on Hugging Face.
