# Seed-1 checkpoint trajectory: internal and external analysis

Date: 2026-07-23

Status: complete descriptive analysis. This is not checkpoint selection. The
primary selected policy remains seed-1 checkpoint-1500, selected on the frozen
internal protocol before any external result was observed.

## Artifact integrity and comparison contract

The checkpoint-500 and checkpoint-1000 external run completed with exit code 0
and exact suite count guards:

- ToolPrivBench: 544 examples per arm
- FORTIS: 2,143 examples per arm (600 task 1 + 1,543 task 2)
- MetaTool registered subset: 1,000 examples per arm

The new arms are in canonical WSL at
`logs/artifacts/ckpt_ablation_external_2026-07-21`. Their six summary JSON files,
launcher log, and run configuration are copied to the F: review mirror at
`logs/ckpt_ablation_external_2026-07-21`. Preserved base and checkpoint-1500
summary JSONs are copied to `logs/external1500_reference` on F:. SHA-256 hashes
were recorded at copy time. All checkpoint arms used the same deterministic
served-model configuration and the same benchmark prompt as base; the adapters
did not receive a stronger prompt.

The internal checkpoint-500 and checkpoint-1000 arms each contain exactly 1,648
episodes (206 tasks x8) at
`logs/artifacts/ckpt_efficiency_206_2026-07-21`. The attempted checkpoint-1500
rerun was redundant and was stopped after 312 episodes; it has no summary and is
not an analysis arm. Checkpoint-1500 internal values below come from the already
complete authoritative 206-task evaluation at
`logs/artifacts/final_routine_eval_v2_2026-07-13_recovery`.

## Internal 206-task trajectory

| Policy | Success | Safe success | Over-privileged success | Non-terminated |
| --- | ---: | ---: | ---: | ---: |
| base | 1,080/1,648 (65.53%) | 1,016 (61.65%) | 64 (3.88%) | 370 |
| seed-1 ckpt-500 | 1,622 (98.42%) | 1,597 (96.91%) | 25 (1.52%) | 5 |
| seed-1 ckpt-1000 | 1,625 (98.60%) | 1,608 (97.57%) | 17 (1.03%) | 6 |
| seed-1 ckpt-1500 | 1,621 (98.36%) | 1,609 (97.63%) | 12 (0.73%) | 5 |

Most internal capability is already present at checkpoint 500. Checkpoint 1000
is one safe success short of checkpoint 1500, although checkpoint 1500 has five
fewer over-privileged successes. This is descriptive sample-efficiency evidence,
not proof that exactly 500 steps is optimal: the arms are checkpoints from one
training trajectory and have sampling noise.

The family-level checkpoint-500 weaknesses are concentrated in scoped cleanup
(37/56 safe), noisy error recovery (35/40 safe), prompt injection (89/96 safe),
run-the-right-test (43/48 safe), and fake external reads (169/176 safe).
Checkpoint 1000 repairs run-the-right-test (48/48 safe) and fake external reads
(173/176 safe), improves scoped cleanup to 46/56 safe, but prompt injection is
lower at 79/96 safe and noisy error recovery is 36/40 safe. This non-monotonicity
is why milestone checkpoints are diagnostic rather than independently selected
policies.

## MetaTool trajectory

| Arm | Correct / 1,000 | Accuracy |
| --- | ---: | ---: |
| base | 819 | 81.9% |
| ckpt-500 | 830 | 83.0% |
| ckpt-1000 | 832 | 83.2% |
| ckpt-1500 | 836 | 83.6% |

Correct counts by MetaTool task type, in the order awareness / multi-tool /
reliability / scenario / similar-tool, are:

- base: 177 / 163 / 162 / 169 / 148
- checkpoint 500: 178 / 165 / 172 / 169 / 146
- checkpoint 1000: 178 / 163 / 173 / 169 / 149
- checkpoint 1500: 178 / 166 / 174 / 169 / 149

General tool-selection transfer appears mostly by checkpoint 500 and then rises
gradually. The gains are modest and should be described that way.

## ToolPrivBench trajectory

Lower over-privilege rate is better.

| Arm | Over-privileged / 544 | Rate |
| --- | ---: | ---: |
| base | 246 | 45.22% |
| ckpt-500 | 256 | 47.06% |
| ckpt-1000 | 246 | 45.22% |
| ckpt-1500 | 245 | 45.04% |

The aggregate conceals opposing category behavior:

| Category | n | Base | ckpt-500 | ckpt-1000 | ckpt-1500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Authority Escalation | 139 | 82 (58.99%) | 85 (61.15%) | 81 (58.27%) | 85 (61.15%) |
| Data Over-Exposure | 99 | 46 (46.46%) | 53 (53.54%) | 43 (43.43%) | 42 (42.42%) |
| Safety Bypass | 116 | 61 (52.59%) | 62 (53.45%) | 72 (62.07%) | 69 (59.48%) |
| Scope Expansion | 99 | 25 (25.25%) | 21 (21.21%) | 19 (19.19%) | 19 (19.19%) |
| Temporal Persistence | 91 | 32 (35.16%) | 35 (38.46%) | 31 (34.07%) | 30 (32.97%) |

Scope expansion improves early and remains improved. Data exposure improves
after checkpoint 500. Safety bypass is approximately base-level at checkpoint
500, becomes substantially worse at checkpoint 1000, and partly recovers by
checkpoint 1500. Authority escalation is non-monotonic and checkpoint 1500 is
worse than base despite checkpoint 1000 being slightly better. Therefore the
near-tied overall rate is not evidence of uniform least-privilege transfer.

## FORTIS trajectory

### Task 1

| Arm | Exact / 600 | Exact rate | Over-privileged / 600 |
| --- | ---: | ---: | ---: |
| base | 225 | 37.50% | 307 (51.17%) |
| ckpt-500 | 237 | 39.50% | 294 (49.00%) |
| ckpt-1000 | 236 | 39.33% | 295 (49.17%) |
| ckpt-1500 | 237 | 39.50% | 297 (49.50%) |

The direct skill-choice gain appears by checkpoint 500 and is stable.

### Task 2

| Arm | Exact / 1,543 | Exact rate | Over-privileged | No action | Action-conditional OPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 253 | 16.40% | 546 (35.39%) | 286 | 43.44% |
| ckpt-500 | 293 | 18.99% | 593 (38.43%) | 177 | 43.41% |
| ckpt-1000 | 334 | 21.65% | 610 (39.53%) | 97 | 42.19% |
| ckpt-1500 | 308 | 19.96% | 571 (37.01%) | 157 | 41.20% |

The unconditional over-privilege count rises partly because the model becomes
more action-capable: no-action cases drop sharply. Conditional on taking an
action, over-privilege improves gradually. That does not erase the regressions;
they are localized in the convenience and filesystem slices.

Task-2 category exact / over-privileged / no-action counts:

| Category (n) | Base | ckpt-500 | ckpt-1000 | ckpt-1500 |
| --- | ---: | ---: | ---: | ---: |
| convenience (464) | 48 / 241 / 144 | 66 / 278 / 89 | 86 / 302 / 43 | 78 / 280 / 72 |
| feature_gap (150) | 11 / 109 / 26 | 11 / 111 / 25 | 13 / 118 / 12 | 13 / 110 / 22 |
| min_correct (654) | 125 / 141 / 35 | 134 / 142 / 10 | 138 / 127 / 7 | 134 / 124 / 7 |
| over_priv_trap (275) | 69 / 55 / 81 | 82 / 62 / 53 | 97 / 63 / 35 | 83 / 57 / 56 |

Task-2 domain exact / over-privileged / no-action counts:

| Domain (n) | Base | ckpt-500 | ckpt-1000 | ckpt-1500 |
| --- | ---: | ---: | ---: | ---: |
| ecommerce (499) | 0 / 225 / 21 | 0 / 227 / 19 | 0 / 218 / 15 | 0 / 217 / 15 |
| email (524) | 193 / 186 / 91 | 219 / 190 / 52 | 246 / 198 / 12 | 233 / 180 / 38 |
| filesystem (520) | 60 / 135 / 174 | 74 / 176 / 106 | 88 / 194 / 70 | 75 / 174 / 104 |

## Interpretation and corrective experiment

The combined trajectory argues against a simple “more steps made everything
worse” story. General task completion, internal safe success, FORTIS exact
selection, and MetaTool improve early. Several static least-privilege categories
also improve. The reproducible weakness is narrower: when a minimum-authority
operation fails or a convenient broader path remains available, the original
training distribution did not supply enough varied dynamic authority contrasts.
Safety-bypass and authority behavior then varies non-monotonically across
checkpoints and FORTIS convenience/filesystem cases remain weak.

That diagnosis motivates, but does not guarantee, the frozen post-hoc repair:
200 disjoint training tasks covering transient narrow retries, delayed retries,
equal-authority alternatives, successful-but-overprivileged broad shortcuts,
convenience traps, and genuine escalation; 50 disjoint held-out probes are the
only checkpoint-selection set. The original checkpoint-1500 remains the primary
result. Any continuation result must be reported as a selected-seed post-hoc
before/after experiment, including a negative outcome if the repair does not
narrow the external gaps without internal regression.
