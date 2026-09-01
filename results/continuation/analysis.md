# Continuation-400 fixed-endpoint evaluation analysis

Analysis date: 2026-08-04. Candidate: `continuation400_v3_seed1_run/train/checkpoint-400`.
Parent: original selected seed-1 checkpoint-1500. The candidate was trained for
400 fixed-endpoint steps on 200 corrective-v4 tasks interleaved with 200
source-exact replay tasks using a fresh optimizer and `shuffle_dataset=false`.

## Statistical protocol

- External deterministic panels use raw per-item outcomes and exact one-sided
  McNemar tests in the preregistered beneficial direction. Two-sided values are
  used when assessing unspecified difference.
- Internal panels have four or eight generations per task. Their primary
  uncertainty unit is therefore the task, using paired task-level bootstrap
  intervals and sign-randomization tests. Episode-level tests are not used as
  the primary inference.
- Family tests are exploratory. Holm correction is applied within each family
  of five ToolPriv or MetaTool subtests.
- The historical frozen-500 base used generation seed 0, whereas the new
  parent/candidate run used seed 1. Base comparisons therefore use paired task
  clusters, not claimed common-random-number episode pairing.
- No base-policy cell exists for the new lockbox50. Its valid comparison is the
  untouched parent checkpoint.

## Headline results

| Panel | Base | Parent | Continuation-400 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| lockbox50 safe success | not run | 271/400 (67.75%) | **362/400 (90.50%)** | +22.75 pp vs parent; task-bootstrap 95% CI +12.0 to +34.5 pp; one-sided cluster-randomization p < 0.0002 |
| frozen-500 safe success | 1,864/2,896 (64.36%) | 2,863/2,896 (98.86%) | **2,860/2,896 (98.76%)** | -0.10 pp vs parent, not significant; +34.40 pp vs base |
| frozen-500 over-privileged success | 132/2,896 (4.56%) | 13/2,896 (0.45%) | **16/2,896 (0.55%)** | +3 vs parent, not significant; -116 vs base |
| MetaTool accuracy | 819/1,000 (81.9%) | 837/1,000 (83.7%) | **835/1,000 (83.5%)** | -2 vs parent, not significant; +16 vs base, one-sided exact p=0.00350 |
| ToolPriv over-privileged selection | 246/544 (45.22%) | 244/544 (44.85%) | **207/544 (38.05%)** | -37 vs parent, p=3.70e-5; -39 vs base, p=3.23e-5 |

The lockbox `VALIDATION_GUARD.json` is not an authoritative safety summary:
the generic reporter looked up these new task IDs in an unrelated risk archive
and defaulted required risk to zero. The figures above were recomputed from the
preserved raw rollouts using each lockbox task's inline `required_risk`. No GPU
rerun is required, but the reporter must be fixed before paper automation uses
this panel.

## Parent-retention tests

The apparent frozen-500 loss is not statistically distinguishable from zero.
At equal task weight, safe success changed by -0.15 pp (bootstrap 95% CI -0.80
to +0.45 pp; one-sided regression p=0.351), and over-privileged success changed
by +0.05 pp (CI -0.40 to +0.525 pp; regression p=0.501). Exploratory family
declines were also nonsignificant: `scoped_cleanup` p=0.254,
`run_the_right_test` p=0.250, `prompt_injection` p=0.470, and
`escalation_required` p=0.498 in the regression direction.

On lockbox50, the `escalation_required` change (65/80 to 62/80) is not a
significant regression (task-cluster p=0.396; 95% CI -17.5 to +13.75 pp).
The major gains are real: `convenience_trap` +87.5 pp (p=0.00425),
`bounded_retry` +35.42 pp (p=0.0324), and `transient_retry` +13.82 pp
(p=0.0280). These small-family p-values are exploratory.

MetaTool overall did not regress (one-sided parent-to-candidate regression
p=0.416), but its `reliability` subset did: 174/200 to 166/200, with all eight
discordant items moving from correct to incorrect (one-sided exact p=0.003906;
Holm-adjusted across five MetaTool subsets p=0.0195). This is a real narrow
regression even though continuation-400 remains above base overall.

FORTIS changes versus the parent are small. Task 1 exact match is 234 to 238
(p=0.212 for improvement) and over-privilege is 299 to 293 (p=0.105). Task 2
exact match is 317 to 321 (p=0.326), over-privilege is 572 to 566 (p=0.259),
and safe classification is 820 to 810 (regression p=0.148). Parse count fell
1,392 to 1,376 with nominal p=0.0164, but parsing is secondary and this does
not survive correction across the FORTIS endpoints.

## Comparison with base

The continued endpoint remains substantially stronger than base internally.
On frozen-500, task-equal safe success is +32.8 pp (bootstrap 95% CI +30.125
to +35.575 pp) and task-equal over-privileged success is -2.625 pp (CI -3.75
to -1.625 pp); both one-sided task-randomization p < 0.00005. Aggregate rates
differ slightly because safety-heavy tasks receive eight generations and
ordinary tasks four.

On FORTIS Task 1, continuation-400 improves exact match 225 to 238
(p=0.00732) and reduces over-privilege 307 to 293 (p=0.00217). On Task 2 it
improves exact match 253 to 321 (p=3.62e-13) and safe classification 711 to 810
(p=1.23e-14), but over-privilege is 546 to 566. That +20 adverse difference is
borderline but does not cross alpha 0.05 (one-sided regression p=0.0578).

On ToolPriv, the base-to-candidate reduction is paired 66 improvements versus
27 regressions (one-sided p=3.23e-5). By type, counts are: Authority Escalation
82 to 75, Data Over-Exposure 46 to 28, Safety Bypass 61 to 59, Scope Expansion
25 to 17, and Temporal Persistence 32 to 28. After Holm correction across the
five types, Data Over-Exposure and Scope Expansion are individually
significant; the benchmark-wide primary effect is significant. The ToolPriv
success proxy is 536 for base and 533 for the candidate, a nonsignificant
difference (regression p=0.315).

## Verdict

Continuation-400 produces a large, statistically supported correction on the
new cue-neutral lockbox and ToolPriv while retaining frozen-500 performance.
It is not literally regression-free: MetaTool reliability is a significant
narrow regression and FORTIS Task 2 still shows more over-privilege than base.
The defensible claim is targeted corrective generalization with broad internal
retention, not uniform dominance on every secondary slice.
