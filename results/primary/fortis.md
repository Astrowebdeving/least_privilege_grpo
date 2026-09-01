# FORTIS full comparison: independent audit

Date: 2026-07-16  
Artifact: `logs/artifacts/fortis_full_2026-07-16`  
Official FORTIS revision: `68945b89e5a809153a232f96d6c7260569202ffe`

## Integrity and protocol

The run contains the complete official Task 1 (600 skill-selection queries) and
Task 2 (1,543 skill-grounded tool-selection queries) for base and checkpoint-1500
seeds 0, 1, and 2. Each arm has identical query membership. All outputs completed
without transport retries; Task 1 parsed 600/600 for every arm. Source-file hashes,
the exact per-domain manifest, model names, decoding limits, and the official commit
are preserved in every arm's `summary.json`. Inference used temperature zero, disabled
thinking, 128 Task-1 output tokens, and 1,024 Task-2 output tokens. The official
prompts and deterministic classification rules were unchanged; only the official
Batch-API transport was replaced by the local OpenAI-compatible vLLM endpoint.

At least ten raw outputs from each task type were reviewed, including paired
base-over to seed-1-safe, base-safe to seed-1-over, base-no-action to seed-1-exact,
and seed-1 no-action cases. The classifications match the official evaluator.

## Complete results

| Task | Policy | Exact | Under-privileged | Over-privileged | No action | Safe rate |
|---|---|---:|---:|---:|---:|---:|
| Task 1 | Base | 225/600 (37.5%) | 68 | 307 (51.2%) | 0 | 48.8% |
| Task 1 | Seed 0 | 238/600 (39.7%) | 66 | 296 (49.3%) | 0 | 50.7% |
| Task 1 | **Seed 1** | **237/600 (39.5%)** | 66 | **297 (49.5%)** | 0 | **50.5%** |
| Task 1 | Seed 2 | 232/600 (38.7%) | 65 | 303 (50.5%) | 0 | 49.5% |
| Task 2 | Base | 253/1,543 (16.4%) | 458 | 546 (35.4%) | 286 | 46.1% |
| Task 2 | Seed 0 | 350/1,543 (22.7%) | 506 | 649 (42.1%) | 38 | 55.5% |
| Task 2 | **Seed 1** | **308/1,543 (20.0%)** | **507** | **571 (37.0%)** | **157** | **52.8%** |
| Task 2 | Seed 2 | 347/1,543 (22.5%) | 470 | 620 (40.2%) | 106 | 52.9% |

FORTIS defines `under_privilege` as safe but potentially incomplete. Therefore safe
rate is a safety/abstention statistic, not exact task completion, and must always be
reported beside exact match and no action.

## Seed-1 paired result versus base

### Task 1: direct least-privilege skill choice

Seed 1 gains 12 exact matches and removes 10 over-privileged choices: exact match
increases 2.00 percentage points (post-run paired bootstrap 95% interval +0.33 to
+3.67), while over-privilege falls 1.67 points (interval -3.17 to -0.17). Sixteen
base-over cases become non-over under seed 1, while six base-non-over cases become
over. The full transition matrix is:

| Base -> seed 1 | Count |
|---|---:|
| Exact -> exact / under / over | 218 / 1 / 6 |
| Under -> exact / under / over | 6 / 62 / 0 |
| Over -> exact / under / over | 13 / 3 / 291 |

The improvement is not confined to one category: over-privilege changes by 0.0
points on action-hint traps, -1.90 on keyword traps, -2.50 on privilege ladders,
and -2.00 on scope traps. Exact match improves most on privilege ladders (+4.17)
and keyword traps (+2.86). By domain, email is strongest (exact +4.0, over -3.0),
with smaller over-privilege reductions in ecommerce and filesystem (-1.0 each).

Raw examples are semantically genuine. Seed 1 corrects `product-information` to
the minimally sufficient `shop-read` and `bulk-order` to `shop-bulk-order` on
several privilege-ladder queries. Its real regressions include choosing
`wishlist-add`/`wishlist-view` instead of `shop-wishlist` and `email-read` instead
of `email-observe`.

### Task 2: tool use inside an assigned skill

Seed 1 increases exact match by 55 (+3.56 points; paired interval +2.40 to +4.73),
increases under-privileged outputs by 49, and reduces no action by 129 (-8.36
points; paired interval -9.85 to -6.93). It also creates 25 net additional
over-privileged outputs (+1.62 points; paired interval +0.19 to +3.05).

That over-privilege increase needs the paired transition matrix for correct
interpretation:

| Base -> seed 1 | Exact | Under | Over | No action |
|---|---:|---:|---:|---:|
| Exact | 237 | 5 | 9 | 2 |
| Under | 4 | 449 | 5 | 0 |
| Over | 22 | 24 | 495 | 5 |
| No action | 45 | 29 | 62 | 150 |

Only 14 regressions are base-safe to seed-1-over (nine exact and five under).
Sixty-two new over-privileged outputs replace base no-action failures. Conversely,
46 base-over outputs become exact or under-privileged. Seed 1 is substantially more
action-capable, and that exposes both additional correct behavior and additional
unsafe behavior. Among non-no-action outputs, the over-privileged share falls from
546/1,257 (43.4%) for base to 571/1,386 (41.2%) for seed 1. This conditional rate is
a useful post-hoc diagnostic, not a replacement for the official unconditional OPR.

The tradeoff is localized:

- `min_correct`: exact +1.38 points, over -2.60, no action -4.28, safe +6.88;
- `over_privilege_trap`: exact +5.09, over +0.73, no action -9.09, safe +8.36;
- `feature_gap_trap`: nearly flat (exact +1.33, over +0.67);
- `convenience_trap`: exact +6.47 and no action -15.52, but over +8.41.

The domain split identifies the principal weakness. Email improves strongly (exact
+7.63, over -1.15, safe +11.26) and ecommerce reduces over-privilege by 1.60 points.
Filesystem improves exact by 2.88 and safe by 5.96 but increases over-privilege by
7.50 points while reducing no action by 13.46. Convenience-sensitive filesystem
selection is therefore the main FORTIS regression target.

Raw examples again support the labels. Seed 1 converts base `[]` into exact
`forward_message` and `reply_to_message` calls, and converts several broad searches
to task-relevant tool subsets. Genuine regressions include `bulk_set_label` for one
message, `cross_folder_count` for one folder, `audit_export` in place of
`export_folder`, and broader convenience-search tools.

## Cross-seed judgment

Seed 0 is marginally best on Task-1 exact match and Task-1 OPR, but its Task-2 OPR
rises 6.7 points over base. Seed 2 similarly raises Task-2 OPR by 4.8 points and has
54 length-limited Task-2 outputs. Seed 1 raises Task-2 OPR by only 1.6 points while
retaining the Task-1 improvement and a meaningful Task-2 exact/no-action gain. It is
therefore the best balanced trained seed on this benchmark, consistent with its
internal selection, even though it is not uniformly safer than base on every FORTIS
slice.

## Scientific conclusion

FORTIS supplies a real positive external result and a real limitation. The strongest
claim is: the selected RL policy modestly improves direct minimum-sufficient skill
selection, improves exact and non-abstaining tool selection, and lowers conditional
over-privilege among actionful Task-2 outputs. The policy does not achieve a blanket
reduction in unconditional Task-2 over-privilege; its increased actionfulness creates
a small significant OPR increase concentrated in convenience-sensitive filesystem
cases. This should be reported as an explicit utility-safety tradeoff rather than
hidden by safe rate or exact match alone.

These are post-run descriptive intervals and subgroup analyses. They do not turn the
external benchmark into a training-selection set and should not be used to tune the
primary seed or rewrite the frozen v2.0 benchmark.


## Claude independent audit (2026-07-16)

All headline numbers were recomputed from the raw per-scenario JSONs and match
exactly: both task tables (all 4 policies), the T1 paired 16/6, the T2 paired
matrix (newly-over 76 = 62 ex-no-action + 14 ex-safe; 46 over-to-safe), the
conditional OPR 43.4% -> 41.2%, and every per-category/per-domain delta bullet.
Two audit additions that should shape publication wording:

1. **Task-2 `no_action` is 100% unparseable output — zero explicit refusals in
   any arm** (base 286/286 unparsed, seed-0 38/38, seed-1 157/157, seed-2
   106/106; parse rates 81.5% / 97.5% / 89.8% / 93.1%). The official scorer
   classifies unparsed as no_action, which is fine to follow, but the paper
   must say "produced no parseable action", not "declined to act": the
   no-action reduction is a format-validity/actionfulness gain, not a change
   in deliberate abstention.
2. **T1 significance is borderline, but direction is cross-seed consistent.**
   Sign tests: exact p=0.029, OPR p=0.052 (16 vs 6 discordant). The stronger
   evidence is that all three seeds move the same way on both T1 metrics
   (exact 238/237/232 vs 225; over 296/297/303 vs 307). Report as small-but-
   consistent, not decisive per-seed.
