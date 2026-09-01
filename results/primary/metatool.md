# MetaTool registered-subset comparison: corrected independent audit

Date: 2026-07-16  
Artifact: `logs/artifacts/metatool_subset_full_2026-07-16`  
Official MetaTool revision: `35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb`

## Integrity and scoring correction

The run contains the preregistered deterministic subset of 1,000 official prepared
prompts: 200 each from tool-use awareness, similar-tool selection, scenario-specific
selection, reliability/hallucination, and multi-tool selection. Base and all three
checkpoint-1500 seeds use identical sample manifests, temperature zero, disabled
thinking, and 128 output tokens. All arms contain exactly 1,000 raw responses and the
official revision, source hashes, and sample indices are preserved in each summary.
The suite completed normally and released the project vLLM process.

The first report showed 0/200 reliability accuracy for every arm. Raw/source review
proved this was an adapter scoring defect, not a model result. MetaTool constructs the
reliability subset by excluding the source query's original tool from the ten offered
candidates (`reliability_tool_selection`; its generated file is called a hallucination
prompt). The correct answer is consequently `None`. The source row retains its
original `tool` field only as provenance. The v1 adapter incorrectly treated that
excluded name as the expected output, making every answer impossible to score correct.

The evaluator now assigns the reliability target `None`, records scoring version
`2026-07-16-reliability-none-v2`, and re-scores cached raw responses against current
official source rows. A focused regression test asserts this invariant; nine MetaTool
and external-adapter tests pass. The same 4,000 raw model responses were re-scored;
there was no GPU or inference rerun. In the selected reliability sample, the source
tool is absent from the candidate list in all 200/200 rows.

## Corrected results

| Subtask | Base | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| Awareness | 88.5% | 88.0% | **89.0%** | 84.0% |
| Multi-tool | 81.5% | 78.5% | **83.0%** | 80.0% |
| Reliability / unavailable-tool restraint | 81.0% | 86.0% | **87.0%** | 92.5% |
| Scenario | 84.5% | 84.0% | **84.5%** | 82.0% |
| Similar-tool | 74.0% | 74.5% | **74.5%** | 68.0% |
| **Overall** | **81.9%** | **82.2%** | **83.6%** | **81.3%** |

Against base, seed 1 gains 17 correct examples (+1.7 percentage points), with 21
base-wrong/seed-1-right and four base-right/seed-1-wrong pairs. The post-run paired
bootstrap interval is +0.7 to +2.5 points. Seed 0 is essentially flat (+0.3, interval
-0.9 to +1.3); seed 2 is also statistically unresolved overall (-0.6, interval -2.2
to +0.7) despite its high reliability restraint.

Seed-1's paired net gains by subtask are:

| Subtask | Base wrong -> seed 1 right | Base right -> seed 1 wrong | Net |
|---|---:|---:|---:|
| Awareness | 2 | 1 | +1 |
| Multi-tool | 4 | 1 | +3 |
| Reliability | 12 | 0 | +12 |
| Scenario | 0 | 0 | 0 |
| Similar-tool | 3 | 2 | +1 |

The result is not only a formatting change. Manual review of all 25 seed-1/base
correctness disagreements shows semantically meaningful decisions. Seed 1 correctly
returns `None` when no offered tool can answer historical, product, SEO, automation,
or puzzle requests; base often overextends generic `ai_council`, `web_scraper`,
`chatspot`, or similarly named tools. Seed 1 also corrects several multi-tool pairs,
such as `NewsTool` plus `FinanceTool`, and chooses the registered exact tool in several
similar-tool cases. Its four regressions are plausible close alternatives, including
web search instead of `ResearchHelper`, `PDF&URLTool` instead of `universal`, and
`jini` instead of `EarthquakeTool`.

There is one strict parser failure across the 4,000 responses: base emits
`research_helper, research_finder` where the candidate names are `ResearchHelper` and
`ResearchFinder`. The adapter deliberately requires candidate spelling apart from
case; underscore aliases are not silently credited. Every trained-seed response
parses. This single case does not determine the seed-1 gain.

## Interpretation and claim limit

This is a positive external capability-retention result for the selected policy:
seed 1 does not merely preserve aggregate MetaTool subset accuracy, but improves it,
with the largest gain on refusing to invent or stretch an unavailable capability.
That direction is consistent with least-privilege restraint and complements the
FORTIS Task-1 gain. It also helps separate ToolPriv's dynamic error-recovery weakness
from a global tool-selection regression.

The result remains a deterministic registered 1,000-example subset, not a full
MetaTool reproduction or leaderboard comparison. The confidence interval is a
post-run descriptive analysis. It supports capability retention and modest transfer,
not universal general-tool superiority.

## Claude independent audit (2026-07-16)

Recomputed from raw per-scenario JSONs: all subtask accuracies match; the
reliability scorer fix is sound and fair (uniform rule across arms, expected
target is None for all 200 items, stored correctness == independent
recomputation for base 162/200 and seed-1 174/200; sampled base miss genuinely
names an absent tool, sampled seed-1 hit genuinely answers None); paired 21
wins / 4 losses confirmed. Note the strongest cross-seed external signal in
the whole suite is reliability: all three trained seeds improve it
(86.0 / 87.0 / 92.5 vs base 81.0) — unavailable-tool restraint transfers
robustly across seeds.
