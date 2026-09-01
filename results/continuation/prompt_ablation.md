# Continuation-400 ToolPriv prompt ablation

Completed 2026-08-04. All 544 official ToolPriv scenarios ran with temperature
0, tool-order seed 0, the pinned benchmark revision, the continuation-400
adapter, and ToolPriv's official privilege-aware system prompt. The completion
guard passes and vLLM cleanup completed.

## 2 x 2 extension

| Policy | Official default prompt | Official privilege-aware prompt | Prompt effect |
| --- | ---: | ---: | ---: |
| base | 246/544 over-privileged (45.22%) | 224/544 (41.18%) | -4.04 pp |
| original seed-1 parent | 244/544 (44.85%) | 206/544 (37.87%) | -6.99 pp (the preserved default cell differs by one case from the earlier 245-count run) |
| **continuation-400** | **207/544 (38.05%)** | **145/544 (26.65%)** | **-11.40 pp** |

For continuation-400, prompting changes 78 scenarios from over-privileged to
safe and 16 in the opposite direction. The exact one-sided McNemar p-value is
**2.89e-11** (z=6.395). Its success proxy is effectively unchanged, 533 to 532
(p=0.500 in the regression direction).

Under the same privilege-aware prompt, continuation-400 beats:

- base: 224 to 145 over-privileged, -14.52 pp; 100 improvements versus 21
  regressions; exact one-sided p=8.03e-14;
- the original seed-1 parent: 206 to 145, -11.21 pp; 82 improvements versus 21
  regressions; p=5.35e-10.

The scenario-paired interaction is also nonzero. The candidate prompt effect
minus the base prompt effect is -7.35 pp, with paired bootstrap 95% CI -11.95
to -2.76 pp. Thus corrective training and explicit prompting are additive and
the prompt is more effective after corrective training; they are not redundant.

## Type breakdown under the privilege-aware prompt

| ToolPriv type | Base+prompt | Parent+prompt | Continuation+prompt |
| --- | ---: | ---: | ---: |
| Authority Escalation | 76/139 | 71/139 | **53/139** |
| Data Over-Exposure | 47/99 | 37/99 | **20/99** |
| Safety Bypass | 54/116 | 56/116 | **44/116** |
| Scope Expansion | 18/99 | 15/99 | **10/99** |
| Temporal Persistence | 29/91 | 27/91 | **18/91** |

Against base+prompt, the exact one-sided p-values are 5.84e-5, 7.45e-9,
0.0494, 0.0193, and 0.00636 respectively. All five remain at or below 0.05
after Holm correction within this five-type exploratory family (Safety Bypass
is the boundary result at adjusted p approximately 0.0494).

## Publication interpretation

The default-prompt result already shows that the learned correction transfers
without an explicit least-privilege reminder. The prompt cell adds a separate
deployment result: explicit policy guidance and learned privilege-aware
behavior compound, reaching 26.65% ToolPriv over-privilege without sacrificing
the benchmark's success proxy. This supports presenting training as useful in
both prompt-light and explicitly governed deployments.
