# Publication prompt/norm ablations

Primary uncertainty uses the task as the paired cluster. McNemar is a secondary
common-random-number diagnostic because eight generations from one task are dependent.

## Internal 206-task norm ablation

| Prompt | Policy | Safe success | Success | Over-privileged success | Mean excess |
|---|---|---:|---:|---:|---:|
| P_full | base | 1016/1648 (61.7%) | 1080/1648 (65.5%) | 64/1648 (3.9%) | 0.0249 |
| P_full | seed-1 | 1609/1648 (97.6%) | 1621/1648 (98.4%) | 12/1648 (0.7%) | 0.0016 |
| P_none | base | 1004/1648 (60.9%) | 1071/1648 (65.0%) | 67/1648 (4.1%) | 0.0197 |
| P_none | seed-1 | 1609/1648 (97.6%) | 1625/1648 (98.6%) | 16/1648 (1.0%) | 0.0027 |
| P_min | base | 955/1648 (57.9%) | 1029/1648 (62.4%) | 74/1648 (4.5%) | 0.0260 |
| P_min | seed-1 | 1605/1648 (97.4%) | 1629/1648 (98.8%) | 24/1648 (1.5%) | 0.0032 |

## ToolPrivBench 2x2

- base_default: over-privileged selection 45.2%
- base_privilege_aware: over-privileged selection 41.2%
- seed1_default: over-privileged selection 45.0%
- seed1_privilege_aware: over-privileged selection 37.9%
- Prompt effect on base: -4.0%
- Prompt effect on seed-1: -7.2%
- Training effect under default prompt: -0.2%
- Training effect under privilege-aware prompt: -3.3%
- Interaction: -3.1% (paired bootstrap 95% -7.7% to +0.7%)

Official-prompt FORTIS/MetaTool results remain primary. Their P_min cells are
secondary prompt-interaction controls and are preserved in ANALYSIS.json.
