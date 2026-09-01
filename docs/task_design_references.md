# Task Design References

This task harness borrows only structural ideas from external benchmarks. The
actual spike tasks are local fixtures written for least-privilege measurement.
The current catalog has 260 tasks: 200 train and 60 validation. The 198-task
expansion is template-authored and deterministically instantiated, so it is
proposal-scale in count and verification but not equivalent to 260 independent
human-authored environments.

## Patterns To Reuse

- FORTIS frames the skill/tool layer as a privilege boundary and evaluates both
  minimally sufficient capability selection and whether execution expands
  beyond that capability. Use this to keep `required_risk` task-conditioned.
- SkillScope emphasizes task-conditioned least privilege. The same action can
  be acceptable in one task and over-privileged in another, so labels belong on
  task instances, not global command names alone.
- Terminal-Bench tasks are realistic terminal jobs with unique environments,
  human-written solutions, comprehensive tests, categories/tags, and difficulty
  labels. Use that pattern for `family`, `difficulty`, fixture workspace, hidden
  verifier, and oracle trajectory.
- ToolBench/ToolLLM separates instructions, tool/API environment data, and
  solution-path annotations across single-tool and multi-tool scenarios. Use
  this for fake MCP single-tool, same-domain, and cross-server scenarios.
- MCP-Atlas uses natural prompts that avoid handing the agent the tool name,
  exposes plausible distractors, records a minimal reference trajectory, and
  decomposes final answers into atomic claims. The local MCP expansion borrows
  those patterns while keeping deterministic verifiers instead of an LLM judge.
- Toolathlon combines realistic initial application state, multi-app workflows,
  and dedicated execution verifiers. The local service fixtures remain fake and
  compact, but stateful writes and cross-service reads follow that structure.
- SWE-bench evaluates repository edits through containerized harnesses and issue
  prompts. Its known evaluation weaknesses motivate hidden verifier isolation,
  no solution leakage, and diff-based review for plausible but wrong patches.

## Task Quality Lessons From The Frozen Evals

The pitfalls the frozen evals exposed — and the checks to run when adding or
auditing tasks — live in `docs/task_design_pitfalls.md` (what tends to go wrong
plus what to verify), with the worked base-vs-checkpoint example summarized in
`docs/past_results_and_methods.md` (Phase 1). Treat them as checks, not a rigid
template.

TODO(lp-rl): add a "policy-realism" report to the rollout analyzer that flags any
task where base-model success is 0/8 while the oracle passes — likely a task/spec
bug rather than a hard task. Oracle smoke cannot catch this by construction.

## Sources Checked

- FORTIS: https://arxiv.org/abs/2605.09163
- SkillScope: https://arxiv.org/abs/2605.05868
- Terminal-Bench site and task examples: https://www.tbench.ai/
- Terminal-Bench repository: https://github.com/harbor-framework/terminal-bench
- ToolBench repository: https://github.com/OpenBMB/ToolBench
- ToolLLM/ToolBench paper: https://arxiv.org/abs/2307.16789
- MCP-Atlas: https://arxiv.org/abs/2602.00933
- Toolathlon / Tool Decathlon: https://arxiv.org/abs/2510.25726
- SWE-bench repository: https://github.com/SWE-bench/SWE-bench

The fake MCP layer now covers 31 catalog tasks. Exact tool metadata is visible
without risk scores, so the expanded benchmark no longer measures hidden tool
name guessing.

The controlled expansion landed read-vs-write, local-vs-external,
solvable-vs-escalation, safe-vs-lure, targeted-vs-broad, and scoped-cleanup
contrasts. Future additions should follow `docs/task_authoring_standard.md`
and replace saturated templates with independently authored variants.

The first generation pass is `scripts/build_expanded_tasks.py`; its result and
limitations are in `docs/past_results_and_methods.md` (Phase 3). The next ingestion step
should replace saturated template variants with independently authored or
recording-derived tasks, not merely increase the count.
