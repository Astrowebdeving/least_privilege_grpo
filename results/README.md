# Result artifact index

These are compact records copied from completed campaigns. They are included so
headline tables can be traced to preserved configurations, completion guards,
and analyzer output without publishing raw third-party corpora or generations.

- `primary/`: frozen internal evaluation, checkpoint trajectory, terminal-stop
  diagnostic, MetaTool, FORTIS, and prompt/norm-ablation records.
- `continuation/`: continuation-400 campaign, frozen/lockbox/external analysis,
  prompt interaction, and completion guards.
- `causal/`: reward-arm campaign configurations, completion guards, generated
  three-seed base-start analysis, and five-seed production-policy analyses.

The narrative comparison contract and the claims these records support are in
[`docs/RESULTS.md`](../docs/RESULTS.md). Analyzer source is under `scripts/`.
