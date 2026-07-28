# External benchmarks

Transfer is measured against three third-party benchmarks. **None of their data
is redistributed in this repository.** The scripts here are adapters: you fetch
each benchmark yourself, and the adapter runs our policy against it and writes
results in our summary format.

Fetch them into `external/` (git-ignored):

```bash
mkdir -p external
git clone <toolprivbench-upstream> external/toolprivbench
git clone <fortis-upstream>        external/fortis
git clone <metatool-upstream>      external/metatool
```

Pin a commit for anything you intend to report. These benchmarks change, and an
unpinned clone makes numbers unreproducible.

Each upstream carries its own license. Check it before redistributing anything
derived from their prompts or scenarios — that includes raw per-scenario outputs,
which typically embed the source prompts.

## Adapters

| benchmark | run | summarise |
|---|---|---|
| ToolPrivBench | `scripts/run_toolprivbench_suite.sh` | `scripts/summarize_toolprivbench_suite.py` |
| FORTIS | `scripts/run_fortis_suite.sh` | `scripts/summarize_fortis_suite.py` |
| MetaTool | `scripts/run_metatool_subset.sh` | `scripts/summarize_metatool_subset.py` |

`scripts/evaluate_toolprivbench_local.py` runs against a locally served model;
the `*_api.py` variants target a hosted endpoint.

## Disjointness

Before reporting transfer, confirm the training catalog does not share
distinctive vocabulary with the external benchmarks:

```bash
python scripts/check_external_disjointness.py \
  --catalog data/expansion_v2_microslice.json \
  --vocab   data/external_benchmark_vocab.json \
  --strict
```

This checks distinctive-tool vocabulary overlap and equal-length prompt/error
spans across the full corpora. `data/external_benchmark_vocab.json` holds the
extracted vocabulary only — not the benchmark content.

**`--strict` fails until you have fetched the benchmarks**, reporting
`missing_sources`. That is intended: strict mode compares against the full
upstream corpora, which are not in this repository, and it fails closed rather
than silently reporting a vacuous pass. Without `--strict` the vocabulary check
still runs and is meaningful on its own.

## Interpreting the results

Two cautions that materially changed our own reading of these numbers:

**Parse rate is a confound.** If two policies parse different numbers of items,
comparing rates over all items conflates capability with formatting compliance.
Compare over the subset every arm parsed successfully, and report the parse rates
alongside.

**Over-privilege metrics are conditional.** A metric defined only over episodes
that invoked a risky tool improves when a policy calls risky tools less often,
independent of whether its choices got better. Decompose into
`P(risk called) x P(over-privileged | risk called)` before drawing conclusions.

**Never select a checkpoint on these benchmarks.** They exist to measure transfer
to distributions the training signal never saw. Choosing a checkpoint by its
external score destroys exactly the property being measured. Fix the checkpoint
using internal held-out data first, then evaluate externally once.
