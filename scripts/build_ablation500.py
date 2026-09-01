#!/usr/bin/env python3
"""Build the 500-task ablation catalog plus a disjoint held-out validation set.

Selection principle: Phase 0 measured that only ~25% of task groups carry any
gradient at all under dr_grpo, and that this is near-identical across every
reward design (3,688-4,010 of 16,054). No reward design fixes saturation, so a
randomly drawn catalog would spend ~75% of its compute on groups that cannot
move the weights. This builder selects on MEASURED per-task reward variance
under the real reward (A0), computed from preserved rollouts, so the catalog is
gradient-bearing by construction.

Comparability note: the user's reference point is seed-1 checkpoint-500. That
checkpoint was trained on the full 2,000-task catalog, not on this 500. It is
therefore a useful secondary reference but NOT a matched control -- the study
must still train its own A0 arm on this catalog for the clean comparison. This
is recorded in the manifest rather than left implicit.

Usage:
  python scripts/build_ablation500.py            # build
  python scripts/build_ablation500.py --check    # verify reproducibility
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import importlib
P0 = importlib.import_module("scripts.ablation_phase0_rescore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = "2026-08-04-ablation500-v1"
POOL = "data/expansion_v2_microslice.json"
OUT_CATALOG = "data/ablation500_v1.json"
OUT_HELDOUT = "data/ablation500_heldout_v1.json"
OUT_MANIFEST = "data/ablation500_manifest.json"
N_TRAIN, N_HELDOUT = 500, 50
MAD_FLOOR = 0.05          # matches the campaign's selector
MIN_EPISODES = 4          # minimum observations before trusting a variance estimate

# Sets the ablation must never touch.
EXCLUDE_SOURCES = [
    ("frozen500", "data/validation_frozen_2000_2026-06-30.json", "validation_task_ids"),
    ("routine206", "data/validation_frozen_2000_2026-06-30.json", "routine_subset_ids"),
]
EXCLUDE_FILES = ["data/diagnostic_50_v2.txt"]
EXCLUDE_CATALOGS = [
    "data/continuation_confirmatory_v1.json",
    "data/continuation_confirmatory_v2.json",
    "data/continuation_transient_heldout_v3.json",
    "data/continuation_transient_heldout_v4.json",
]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded_ids() -> set[str]:
    out: set[str] = set()
    for _, rel, key in EXCLUDE_SOURCES:
        p = ROOT / rel
        if p.exists():
            out |= set(json.loads(p.read_text(encoding="utf-8")).get(key) or [])
    for rel in EXCLUDE_FILES:
        p = ROOT / rel
        if p.exists():
            out |= {x.strip() for x in p.read_text(encoding="utf-8").replace("\n", ",").split(",") if x.strip()}
    for rel in EXCLUDE_CATALOGS:
        p = ROOT / rel
        if p.exists():
            out |= {r["task_id"] for r in json.loads(p.read_text(encoding="utf-8"))}
    return out


#: Arms whose measured variance is averaged to give an ARM-NEUTRAL selection
#: criterion. Selecting on A0's own MAD enriches the control's preferred signal
#: and biases the study toward finding the control effective.
NEUTRAL_ARMS = ("A0_full", "A2_binary_tag", "A4_no_envelope", "B1_no_privilege")


ROLLOUT_INPUTS = "data/ablation500_rollout_inputs.json"


def rollout_files(roots: list[str]) -> list[pathlib.Path]:
    """The rollout files feeding the variance measurement.

    FROZEN, not globbed live. `logs/artifacts/**` grows every time anything runs
    -- including this study's own smoke tests -- so a live glob makes the catalog
    silently unreproducible. The exact input set is recorded at build time and
    replayed on --check.
    """
    manifest = ROOT / ROLLOUT_INPUTS
    if manifest.exists():
        rec = json.loads(manifest.read_text(encoding="utf-8"))
        files = [ROOT / p for p in rec["files"]]
        missing = [p for p in files if not p.exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} pinned rollout inputs are missing (e.g. {missing[0]}). "
                f"Refusing to silently re-measure on a different corpus.")
        return files
    # Exclude this study's OWN outputs: selecting tasks using rollouts produced
    # by the ablation itself would be circular, and the smoke test alone was
    # enough to change the catalog and trip the launcher's reproducibility gate.
    skip = ("ablation_smoke", "ablation_campaign", "ablation_phase0")
    found = []
    for r in roots:
        for p in sorted((ROOT / r).rglob("rollouts-*.jsonl")):
            if not any(s in str(p) for s in skip):
                found.append(p)
    return sorted(found)


def freeze_rollout_inputs(files: list[pathlib.Path]) -> dict:
    rec = {
        "description": "frozen rollout input set for the ablation catalog; the "
                       "builder must not re-glob logs/artifacts, which grows",
        "count": len(files),
        "files": sorted(str(p.relative_to(ROOT)) for p in files),
        "total_bytes": sum(p.stat().st_size for p in files),
    }
    (ROOT / ROLLOUT_INPUTS).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def measure_variance(roots: list[str]) -> dict[str, dict]:
    """Per-task reward variance averaged across the primary arms (arm-neutral)."""
    tasks = P0.build_task_index(
        ["data/continuation*.json", "data/expansion_v2_microslice.json", "data/spike_tasks.json"])
    files = rollout_files(roots)
    eps = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except Exception:
                    continue
                ep["_file"] = str(f)
                eps.append(ep)
    print(f"  rollout inputs: {len(files)} files, {len(eps)} episodes"
          f"{' (pinned)' if (ROOT / ROLLOUT_INPUTS).exists() else ' (UNPINNED - will freeze)'}")
    globals()["_LAST_FILES"] = files
    fg = P0.fidelity_gate(eps, tasks)
    resolved, excluded = fg["resolved"], fg["excluded"]
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, ep in enumerate(eps):
        spec = resolved.get(i)
        if spec is None or i in excluded:
            continue
        key = (ep["_file"], ep["task_id"])
        for arm in NEUTRAL_ARMS:
            groups[key][arm].append(P0.score_episode(arm, ep, spec))

    per_task_mad: dict[str, list[float]] = defaultdict(list)
    per_task_std: dict[str, list[float]] = defaultdict(list)
    per_task_any: dict[str, list[bool]] = defaultdict(list)
    for (_f, tid), by_arm in groups.items():
        if len(by_arm[NEUTRAL_ARMS[0]]) < MIN_EPISODES:
            continue
        mads, stds, any_live = [], [], False
        for arm in NEUTRAL_ARMS:
            vals = by_arm[arm]
            m = statistics.fmean(vals)
            mads.append(statistics.fmean([abs(v - m) for v in vals]))
            sd = statistics.pstdev(vals)
            stds.append(sd)
            if sd > 0:
                any_live = True
        per_task_mad[tid].append(statistics.fmean(mads))
        per_task_std[tid].append(statistics.fmean(stds))
        per_task_any[tid].append(any_live)
    return {tid: {"mad": statistics.fmean(v),
                  "std": statistics.fmean(per_task_std[tid]),
                  "any_arm_live": any(per_task_any[tid]),
                  "groups": len(v)} for tid, v in per_task_mad.items()}


class ShortPool(Exception):
    """Not enough gradient-bearing tasks; padding must be explicitly requested."""

    def __init__(self, have: int, need: int):
        super().__init__(f"only {have} tasks clear the floor, need {need}")
        self.have, self.need = have, need


def build(measured: dict[str, dict], allow_pad: bool = False) -> tuple[list, list, dict]:
    pool = json.loads((ROOT / POOL).read_text(encoding="utf-8"))
    bad = excluded_ids()
    train_pool = [t for t in pool
                  if t.get("split") == "train" and t["task_id"] not in bad]

    mad_of = lambda t: measured.get(t["task_id"], {}).get("mad", 0.0)
    eligible = [t for t in train_pool if mad_of(t) >= MAD_FLOOR]
    # Deterministic order: by measured MAD desc, then task_id for ties.
    eligible.sort(key=lambda t: (-mad_of(t), t["task_id"]))

    # Round numbers matter for reporting, so pad to the target with the
    # highest-variance sub-floor tasks. These are recorded as
    # `gradient_bearing: false` in the manifest so the paper can state exactly
    # how many of the 500 carry signal instead of implying all of them do.
    need = N_TRAIN + N_HELDOUT
    padding: list = []
    if len(eligible) < need:
        if not allow_pad:
            raise ShortPool(len(eligible), need)
        rest = [t for t in train_pool if mad_of(t) < MAD_FLOOR]
        rest.sort(key=lambda t: (-mad_of(t), t["task_id"]))
        padding = rest[: need - len(eligible)]

    selected = eligible + padding
    by_family: dict[str, list] = defaultdict(list)
    for t in selected:
        by_family[t["family"]].append(t)

    # Round-robin across families so no family is crowded out by a large one.
    picked: list = []
    fams = sorted(by_family, key=lambda f: (-len(by_family[f]), f))
    cursor = {f: 0 for f in fams}
    while len(picked) < need and any(cursor[f] < len(by_family[f]) for f in fams):
        for f in fams:
            if len(picked) >= need:
                break
            if cursor[f] < len(by_family[f]):
                picked.append(by_family[f][cursor[f]])
                cursor[f] += 1

    # Split by TEMPLATE GROUP, not task_id. Task-id-disjoint is not enough: tasks
    # sharing a template_group_id are near-duplicates, so a held-out task whose
    # template was trained on measures memorisation, not generalisation. The
    # previous build leaked 26 template groups across the split.
    tg_of = lambda t: t.get("template_group_id") or f"__solo__{t['task_id']}"
    by_tg: dict[str, list] = defaultdict(list)
    for t in picked:
        by_tg[tg_of(t)].append(t)
    # Stride over template groups (not tasks) so held-out spans the same families
    # and variance range while keeping whole groups on one side.
    tg_order = sorted(by_tg, key=lambda g: (-len(by_tg[g]), g))
    held, train = [], []
    stride = max(1, len(tg_order) // max(1, N_HELDOUT))
    for i, g in enumerate(tg_order):
        rows = by_tg[g]
        if i % stride == 0 and len(held) + len(rows) <= N_HELDOUT:
            held.extend(rows)
        else:
            train.extend(rows)
    train = train[:N_TRAIN]

    gb = {t["task_id"] for t in eligible}
    live = lambda rows: sum(1 for t in rows if measured.get(t["task_id"], {}).get("any_arm_live"))
    stats = {
        "eligible_pool": len(eligible),
        # Reported per split, never pooled: an earlier manifest mixed a
        # 550-task padding count with a 500-task gradient-bearing count.
        "train_padded": sum(1 for t in train if t["task_id"] not in gb),
        "heldout_padded": sum(1 for t in held if t["task_id"] not in gb),
        "train_above_mad_floor": sum(1 for t in train if t["task_id"] in gb),
        "heldout_above_mad_floor": sum(1 for t in held if t["task_id"] in gb),
        # MAD>=floor is a selector threshold, NOT the definition of carrying
        # gradient -- any within-group variation contributes. Report both.
        "train_any_variance": live(train),
        "heldout_any_variance": live(held),
        "template_groups_train": len({t.get("template_group_id") for t in train}),
        "template_groups_heldout": len({t.get("template_group_id") for t in held}),
        "template_group_overlap": len(
            {t.get("template_group_id") for t in train if t.get("template_group_id")}
            & {t.get("template_group_id") for t in held if t.get("template_group_id")}),
        "train_pool": len(train_pool),
        "excluded_ids": len(bad),
        "families_train": {k: v for k, v in sorted(
            __import__("collections").Counter(t["family"] for t in train).items())},
        "families_heldout": {k: v for k, v in sorted(
            __import__("collections").Counter(t["family"] for t in held).items())},
        "mad_train_mean": round(statistics.fmean(
            measured[t["task_id"]]["mad"] for t in train), 4) if train else 0.0,
        "mad_heldout_mean": round(statistics.fmean(
            measured[t["task_id"]]["mad"] for t in held), 4) if held else 0.0,
    }
    return train, held, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["logs/artifacts"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pad-to-target", action="store_true",
                    help="opt in to padding with sub-floor tasks when the pool "
                         "lacks enough above-floor ones (recorded in the manifest)")
    args = ap.parse_args()

    print("measuring per-task variance under the real reward ...")
    measured = measure_variance(args.roots)
    print(f"  tasks with a usable variance estimate: {len(measured)}")

    try:
        train, held, stats = build(measured, allow_pad=args.pad_to_target)
    except ShortPool as e:
        print(f"REFUSING: {e}.\n"
              f"  Padding would add tasks below the MAD floor, which cost generation "
              f"time and contribute little or no update. Re-run with --pad-to-target "
              f"to opt in explicitly (the manifest will record how many were padded), "
              f"or with a lower --mad-floor.", file=sys.stderr)
        return 2
    print(f"  train={len(train)} heldout={len(held)}")
    for k, v in stats.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    if stats["template_group_overlap"] != 0:
        print(f"REFUSING: {stats['template_group_overlap']} template groups span "
              f"train and held-out; the held-out set would measure memorisation.",
              file=sys.stderr)
        return 2
    if len(held) < N_HELDOUT or len(train) < N_TRAIN:
        print(f"REFUSING: built {len(train)}/{N_TRAIN} train and {len(held)}/{N_HELDOUT} "
              f"held-out.", file=sys.stderr)
        return 2

    ids_t = {t["task_id"] for t in train}
    ids_h = {t["task_id"] for t in held}
    assert not (ids_t & ids_h), "train/heldout overlap"
    assert not (ids_t & excluded_ids()), "train touches an excluded evaluation set"
    assert not (ids_h & excluded_ids()), "heldout touches an excluded evaluation set"

    blob_t = json.dumps(train, indent=2, sort_keys=True) + "\n"
    blob_h = json.dumps(held, indent=2, sort_keys=True) + "\n"
    if args.check:
        cur_t = (ROOT / OUT_CATALOG).read_text(encoding="utf-8")
        cur_h = (ROOT / OUT_HELDOUT).read_text(encoding="utf-8")
        ok = (cur_t == blob_t) and (cur_h == blob_h)
        print("CHECK:", "reproduces byte-for-byte" if ok else "MISMATCH")
        return 0 if ok else 1

    if not (ROOT / ROLLOUT_INPUTS).exists():
        rec = freeze_rollout_inputs(globals().get("_LAST_FILES") or [])
        print(f"  froze rollout inputs: {rec['count']} files -> {ROLLOUT_INPUTS}")
    (ROOT / OUT_CATALOG).write_text(blob_t, encoding="utf-8")
    (ROOT / OUT_HELDOUT).write_text(blob_h, encoding="utf-8")
    manifest = {
        "seed": SEED,
        "built_at": "2026-08-04",
        "purpose": "causal-origin ablation of the 6-dim risk framework",
        "selection": f"per-task MAD averaged across arms {list(NEUTRAL_ARMS)} "
                     f"(arm-neutral, so the control is not favoured), floor {MAD_FLOOR}; "
                     f"family round-robin; train/heldout split by template_group_id",
        "train_count": len(train), "heldout_count": len(held),
        "catalog_sha256": hashlib.sha256(blob_t.encode()).hexdigest(),
        "heldout_sha256": hashlib.sha256(blob_h.encode()).hexdigest(),
        "pool": POOL, "pool_sha256": sha(ROOT / POOL),
        "comparability_caveat":
            "seed-1 checkpoint-500 is a SECONDARY reference only: it was trained on the "
            "full 2000-task catalog, not this 500, so it is not a matched control. The "
            "study must train its own A0 arm on this catalog for the primary comparison.",
        "stats": stats,
    }
    (ROOT / OUT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_CATALOG} sha={manifest['catalog_sha256'][:16]}")
    print(f"wrote {OUT_HELDOUT} sha={manifest['heldout_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
