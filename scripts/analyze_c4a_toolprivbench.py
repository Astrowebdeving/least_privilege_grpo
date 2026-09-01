#!/usr/bin/env python3
"""Analysis of the c4a reward-arm campaign on ToolPrivBench.

Two questions, two different tests, because they have wildly different power:

  1. Does continuation training reduce over-privileged tool selection?
     Paired per-item comparison against the parent checkpoint, exact McNemar on
     544 items. Well powered.

  2. Do the reward arms differ from each other?
     The unit of randomisation is the SEED, not the item -- two checkpoints from
     the same arm differ only by seed, so items are not exchangeable across
     arms. Seed-level exact permutation. With 2 seeds/arm this is essentially
     powerless, and the script says so numerically rather than burying it.

ToolPrivBench is REPORT-ONLY. Step 400 is fixed for every arm, so nothing here
selects a checkpoint.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import statistics
from collections import OrderedDict

EXT_DEFAULT = "logs/artifacts/cont400_arm_campaign/toolprivbench"
# Cross-run reference points. Kept for context only -- see the replication-noise
# note below; the in-run parent is the baseline that actually gets used.
REFERENCE = OrderedDict([
    ("parent (earlier run, default prompt)", (244, 544)),
    ("cont400 (earlier run, default prompt)", (207, 544)),
    ("cont400 + privilege-aware prompt", (145, 544)),
])


def load_items(policy_dir: pathlib.Path) -> dict[str, bool]:
    """scenario_id -> over_privileged_selection, for one policy."""
    out: dict[str, bool] = {}
    for p in policy_dir.glob("*.json"):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        sid = d.get("scenario_id") or p.stem
        out[sid] = bool(d.get("over_privileged_selection"))
    return out


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar. b_only = a-true/b-false, c_only = a-false/b-true."""
    keys = sorted(set(a) & set(b))
    b_only = sum(1 for k in keys if a[k] and not b[k])
    c_only = sum(1 for k in keys if not a[k] and b[k])
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    lo = min(b_only, c_only)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return b_only, c_only, min(1.0, 2 * tail)


def perm_test_seeds(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Exact permutation on seed-level rates. Returns (obs_diff, p, n_perms)."""
    obs = abs(statistics.fmean(x) - statistics.fmean(y))
    pool = x + y
    k = len(x)
    diffs = []
    for idx in itertools.combinations(range(len(pool)), k):
        left = [pool[i] for i in idx]
        right = [pool[i] for i in range(len(pool)) if i not in idx]
        diffs.append(abs(statistics.fmean(left) - statistics.fmean(right)))
    p = sum(1 for d in diffs if d >= obs - 1e-12) / len(diffs)
    return obs, p, len(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=EXT_DEFAULT)
    ap.add_argument("--arms", default="A0 A4 A2 B1")
    ap.add_argument("--seeds", default="1 2")
    args = ap.parse_args()

    ext = pathlib.Path(args.ext)
    arms = args.arms.split()
    seeds = args.seeds.split()

    policies: dict[str, dict[str, bool]] = {}
    for name in ["parent"] + [f"{a}-s{s}" for a in arms for s in seeds]:
        d = ext / name
        if (d / "summary.json").exists():
            policies[name] = load_items(d)

    missing = [f"{a}-s{s}" for a in arms for s in seeds
               if f"{a}-s{s}" not in policies]
    print(f"loaded {len(policies)} policies from {ext}")
    if missing:
        print(f"  NOT YET COMPLETE: {', '.join(missing)}")
    if "parent" not in policies:
        print("  parent missing -- cannot anchor the comparison")
        return 1

    n_items = len(policies["parent"])
    par = policies["parent"]
    par_n = sum(par.values())
    print(f"  items per policy: {n_items}")

    print("\n== over-privileged selections (all measured in THIS run)")
    print(f"  {'policy':10s} {'count':>9s} {'rate':>8s} {'vs parent':>10s}")
    print(f"  {'parent':10s} {par_n:>4d}/{n_items:<4d} {par_n/n_items:>8.4f} {'--':>10s}")
    for name, items in policies.items():
        if name == "parent":
            continue
        n = sum(items.values())
        print(f"  {name:10s} {n:>4d}/{n_items:<4d} {n/n_items:>8.4f} "
              f"{100*(n-par_n)/n_items:>+9.1f}p")

    print("\n== cross-run reference (do NOT subtract these from the above)")
    for label, (ov, tot) in REFERENCE.items():
        print(f"  {label:38s} {ov:4d}/{tot} = {ov/tot:.4f}")
    print("  Same adapter re-scored across runs differed by up to 8/544 (1.5 pts),")
    print("  so cross-run deltas under ~1.5 pts are not interpretable.")

    print("\n== Q1: does training reduce over-privilege? (exact McNemar, paired)")
    print(f"  {'policy':10s} {'par-only':>9s} {'pol-only':>9s} {'p':>10s}")
    for name, items in policies.items():
        if name == "parent":
            continue
        b, c, p = mcnemar_exact(par, items)
        star = " *" if p < 0.05 else ""
        print(f"  {name:10s} {b:>9d} {c:>9d} {p:>10.2e}{star}")
    print("  par-only = parent over-privileged where the trained model was not.")

    print("\n== Q2: do the arms differ? (seed-level exact permutation)")
    rates: dict[str, list[float]] = {}
    for a in arms:
        vals = [sum(policies[f"{a}-s{s}"].values()) / n_items
                for s in seeds if f"{a}-s{s}" in policies]
        if vals:
            rates[a] = vals
    for a, vals in rates.items():
        spread = (max(vals) - min(vals)) * n_items if len(vals) > 1 else 0
        print(f"  {a:4s} seeds={[f'{v:.4f}' for v in vals]} "
              f"mean={statistics.fmean(vals):.4f} spread={spread:.0f} items")
    print()
    done = [a for a in arms if len(rates.get(a, [])) == len(seeds)]
    for a, b_ in itertools.combinations(done, 2):
        obs, p, nperm = perm_test_seeds(rates[a], rates[b_])
        floor = 2 / nperm
        print(f"  {a} vs {b_}: diff={obs*100:5.2f}p  p={p:.3f}  "
              f"(exact, {nperm} perms; smallest possible p = {floor:.3f})")
    if done:
        nperm = math.comb(2 * len(seeds), len(seeds))
        floor = 2 / nperm
        print(f"\n  With {len(seeds)} seeds/arm the exact two-sided permutation "
              f"floor is {floor:.3f} ({nperm} allocations).")
        if floor >= 0.05:
            print("  No arm comparison can attain p<0.05; non-significance is a "
                  "power limitation, not evidence of equivalence.")
        else:
            print("  Arm comparisons can attain p<0.05, but inference remains at "
                  "the seed level rather than the 544-item level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
