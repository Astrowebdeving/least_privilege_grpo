#!/usr/bin/env python3
"""Internal held-out evidence for the c4a reward arms.

Independent of ToolPrivBench: these are the campaign's own held-out rollouts
(50 tasks x 8 generations per cell), scored for every arm under the CONTROL
reward (A0) with gating ON, so the measuring instrument does not vary by arm.

Over-privilege is near-saturated on this set, so it cannot rank the arms on
safety -- that is what ToolPrivBench is for. What this set CAN do is corroborate
the capability side: whether the coarse binary-tag penalty (A2) costs task
success relative to the graded 6-dim penalty (A0). Two independent measurements
agreeing is worth more than either alone.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import pathlib
import statistics


def load_cell(d: pathlib.Path) -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(d / "heldout" / "rollouts" / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    eps.append(json.loads(line))
    return eps


def summarise(eps: list[dict]) -> dict[str, float]:
    def m(key: str) -> float:
        vals = [e.get("reward", {}).get(key) for e in eps]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return statistics.fmean(vals) if vals else float("nan")

    out = {
        "n": len(eps),
        "success": m("success"),
        "evidence": m("evidence"),
        "privilege_penalty": m("privilege_penalty"),
        "reward": m("reward"),
    }
    # fraction of episodes that touched any privilege at all
    pens = [e.get("reward", {}).get("privilege_penalty", 0.0) for e in eps]
    out["any_privilege"] = statistics.fmean(
        [1.0 if (isinstance(p, (int, float)) and p > 0) else 0.0 for p in pens]
    ) if pens else float("nan")
    return out


def perm(x: list[float], y: list[float]) -> tuple[float, float, int]:
    obs = abs(statistics.fmean(x) - statistics.fmean(y))
    pool, k = x + y, len(x)
    diffs = []
    for idx in itertools.combinations(range(len(pool)), k):
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in idx]
        diffs.append(abs(statistics.fmean(a) - statistics.fmean(b)))
    return obs, sum(1 for d in diffs if d >= obs - 1e-12) / len(diffs), len(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/artifacts/cont400_arm_campaign")
    ap.add_argument("--arms", default="A0 A2 A4 B1")
    ap.add_argument("--seeds", default="1 2")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    arms, seeds = args.arms.split(), args.seeds.split()

    cells: dict[str, dict[str, float]] = {}
    for a in arms:
        for s in seeds:
            d = root / f"{a}-seed{s}"
            if not d.exists():
                continue
            eps = load_cell(d)
            if eps:
                cells[f"{a}-s{s}"] = summarise(eps)

    if not cells:
        print("no held-out rollouts found")
        return 1

    print("== per cell (control scorer A0, gating on, for every arm)")
    hdr = f"  {'cell':9s} {'n':>5s} {'success':>8s} {'evidence':>9s} {'priv_pen':>9s} {'reward':>8s}"
    print(hdr)
    for k, v in cells.items():
        print(f"  {k:9s} {v['n']:>5.0f} {v['success']:>8.4f} {v['evidence']:>9.4f} "
              f"{v['privilege_penalty']:>9.4f} {v['reward']:>8.4f}")

    print("\n== arm means")
    per_arm: dict[str, dict[str, list[float]]] = {}
    for a in arms:
        ks = [k for k in cells if k.startswith(a + "-")]
        if not ks:
            continue
        per_arm[a] = {
            metric: [cells[k][metric] for k in ks]
            for metric in ("success", "evidence", "privilege_penalty", "reward")
        }
        print(f"  {a:3s} success={statistics.fmean(per_arm[a]['success']):.4f} "
              f"evidence={statistics.fmean(per_arm[a]['evidence']):.4f} "
              f"priv_pen={statistics.fmean(per_arm[a]['privilege_penalty']):.4f} "
              f"reward={statistics.fmean(per_arm[a]['reward']):.4f}")

    print("\n== does the coarse penalty cost task success? (seed-level exact permutation)")
    for a, b in itertools.combinations(sorted(per_arm), 2):
        if len(per_arm[a]["success"]) != len(per_arm[b]["success"]):
            continue
        obs, p, n = perm(per_arm[a]["success"], per_arm[b]["success"])
        print(f"  {a} vs {b}: success diff={obs:.4f}  p={p:.3f}  "
              f"({n} perms, floor {2/n:.3f})")
    seed_count = len(seeds)
    nperm = math.comb(2 * seed_count, seed_count)
    floor = 2 / nperm
    print(f"\n  With {seed_count} seeds/arm the exact two-sided permutation floor is "
          f"{floor:.3f} ({nperm} allocations).")
    if floor >= 0.05:
        print("  The design cannot attain p<0.05; read direction and cross-panel "
              "agreement, not non-significance as equivalence.")
    else:
        print("  The design can attain p<0.05, but seed-level spread remains the "
              "relevant training uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
