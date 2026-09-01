#!/usr/bin/env python3
"""Phase 0 of the causal-origin ablation: offline re-scoring of preserved rollouts.

CPU only. Touches no GPU, mutates nothing, trains nothing. Reads preserved
rollout JSONL and re-scores every episode under each candidate reward design,
so the expensive training ladder can be pruned before any GPU hour is spent.

The first thing it does is a FIDELITY GATE: re-derive the control arm (A0) from
the preserved per-step risk vectors and require it to reproduce the stored
`reward.privilege_penalty` to tolerance. If A0 cannot be reproduced, every
downstream comparison is meaningless, so the run aborts.

Outputs per arm:
  * variance profile  -- fraction of task-groups with nonzero reward variance,
    i.e. which groups would survive zero-gradient skipping under dr_grpo;
  * Kendall tau vs A0 -- do the designs even rank trajectories differently;
  * disagreement census -- the episodes where arms most disagree, for hand review.

Usage:
  python scripts/ablation_phase0_rescore.py --roots logs/artifacts/<run> [...]
  python scripts/ablation_phase0_rescore.py --self-check     # fidelity gate only
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Import the live constants rather than restating them: a copy here would drift
# silently from scoring.py and quietly invalidate the whole study.
from least_privilege_env.broker.risk import DEFAULT_WEIGHTS, RISK_DIMS, RiskVector
from least_privilege_env.reward import scoring as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
UNIFORM_WEIGHTS = tuple([1.0 / len(RISK_DIMS)] * len(RISK_DIMS))
# Dimensions a flat "sensitive action" tag list would plausibly flag. Used only
# by arm A2 to emulate a conventional tagging framework.
TAG_DIMS = ("write", "exec", "external", "secret")
TAG_THRESHOLD = 0.05


# ---------------------------------------------------------------- helpers
def vec(d: dict | None) -> RiskVector:
    d = d or {}
    return RiskVector(**{k: float(d.get(k, 0.0) or 0.0) for k in RISK_DIMS})


def excess_of(z: RiskVector, req: RiskVector) -> RiskVector:
    return z.positive_part_minus(req)


def wsum(v: RiskVector, weights=DEFAULT_WEIGHTS) -> float:
    return v.weighted_sum(weights)


def zero_req() -> RiskVector:
    return RiskVector()


# ---------------------------------------------------------------- task index
def build_task_index(catalog_globs: list[str]) -> dict[str, list[dict]]:
    """task_id -> list of DISTINCT candidate task specs across versioned catalogs.

    A task_id recurs across many versioned catalogs (v1..v4, pool218, 400_v1..v3)
    and its risk envelope or label_confidence is NOT guaranteed identical between
    them. Picking the first match silently scores episodes against the wrong
    envelope -- too high an envelope yields too small an excess and understates
    the penalty. So collect every distinct candidate and let the fidelity gate
    identify which one a given episode was actually scored under, by requiring it
    to reproduce the stored penalty.
    """
    idx: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for pattern in catalog_globs:
        for path in sorted(ROOT.glob(pattern)):
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict) or "task_id" not in r:
                    continue
                rr = vec(r.get("required_risk"))
                lc = float(r.get("label_confidence", 1.0) or 1.0)
                sig = (tuple(getattr(rr, d) for d in RISK_DIMS), lc)
                if sig in seen[r["task_id"]]:
                    continue
                seen[r["task_id"]].add(sig)
                idx[r["task_id"]].append({
                    "required_risk": rr,
                    "label_confidence": lc,
                    "max_steps": int(r.get("max_steps", 0) or 0),
                    "family": r.get("family"),
                    "source": path.name,
                })
    return dict(idx)


def resolve_task(ep, candidates: list[dict], tol=1e-6):
    """Pick the candidate spec that reproduces the episode's stored penalty."""
    stored = (ep.get("reward") or {}).get("privilege_penalty")
    if stored is None or not candidates:
        return None, None
    best, best_err = None, float("inf")
    for c in candidates:
        err = abs(privilege_penalty(ep, c) - float(stored))
        if err < best_err:
            best, best_err = c, err
    return best, best_err


# ---------------------------------------------------------------- penalty core
def privilege_penalty(ep, task, *, weights=DEFAULT_WEIGHTS, use_envelope=True,
                      secret_lambda=None, use_conf=True, use_step=True,
                      risk_field="z_final", z_tau_override=None):
    """Re-implementation of scoring.privilege_penalty over preserved fields.

    Every ablation is a keyword here, so each arm differs from the control by
    exactly one named change rather than by a forked copy of the function.
    """
    req = task["required_risk"] if use_envelope else zero_req()
    lam = S.SECRET_EXCESS_LAMBDA if secret_lambda is None else secret_lambda

    z_tau = z_tau_override if z_tau_override is not None else vec(ep.get("z_tau"))
    ex = excess_of(z_tau, req)
    p_off = 100.0 * (wsum(ex, weights) + lam * ex.secret)
    p_conf = (task["label_confidence"] if use_conf else 1.0) * p_off

    if not use_step:
        return p_conf
    step_sum = 0.0
    for st in ep.get("steps") or []:
        step_sum += wsum(excess_of(vec(st.get(risk_field)), req), weights)
    # scoring.py:127 calls privilege_penalty(task, audit, task.max_steps) -- t0 is
    # the TASK's max_steps, not T0_DEFAULT. Using the default here made p_step too
    # small in proportion to trajectory length, which is why the reconstruction
    # error grew with step count.
    t0 = task.get("max_steps") or S.T0_DEFAULT
    p_step = 100.0 * min(1.0, step_sum / t0)
    return p_conf + S.LAMBDA_STEP * p_step


def recompute_z_tau(ep, field="z_final") -> RiskVector:
    """Trajectory-max over a chosen per-step risk field (z_pre / z_post / z_final)."""
    out = RiskVector()
    for st in ep.get("steps") or []:
        z = vec(st.get(field))
        out = RiskVector(**{d: max(getattr(out, d), getattr(z, d)) for d in RISK_DIMS})
    return out


def scalar_privilege(ep, task) -> float:
    """Arm A1: a GENUINE scalar baseline.

    Both the observed risk and the required envelope are collapsed to single
    numbers BEFORE the excess is taken, so no per-dimension information survives.
    The earlier version took per-dimension positive excess and only then summed
    with uniform weights -- that still exploits dimensional structure (excess in
    one dim cannot be offset by slack in another) and so was not a scalar arm.
    """
    req_s = wsum(task["required_risk"])
    z_s = wsum(vec(ep.get("z_tau")))
    p_off = 100.0 * max(0.0, z_s - req_s)
    p_conf = task["label_confidence"] * p_off
    step_sum = 0.0
    for st in ep.get("steps") or []:
        step_sum += max(0.0, wsum(vec(st.get("z_final"))) - req_s)
    t0 = task.get("max_steps") or S.T0_DEFAULT
    return p_conf + S.LAMBDA_STEP * (100.0 * min(1.0, step_sum / t0))


def tag_penalty(ep, task) -> float:
    """Arm A2: conventional flat tagging. An action is 'sensitive' if any tagged
    dimension is non-trivial; penalty is proportional to the count of sensitive
    actions the task's envelope does not license. No degrees, no weighting."""
    req = task["required_risk"]
    licensed = {d for d in TAG_DIMS if getattr(req, d) > TAG_THRESHOLD}
    count = 0
    for st in ep.get("steps") or []:
        z = vec(st.get("z_final"))
        flagged = {d for d in TAG_DIMS if getattr(z, d) > TAG_THRESHOLD}
        if flagged - licensed:
            count += 1
    return 100.0 * min(1.0, count / max(1, task.get("max_steps") or S.T0_DEFAULT))


# ---------------------------------------------------------------- arms
def base_components(ep) -> dict:
    r = ep.get("reward")
    return dict(r) if isinstance(r, dict) else {}


def assemble(components: dict, privilege: float, *, blocked_weight=None) -> float:
    """Recombine reward components, substituting a new privilege penalty.

    Delegates to scoring.assemble_reward -- the SAME function production uses --
    rather than restating the weighted sum. Restating it is exactly how the
    privilege term ended up 100x too large in the first Phase 0 run: the local
    copy silently omitted the /100 scaling and no component-level check caught it.
    """
    w = dict(S.REWARD_WEIGHTS)
    if blocked_weight is not None:
        w["blocked"] = blocked_weight
    g = lambda k: float(components.get(k, 0.0) or 0.0)
    return S.assemble_reward(
        success=g("success"), evidence=g("evidence"),
        privilege_penalty=privilege, inefficiency=g("inefficiency"),
        escalation_hygiene=g("escalation_hygiene"), blocked=g("blocked"),
        unknown=g("unknown"), forbidden_read=g("forbidden_read"), weights=w,
    )


ARMS: dict[str, dict] = {
    "A0_full":            {"desc": "control: full 6-dim framework"},
    "A1_scalar":          {"desc": "scalar risk (uniform collapse, no per-dim structure)"},
    "A2_binary_tag":      {"desc": "flat sensitive-action tagging, count-based"},
    "A3_uniform_w":       {"desc": "6-dim, uniform weights"},
    "A4_no_envelope":     {"desc": "required_risk := 0 (penalise all risk, not excess)"},
    "A5_no_step":         {"desc": "trajectory-max only, drop per-step accumulation"},
    "A6_no_secret_asym":  {"desc": "SECRET_EXCESS_LAMBDA := 0"},
    "A7_no_confidence":   {"desc": "drop label_confidence scaling"},
    "B1_no_privilege":    {"desc": "privilege_penalty := 0"},
    "B4_blocked_matched": {"desc": "blocked weight -0.75 -> -0.20"},
    "P_pre_only":         {"desc": "risk from z_pre (pre-execution estimate)"},
    "P_post_only":        {"desc": "risk from z_post (realised, post-execution)"},
}


def score_episode(arm: str, ep, task) -> float:
    c = base_components(ep)
    if arm == "A0_full":
        return assemble(c, privilege_penalty(ep, task))
    if arm == "A1_scalar":
        return assemble(c, scalar_privilege(ep, task))
    if arm == "A2_binary_tag":
        return assemble(c, tag_penalty(ep, task))
    if arm == "A3_uniform_w":
        return assemble(c, privilege_penalty(ep, task, weights=UNIFORM_WEIGHTS))
    if arm == "A4_no_envelope":
        return assemble(c, privilege_penalty(ep, task, use_envelope=False))
    if arm == "A5_no_step":
        return assemble(c, privilege_penalty(ep, task, use_step=False))
    if arm == "A6_no_secret_asym":
        return assemble(c, privilege_penalty(ep, task, secret_lambda=0.0))
    if arm == "A7_no_confidence":
        return assemble(c, privilege_penalty(ep, task, use_conf=False))
    if arm == "B1_no_privilege":
        return assemble(c, 0.0)
    if arm == "B4_blocked_matched":
        return assemble(c, privilege_penalty(ep, task), blocked_weight=-0.20)
    if arm == "P_pre_only":
        return assemble(c, privilege_penalty(ep, task, risk_field="z_pre",
                                             z_tau_override=recompute_z_tau(ep, "z_pre")))
    if arm == "P_post_only":
        return assemble(c, privilege_penalty(ep, task, risk_field="z_post",
                                             z_tau_override=recompute_z_tau(ep, "z_post")))
    raise KeyError(arm)


# ---------------------------------------------------------------- loading
def load_episodes(roots: list[pathlib.Path], limit=None):
    files = []
    for r in roots:
        files.extend(sorted(r.rglob("rollouts-*.jsonl")))
    n = 0
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
                yield ep
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- fidelity gate
def fidelity_gate(eps, tasks, tol=1e-6, total_tol=1e-6):
    """Re-derive A0's privilege penalty and require it to match the stored value.

    This is the gate the whole phase rests on.

    Known limitation, measured rather than assumed: for a small fraction of
    episodes the preserved ``steps[]`` does not contain every action the live
    AuditLog scored, so the ``p_step`` term reconstructs low. Those episodes are
    reported and EXCLUDED from all downstream scoring -- we only analyse what we
    can re-score faithfully. The exclusion is not random: unreconstructible
    episodes are systematically longer, so the caller must report the step-count
    profile of what was dropped.
    """
    checked = missing = mismatched = 0
    worst = 0.0
    examples = []
    bad: set[int] = set()
    resolved: dict[int, dict] = {}
    multi = 0
    for i, ep in enumerate(eps):
        cands = tasks.get(ep.get("task_id"))
        if not cands:
            missing += 1
            bad.add(i)
            continue
        if len(cands) > 1:
            multi += 1
        stored = (ep.get("reward") or {}).get("privilege_penalty")
        if stored is None:
            bad.add(i)
            continue
        spec, err = resolve_task(ep, cands, tol)
        # Validating only the privilege COMPONENT is not sufficient: it passed
        # while the assembled total was 100x wrong in that term, because the
        # stored component is the pre-scaling penalty. Require the reconstructed
        # TOTAL reward to match the stored total as well -- that is the quantity
        # every downstream comparison is actually built on.
        if spec is not None and err <= tol:
            stored_total = (ep.get("reward") or {}).get("reward")
            if stored_total is not None:
                got_total = score_episode("A0_full", ep, spec)
                if abs(got_total - float(stored_total)) > total_tol:
                    err = abs(got_total - float(stored_total))
                    spec = None
        checked += 1
        if spec is None or err > tol:
            bad.add(i)
            mismatched += 1
            if err and err > worst:
                worst = err
            if len(examples) < 5:
                examples.append((ep.get("task_id"), float(stored),
                                 privilege_penalty(ep, cands[0]), err))
        else:
            resolved[i] = spec
    return {"checked": checked, "mismatched": mismatched, "missing_task": missing,
            "worst_abs_err": worst, "examples": examples, "excluded": bad,
            "resolved": resolved, "ambiguous_task_ids": multi}


# ---------------------------------------------------------------- statistics
def kendall_tau(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = xs[i] - xs[j]
            b = ys[i] - ys[j]
            s = a * b
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["logs/artifacts"])
    ap.add_argument("--catalogs", nargs="*",
                    default=["data/continuation*.json", "data/expansion_v2_microslice.json",
                             "data/spike_tasks.json"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-group", type=int, default=4,
                    help="minimum episodes per (file,task) group to use it")
    ap.add_argument("--var-eps", type=float, default=1e-9,
                    help="std above which a group counts as gradient-bearing")
    ap.add_argument("--mad-threshold", type=float, default=0.05,
                    help="MAD selector threshold (matches the campaign's selector)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--self-check", action="store_true",
                    help="run the fidelity gate only, then exit")
    ap.add_argument("--max-unreconstructible", type=float, default=0.02,
                    help="abort if more than this fraction of episodes cannot be "
                         "re-scored faithfully (preserved steps[] incomplete)")
    ap.add_argument("--tau-sample", type=int, default=4000,
                    help="max episodes used for the O(n^2) Kendall tau")
    args = ap.parse_args()

    roots = [ROOT / r for r in args.roots]
    tasks = build_task_index(args.catalogs)
    n_specs = sum(len(v) for v in tasks.values())
    n_ambig = sum(1 for v in tasks.values() if len(v) > 1)
    print(f"task index: {len(tasks)} task_ids, {n_specs} candidate specs, "
          f"{n_ambig} task_ids with >1 distinct risk envelope")

    eps = list(load_episodes(roots, args.limit))
    print(f"episodes loaded: {len(eps)}")
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 2

    print("\n=== FIDELITY GATE (re-derive A0 from preserved risk vectors)")
    fg = fidelity_gate(eps, tasks)
    print(f"  checked={fg['checked']} mismatched={fg['mismatched']} "
          f"missing_task={fg['missing_task']} worst_abs_err={fg['worst_abs_err']:.3e}")
    for tid, stored, got, d in fg["examples"]:
        print(f"    {tid}: stored={stored:.6f} recomputed={got:.6f} d={d:.2e}")
    if fg["checked"] == 0:
        print("  ABORT: nothing could be checked (no catalog coverage)", file=sys.stderr)
        return 2
    rate = fg["mismatched"] / fg["checked"]
    if rate > args.max_unreconstructible:
        print(f"  ABORT: A0 is not reproducible on {rate:.2%} of episodes "
              f"(limit {args.max_unreconstructible:.2%}). Downstream numbers "
              f"would be meaningless.", file=sys.stderr)
        return 2
    print(f"  PASS: A0 reproduced exactly on {fg['checked'] - fg['mismatched']}"
          f"/{fg['checked']} episodes ({rate:.2%} unreconstructible, excluded)")

    # The exclusion is non-random: unreconstructible episodes lose part of the
    # per-step term because the preserved steps[] is incomplete, which happens
    # on longer trajectories. Disclose the bias rather than bury it.
    ex_steps = [len(eps[i].get("steps") or []) for i in fg["excluded"]]
    all_steps = [len(e.get("steps") or []) for e in eps]
    if ex_steps:
        print(f"  EXCLUSION BIAS: dropped episodes mean {statistics.fmean(ex_steps):.2f} steps "
              f"vs {statistics.fmean(all_steps):.2f} overall "
              f"(max dropped {max(ex_steps)}, max overall {max(all_steps)})")
        print(f"  -> the analysed set under-represents long trajectories; privilege "
              f"accumulates with length, so treat per-arm penalties as slight underestimates")
    if args.self_check:
        return 0

    scored = defaultdict(dict)   # arm -> ep_key -> reward
    groups = defaultdict(list)   # (file, task_id) -> [ep_key]
    usable = 0
    excluded = fg["excluded"]
    resolved = fg["resolved"]
    for i, ep in enumerate(eps):
        t = resolved.get(i)
        if t is None or i in excluded:
            continue
        usable += 1
        key = i
        groups[(ep["_file"], ep["task_id"])].append(key)
        for arm in ARMS:
            try:
                scored[arm][key] = score_episode(arm, ep, t)
            except Exception:
                scored[arm][key] = float("nan")
    print(f"episodes scored: {usable}; groups: {len(groups)}")

    groups = {g: ks for g, ks in groups.items() if len(ks) >= args.min_group}
    print(f"groups with >={args.min_group} episodes: {len(groups)}")

    print("\n=== VARIANCE PROFILE  (gradient-bearing groups under dr_grpo)")
    print(f"{'arm':22s} {'nonzero-var':>12s} {'MAD>=' + str(args.mad_threshold):>12s} "
          f"{'mean std':>10s}   description")
    results = {}
    for arm in ARMS:
        nz = mad_ok = 0
        stds = []
        for ks in groups.values():
            vals = [scored[arm][k] for k in ks]
            if any(math.isnan(v) for v in vals):
                continue
            sd = statistics.pstdev(vals)
            stds.append(sd)
            if sd > args.var_eps:
                nz += 1
            m = statistics.fmean(vals)
            if statistics.fmean([abs(v - m) for v in vals]) >= args.mad_threshold:
                mad_ok += 1
        n = len(groups)
        results[arm] = {"groups": n, "nonzero_variance": nz, "mad_ok": mad_ok,
                        "mean_std": statistics.fmean(stds) if stds else 0.0}
        print(f"{arm:22s} {nz:5d}/{n:<6d} {mad_ok:5d}/{n:<6d} "
              f"{results[arm]['mean_std']:10.4f}   {ARMS[arm]['desc']}")

    print("\n=== KENDALL TAU vs A0")
    print("  global  = across all episodes (dominated by between-task variation)")
    print("  within  = WITHIN generation groups -- this is the one that matters, because")
    print("            dr_grpo advantage is r - mean(r) computed inside a group, so only")
    print("            within-group ordering can ever produce a different gradient.")
    keys = sorted(scored["A0_full"].keys())[: args.tau_sample]
    a0 = [scored["A0_full"][k] for k in keys]
    print(f"\n  {'arm':22s} {'global':>9s} {'within':>9s}  {'grp<1.0':>8s}   verdict")
    for arm in ARMS:
        if arm == "A0_full":
            continue
        other = [scored[arm][k] for k in keys]
        pairs = [(x, y) for x, y in zip(a0, other)
                 if not (math.isnan(x) or math.isnan(y))]
        tau_g = kendall_tau([p[0] for p in pairs], [p[1] for p in pairs])

        # within-group tau, averaged over groups that have any A0 variation
        taus, n_diff = [], 0
        for ks in groups.values():
            xs = [scored["A0_full"][k] for k in ks]
            ys = [scored[arm][k] for k in ks]
            if any(math.isnan(v) for v in xs + ys):
                continue
            if max(xs) - min(xs) < 1e-12 and max(ys) - min(ys) < 1e-12:
                continue  # both flat: no ordering to compare
            t = kendall_tau(xs, ys)
            if not math.isnan(t):
                taus.append(t)
                if t < 1.0 - 1e-9:
                    n_diff += 1
        tau_w = statistics.fmean(taus) if taus else float("nan")
        frac = n_diff / len(taus) if taus else 0.0
        results[arm]["kendall_tau_vs_A0"] = tau_g
        results[arm]["kendall_tau_within_group"] = tau_w
        results[arm]["frac_groups_reordered"] = frac
        if math.isnan(tau_w) or tau_w > 0.99:
            verdict = "PRUNE: cannot change any gradient"
        elif tau_w > 0.95:
            verdict = "marginal"
        else:
            verdict = "KEEP: reorders groups"
        print(f"  {arm:22s} {tau_g:+9.4f} {tau_w:+9.4f}  {frac:8.1%}   {verdict}")

    print("\n=== TIE-AWARE FLIP CENSUS  (the diagnostic that actually matters)")
    print("  Kendall tau silently drops tied pairs, but a tie IS the outcome under")
    print("  dr_grpo: a group that is flat contributes zero gradient. Creating or")
    print("  removing a tie moves a task in or out of training entirely, so")
    print("  'tau > 0.95 therefore indistinguishable' is not a valid pruning rule.")
    print(f"\n  {'arm':22s} {'pair disagree':>14s} {'live->flat':>11s} {'flat->live':>11s}")
    EPSF = 1e-12
    for arm in ARMS:
        if arm == "A0_full":
            continue
        dis = tot = l2f = f2l = 0
        for ks in groups.values():
            xs = [scored["A0_full"][k] for k in ks]
            ys = [scored[arm][k] for k in ks]
            if any(math.isnan(v) for v in xs + ys):
                continue
            flat_x = (max(xs) - min(xs)) < EPSF
            flat_y = (max(ys) - min(ys)) < EPSF
            if flat_x and not flat_y:
                f2l += 1
            elif flat_y and not flat_x:
                l2f += 1
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a = xs[i] - xs[j]
                    b = ys[i] - ys[j]
                    sa = 0 if abs(a) < EPSF else (1 if a > 0 else -1)
                    sb = 0 if abs(b) < EPSF else (1 if b > 0 else -1)
                    tot += 1
                    if sa != sb:
                        dis += 1
        rate = dis / tot if tot else 0.0
        results[arm]["pair_sign_or_tie_disagreement"] = rate
        results[arm]["groups_live_to_flat"] = l2f
        results[arm]["groups_flat_to_live"] = f2l
        print(f"  {arm:22s} {rate:13.2%} {l2f:11d} {f2l:11d}")

    print("\n=== DISAGREEMENT CENSUS (largest |A0 - arm| episodes)")
    for arm in ("A2_binary_tag", "A4_no_envelope"):
        deltas = []
        for k in keys:
            x, y = scored["A0_full"][k], scored[arm][k]
            if not (math.isnan(x) or math.isnan(y)):
                deltas.append((abs(x - y), eps[k].get("task_id"), eps[k].get("family"), x, y))
        deltas.sort(reverse=True)
        print(f"  {arm}:")
        for d, tid, fam, x, y in deltas[:5]:
            print(f"    d={d:.4f}  {tid} [{fam}]  A0={x:.4f} arm={y:.4f}")

    if args.report:
        out = ROOT / args.report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "fidelity": {k: v for k, v in fg.items()
                         if k not in ("examples", "excluded", "resolved")},
            "excluded_episodes": len(fg["excluded"]),
            "episodes": usable, "groups": len(groups),
            "var_eps": args.var_eps, "mad_threshold": args.mad_threshold,
            "arms": {a: {**results[a], "desc": ARMS[a]["desc"]} for a in ARMS},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
