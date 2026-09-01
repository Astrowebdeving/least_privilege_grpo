"""Pin that reward assembly has exactly one implementation.

The 2026-08-04 ablation bug: an offline re-scorer restated the weighted sum and
omitted the /100 privilege scaling, making that term 100x too large. Every
downstream conclusion was wrong, and a component-level fidelity check passed
anyway because the stored component is the PRE-scaling penalty.

These tests pin (a) the extraction is behaviour-preserving, and (b) the ablation
harness delegates rather than restating.
"""
from __future__ import annotations

import pathlib
import re

from least_privilege_env.reward import scoring as S

REPO = pathlib.Path(__file__).resolve().parents[1]

COMPONENTS = dict(success=1.0, evidence=0.85, privilege_penalty=55.1,
                  inefficiency=0.17, escalation_hygiene=0.1, blocked=0.0,
                  unknown=0.0, forbidden_read=0.0)


def _literal_expected(c: dict, w: dict | None = None) -> float:
    """The assembly written out longhand, as scoring.py had it before extraction."""
    w = w or S.REWARD_WEIGHTS
    return (
        w["success"] * c["success"]
        + w["evidence"] * c["evidence"]
        + w["privilege_penalty"] * (c["privilege_penalty"] / 100)
        + w["inefficiency"] * c["inefficiency"]
        + w["escalation_hygiene"] * c["escalation_hygiene"]
        + w["blocked"] * c["blocked"]
        + w["unknown"] * c["unknown"]
        + w["forbidden_read"] * c["forbidden_read"]
    )


def test_extraction_is_behaviour_preserving():
    assert S.assemble_reward(**COMPONENTS) == _literal_expected(COMPONENTS)


def test_privilege_scale_is_one_hundred():
    """The exact constant the ablation bug got wrong."""
    assert S.PRIVILEGE_SCALE == 100.0
    a = S.assemble_reward(**{**COMPONENTS, "privilege_penalty": 0.0})
    b = S.assemble_reward(**{**COMPONENTS, "privilege_penalty": 100.0})
    delta = b - a
    assert abs(delta - S.REWARD_WEIGHTS["privilege_penalty"]) < 1e-12, (
        "a penalty of 100 must move the reward by exactly one privilege weight; "
        f"moved {delta}")


def test_weight_override_only_changes_that_term():
    w = dict(S.REWARD_WEIGHTS)
    w["blocked"] = -0.20
    got = S.assemble_reward(**{**COMPONENTS, "blocked": 1.0}, weights=w)
    assert got == _literal_expected({**COMPONENTS, "blocked": 1.0}, w)


def test_scoring_does_not_restate_the_weighted_sum():
    """total_reward must delegate, so there is one implementation to get wrong."""
    src = (REPO / "least_privilege_env" / "reward" / "scoring.py").read_text(encoding="utf-8")
    body = src.split("def total_reward", 1)[1]
    assert "assemble_reward(" in body, "total_reward no longer delegates to assemble_reward"
    assert "/ 100" not in body.split("return RewardBreakdown", 1)[0], \
        "total_reward restates the /100 scaling instead of delegating"


def test_ablation_harness_delegates_to_production_assembly():
    src = (REPO / "scripts" / "ablation_phase0_rescore.py").read_text(encoding="utf-8")
    assert "S.assemble_reward(" in src, \
        "the ablation harness must call scoring.assemble_reward, not restate it"
    assert not re.search(r'total\s*\+=\s*weight\s*\*\s*privilege\b', src), \
        "the ablation harness restates the privilege term again"
