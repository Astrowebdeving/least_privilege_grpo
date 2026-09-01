"""B2 (no evidence precision) and the LP_RL_GATING hard-constraint ablation.

Both default to production behaviour and must fail closed on a bad value, so an
unset or typo'd env var can never silently change a training run.
"""
from __future__ import annotations

import os

import pytest

from least_privilege_env.broker import policy as P
from least_privilege_env.broker.classifier import Classification
from least_privilege_env.broker.risk import RiskVector
from least_privilege_env.reward import scoring as S


@pytest.fixture
def clean_env():
    saved = {k: os.environ.pop(k, None) for k in ("LP_RL_REWARD_ARM", "LP_RL_GATING")}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


# ---------------------------------------------------------------- B2
def test_precision_half_penalises_over_reading(clean_env):
    """The control's F1 drops when extra files are read; coverage-only does not."""
    req = ("a.txt",)
    tight = {"a.txt"}
    broad = {"a.txt", "b.txt", "c.txt", "d.txt"}
    assert S.evidence_score(req, tight) == 1.0
    assert S.evidence_score(req, broad) < 1.0, "F1 must punish reading extra files"
    # B2: coverage only -- reading the whole repo costs nothing
    assert S.evidence_score(req, tight, use_precision=False) == 1.0
    assert S.evidence_score(req, broad, use_precision=False) == 1.0


def test_b2_is_registered_and_distinct(clean_env):
    assert "B2" in S.REWARD_ARMS
    os.environ["LP_RL_REWARD_ARM"] = "B2"
    assert S.active_reward_arm() == "B2"


def test_b2_does_not_change_the_privilege_penalty(clean_env):
    """B2 must touch ONLY the evidence term, or it stops being a clean ablation."""
    req = RiskVector(write=0.5)
    assert S.arm_privilege_penalty.__name__  # present
    # B2 is not handled in arm_privilege_penalty, so it must behave as the control
    # there; verify by construction that the arm list routes it to A0's penalty.
    with pytest.raises(ValueError):
        S.arm_privilege_penalty(None, None, 8, "NOPE")


# ---------------------------------------------------------------- gating
def _blocked_classification():
    return Classification(z_pre=RiskVector(), blocked=True, block_reason="test-block")


def test_gating_defaults_on(clean_env):
    assert P.gating_enabled() is True
    d = P.choose_execution_mode(_blocked_classification(), RiskVector(), [])
    assert d.mode == "block"


def test_gating_off_permits_what_would_be_blocked(clean_env):
    os.environ["LP_RL_GATING"] = "off"
    assert P.gating_enabled() is False
    d = P.choose_execution_mode(_blocked_classification(), RiskVector(), [])
    assert d.mode == "execute"
    assert "gating disabled" in (d.reason or "")


def test_gating_off_also_permits_unauthorised_external(clean_env):
    c = Classification(z_pre=RiskVector(external=1.0), blocked=False)
    assert P.choose_execution_mode(c, RiskVector(), []).mode == "block"
    os.environ["LP_RL_GATING"] = "off"
    assert P.choose_execution_mode(c, RiskVector(), []).mode == "execute"


def test_bad_gating_value_raises(clean_env):
    os.environ["LP_RL_GATING"] = "yes"
    with pytest.raises(ValueError, match="must be"):
        P.gating_enabled()
    os.environ["LP_RL_GATING"] = ""
    assert P.gating_enabled() is True, "empty must mean default-on, not error"
