"""Verify that zero-variance generation groups contribute exactly zero update.

The ablation catalog deliberately contains tasks measured to produce no reward
variance (padding to a round 500). That is only safe if such groups cannot
perturb the weights. Under this project's config -- loss_type="dr_grpo",
scale_rewards="none", beta=0.0 -- TRL computes

    advantages = rewards - mean_grouped_rewards          # grpo_trainer.py:2165
    (no division by std, because scale_rewards == "none") # grpo_trainer.py:2167

so a group whose rewards are all equal yields advantages of exactly 0.0, and
    per_token_loss = -clamped_ratios * advantages * per_token_logps  # :2524
is then exactly 0 for every token in that group.

These tests pin that behaviour so a TRL upgrade cannot silently start
normalising by std (which would turn a 0/0 group into a NaN or a spurious
non-zero update) without failing here first.
"""
from __future__ import annotations

import pathlib
import re

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]


def _advantages(rewards: torch.Tensor, num_generations: int,
                scale_rewards: str = "none") -> torch.Tensor:
    """Mirror of TRL 1.5.1 grpo_trainer.py:2145-2167 for scale_rewards in {group,none}."""
    mean_grouped = rewards.view(-1, num_generations).mean(dim=1)
    mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
    adv = rewards - mean_grouped
    if scale_rewards not in ("none",):
        std = rewards.view(-1, num_generations).std(dim=1)
        std = std.repeat_interleave(num_generations, dim=0)
        adv = adv / (std + 1e-4)
    return adv


def test_zero_variance_group_gives_exactly_zero_advantage():
    # group 0: saturated (all identical) -- the padded/dead tasks
    # group 1: varied -- a gradient-bearing task
    rewards = torch.tensor([0.79, 0.79, 0.79, 0.79,
                            0.10, 0.55, 0.80, 0.30])
    adv = _advantages(rewards, num_generations=4)
    dead, live = adv[:4], adv[4:]

    assert torch.equal(dead, torch.zeros(4)), f"expected exact zeros, got {dead}"
    assert dead.abs().max().item() == 0.0
    assert live.abs().max().item() > 0.0, "the varied group must retain signal"


def test_zero_advantage_yields_zero_token_loss():
    """advantages == 0 kills the loss regardless of ratios/logprobs (:2524)."""
    adv = torch.zeros(4, 1)
    ratios = torch.rand(4, 6) + 0.5
    logps = torch.randn(4, 6)
    per_token_loss = -ratios * adv * logps
    assert torch.equal(per_token_loss, torch.zeros(4, 6))


def test_zero_variance_advantage_is_negligible_not_always_exactly_zero():
    """float32 caveat, measured rather than assumed.

    For values exactly representable in float32 the subtraction cancels to a
    true 0.0. For others (e.g. 0.42) the group mean is off by an ulp and the
    advantage lands at ~3e-08 instead. That is ~7 orders of magnitude below a
    typical live advantage, so it is inert in practice -- but "exactly zero"
    is not guaranteed for every reward value, and claiming so would be wrong.
    """
    exact = _advantages(torch.tensor([0.75, 0.75, 0.75, 0.75]), num_generations=4)
    assert torch.equal(exact, torch.zeros(4))

    inexact = _advantages(torch.full((8,), 0.42), num_generations=8)
    assert torch.isfinite(inexact).all(), "must never be NaN: no 0/0 when scale_rewards='none'"
    assert inexact.abs().max().item() < 1e-6, \
        f"residual should be float32 noise, got {inexact.abs().max().item():.3e}"


def test_our_training_config_still_disables_reward_scaling():
    """If any of these flip, zero-variance groups stop being inert.

    The config passes scale_rewards=False (a bool); GRPOConfig.__post_init__
    maps {True: 'group', False: 'none'}, so False is the 'no scaling' setting.
    Accept either spelling but reject 'group'/'batch'/True.
    """
    src = (REPO / "train" / "train_trl_drgrpo.py").read_text(encoding="utf-8")
    m = re.search(r'scale_rewards\s*=\s*(False|["\']none["\'])', src)
    assert m, "scale_rewards is no longer False/'none'; advantages would be std-normalised, " \
              "which also invalidates the Phase 0 reward-magnitude analysis"
    assert re.search(r'loss_type\s*=\s*["\']dr_grpo["\']', src), "loss_type is no longer dr_grpo"

    # beta is not set in the script -- it relies on the TRL default. Assert on the
    # EFFECTIVE config rather than the source text, so an upstream default change
    # (which would silently reintroduce a KL term) fails here.
    GRPOConfig = pytest.importorskip("trl").GRPOConfig
    cfg = GRPOConfig(output_dir="/tmp/_lp_effective_cfg",
                     loss_type="dr_grpo", scale_rewards=False)
    assert cfg.beta == 0.0, (
        f"TRL default beta is now {cfg.beta}, not 0.0. A KL term adds gradient even "
        f"where the advantage is zero, so zero-variance groups would stop being inert "
        f"and the training script must set beta=0.0 explicitly.")
    assert cfg.scale_rewards == "none"
    assert cfg.loss_type == "dr_grpo"


def test_grpo_config_maps_false_to_none():
    """Pin the bool->str normalisation the config depends on (grpo_config.py:899)."""
    GRPOConfig = pytest.importorskip("trl").GRPOConfig
    assert GRPOConfig(output_dir="/tmp/_lp_cfg_probe", scale_rewards=False).scale_rewards == "none"
    assert GRPOConfig(output_dir="/tmp/_lp_cfg_probe", scale_rewards=True).scale_rewards == "group"


def test_trl_still_skips_std_division_when_scale_rewards_is_none():
    """Pin the upstream branch this whole argument rests on."""
    trl = pytest.importorskip("trl")
    src = (pathlib.Path(trl.__file__).parent / "trainer" / "grpo_trainer.py") \
        .read_text(encoding="utf-8")
    assert "advantages = rewards - mean_grouped_rewards" in src, \
        "TRL changed its advantage formula; re-verify the zero-gradient claim"
    assert 'self.scale_rewards in ["group", "none"]' in src, \
        "TRL changed how scale_rewards='none' is branched; re-verify"
