# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for the advantage-normalization axis under async-HPT.

The historical validation pinned ``algorithm.norm_adv_by_std_in_grpo=False`` (Dr.GRPO
phase-1 contract, DR-001). Improvement_RL.md §5.10 (M4) revises this: both estimator
contracts are admissible because HPT's SFT rows are singleton advantage groups and
``compute_grpo_outcome_advantage`` special-cases ``len==1`` groups to ``mean=0/std=1``,
which preserves the SFT terminal pseudo-reward ``beta_r`` as the row advantage under
either mode. These tests pin (a) the relaxed-but-explicit validation behavior and
(b) the singleton pass-through property the relaxation relies on.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _make_hpt_config(*, norm_adv_by_std_in_grpo):
    """Minimal config accepted by validate_async_hpt_config (M-anchor shape)."""
    return OmegaConf.create(
        {
            "async_hpt": {
                "enabled": True,
                "tau_dataset_path": "/tmp/tau.parquet",
                "tau_messages_key": "tau_messages",
                "gamma": 0.0,
                "alpha": 1.0,
                "beta": 0.3,
                "sft_beta_mode": "constant",
                "loss_aggregation": "branch_blind",
                "sft_entropy_enabled": False,
                "sft_kl_enabled": False,
                "fail_on_missing_tau": True,
                "rl_old_logprob_source": "entry",
                "entry_proximal": "recent",
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
                "rollout_correction": {
                    "rollout_is": "token",
                    "rollout_rs": None,
                    "rollout_is_threshold": 2.0,
                    "bypass_mode": False,
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "calculate_log_probs": True,
                    "n": 8,
                },
                "actor": {
                    "loss_agg_mode": "seq-mean-token-sum-norm",
                    "loss_scale_factor": 8192,
                    "clip_ratio_low": 10.0,
                    "clip_ratio_high": 0.28,
                    "policy_loss": {
                        "loss_mode": "cispo",
                    },
                },
            },
        }
    )


def test_validate_accepts_both_advantage_normalization_contracts():
    # Dr.GRPO (False) is the D0-lineage value; GRPO std-normalized (True) is the
    # reference-parity value adopted by M4. Both must validate.
    validate_async_hpt_config(_make_hpt_config(norm_adv_by_std_in_grpo=False))
    validate_async_hpt_config(_make_hpt_config(norm_adv_by_std_in_grpo=True))


def test_validate_rejects_unset_advantage_normalization():
    # The launcher must state the estimator contract explicitly; a missing key silently
    # inheriting upstream verl's default (True) is exactly the ambiguity we forbid.
    with pytest.raises(ValueError, match="norm_adv_by_std_in_grpo"):
        validate_async_hpt_config(_make_hpt_config(norm_adv_by_std_in_grpo=None))


def _advantage_scalars(norm_adv_by_std_in_grpo: bool) -> torch.Tensor:
    """One 8-row RL group (1 success / 7 fail) + one singleton SFT row (beta_r=0.3)."""
    n_tokens = 4
    rewards = torch.zeros(9, n_tokens)
    rewards[0, -1] = 1.0  # the lone RL success (terminal reward)
    rewards[8, -1] = 0.3  # SFT pseudo-reward beta_r on the singleton row
    response_mask = torch.ones(9, n_tokens)
    index = np.array(["rl"] * 8 + ["sft"], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    return advantages[:, 0]  # scalar advantage broadcast per row


def test_singleton_sft_advantage_is_beta_r_under_both_modes():
    # The load-bearing property behind the gate relaxation: singleton groups get
    # mean=0/std=1, so beta_r survives std normalization untouched.
    for mode in (False, True):
        scalars = _advantage_scalars(mode)
        assert scalars[8].item() == pytest.approx(0.3, rel=1e-4), (
            f"SFT beta_r must pass through unchanged (mode={mode})"
        )


def test_rl_group_advantage_is_std_scaled_only_in_grpo_mode():
    unnormalized = _advantage_scalars(False)
    normalized = _advantage_scalars(True)

    # Dr.GRPO: success advantage = 1 - mean = 0.875; fails = -0.125.
    assert unnormalized[0].item() == pytest.approx(0.875, rel=1e-4)
    assert unnormalized[1].item() == pytest.approx(-0.125, rel=1e-4)

    # GRPO: same values divided by the group std (Bessel-corrected, n=8) + epsilon.
    group_std = torch.std(torch.tensor([1.0] + [0.0] * 7)).item()
    assert normalized[0].item() == pytest.approx(0.875 / (group_std + 1e-6), rel=1e-3)
    assert normalized[1].item() == pytest.approx(-0.125 / (group_std + 1e-6), rel=1e-3)
    # The sparse-success amplification M4 buys: ~2.5-3x on a 1/8 group.
    assert normalized[0].item() / unnormalized[0].item() > 2.0
