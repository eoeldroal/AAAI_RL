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

"""CPU tests for entropy-resolved clip diagnostics (analysis metrics, no grad path).

Covers the two questions these metrics exist to answer:
  - per-token entropy (de-confounding the sum-normed ``actor/entropy_loss``), and
  - whether the low aggregate clip fraction concentrates on high-entropy pivotal tokens.

Includes fail-closed cases: all-SFT (empty RL) microbatches must not crash, and SFT tokens
must never leak into RL-only diagnostics.
"""

import math

import torch

from verl.trainer.ppo.core_algos import compute_entropy_clip_diagnostics, pg_clip_active_mask


def test_pg_clip_active_mask_upper_and_lower_side():
    # band [0.8, 1.2]; ratios below / in / above / just-inside the band.
    ratio = torch.tensor([[0.5, 1.0, 1.5, 1.1]])

    # A > 0: only ratio above 1.2 freezes (upper clip suppresses probability growth).
    mask_pos = pg_clip_active_mask(ratio, torch.ones_like(ratio), 0.2, 0.2)
    assert mask_pos.tolist() == [[False, False, True, False]]

    # A < 0: only ratio below 0.8 freezes (lower side).
    mask_neg = pg_clip_active_mask(ratio, -torch.ones_like(ratio), 0.2, 0.2)
    assert mask_neg.tolist() == [[True, False, False, False]]


def test_entropy_clip_diag_basic_values():
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])
    rl_mask = torch.ones_like(entropy, dtype=torch.bool)
    # Clip only the high-entropy pivotal token (idx 4): ratio 1.5 > 1.2.
    old_log_prob = torch.zeros_like(entropy)
    log_prob = old_log_prob + torch.tensor([[0.0, 0.0, 0.0, 0.0, math.log(1.5)]])
    adv = torch.ones_like(entropy)

    d = compute_entropy_clip_diagnostics(entropy, log_prob, old_log_prob, adv, rl_mask, 0.2, 0.2)

    assert abs(d["actor/entropy_mean"] - 0.6) < 1e-5  # mean of all five
    assert abs(d["actor/entropy_top20_mean"] - 2.0) < 1e-5  # top 20% of 5 tokens = the 2.0 token
    assert abs(d["actor/pg_clipfrac_top20entropy"] - 1.0) < 1e-6  # that token is clipped


def test_low_entropy_clip_does_not_inflate_pivotal_clipfrac():
    # The whole point: a clip event on a LOW-entropy token must NOT show up in the pivotal
    # clip fraction, even though the aggregate pg_clipfrac would be 1/5 = 0.2.
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])
    rl_mask = torch.ones_like(entropy, dtype=torch.bool)
    old_log_prob = torch.zeros_like(entropy)
    log_prob = old_log_prob + torch.tensor([[math.log(1.5), 0.0, 0.0, 0.0, 0.0]])  # clip idx 0
    adv = torch.ones_like(entropy)

    d = compute_entropy_clip_diagnostics(entropy, log_prob, old_log_prob, adv, rl_mask, 0.2, 0.2)

    assert abs(d["actor/pg_clipfrac_top20entropy"] - 0.0) < 1e-6


def test_empty_rl_mask_returns_empty_dict():
    # All-SFT microbatch (early HPT training): must not crash, must emit nothing.
    entropy = torch.tensor([[0.1, 0.2, 0.3]])
    zeros = torch.zeros_like(entropy)
    rl_mask = torch.zeros_like(entropy, dtype=torch.bool)

    out = compute_entropy_clip_diagnostics(entropy, zeros, zeros, torch.ones_like(entropy), rl_mask, 0.2, 0.2)
    assert out == {}


def test_sft_tokens_excluded_from_rl_diagnostics():
    # idx 2 is an SFT token (rl_mask False): its huge entropy and huge would-be clip must not
    # leak into RL-only diagnostics.
    entropy = torch.tensor([[0.1, 0.2, 9.9]])
    rl_mask = torch.tensor([[True, True, False]])
    old_log_prob = torch.zeros_like(entropy)
    log_prob = old_log_prob + torch.tensor([[0.0, 0.0, math.log(10.0)]])  # SFT token would clip
    adv = torch.ones_like(entropy)

    d = compute_entropy_clip_diagnostics(entropy, log_prob, old_log_prob, adv, rl_mask, 0.2, 0.2)

    assert abs(d["actor/entropy_mean"] - 0.15) < 1e-5  # mean of [0.1, 0.2] only, not 9.9
    assert d["actor/pg_clipfrac_top20entropy"] == 0.0  # SFT clip excluded; RL tokens unclipped


def test_diagnostics_are_plain_floats_and_grad_safe():
    # Metrics must be detached python floats and safe to compute from graph tensors.
    entropy = torch.rand(2, 6, requires_grad=True)
    log_prob = torch.rand(2, 6, requires_grad=True)
    old_log_prob = torch.rand(2, 6)
    adv = torch.randn(2, 6)
    rl_mask = torch.ones(2, 6, dtype=torch.bool)

    d = compute_entropy_clip_diagnostics(entropy, log_prob, old_log_prob, adv, rl_mask, 0.2, 0.28)

    assert set(d) == {"actor/entropy_mean", "actor/entropy_top20_mean", "actor/pg_clipfrac_top20entropy"}
    assert all(isinstance(v, float) for v in d.values())
