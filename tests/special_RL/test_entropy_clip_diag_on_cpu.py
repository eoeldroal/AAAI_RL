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

"""CPU tests for the §11 entropy-resolved clip diagnostics (analysis metrics, no grad path).

The diagnostics answer two questions the aggregate ``actor/pg_clipfrac`` and the count-confounded
``actor/entropy_loss`` cannot: (a) is per-token entropy actually collapsing, and (b) does the low
aggregate clip fraction concentrate on the high-entropy pivotal minority.

They are emitted as token-weighted **sum/count components** (SUM-aggregated) rather than
per-microbatch means. That design is load-bearing:

  * ``compute_entropy_clip_diagnostics`` returns all five keys ALWAYS (0 on an all-SFT microbatch).
    The earlier ``return {}`` made the keys present on only a rank-dependent subset of microbatches,
    so ``Metric.aggregate_dp`` saw unequal per-rank value counts (e.g. ``[3, 5]``) and raised — the
    exact production crash reproduced below.
  * ``finalize_entropy_clip_diagnostics`` recovers each reported ratio as sum/count AFTER
    reduction. The shared aggregation factors cancel in the quotient, so the result is the exact
    token-weighted mean, not a confounded per-microbatch mean.
"""

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_entropy_clip_diagnostics,
    finalize_entropy_clip_diagnostics,
    pg_clip_active_mask,
)
from verl.utils.metric import AggregationType, Metric, reduce_metrics

_COMPONENTS = {
    "actor/_entropy_rl_sum",
    "actor/_entropy_rl_count",
    "actor/_entropy_top20_sum",
    "actor/_entropy_top20_count",
    "actor/_pg_clip_top20entropy_sum",
}


def _diag(entropy, log_prob, old_log_prob, adv, rl_mask, lo=0.2, hi=0.2):
    d = compute_entropy_clip_diagnostics(entropy, log_prob, old_log_prob, adv, rl_mask, lo, hi)
    return {k: float(v) for k, v in d.items()}


# --- pg_clip_active_mask: the shared clip predicate (unchanged) ---
def test_pg_clip_active_mask_upper_and_lower_side():
    # band [0.8, 1.2]; ratios below / in / above / just-inside.
    ratio = torch.tensor([[0.5, 1.0, 1.5, 1.1]])
    # A > 0: only ratio above 1.2 freezes (upper clip suppresses probability growth).
    assert pg_clip_active_mask(ratio, torch.ones_like(ratio), 0.2, 0.2).tolist() == [[False, False, True, False]]
    # A < 0: only ratio below 0.8 freezes (lower side).
    assert pg_clip_active_mask(ratio, -torch.ones_like(ratio), 0.2, 0.2).tolist() == [[True, False, False, False]]


# --- compute_entropy_clip_diagnostics: always emits the five components ---
def test_diag_emits_all_components_with_rl_tokens():
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])
    rl_mask = torch.ones_like(entropy, dtype=torch.bool)
    old = torch.zeros_like(entropy)
    lp = old + torch.tensor([[0.0, 0.0, 0.0, 0.0, math.log(1.5)]])  # clip the high-entropy token
    d = _diag(entropy, lp, old, torch.ones_like(entropy), rl_mask)

    assert set(d) == _COMPONENTS
    assert d["actor/_entropy_rl_count"] == 5.0
    assert d["actor/_entropy_rl_sum"] == pytest.approx(3.0)  # 0.1+0.2+0.3+0.4+2.0
    assert d["actor/_entropy_top20_count"] == 1.0  # top 20% of 5 tokens = the single 2.0 token
    assert d["actor/_entropy_top20_sum"] == pytest.approx(2.0)
    assert d["actor/_pg_clip_top20entropy_sum"] == pytest.approx(1.0)  # that token is clipped


def test_diag_all_sft_microbatch_emits_zeros_not_empty():
    # THE crash fix: an all-SFT microbatch must still emit all five keys (as 0), never {}, so the
    # per-rank value count stays uniform across DP ranks for Metric.aggregate_dp.
    entropy = torch.tensor([[0.1, 0.2, 0.3]])
    z = torch.zeros_like(entropy)
    rl_mask = torch.zeros_like(entropy, dtype=torch.bool)
    d = _diag(entropy, z, z, torch.ones_like(entropy), rl_mask)

    assert set(d) == _COMPONENTS
    assert all(v == 0.0 for v in d.values())


def test_diag_sft_tokens_excluded_from_components():
    # idx 2 is an SFT token (rl_mask False): its huge entropy and would-be clip must not leak in.
    entropy = torch.tensor([[0.1, 0.2, 9.9]])
    rl_mask = torch.tensor([[True, True, False]])
    old = torch.zeros_like(entropy)
    lp = old + torch.tensor([[0.0, 0.0, math.log(10.0)]])  # the SFT token would clip
    d = _diag(entropy, lp, old, torch.ones_like(entropy), rl_mask)

    assert d["actor/_entropy_rl_count"] == 2.0
    assert d["actor/_entropy_rl_sum"] == pytest.approx(0.3)  # 0.1+0.2, not 9.9
    assert d["actor/_pg_clip_top20entropy_sum"] == 0.0  # SFT clip excluded; RL tokens unclipped


def test_diag_low_entropy_clip_not_in_pivotal_sum():
    # A clip on a LOW-entropy token must not enter the pivotal-tail clip sum, even though the
    # aggregate pg_clipfrac would count it (1/5).
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.0]])
    rl_mask = torch.ones_like(entropy, dtype=torch.bool)
    old = torch.zeros_like(entropy)
    lp = old + torch.tensor([[math.log(1.5), 0.0, 0.0, 0.0, 0.0]])  # clip idx 0 (low entropy)
    d = _diag(entropy, lp, old, torch.ones_like(entropy), rl_mask)

    assert d["actor/_pg_clip_top20entropy_sum"] == 0.0


# --- finalize_entropy_clip_diagnostics: recover ratios, handle degenerate / absent cases ---
def test_finalize_recovers_ratios_and_pops_components():
    metrics = {
        "actor/_entropy_rl_sum": 3.0,
        "actor/_entropy_rl_count": 5.0,
        "actor/_entropy_top20_sum": 2.0,
        "actor/_entropy_top20_count": 1.0,
        "actor/_pg_clip_top20entropy_sum": 1.0,
        "actor/pg_loss": -0.42,  # an unrelated metric must pass through untouched
    }
    out = finalize_entropy_clip_diagnostics(metrics)

    assert out["actor/entropy_mean"] == pytest.approx(0.6)  # 3.0 / 5.0
    assert out["actor/entropy_top20_mean"] == pytest.approx(2.0)  # 2.0 / 1.0
    assert out["actor/pg_clipfrac_top20entropy"] == pytest.approx(1.0)  # 1.0 / 1.0
    assert not (_COMPONENTS & set(out))  # raw components popped
    assert out["actor/pg_loss"] == -0.42


def test_finalize_all_sft_step_omits_derived_keys():
    # A whole update step with zero RL tokens: omit the derived metrics rather than emit a
    # misleading 0 (which would re-introduce the routing-fraction confound the metric removes).
    out = finalize_entropy_clip_diagnostics({k: 0.0 for k in _COMPONENTS})

    assert "actor/entropy_mean" not in out
    assert "actor/entropy_top20_mean" not in out
    assert "actor/pg_clipfrac_top20entropy" not in out
    assert not (_COMPONENTS & set(out))  # components still popped


def test_finalize_noop_when_components_absent():
    # Non-HPT paths never set the components; finalize must be a pure pass-through.
    metrics = {"actor/pg_loss": 1.0, "actor/pg_clipfrac": 0.1}
    assert finalize_entropy_clip_diagnostics(dict(metrics)) == metrics


# --- the crash condition + end-to-end token-weighting through the real aggregation path ---
def test_aggregate_dp_raises_on_unequal_per_rank_counts():
    # Reproduces the exact production failure: rank value counts [3, 5] from conditional emission.
    m3 = Metric(aggregation=AggregationType.SUM, value=1.0)
    m3.append(1.0)
    m3.append(1.0)  # 3 values
    m5 = Metric(aggregation=AggregationType.SUM, value=1.0)
    for _ in range(4):
        m5.append(1.0)  # 5 values
    with pytest.raises(ValueError, match="same number of values"):
        Metric.aggregate_dp([m3, m5])


def _sum_metric(values):
    m = Metric(aggregation=AggregationType.SUM, value=values[0])
    for v in values[1:]:
        m.append(v)
    return m


def test_end_to_end_token_weighted_mean_survives_aggregate_dp():
    # Two DP ranks x two microbatches each, with uneven RL-token counts per (rank, chunk) and one
    # all-SFT chunk. Because every chunk emits the components (0 on all-SFT), the per-rank value
    # counts are EQUAL (2 each) so aggregate_dp does not raise; and the finalized ratio is the exact
    # token-weighted mean, not the naive per-chunk mean.
    #   rank0: chunk0 RL entropy [1,1,1,1] (sum 4, n 4); chunk1 all-SFT (sum 0, n 0)
    #   rank1: chunk0 RL entropy [3,3]     (sum 6, n 2); chunk1 RL entropy [5] (sum 5, n 1)
    # token-weighted mean = (4+0+6+5) / (4+0+2+1) = 15/7 ; naive per-chunk mean = (1+3+5)/3 = 3.0
    sum_dp = Metric.aggregate_dp([_sum_metric([4.0, 0.0]), _sum_metric([6.0, 5.0])])
    count_dp = Metric.aggregate_dp([_sum_metric([4.0, 0.0]), _sum_metric([2.0, 1.0])])

    # single mini-batch iteration -> reduction is identity; feed the aggregated scalars to finalize.
    out = finalize_entropy_clip_diagnostics(
        {
            "actor/_entropy_rl_sum": sum_dp,
            "actor/_entropy_rl_count": count_dp,
            "actor/_entropy_top20_sum": sum_dp,
            "actor/_entropy_top20_count": count_dp,
            "actor/_pg_clip_top20entropy_sum": count_dp,  # numerator == count -> ratio 1.0
        }
    )

    assert out["actor/entropy_mean"] == pytest.approx(15.0 / 7.0)
    assert out["actor/entropy_mean"] != pytest.approx(3.0)  # NOT the naive per-chunk mean
    assert out["actor/entropy_top20_mean"] == pytest.approx(15.0 / 7.0)
    assert out["actor/pg_clipfrac_top20entropy"] == pytest.approx(1.0)


def test_component_keys_reduce_by_mean_not_max_min():
    # reduce_metrics switches to np.max / np.min when a key merely CONTAINS "max" / "min". The
    # sum/count cancellation only holds if numerator AND denominator both reduce by mean, so guard
    # against a future rename that would silently break the token weighting.
    for k in _COMPONENTS:
        assert "max" not in k and "min" not in k
    reduced = reduce_metrics({k: [2.0, 4.0] for k in _COMPONENTS})
    for k in _COMPONENTS:
        assert reduced[k] == pytest.approx(3.0)  # mean of [2, 4], not max 4 or min 2


def test_full_pipeline_two_iterations_through_real_reduce_metrics():
    # Faithful end-to-end: two mini-batch iterations, each DP-aggregated per key, accumulated into a
    # per-iteration list, reduced by the REAL reduce_metrics, then finalized. The token-weighted
    # ratio must survive aggregate_dp (rank-mean) AND reduce_metrics (iteration-mean) — every shared
    # factor cancels in sum/count. This is the correctness claim the single-iteration test cannot make.
    #   iter0: rank0 sums/counts [4,0]/[4,0], rank1 [6,5]/[2,1]  (2 chunks/rank; one all-SFT)
    #   iter1: rank0 [6]/[3],                 rank1 [4]/[1]        (1 chunk/rank)
    # RL tokens overall: 4@1.0, 2@3.0, 1@5.0, 3@2.0, 1@4.0 -> sum 25, count 11 -> mean 25/11.
    sum0 = Metric.aggregate_dp([_sum_metric([4.0, 0.0]), _sum_metric([6.0, 5.0])])
    cnt0 = Metric.aggregate_dp([_sum_metric([4.0, 0.0]), _sum_metric([2.0, 1.0])])
    sum1 = Metric.aggregate_dp([_sum_metric([6.0]), _sum_metric([4.0])])
    cnt1 = Metric.aggregate_dp([_sum_metric([3.0]), _sum_metric([1.0])])

    reduced = reduce_metrics(
        {
            "actor/_entropy_rl_sum": [sum0, sum1],
            "actor/_entropy_rl_count": [cnt0, cnt1],
            "actor/_entropy_top20_sum": [sum0, sum1],
            "actor/_entropy_top20_count": [cnt0, cnt1],
            "actor/_pg_clip_top20entropy_sum": [cnt0, cnt1],  # numerator == count -> ratio 1.0
        }
    )
    out = finalize_entropy_clip_diagnostics(reduced)

    assert out["actor/entropy_mean"] == pytest.approx(25.0 / 11.0)
    assert out["actor/entropy_mean"] != pytest.approx((1.0 + 3.0 + 5.0 + 2.0 + 4.0) / 5.0)  # not naive
    assert out["actor/entropy_top20_mean"] == pytest.approx(25.0 / 11.0)
    assert out["actor/pg_clipfrac_top20entropy"] == pytest.approx(1.0)


def test_components_are_detached_scalar_tensors():
    entropy = torch.rand(2, 6, requires_grad=True)
    log_prob = torch.rand(2, 6, requires_grad=True)
    old = torch.rand(2, 6)
    adv = torch.randn(2, 6)
    rl_mask = torch.ones(2, 6, dtype=torch.bool)

    d = compute_entropy_clip_diagnostics(entropy, log_prob, old, adv, rl_mask, 0.2, 0.28)

    assert set(d) == _COMPONENTS
    for v in d.values():  # scalar, detached -> safe to feed Metric (which .item()s them)
        assert isinstance(v, torch.Tensor) and v.numel() == 1
        assert not v.requires_grad
