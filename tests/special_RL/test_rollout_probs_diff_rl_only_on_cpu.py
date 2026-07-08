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
"""Contract tests for the RL-only rollout_probs_diff gauge (Improvement_RL.md §5.11).

async-HPT materializes SFT rows with a PLACEHOLDER rollout_log_probs of zeros
(hpt_assembler.py) because those rows never went through the rollout engine. The
debug metric ``training/rollout_probs_diff_*`` masks by response tokens only, so on an
HPT batch it averages ``|actor_prob - exp(0)=1|`` over the SFT tokens and reports a
large, meaningless mismatch (M4 logged ~0.05-0.16). These tests pin the fix:
``calculate_debug_metrics(data, exclude_rows=hpt_is_sft)`` additionally emits ``_rl``
keys computed over RL rows only — the genuine rollout-engine-vs-trainer precision
gauge that L1 (fp32 lm_head) is meant to move — while the original all-row keys stay
unchanged for back-compat and cross-run comparability.
"""

import math

import pytest
import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics


def _make_batch(*, n_rl: int, n_sft: int, resp_len: int = 4):
    """Build a batch where RL rows have a small actor/rollout logprob gap and SFT rows
    carry the placeholder rollout_log_probs=0 (so exp(0)=1 vs a real actor prob)."""
    n = n_rl + n_sft
    # actor logprobs: modest values (prob ~ exp(-0.5) ~ 0.61) for every row/token.
    actor = torch.full((n, resp_len), -0.5)
    rollout = torch.empty((n, resp_len))
    # RL rows: rollout logprob differs from actor by a tiny amount (bf16-scale mismatch).
    rollout[:n_rl] = -0.5 + 0.01
    # SFT rows: placeholder zeros -> rollout prob = exp(0) = 1.0 (the pollution source).
    rollout[n_rl:] = 0.0
    response_mask = torch.ones((n, resp_len), dtype=torch.int64)
    hpt_is_sft = torch.zeros(n, dtype=torch.bool)
    hpt_is_sft[n_rl:] = True
    responses = torch.zeros((n, resp_len), dtype=torch.int64)
    return DataProto.from_dict(
        tensors={
            "old_log_probs": actor,
            "rollout_log_probs": rollout,
            "response_mask": response_mask,
            "responses": responses,
            "hpt_is_sft": hpt_is_sft,
        }
    ), hpt_is_sft


def test_base_keys_unchanged_when_no_exclusion():
    # Non-HPT path (exclude_rows=None): only the original keys, no "_rl" keys — proves the
    # change is purely additive and cannot alter existing (non-HPT) run behavior.
    data, _ = _make_batch(n_rl=6, n_sft=0)
    out = calculate_debug_metrics(data)
    assert out["training/rollout_probs_diff_valid"] == 1
    assert not any(k.endswith("_rl") or k.endswith("_rl_mean") or "rl_" in k for k in out)
    assert "training/rollout_probs_diff_sft_rows_excluded" not in out


def test_sft_placeholder_pollutes_all_row_metric():
    # The documented bug: with SFT rows present, the all-row mean is dominated by the
    # |actor_prob - 1.0| placeholder gap, not the true RL mismatch.
    data, sft = _make_batch(n_rl=4, n_sft=4)
    out = calculate_debug_metrics(data, exclude_rows=sft)
    # actor prob = exp(-0.5) ~= 0.6065; SFT placeholder rollout prob = 1.0 -> gap ~= 0.3935.
    # RL gap: exp(-0.49) - exp(-0.5) ~= 0.0061. All-row mean sits between, pulled up by SFT.
    assert out["training/rollout_probs_diff_mean"] > 0.1, "all-row metric should be polluted by SFT"


def test_rl_only_metric_excludes_sft_placeholder():
    data, sft = _make_batch(n_rl=4, n_sft=4)
    out = calculate_debug_metrics(data, exclude_rows=sft)
    # RL-only mean reflects only the tiny bf16-scale RL gap.
    rl_mean = out["training/rollout_probs_diff_rl_mean"]
    expected = abs(math.exp(-0.49) - math.exp(-0.5))
    assert rl_mean == pytest.approx(expected, rel=1e-3)
    assert rl_mean < 0.02
    # and it is far cleaner than the polluted all-row value
    assert rl_mean < out["training/rollout_probs_diff_mean"] / 10
    assert out["training/rollout_probs_diff_rl_valid"] == 1
    assert out["training/rollout_probs_diff_sft_rows_excluded"] == 4


def test_all_sft_batch_reports_rl_invalid_not_crash():
    # A fully-SFT batch (early training, everything routed to SFT) must not crash the
    # empty-reduction path: base keys stay valid (SFT tokens are real response tokens),
    # but the RL-only gauge reports valid=0.
    data, sft = _make_batch(n_rl=0, n_sft=5)
    out = calculate_debug_metrics(data, exclude_rows=sft)
    assert out["training/rollout_probs_diff_valid"] == 1
    assert out["training/rollout_probs_diff_rl_valid"] == 0
    assert math.isnan(out["training/rollout_probs_diff_rl_mean"])
    assert out["training/rollout_probs_diff_sft_rows_excluded"] == 5
