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

"""Consolidated CPU tests for HPT truncation-as-failure handling (docs/Improvement_RL.md §5-6).

Single module covering the whole feature, organized components -> integration:

  Section A — P0-1 reward gate               (reward_loop/reward_manager/dapo.py::run_single)
  Section B — routing score-position fix     (hpt_gate.extract_score_values / count_successful_rollouts)
  Section C — P0-2 truncated-advantage zeroing (FullyAsyncTrainer._fit_filter_truncated_rl_advantage)
  Section D — zeroing == physical removal    (compute_policy_loss_cispo, fixed seq-mean denominator)
  Section E — END-TO-END pipeline integration (gate -> route -> advantage -> P0-2 -> loss)

Why the pieces belong together (the failure they jointly prevent, Improvement_RL.md §5.7): a reward
lives at the response's TERMINAL token, not the last tensor index. Routing must reduce it with
sum(-1) (Section B) or every early-terminating correct rollout scores 0 and its group is routed to
SFT; combined with the P0-1 gate zeroing truncated rewards (Section A), that drove on-policy success
to exactly 0 and collapsed the run to pure SFT. Section E walks a synthetic group through all stages
so a regression in their interaction is caught, not just in one stage.
"""

import asyncio
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.hpt_gate import count_successful_rollouts, extract_score_values
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_cispo
from verl.workers.config import ActorConfig

# Integration constants (Section E); the component sections use their own small widths.
GAMMA = 0.0  # HPT routes to SFT iff success_probability <= gamma
MAX_RESP = 16  # generation budget; a rollout of length MAX_RESP is "truncated"
CLEAN_LEN = 8  # clean (early-terminating) rollouts stop at half budget


# ==================================================================================================
# Shared helpers
# ==================================================================================================
def _make_reward_manager(*, zero_reward_if_truncated, max_resp_len, score_result, overlong_cfg=None):
    """reward-loop DAPORewardManager, bypassing __init__ (no tokenizer/model load)."""
    from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager

    mgr = object.__new__(DAPORewardManager)
    mgr.overlong_buffer_cfg = overlong_cfg
    mgr.max_resp_len = max_resp_len
    mgr.zero_reward_if_truncated = zero_reward_if_truncated
    mgr.compute_score_in_executor = False
    mgr.is_async_reward_score = False
    mgr.reward_router_address = None
    mgr.reward_model_tokenizer = None
    mgr.compute_score = lambda **kwargs: (dict(score_result) if isinstance(score_result, dict) else score_result)

    class _StubTok:
        def decode(self, ids, skip_special_tokens=True):
            return "x"

    mgr.tokenizer = _StubTok()
    return mgr


def _make_reward_item(*, prompt_len, resp_len, valid_resp):
    total = prompt_len + resp_len
    attention_mask = torch.zeros(1, total, dtype=torch.long)
    attention_mask[0, :prompt_len] = 1
    attention_mask[0, prompt_len : prompt_len + valid_resp] = 1
    responses = torch.ones(1, resp_len, dtype=torch.long)
    return DataProto.from_dict(
        tensors={"responses": responses, "attention_mask": attention_mask},
        non_tensors={
            "data_source": np.array(["math"], dtype=object),
            "reward_model": np.array([{"ground_truth": "1"}], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )


def _run_single(mgr, data):
    async def _driver():
        mgr.loop = asyncio.get_running_loop()
        return await mgr.run_single(data)

    return asyncio.run(_driver())


def _make_trainer(config_dict):
    """FullyAsyncTrainer, bypassing __init__, for exercising _fit_filter_truncated_rl_advantage."""
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    meta = getattr(FullyAsyncTrainer, "__ray_metadata__", None)
    trainer_cls = meta.modified_class if meta is not None else FullyAsyncTrainer
    trainer = object.__new__(trainer_cls)
    trainer.config = OmegaConf.create(config_dict)
    trainer.metrics = {}
    return trainer


def _trainer(*, enabled=True, max_response_length=4):
    return _make_trainer(
        {
            "reward": {"reward_kwargs": {"zero_truncated_rl_advantage": enabled}},
            "data": {"max_response_length": max_response_length},
        }
    )


def _adv_batch(row_adv, response_mask, hpt_is_sft):
    advantages = torch.tensor(row_adv, dtype=torch.float32).unsqueeze(-1).expand(-1, response_mask.shape[1]).clone()
    advantages = advantages * response_mask
    tensors = {"advantages": advantages, "response_mask": response_mask}
    if hpt_is_sft is not None:
        tensors["hpt_is_sft"] = torch.tensor(hpt_is_sft)
    return DataProto.from_dict(tensors=tensors)


def _payload(rm_scores):
    return DataProto.from_dict(tensors={"rm_scores": rm_scores})


def _cispo_config(global_batch_size, loss_scale_factor):
    return ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_mini_batch_size=1, ppo_micro_batch_size=1,
        clip_ratio=0.2, clip_ratio_low=10.0, clip_ratio_high=0.28, clip_ratio_c=10.0,
        loss_agg_mode="seq-mean-token-sum-norm", use_kl_loss=False, entropy_coeff=0.0,
        global_batch_info={
            "dp_size": 1, "batch_num_tokens": None,
            "global_batch_size": global_batch_size, "loss_scale_factor": loss_scale_factor,
        },
        policy_loss={"loss_mode": "cispo"},
    )


def _cispo_grad(*, advantages, global_batch_size, loss_scale_factor=8.0):
    """Run the real CISPO loss; return per-token log_prob gradient tensor."""
    n_rows, T = advantages.shape
    response_mask = torch.ones(n_rows, T, dtype=torch.bool)
    log_prob = torch.zeros(n_rows, T, requires_grad=True)
    pg_loss, _ = compute_policy_loss_cispo(
        old_log_prob=torch.zeros(n_rows, T), log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, loss_agg_mode="seq-mean-token-sum-norm",
        config=_cispo_config(global_batch_size, loss_scale_factor),
    )
    pg_loss.backward()
    return log_prob.grad


# ==================================================================================================
# Section A — P0-1 reward gate (truncated => reward 0, raw acc preserved)
# ==================================================================================================
def test_p0_1_truncated_correct_is_zeroed_but_acc_preserved():
    mgr = _make_reward_manager(zero_reward_if_truncated=True, max_resp_len=4, score_result={"score": 1.0, "acc": 1.0})
    out = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))
    assert out["reward_score"] == 0.0
    assert out["reward_extra_info"]["is_truncated"] is True
    assert out["reward_extra_info"]["acc"] == 1.0  # raw correctness preserved for observability


def test_p0_1_truncated_incorrect_stays_zero_and_flagged():
    mgr = _make_reward_manager(zero_reward_if_truncated=True, max_resp_len=4, score_result={"score": 0.0, "acc": 0.0})
    out = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))
    assert out["reward_score"] == 0.0
    assert out["reward_extra_info"]["is_truncated"] is True


def test_p0_1_terminated_response_keeps_reward():
    mgr = _make_reward_manager(zero_reward_if_truncated=True, max_resp_len=4, score_result={"score": 1.0, "acc": 1.0})
    out = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=3))
    assert out["reward_score"] == 1.0
    assert out["reward_extra_info"]["is_truncated"] is False


def test_p0_1_boundary_is_at_the_cap_not_the_tensor_width():
    mgr = _make_reward_manager(zero_reward_if_truncated=True, max_resp_len=4, score_result={"score": 1.0, "acc": 1.0})
    at_cap = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=6, valid_resp=4))
    below = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=6, valid_resp=3))
    assert at_cap["reward_score"] == 0.0 and at_cap["reward_extra_info"]["is_truncated"] is True
    assert below["reward_score"] == 1.0 and below["reward_extra_info"]["is_truncated"] is False


def test_p0_1_max_resp_len_none_falls_back_to_tensor_width():
    mgr = _make_reward_manager(
        zero_reward_if_truncated=True, max_resp_len=None, score_result={"score": 1.0, "acc": 1.0}
    )
    assert _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))["reward_score"] == 0.0
    assert _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=2))["reward_score"] == 1.0


def test_p0_1_handles_scalar_score_result():
    mgr = _make_reward_manager(zero_reward_if_truncated=True, max_resp_len=4, score_result=1.0)
    out = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))
    assert out["reward_score"] == 0.0
    assert out["reward_extra_info"]["acc"] == 1.0
    assert out["reward_extra_info"]["is_truncated"] is True


def test_p0_1_truncation_zeros_after_overlong_penalty():
    overlong = types.SimpleNamespace(enable=True, len=2, penalty_factor=1.0, log=False)
    mgr = _make_reward_manager(
        zero_reward_if_truncated=True, max_resp_len=4, score_result={"score": 1.0, "acc": 1.0}, overlong_cfg=overlong
    )
    assert _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))["reward_score"] == 0.0


def test_p0_1_gate_is_opt_in():
    mgr = _make_reward_manager(zero_reward_if_truncated=False, max_resp_len=4, score_result={"score": 1.0, "acc": 1.0})
    out = _run_single(mgr, _make_reward_item(prompt_len=2, resp_len=4, valid_resp=4))
    assert out["reward_score"] == 1.0
    assert "is_truncated" not in out["reward_extra_info"]


# ==================================================================================================
# Section B — routing score-position fix (reward at terminal token, not last index)
# ==================================================================================================
def test_reward_at_terminal_token_is_counted_regardless_of_position():
    rm = torch.zeros(3, 10)
    rm[0, 5] = 1.0  # clean correct: reward at terminal token 5 (indices 6..9 are padding)
    rm[1, 9] = 1.0  # truncated correct: reward at the last index 9
    assert extract_score_values(_payload(rm), score_key="reward_score") == [1.0, 1.0, 0.0]
    assert count_successful_rollouts(_payload(rm), score_key="reward_score", success_threshold=0.0) == (2, 3)


def test_clean_correct_group_routes_to_rl_not_sft():
    rm = torch.zeros(8, 10)
    rm[3, 4] = 1.0  # single clean-correct rollout, reward at terminal token 4 (not the last index)
    success_count, total = count_successful_rollouts(_payload(rm), score_key="reward_score", success_threshold=0.0)
    assert success_count == 1 and total == 8
    assert success_count / total > GAMMA  # -> RL, not SFT


def test_all_zero_group_scores_zero():
    rm = torch.zeros(8, 10)
    assert count_successful_rollouts(_payload(rm), score_key="reward_score", success_threshold=0.0) == (0, 8)


def test_1d_rm_scores_uses_terminal_reward():
    rm = torch.zeros(10)
    rm[3] = 1.0
    assert extract_score_values(_payload(rm.unsqueeze(0)), score_key="reward_score") == [1.0]


# ==================================================================================================
# Section C — P0-2 truncated-advantage zeroing (after compute_advantage)
# ==================================================================================================
def test_p0_2_zeros_truncated_rl_rows_only_and_reports_rl_denominator():
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    batch = _adv_batch([-0.25, 0.75, 0.9, 0.5], mask, [False, False, True, False])
    tr = _trainer()
    tr._fit_filter_truncated_rl_advantage(batch)
    adv = batch.batch["advantages"]
    assert torch.all(adv[0] == 0.0)  # RL truncated -> zeroed
    assert torch.allclose(adv[1][:3], torch.full((3,), 0.75))  # RL terminated -> untouched
    assert torch.allclose(adv[2][:3], torch.full((3,), 0.9))  # SFT tau -> untouched despite length==cap
    assert torch.all(adv[3] == 0.0)  # RL truncated -> zeroed
    assert tr.metrics["hpt/truncated_rl_rows_zeroed"] == 2
    assert tr.metrics["hpt/truncated_rl_frac"] == pytest.approx(2 / 3)  # 2 of 3 RL rows (SFT excluded)
    assert batch.batch["hpt_is_truncated_rl"].tolist() == [True, False, False, True]


def test_p0_2_all_sft_batch_is_never_touched():
    mask = torch.ones(3, 4, dtype=torch.bool)
    batch = _adv_batch([0.3, 0.3, 0.3], mask, [True, True, True])
    before = batch.batch["advantages"].clone()
    tr = _trainer()
    tr._fit_filter_truncated_rl_advantage(batch)
    assert torch.allclose(batch.batch["advantages"], before)
    assert tr.metrics["hpt/truncated_rl_rows_zeroed"] == 0
    assert tr.metrics["hpt/truncated_rl_frac"] == 0.0  # n_rl == 0 -> guarded division


def test_p0_2_base_rl_without_hpt_key_treats_all_as_rl():
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    batch = _adv_batch([-0.25, 0.75], mask, None)
    tr = _trainer()
    tr._fit_filter_truncated_rl_advantage(batch)
    assert torch.all(batch.batch["advantages"][0] == 0.0)
    assert torch.allclose(batch.batch["advantages"][1][:3], torch.full((3,), 0.75))
    assert tr.metrics["hpt/truncated_rl_frac"] == pytest.approx(1 / 2)


def test_p0_2_no_truncated_rows_is_a_noop():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    batch = _adv_batch([0.75, -0.25], mask, [False, False])
    before = batch.batch["advantages"].clone()
    tr = _trainer()
    tr._fit_filter_truncated_rl_advantage(batch)
    assert torch.allclose(batch.batch["advantages"], before)
    assert tr.metrics["hpt/truncated_rl_rows_zeroed"] == 0


def test_p0_2_boundary_length_at_cap_vs_below():
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    batch = _adv_batch([0.5, 0.5], mask, [False, False])
    _trainer()._fit_filter_truncated_rl_advantage(batch)
    assert torch.all(batch.batch["advantages"][0] == 0.0)  # exactly at cap -> truncated
    assert torch.allclose(batch.batch["advantages"][1][:3], torch.full((3,), 0.5))  # below cap -> kept


def test_p0_2_is_opt_in():
    mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _adv_batch([0.75, -0.25], mask, [False, False])
    before = batch.batch["advantages"].clone()
    _trainer(enabled=False)._fit_filter_truncated_rl_advantage(batch)
    assert torch.allclose(batch.batch["advantages"], before)


def test_p0_2_missing_config_keys_are_safe_noops():
    mask = torch.ones(2, 4, dtype=torch.bool)
    batch = _adv_batch([0.75, -0.25], mask, [False, False])
    before = batch.batch["advantages"].clone()
    _make_trainer({"reward": {}, "data": {"max_response_length": 4}})._fit_filter_truncated_rl_advantage(batch)
    assert torch.allclose(batch.batch["advantages"], before)


def test_p0_2_missing_advantages_key_is_guarded():
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones(2, 4, dtype=torch.bool)})
    _trainer()._fit_filter_truncated_rl_advantage(batch)  # must not raise


def test_p0_2_ordering_invariant_baseline_reflects_truncated_failures():
    """§5.5.2: advantage computed with truncated rows present (as failures) THEN zeroed."""
    resp_len = 4
    token_level_rewards = torch.zeros(4, resp_len)
    token_level_rewards[0, -1] = 1.0
    response_mask = torch.ones(4, resp_len, dtype=torch.bool)
    index = np.array(["g0", "g0", "g0", "g0"], dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask, index=index,
        norm_adv_by_std_in_grpo=False,
    )
    assert advantages[0].mean().item() == pytest.approx(0.75)  # baseline 0.25 reflects 3/4 failures
    assert advantages[1].mean().item() == pytest.approx(-0.25)

    mask = response_mask.clone()
    mask[0, -1] = 0  # clean row terminates early -> survives; the 3 failures are length-truncated
    batch = DataProto.from_dict(
        tensors={"advantages": advantages, "response_mask": mask, "hpt_is_sft": torch.zeros(4, dtype=torch.bool)}
    )
    _trainer(max_response_length=resp_len)._fit_filter_truncated_rl_advantage(batch)
    assert batch.batch["advantages"][0][:3].mean().item() == pytest.approx(0.75)
    assert torch.all(batch.batch["advantages"][1] == 0.0)


# ==================================================================================================
# Section D — advantage-zeroing == physical removal under fixed seq-mean denominator (§5.6)
# ==================================================================================================
def test_zeroing_is_gradient_identical_to_removal_under_fixed_denominator():
    G = 4
    zeroed = _cispo_grad(
        advantages=torch.tensor([[0.75, 0.75], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), global_batch_size=G
    )
    removed = _cispo_grad(advantages=torch.tensor([[0.75, 0.75]]), global_batch_size=G)
    assert torch.allclose(zeroed[0], removed[0])  # surviving clean row: identical gradient
    assert torch.all(zeroed[1:] == 0.0)  # zeroed rows: no gradient at all


def test_live_denominator_would_reintroduce_dilution():
    """Guard: if seq-mean ever normalized by the LIVE non-empty-seq count (global_batch_size=None),
    dead rows would dilute the survivor by the row-count ratio. Fails loudly if that regresses."""
    zeroed = _cispo_grad(advantages=torch.tensor([[0.75, 0.75], [0.0, 0.0], [0.0, 0.0]]), global_batch_size=None)
    removed = _cispo_grad(advantages=torch.tensor([[0.75, 0.75]]), global_batch_size=None)
    assert not torch.allclose(zeroed[0], removed[0])
    assert torch.allclose(zeroed[0] * 3.0, removed[0])  # exactly the 3:1 dilution


# ==================================================================================================
# Section E — END-TO-END pipeline: gate -> route -> advantage -> P0-2 -> CISPO loss
# ==================================================================================================
def _gate_reward(correct, truncated, *, zero_if_truncated):
    raw = 1.0 if correct else 0.0
    return 0.0 if (truncated and zero_if_truncated) else raw


def _build_group(specs, *, zero_if_truncated=True):
    n = len(specs)
    rm = torch.zeros(n, MAX_RESP)
    response_mask = torch.zeros(n, MAX_RESP)
    for i, (correct, truncated) in enumerate(specs):
        valid_len = MAX_RESP if truncated else CLEAN_LEN
        response_mask[i, :valid_len] = 1.0
        rm[i, valid_len - 1] = _gate_reward(correct, truncated, zero_if_truncated=zero_if_truncated)
    return rm, response_mask


def _route(rm_scores):
    success_count, total = count_successful_rollouts(
        _payload(rm_scores), score_key="reward_score", success_threshold=0.0
    )
    return success_count, total, (success_count / total) <= GAMMA


def _legacy_route_success_count(rm_scores):
    """The OLD buggy extraction (rm_scores[-1]) — for the regression contrast only."""
    return int(sum(1 for row in rm_scores if float(row[-1]) > 0.0))


def _full_pipeline(specs):
    rm, response_mask = _build_group(specs)
    success_count, total, route_to_sft = _route(rm)
    out = {"rm": rm, "response_mask": response_mask, "success_count": success_count, "route_to_sft": route_to_sft}
    if route_to_sft:
        return out
    index = np.array(["g0"] * len(specs), dtype=object)
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rm, response_mask=response_mask.bool(), index=index, norm_adv_by_std_in_grpo=False
    )
    batch = DataProto.from_dict(
        tensors={
            "advantages": adv.clone(),
            "response_mask": response_mask,
            "hpt_is_sft": torch.zeros(len(specs), dtype=torch.bool),
        }
    )
    tr = _make_trainer(
        {"reward": {"reward_kwargs": {"zero_truncated_rl_advantage": True}}, "data": {"max_response_length": MAX_RESP}}
    )
    tr._fit_filter_truncated_rl_advantage(batch)
    adv_post = batch.batch["advantages"]
    grad = _cispo_grad(advantages=adv_post, global_batch_size=8, loss_scale_factor=float(MAX_RESP)).abs().sum(dim=-1)
    out.update({"adv_pre": adv, "adv_post": adv_post, "p0_2_metrics": tr.metrics, "grad": grad})
    return out


def test_e2e_clean_correct_group_trains():
    """Linchpin: 1 clean-correct + 3 truncated-correct + 4 wrong. Only the clean-correct carries
    reward>0 after the gate; the pipeline must route RL, give it a positive gradient, and give every
    truncated row exactly zero gradient."""
    specs = [(True, False)] + [(True, True)] * 3 + [(False, True)] * 2 + [(False, False)] * 2
    r = _full_pipeline(specs)
    assert r["success_count"] == 1 and r["route_to_sft"] is False  # clean-correct seen -> RL
    assert _legacy_route_success_count(r["rm"]) == 0  # old rm_scores[-1] would have -> SFT (regression guard)

    row_adv_pre = r["adv_pre"].sum(-1) / r["response_mask"].sum(-1)
    assert row_adv_pre[0].item() == pytest.approx(0.875, abs=1e-4)  # baseline 1/8 -> clean-correct +0.875
    assert row_adv_pre[1].item() == pytest.approx(-0.125, abs=1e-4)

    truncated_idx = [1, 2, 3, 4, 5]
    for i in truncated_idx:
        assert torch.all(r["adv_post"][i] == 0.0)
    assert torch.any(r["adv_post"][0] != 0.0)
    assert r["p0_2_metrics"]["hpt/truncated_rl_rows_zeroed"] == len(truncated_idx)

    assert r["grad"][0].item() > 0.0  # clean-correct learns
    for i in truncated_idx:
        assert r["grad"][i].item() == 0.0  # truncated rollouts contribute no policy gradient


def test_e2e_only_truncated_correct_group_routes_to_sft():
    r = _full_pipeline([(True, True)] * 5 + [(False, True)] * 3)
    assert r["success_count"] == 0 and r["route_to_sft"] is True


def test_e2e_all_incorrect_group_routes_to_sft():
    r = _full_pipeline([(False, False)] * 4 + [(False, True)] * 4)
    assert r["success_count"] == 0 and r["route_to_sft"] is True


def test_e2e_clean_correct_reward_survives_the_gate():
    assert float(_build_group([(True, False)])[0].sum()) == 1.0  # clean-correct preserved
    assert float(_build_group([(True, True)])[0].sum()) == 0.0  # truncated-correct gated to 0


def test_e2e_reproduces_collapse_under_old_routing_and_rescue_under_fix():
    """Realistic mostly-truncated group (15% clean, like the run): old [-1] routing scores 0 -> SFT
    collapse; fixed sum(-1) rescues the clean-correct -> RL."""
    specs = [(True, False)] + [(True, True)] * 4 + [(False, True)] * 3
    rm, _ = _build_group(specs)
    assert _legacy_route_success_count(rm) == 0  # OLD -> pure-SFT collapse
    success_count, _, route_to_sft = _route(rm)
    assert success_count == 1 and route_to_sft is False  # FIXED -> RL
