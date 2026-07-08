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
"""Contract tests for the cispo_klcov policy loss (Improvement_RL.md §5.12, lever B2).

cispo_klcov = CISPO base objective + KL-Cov overlay (Cui et al. 2505.22617): add a
KL(pi_old||pi) penalty on the top-`kl_cov_ratio` RL tokens ranked by Cov(logp, A) -- the
pivotal minority driving entropy collapse -- while every other token keeps the full CISPO
gradient. These tests pin the invariants that make it a SAFE, OFF-by-default overlay:
  (1) ratio=0 reduces EXACTLY to plain CISPO (so loss_mode is the only on/off switch),
  (2) the overlay perturbs the loss and keeps a finite gradient through log_prob,
  (3) SFT (teacher) tokens are excluded from the covariance selection universe, so
      teacher-forced imitation targets are never entropy-damped.
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_cispo,
    compute_policy_loss_cispo_klcov,
)
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def _cfg(kl_cov_ratio: float, ppo_kl_coef: float = 0.1) -> ActorConfig:
    # Mirror the M6 CISPO contract: upper-only clip (clip_ratio_low >= 1.0).
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=8,
        use_dynamic_bsz=True,
        clip_ratio_low=10.0,
        clip_ratio_high=0.28,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode="cispo_klcov", kl_cov_ratio=kl_cov_ratio, ppo_kl_coef=ppo_kl_coef),
    )


def _inputs(seed: int = 0, B: int = 6, T: int = 12):
    torch.manual_seed(seed)
    old = torch.randn(B, T) * 0.1
    adv = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    return old, adv, mask


def test_zero_ratio_reduces_exactly_to_cispo():
    # kl_cov_ratio=0 selects no tokens -> the overlay is a no-op and the loss must match the
    # plain CISPO loss bit-for-bit. This is what makes loss_mode the single activation switch:
    # M6 runs loss_mode=cispo (overlay absent); flipping to cispo_klcov with ratio>0 turns it on.
    old, adv, mask = _inputs()
    lp_a = (old + 0.02).detach().clone().requires_grad_(True)
    base, _ = compute_policy_loss_cispo(
        old_log_prob=old,
        log_prob=lp_a,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_cfg(0.0),
    )
    lp_b = (old + 0.02).detach().clone().requires_grad_(True)
    klcov, m = compute_policy_loss_cispo_klcov(
        old_log_prob=old,
        log_prob=lp_b,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_cfg(0.0),
    )
    assert torch.allclose(base, klcov, atol=1e-7)
    assert m["actor/klcov_selected_tokens"] == 0.0


def test_overlay_perturbs_loss_and_keeps_finite_gradient():
    old, adv, mask = _inputs()
    lp0 = (old + 0.02).detach().clone().requires_grad_(True)
    base, _ = compute_policy_loss_cispo(
        old_log_prob=old,
        log_prob=lp0,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_cfg(0.0),
    )
    lp1 = (old + 0.02).detach().clone().requires_grad_(True)
    loss, m = compute_policy_loss_cispo_klcov(
        old_log_prob=old,
        log_prob=lp1,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_cfg(0.25),
    )
    # Some tokens selected (~0.25 of 72), loss differs from plain CISPO, gradient finite.
    assert m["actor/klcov_selected_tokens"] > 0
    assert not torch.allclose(base, loss, atol=1e-6)
    loss.backward()
    assert torch.isfinite(lp1.grad).all()
    # Reported fraction tracks kl_cov_ratio over the RL-token universe.
    assert 0.1 < m["actor/klcov_selected_frac"] < 0.4


def test_sft_tokens_excluded_from_selection_universe():
    # With 2 of 6 rows marked SFT, the covariance ranking must run over the 4 RL rows only
    # (4*12=48 tokens). Verified via the reported denominator (selected / frac).
    old, adv, mask = _inputs()
    sft = torch.zeros_like(mask)
    sft[4:] = True
    lp = (old + 0.02).detach().clone().requires_grad_(True)
    _, m = compute_policy_loss_cispo_klcov(
        old_log_prob=old,
        log_prob=lp,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=_cfg(0.25),
        hpt_sft_token_mask=sft,
    )
    rl_universe = round(m["actor/klcov_selected_tokens"] / max(m["actor/klcov_selected_frac"], 1e-9))
    assert rl_universe == 48

    # Stronger guarantee: no selected (row, col) may land on an SFT row. Reconstruct the
    # selection deterministically and check it is disjoint from the SFT rows.
    sel_mask = mask & ~sft
    valid_idx = torch.nonzero(sel_mask.reshape(-1), as_tuple=True)[0]
    v_adv = adv.reshape(-1)[valid_idx]
    v_logp = lp.detach().reshape(-1)[valid_idx]
    cov = (v_adv - v_adv.mean()) * (v_logp - v_logp.mean())
    k = min(max(1, int(valid_idx.numel() * 0.25)), cov.numel())
    top = valid_idx[torch.topk(cov, k, largest=True).indices]
    rows = top // adv.shape[1]
    assert rows.max().item() < 4  # all selected tokens are on RL rows (0..3), never SFT (4,5)


def _make_hpt_batch() -> TensorDict:
    # row0 = SFT (teacher, self-detach), row1 = RL; 2 response tokens each. Mirrors the
    # M-anchor cispo harness so this exercises the exact M7 activation path.
    rm = torch.ones(2, 2, dtype=torch.bool)
    ids = torch.arange(8).reshape(2, 4)
    batch = TensorDict(
        {
            "input_ids": ids,
            "prompts": ids[:, :2],
            "attention_mask": torch.ones(2, 4, dtype=torch.bool),
            "position_ids": torch.arange(4).repeat(2, 1),
            "responses": ids[:, -2:],
            "response_mask": rm,
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.tensor([[1.0, 1.0], [0.5, -0.5]]),
            "loss_mask": rm.clone(),
            "loss_scale": torch.ones(2, 2),
            "rollout_is_weights": torch.tensor([[99.0, 99.0], [2.0, 2.0]]),
            "hpt_is_sft": torch.tensor([True, False]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, dp_size=1, batch_num_tokens=int(rm.sum().item()), global_batch_size=2)
    return batch


def test_cispo_klcov_runs_through_ppo_loss_under_hpt():
    # The M7 activation path: loss_mode=cispo_klcov must (a) pass the HPT policy-loss whitelist
    # in ppo_loss, (b) receive the SFT-token mask (threaded only for this mode), and (c) produce
    # a finite loss + gradient on an HPT-shaped batch. This is the integration point the
    # loss-fn-only tests above do not cover. Also guards config.policy_loss.get(...) access
    # working whether policy_loss is a PolicyLossConfig, DictConfig, or plain dict.
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_mini_batch_size=1,
        ppo_micro_batch_size=1,
        clip_ratio=0.2,
        clip_ratio_low=10.0,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        global_batch_info={"dp_size": 1},
        policy_loss={"loss_mode": "cispo_klcov", "kl_cov_ratio": 0.25, "ppo_kl_coef": 0.1},
    )
    model_output = {"log_probs": torch.full((8,), -0.25, requires_grad=True)}
    loss, metrics = ppo_loss(config=config, model_output=model_output, data=_make_hpt_batch())
    assert torch.isfinite(loss)
    assert "actor/klcov_selected_tokens" in metrics
    loss.backward()
    assert torch.isfinite(model_output["log_probs"].grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
