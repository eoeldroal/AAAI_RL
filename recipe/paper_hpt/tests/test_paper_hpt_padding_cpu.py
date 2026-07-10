# Copyright 2026
#
# TRUE-padding inertness proofs for the sync-HPT recipe.
#
# The original mix_actor hard-DROPS its `whether_pad` rows before the loss; we
# must instead mask them (modern make_iterator can't take ragged minis). These
# tests prove the masking is EXACTLY equivalent to dropping on every path a pad
# row touches: the dual loss (RL / SFT / entropy / ppo_kl, numerators AND
# denominators), GRPO advantage, and the engine's response-slice math. Pad rows
# are seeded with pathological values (huge log-probs / advantages / entropy) so
# any leak would be loud.

import numpy as np
import pytest
import torch

from recipe.paper_hpt import paper_hpt_routing as routing
from recipe.paper_hpt.paper_hpt_loss import paper_hpt_dual_loss_core
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

P, R = 4, 6  # prompt / response widths


def _rl_batch(scores, uids, puids):
    n = len(uids)
    prompts = torch.arange(n * P).reshape(n, P) + 1
    responses = torch.arange(n * R).reshape(n, R) + 100
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.ones(n, P + R, dtype=torch.long)
    attention_mask[:, 0] = 0  # left-padded prompts: first prompt col invalid
    position_ids = torch.arange(P + R).unsqueeze(0).repeat(n, 1)
    return DataProto.from_single_dict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": torch.ones(n, R),
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
            "old_log_probs": torch.randn(n, R),
            "uid": np.array(uids, dtype=object),
            "prompt_uid": np.array(puids, dtype=object),
        }
    )


def _routed(pad_to_multiple=None, pad_spread=False):
    # p0 solved (2 RL rows) + p1 unsolved (1 SFT row from tau) = 3 real rows
    b = _rl_batch(
        [[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ["p0", "p0", "p1", "p1"],
        ["u0", "u0", "u1", "u1"],
    )
    return routing.route_generated_batch_synchronous(
        b, {"u1": [7, 8, 9]}, gamma=0.0, pad_id=0,
        pad_to_multiple=pad_to_multiple, pad_spread=pad_spread,
    )


# --------------------------------------------------------------------------- #
# pad-row construction hygiene
# --------------------------------------------------------------------------- #
def test_pad_row_hygiene():
    routed, m = _routed(pad_to_multiple=8)  # 3 real -> 8 total, appended (no spread)
    assert m["hpt/pad_rows"] == 5.0 and len(routed) == 8
    pads = routed.select_idxs(list(range(3, 8))).batch
    # attention: EXACTLY one valid token, at the last prompt position
    assert pads["attention_mask"].sum().item() == 5
    assert bool((pads["attention_mask"][:, P - 1] == 1).all())
    # response fully inert: mask 0, tokens = pad_id, input_ids response region = pad_id
    assert pads["response_mask"].abs().sum().item() == 0
    assert bool((pads["responses"] == 0).all())
    assert bool((pads["input_ids"][:, P:] == 0).all())
    # reward / log-prob zeroed; not SFT; dummy uid
    assert pads["token_level_scores"].abs().sum().item() == 0
    assert pads["old_log_probs"].abs().sum().item() == 0
    assert not bool(pads["hpt_is_sft"].any())
    assert all(str(u).startswith("__pad_") for u in routed.non_tensor_batch["uid"][3:])
    # position_ids recomputed for the 1-token mask (all zeros after clip)
    assert pads["position_ids"].abs().sum().item() == 0


# --------------------------------------------------------------------------- #
# loss: bit-level inertness even with pathological pad-row values
# --------------------------------------------------------------------------- #
def _loss(log_prob, entropy, response_mask, old_log_prob, advantages, hpt_is_sft):
    return paper_hpt_dual_loss_core(
        log_prob=log_prob, entropy=entropy, response_mask=response_mask,
        old_log_prob=old_log_prob, advantages=advantages, hpt_is_sft=hpt_is_sft,
        beta=0.3, loss_scale_factor=8192, entropy_coeff=0.001,
    )


def test_loss_exactly_invariant_to_pads():
    torch.manual_seed(0)
    n_real, n_pad = 3, 5
    log_prob = torch.randn(n_real, R)
    entropy = torch.rand(n_real, R)
    response_mask = torch.ones(n_real, R)
    old_log_prob = torch.randn(n_real, R)
    advantages = torch.randn(n_real, R)
    hpt_is_sft = torch.tensor([False, False, True])

    loss_a, raw_a = _loss(log_prob, entropy, response_mask, old_log_prob, advantages, hpt_is_sft)

    # append pads with PATHOLOGICAL values but response_mask=0 (the only guard)
    def pad_rows(base, fill):
        return torch.cat([base, torch.full((n_pad, R), fill)], dim=0)

    loss_b, raw_b = _loss(
        pad_rows(log_prob, 1e3),
        pad_rows(entropy, 1e5),
        torch.cat([response_mask, torch.zeros(n_pad, R)], dim=0),
        pad_rows(old_log_prob, -1e3),
        pad_rows(advantages, 1e6),
        torch.cat([hpt_is_sft, torch.zeros(n_pad, dtype=torch.bool)]),
    )

    assert torch.allclose(loss_a, loss_b, atol=1e-10, rtol=0), f"{loss_a} != {loss_b}"
    for k in raw_a:
        assert torch.allclose(raw_a[k], raw_b[k], atol=1e-10, rtol=0), (
            f"metric {k} leaked pad contribution: {raw_a[k]} != {raw_b[k]}"
        )


def test_all_pad_micro_batch_is_finite_zero():
    # An all-pad micro-batch (possible per-rank slice) must give loss 0, no NaN.
    n = 4
    loss, raw = _loss(
        torch.full((n, R), 123.0), torch.full((n, R), 9.0), torch.zeros(n, R),
        torch.full((n, R), -55.0), torch.full((n, R), 7.0),
        torch.zeros(n, dtype=torch.bool),
    )
    assert torch.isfinite(loss) and loss.item() == 0.0
    for k, v in raw.items():
        assert torch.isfinite(v), f"{k} not finite on all-pad micro"


# --------------------------------------------------------------------------- #
# advantage: pads exactly zero, real advantages exactly unchanged
# --------------------------------------------------------------------------- #
def test_pads_zero_advantage_and_real_unchanged():
    real, _ = _routed(pad_to_multiple=None)
    padded, _ = _routed(pad_to_multiple=8)

    def adv(b):
        return compute_grpo_outcome_advantage(
            token_level_rewards=b.batch["token_level_scores"],
            response_mask=b.batch["response_mask"],
            index=b.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=False,
        )[0]

    a_real, a_padded = adv(real), adv(padded)
    assert torch.equal(a_real, a_padded[:3])  # real rows bit-identical
    assert a_padded[3:].abs().sum().item() == 0.0  # pads exactly zero


# --------------------------------------------------------------------------- #
# spread: even real-row density over rank chunks, order preserved
# --------------------------------------------------------------------------- #
def test_spread_indices_even_density_and_order():
    src = routing._spread_indices(55, 512)
    assert sorted(src) == list(range(512))
    real_pos = [t for t, s in enumerate(src) if s < 55]
    # order of real rows preserved
    assert [src[t] for t in real_pos] == list(range(55))
    # each of the 8 rank chunks (64 rows) gets 6-7 real rows
    for r in range(8):
        cnt = sum(1 for t in real_pos if r * 64 <= t < (r + 1) * 64)
        assert cnt in (6, 7), f"rank {r} got {cnt} real rows"


def test_route_spread_end_to_end():
    routed, m = _routed(pad_to_multiple=8, pad_spread=True)
    assert len(routed) == 8 and m["hpt/pad_rows"] == 5.0
    uids = [str(u) for u in routed.non_tensor_batch["uid"]]
    real_uids = [u for u in uids if not u.startswith("__pad_")]
    assert real_uids == ["p0", "p0", "p1"]  # all real rows present, order kept
    # both halves (2 "rank chunks" of 4) contain at least one real row
    for half in (uids[:4], uids[4:]):
        assert any(not u.startswith("__pad_") for u in half)
    # pads inert after the permutation too
    pad_idx = [i for i, u in enumerate(uids) if u.startswith("__pad_")]
    pads = routed.select_idxs(pad_idx).batch
    assert pads["response_mask"].abs().sum().item() == 0
    assert pads["attention_mask"].sum().item() == len(pad_idx)
    # real-batch observability metrics computed PRE-pad
    assert m["hpt/real_rows"] == 3.0
    assert m["hpt/real_response_length_mean"] == pytest.approx((6 + 6 + 3) / 3)


# --------------------------------------------------------------------------- #
# core trainer metrics: the fork's compute_data_metrics demands the async-HPT
# instrumentation fields whenever `hpt_is_sft` is in the batch. Run the REAL
# function end-to-end on a routed+padded batch (this is the exact call that
# crashed the first launch) and prove pads don't distort the group metrics.
# --------------------------------------------------------------------------- #
def _with_advantages(batch):
    adv, ret = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_scores"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=False,
    )
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()
    batch.batch["advantages"] = adv
    batch.batch["returns"] = ret
    return batch


def test_compute_data_metrics_end_to_end_with_pads():
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    m_real = compute_data_metrics(batch=_with_advantages(_routed()[0]), use_critic=False)
    m_pad = compute_data_metrics(
        batch=_with_advantages(_routed(pad_to_multiple=8, pad_spread=True)[0]), use_critic=False
    )

    # the instrumentation contract is satisfied and group-deduped: p0 = 1/2, p1 = 0/2
    for m in (m_real, m_pad):
        assert m["hpt/onpolicy_success_rate"] == pytest.approx(0.25)
        assert m["hpt/onpolicy_num_groups"] == 2.0
    # pad rows have response_length 0 -> counted "aborted" and EXCLUDED from
    # score/reward means; group dedup absorbs pad clones -> identical readings
    for k in ("critic/score/mean", "critic/rewards/mean", "critic/advantages/mean"):
        assert m_pad[k] == pytest.approx(m_real[k]), k


def test_routing_sets_instrumentation_fields():
    routed, _ = _routed(pad_to_multiple=8, pad_spread=True)
    sp = routed.non_tensor_batch["hpt_success_probability"]
    gid = routed.non_tensor_batch["hpt_group_uid"]
    assert len(sp) == len(gid) == len(routed)
    by_gid = {}
    for g, p in zip(gid, sp, strict=False):
        by_gid.setdefault(str(g), set()).add(float(p))
    # constant within group; pads inherited a REAL group id (never __pad_)
    assert all(len(v) == 1 for v in by_gid.values())
    assert set(by_gid) == {"p0", "p1"}
    assert by_gid["p0"] == {0.5} and by_gid["p1"] == {0.0}


# --------------------------------------------------------------------------- #
# engine response-slice math: a 1-valid-token pad row yields an all-zero,
# correctly shaped log_prob row (the exact contract of no_padding_2_padding)
# --------------------------------------------------------------------------- #
def test_response_slice_math_with_pad_rows():
    from tensordict import TensorDict

    from verl.workers.utils.padding import no_padding_2_padding

    routed, _ = _routed(pad_to_multiple=4)  # 3 real + 1 pad
    b = routed.batch
    attn = b["attention_mask"]
    prompt_lens = attn[:, :P].sum(dim=1)
    response_lens = attn[:, P:].sum(dim=1)
    assert response_lens[-1].item() == 0 and prompt_lens[-1].item() == 1  # the pad row

    # fake per-valid-token model output: value = global token index + 1 (nonzero)
    total = int((prompt_lens + response_lens).sum().item())
    values = torch.arange(1, total + 1, dtype=torch.float32)

    data = TensorDict(
        {"prompts": b["prompts"], "responses": b["responses"], "attention_mask": attn},
        batch_size=[len(routed)],
    )
    out = no_padding_2_padding(values, data)  # must not trip its asserts
    assert out.shape == (len(routed), R)
    assert out[-1].abs().sum().item() == 0.0  # pad row: all zeros
    # real rows: left-shifted slice of their own token span, then right-padded
    offsets = (prompt_lens + response_lens).cumsum(0)
    for i in range(3):
        rl = int(response_lens[i].item())
        expect = values[offsets[i] - rl - 1 : offsets[i] - 1]
        assert torch.equal(out[i, :rl], expect)
        assert out[i, rl:].abs().sum().item() == 0.0
