# Copyright 2026
#
# Dual-loss core tests (CPU). This is the heart of the paper reproduction: the
# explicit two-term loss (RL sum/L + beta*masked_mean SFT - entropy). Tests check
# it against the paper's LITERAL formulas (mix_core_alg compute_sft_pure_loss /
# compute_policy_loss) and pin the multi-row SFT uniformity that the previous
# reward-injection (length_inverse) design got wrong.
#
# Run:
#   CUDA_VISIBLE_DEVICES="" /home/sogang_nlpy/miniconda3/envs/RL/bin/python \
#       -m pytest -q recipe/paper_hpt/tests/test_paper_hpt_loss_cpu.py

import pytest
import torch

from recipe.paper_hpt.paper_hpt_loss import paper_hpt_dual_loss_core


def _masked_mean(x, m):
    m = m.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1)


def _paper_reference(log_prob, entropy, mask, old, adv, is_sft, *, beta, L, ec):
    """The paper's literal per-microbatch policy loss (mix_actor + mix_core_alg)."""
    sft_tok = is_sft.bool().unsqueeze(-1) & mask.bool()
    rl_tok = (~is_sft.bool()).unsqueeze(-1) & mask.bool()
    ratio = torch.exp(torch.clamp(log_prob - old, -20, 20))
    # compute_policy_loss(loss_remove_clip=True, loss_remove_token_mean=True): sum / L
    rl = (-adv * ratio * rl_tok).sum() / L
    # compute_sft_pure_loss = masked_mean(-logpi) over SFT tokens
    sft = beta * (-(log_prob * sft_tok).sum() / sft_tok.sum().clamp_min(1))
    ent = _masked_mean(entropy, mask)
    return (rl + sft - ec * ent)


def _rand_inputs(bsz, T, seed=0, all_masked=True):
    torch.manual_seed(seed)
    log_prob = torch.randn(bsz, T, requires_grad=True)
    entropy = torch.rand(bsz, T)
    old = torch.randn(bsz, T)
    adv = torch.randn(bsz, T)
    mask = torch.ones(bsz, T) if all_masked else (torch.rand(bsz, T) > 0.3).float()
    return log_prob, entropy, old, adv, mask


# --------------------------------------------------------------------------- #
# matches the paper's literal formulas
# --------------------------------------------------------------------------- #
def test_core_matches_paper_reference_mixed_batch():
    log_prob, entropy, old, adv, mask = _rand_inputs(4, 5, seed=1)
    is_sft = torch.tensor([False, True, False, True])
    beta, L, ec = 0.3, 8, 0.001
    loss, raw = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, old, adv, is_sft,
        beta=beta, loss_scale_factor=L, entropy_coeff=ec,
    )
    ref = _paper_reference(log_prob, entropy, mask, old, adv, is_sft, beta=beta, L=L, ec=ec)
    assert torch.allclose(loss, ref, atol=1e-6)


def test_core_matches_paper_with_partial_masks():
    log_prob, entropy, old, adv, mask = _rand_inputs(6, 7, seed=2, all_masked=False)
    is_sft = torch.tensor([True, False, True, False, True, False])
    beta, L, ec = 0.5, 16, 0.01
    loss, _ = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, old, adv, is_sft,
        beta=beta, loss_scale_factor=L, entropy_coeff=ec,
    )
    ref = _paper_reference(log_prob, entropy, mask, old, adv, is_sft, beta=beta, L=L, ec=ec)
    assert torch.allclose(loss, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# RL branch: no-clip, sum / constant L
# --------------------------------------------------------------------------- #
def test_rl_branch_is_unclipped_sum_over_L():
    # single RL row, ratio far from 1 (clip would trigger); no SFT.
    old = torch.zeros(1, 3)
    log_prob = torch.full((1, 3), torch.log(torch.tensor(4.0)).item())  # ratio 4
    adv = torch.full((1, 3), 2.0)
    mask = torch.ones(1, 3)
    is_sft = torch.tensor([False])
    loss, raw = paper_hpt_dual_loss_core(
        log_prob, None, mask, old, adv, is_sft,
        beta=0.3, loss_scale_factor=10, entropy_coeff=0.0,
    )
    # sum(-A*ratio)/L = 3 * (-2*4) / 10 = -24/10 = -2.4  (no clipping)
    assert loss.item() == pytest.approx(-2.4, rel=1e-5)
    assert raw["actor/pg_clipfrac"].item() == 0.0
    assert raw["actor/sft_loss"].item() == 0.0


# --------------------------------------------------------------------------- #
# ★ SFT branch: uniform per-token across rows of different length (the fix)
# --------------------------------------------------------------------------- #
def test_sft_gradient_is_uniform_across_different_length_rows():
    # 2 SFT rows: row0 N=2, row1 N=4. Paper masked_mean => uniform -beta/T per token.
    log_prob = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    is_sft = torch.tensor([True, True])
    beta, L = 0.3, 8
    loss, _ = paper_hpt_dual_loss_core(
        log_prob, None, mask, torch.zeros(2, 4), torch.zeros(2, 4), is_sft,
        beta=beta, loss_scale_factor=L, entropy_coeff=0.0,
    )
    (g,) = torch.autograd.grad(loss, log_prob)
    T = 6  # total SFT tokens
    # every masked token has the SAME gradient -beta/T (uniform), UNLIKE length_inverse
    assert torch.allclose(g[0, 0], g[1, 0], atol=1e-8)
    assert g[0, 0].item() == pytest.approx(-beta / T, rel=1e-6)
    assert g[1, 0].item() == pytest.approx(-beta / T, rel=1e-6)
    # masked-out tokens get zero gradient
    assert g[0, 2].item() == 0.0


def test_sft_branch_equals_beta_times_mean_nll():
    log_prob = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=False)
    mask = torch.ones(1, 3)
    is_sft = torch.tensor([True])
    loss, raw = paper_hpt_dual_loss_core(
        log_prob, None, mask, torch.zeros(1, 3), torch.zeros(1, 3), is_sft,
        beta=0.5, loss_scale_factor=8, entropy_coeff=0.0,
    )
    # sft_nll = mean(-logp) = mean(1,2,3) = 2 ; loss = beta*2 = 1.0
    assert raw["hpt/sft_nll"].item() == pytest.approx(2.0)
    assert loss.item() == pytest.approx(0.5 * 2.0)


# --------------------------------------------------------------------------- #
# entropy over ALL rows (incl SFT), gradient-scale contract, RL:SFT balance
# --------------------------------------------------------------------------- #
def test_entropy_covers_all_rows_including_sft():
    log_prob = torch.zeros(2, 2)
    entropy = torch.tensor([[1.0, 1.0], [3.0, 3.0]])  # row0 RL, row1 SFT
    mask = torch.ones(2, 2)
    is_sft = torch.tensor([False, True])
    _, raw = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, torch.zeros(2, 2), torch.zeros(2, 2), is_sft,
        beta=0.3, loss_scale_factor=8, entropy_coeff=0.001,
    )
    # entropy mean over ALL 4 tokens incl the SFT row = mean(1,1,3,3) = 2.0
    assert raw["actor/entropy"].item() == pytest.approx(2.0)


def test_no_dp_size_multiplier_contract():
    # Gradient-scale parity with the paper: FSDP averages gradients over ranks in
    # BOTH frameworks, so the loss must be the LOCAL per-micro value. A dp_size
    # multiplier would make gradients Rx the paper's and fire grad-clip@1.0 at 1/R
    # of the paper's threshold. Pin the contract: the core takes no dp_size.
    log_prob, entropy, old, adv, mask = _rand_inputs(4, 5, seed=3)
    is_sft = torch.tensor([False, True, False, True])
    kw = dict(beta=0.3, loss_scale_factor=8, entropy_coeff=0.001)
    with pytest.raises(TypeError):
        paper_hpt_dual_loss_core(log_prob, entropy, mask, old, adv, is_sft, dp_size=4, **kw)
    loss, _ = paper_hpt_dual_loss_core(log_prob, entropy, mask, old, adv, is_sft, **kw)
    ref = _paper_reference(log_prob, entropy, mask, old, adv, is_sft, beta=0.3, L=8, ec=0.001)
    assert torch.allclose(loss, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_pure_rl_batch_no_sft_rows():
    log_prob, entropy, old, adv, mask = _rand_inputs(3, 4, seed=4)
    is_sft = torch.tensor([False, False, False])
    loss, raw = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, old, adv, is_sft,
        beta=0.3, loss_scale_factor=8, entropy_coeff=0.001,
    )
    assert raw["actor/sft_loss"].item() == 0.0
    assert raw["hpt/sft_response_token_count"].item() == 0.0
    assert torch.isfinite(loss)


def test_pure_sft_batch_no_rl_rows():
    log_prob, entropy, old, adv, mask = _rand_inputs(3, 4, seed=5)
    is_sft = torch.tensor([True, True, True])
    loss, raw = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, old, adv, is_sft,
        beta=0.3, loss_scale_factor=8, entropy_coeff=0.0,
    )
    assert raw["actor/rl_loss"].item() == 0.0
    assert raw["hpt/rl_response_token_count"].item() == 0.0
    assert torch.isfinite(loss)


def test_token_counts_reported():
    log_prob, entropy, old, adv, mask = _rand_inputs(4, 5, seed=6)
    is_sft = torch.tensor([True, False, True, False])
    _, raw = paper_hpt_dual_loss_core(
        log_prob, entropy, mask, old, adv, is_sft,
        beta=0.3, loss_scale_factor=8, entropy_coeff=0.001,
    )
    assert raw["hpt/sft_response_token_count"].item() == 10.0  # 2 sft rows * 5 tokens
    assert raw["hpt/rl_response_token_count"].item() == 10.0
