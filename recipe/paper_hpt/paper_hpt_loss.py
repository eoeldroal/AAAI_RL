# Copyright 2026
#
# Paper-faithful EXPLICIT dual-loss for the isolated HPT reproduction.
#
# Reproduces the original UPT/HPT actor loss (mix_src/mix_actor.py +
# mix_core_alg.py) exactly, as a SEPARATE two-term objective — NOT the async
# fork's reward-injection/self-detach mechanism (which, under seq-mean-token-sum-norm
# aggregation, does not reproduce the paper's SFT normalization; see README §Mechanism):
#
#   L = rl_loss + beta * sft_loss - entropy_coeff * entropy_loss
#     rl_loss  = sum_over_RL_tokens( -A * (pi/pi_old) ) / L        # loss_remove_clip=True
#                                                                   # + loss_remove_token_mean=True (sum / constant L)
#     sft_loss = masked_mean_over_SFT_tokens( -log pi )            # compute_sft_pure_loss
#     entropy  = masked_mean_over_ALL_response_tokens( entropy )   # paper applies to all rows
#
# Rows are split by the per-row `hpt_is_sft` flag set by the gate/routing. The RL
# term uses the row `advantages` (Dr.GRPO, computed upstream on RL rows); the SFT
# term is pure NLL and ignores advantages entirely (matching the paper — SFT rows
# never go through the advantage). beta is read from batch meta (`paper_hpt_beta`),
# L from `config.loss_scale_factor`.
#
# Gradient-scale contract (CRITICAL for the paper speed comparison):
# the original (mix_actor, gradient_accumulation = 64//64 = 1) backwards each
# micro-batch's LOCAL loss unscaled and lets FSDP AVERAGE gradients over the R
# DP ranks, so its per-optimizer-step gradient is
#     (1/R) * sum_{ranks,micros} grad[ sum_RL/L + beta*mean_SFT - c*mean_ent ].
# The modern engine does the same two things (plain per-micro loss.backward(),
# FSDP2 mean-reduce), so this loss must return EXACTLY the local per-micro value
# with NO dp_size multiplier. (The SHARED ppo_loss multiplies by dp_size only
# because its denominator is the GLOBAL all-reduced batch_num_tokens; our
# denominators are local, like the paper's.) A dp_size multiplier here would make
# gradients Rx the paper's -- invisible to Adam's scale invariance but NOT to
# grad-clip@1.0, which would fire at 1/R of the paper's threshold and
# systematically shrink updates, invalidating any learning-speed comparison.
# Optimizer parity is otherwise exact: AdamW(lr 5e-6, betas (0.9,0.999), wd 0.01),
# constant schedule, warmup 0, clip 1.0, ceil(routed/512) optimizer steps per
# iteration. Wired in via engine_workers' `actor.custom_loss_fn` hook, so no
# shared loss path is altered.

from typing import Any

import torch

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_sum
from verl.workers.utils.padding import no_padding_2_padding

_HPT_LOSS_FIELD = "hpt_is_sft"
PAPER_HPT_BETA_KEY = "paper_hpt_beta"


def paper_hpt_dual_loss_core(
    log_prob: torch.Tensor,
    entropy: torch.Tensor | None,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    hpt_is_sft: torch.Tensor,
    *,
    beta: float,
    loss_scale_factor: int,
    entropy_coeff: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure math of the paper dual-loss (all tensors already padded; unit-testable).

    All args are (bsz, resp_len) except `hpt_is_sft` which is (bsz,). Returns the
    scalar policy_loss and a dict of raw scalar tensors for metrics.
    """
    response_mask = response_mask.to(bool)
    hpt_is_sft = hpt_is_sft.to(bool)
    sft_tok = hpt_is_sft.unsqueeze(-1) & response_mask  # SFT demonstration tokens
    rl_tok = (~hpt_is_sft).unsqueeze(-1) & response_mask  # on-policy RL tokens

    # RL: pure -A*ratio (no clip), summed / constant L (Dr.GRPO sum-norm, loss_remove_token_mean).
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    rl_losses = -advantages * ratio
    rl_loss = masked_sum(rl_losses, rl_tok) / loss_scale_factor

    # SFT: beta * masked_mean(-logpi) pooled over ALL SFT tokens (uniform per token).
    sft_nll = -masked_sum(log_prob, sft_tok) / sft_tok.sum().clamp_min(1)
    sft_loss = beta * sft_nll

    pg_loss = rl_loss + sft_loss

    # entropy: masked_mean over ALL response tokens (paper includes SFT rows).
    if entropy is not None:
        entropy_loss = masked_sum(entropy, response_mask) / response_mask.sum().clamp_min(1)
        policy_loss = pg_loss - entropy_coeff * entropy_loss
    else:
        entropy_loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        policy_loss = pg_loss

    # NO dp_size multiplier: FSDP averages gradients over ranks, exactly like the
    # paper's FSDP1 run -- see the gradient-scale contract in the module docstring.

    ppo_kl = masked_sum(-negative_approx_kl, rl_tok) / rl_tok.sum().clamp_min(1)

    # Two metric families (all response_mask-masked, so pad rows contribute NOTHING):
    #  - MEAN metrics: per-micro masked means, averaged over (ranks x micros) by the
    #    engine. Convention-compatible, but micros with few/no real tokens pull the
    #    average toward 0 (pad spreading keeps this small, not exactly zero).
    #  - hpt/*_sum + token counts: SUM-aggregated numerator/denominator pairs. The
    #    engine means SUMs over ranks then sums over micros, so the ratio of two SUM
    #    metrics is EXACT and dilution-proof:
    #      true entropy = hpt/entropy_sum / hpt/response_token_count
    #      true sft NLL = hpt/sft_nll_sum / hpt/sft_response_token_count
    #      true ppo_kl  = hpt/ppo_kl_sum  / hpt/rl_response_token_count
    #    (NB the trainer-level `actor/entropy` logged from the old_log_prob pass is
    #    computed PRE-routing on real rollout rows only — already pad-free.)
    raw = {
        "actor/pg_loss": pg_loss.detach(),
        "actor/rl_loss": rl_loss.detach(),
        "actor/sft_loss": sft_loss.detach(),
        "actor/entropy": entropy_loss.detach(),
        "actor/ppo_kl": ppo_kl.detach(),
        "actor/pg_clipfrac": torch.zeros((), device=log_prob.device),
        "hpt/sft_nll": sft_nll.detach(),
        "hpt/sft_response_token_count": sft_tok.to(torch.float32).sum().detach(),
        "hpt/rl_response_token_count": rl_tok.to(torch.float32).sum().detach(),
        "hpt/response_token_count": response_mask.to(torch.float32).sum().detach(),
        "hpt/entropy_sum": (
            masked_sum(entropy, response_mask).detach()
            if entropy is not None
            else torch.zeros((), device=log_prob.device)
        ),
        "hpt/sft_nll_sum": masked_sum(-log_prob, sft_tok).detach(),
        "hpt/ppo_kl_sum": masked_sum(-negative_approx_kl, rl_tok).detach(),
    }
    return policy_loss, raw


def paper_hpt_dual_loss(config, model_output, data, dp_group=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Explicit paper HPT dual-loss (thin plumbing over paper_hpt_dual_loss_core)."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    beta = float(tu.get_non_tensor_data(data=data, key=PAPER_HPT_BETA_KEY, default=0.3))
    loss_scale_factor = config.loss_scale_factor  # L = max_response_length (constant RL divisor)
    if loss_scale_factor is None:
        raise ValueError(
            "paper_hpt_dual_loss requires actor.loss_scale_factor (= max_response_length) "
            "for the paper's loss_remove_token_mean=True (sum / constant L)."
        )
    if _HPT_LOSS_FIELD not in data:
        raise ValueError("paper_hpt_dual_loss requires the per-row 'hpt_is_sft' flag.")

    data = data.select("response_mask", "old_log_probs", "advantages", _HPT_LOSS_FIELD).to_padded_tensor()
    policy_loss, raw = paper_hpt_dual_loss_core(
        log_prob=log_prob,
        entropy=entropy,
        response_mask=data["response_mask"],
        old_log_prob=data["old_log_probs"],
        advantages=data["advantages"],
        hpt_is_sft=data[_HPT_LOSS_FIELD],
        beta=beta,
        loss_scale_factor=int(loss_scale_factor),
        entropy_coeff=config.entropy_coeff,
    )
    sum_keys = {
        "hpt/sft_response_token_count",
        "hpt/rl_response_token_count",
        "hpt/response_token_count",
        "hpt/entropy_sum",
        "hpt/sft_nll_sum",
        "hpt/ppo_kl_sum",
    }
    metrics = {
        k: Metric(value=v, aggregation=AggregationType.SUM if k in sum_keys else AggregationType.MEAN)
        for k, v in raw.items()
    }
    return policy_loss, metrics
