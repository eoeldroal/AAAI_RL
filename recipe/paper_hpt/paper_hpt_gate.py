# Copyright 2026
#
# Paper-faithful HPT gate (pure, driver-side).
#
# Reproduces the original UPT/HPT routing decision (mix_src/mix_trainer.py
# `select_on_off_ada_balance`) for the switch strategy with
# switch_gate == switch_gate_off (the paper's Qwen setting, gamma == 0):
#
#   For each prompt (uid) with n on-policy rollouts, let P = (#correct) / n.
#     * P <= gamma  -> UNSOLVED: this prompt is routed to SFT (its rollouts are
#                      replaced by one demonstration row; see paper_hpt_routing).
#     * P >  gamma  -> SOLVED:   this prompt stays on-policy RL.
#
# With the EXPLICIT dual-loss (paper_hpt_loss.paper_hpt_dual_loss) the SFT branch
# is pure beta*masked_mean(-logpi); there is NO synthetic reward / advantage
# injection, so the gate only makes the SFT/RL decision. Everything here is pure
# (tensors / numpy), unit-testable on CPU.

from __future__ import annotations

import numpy as np
import torch


def group_success_counts(
    scores: torch.Tensor,
    uids: np.ndarray,
    *,
    success_value: float = 1.0,
) -> dict[str, tuple[int, int]]:
    """Per-uid (num_correct, group_size) from per-row scalar scores.

    Args:
        scores: (bsz,) per-row scalar reward (e.g. token_level_scores.sum(-1)).
        uids:   (bsz,) object array of prompt-group ids (one shared id per prompt).
        success_value: score counted as a success (binary reward => 1.0).
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be rank-1 (bsz,), got shape {tuple(scores.shape)}.")
    if len(uids) != scores.shape[0]:
        raise ValueError(f"uids length {len(uids)} != scores rows {scores.shape[0]}.")
    out: dict[str, tuple[int, int]] = {}
    for uid in np.unique(uids):
        mask = uids == uid
        grp = scores[torch.as_tensor(mask, device=scores.device)]
        num_correct = int((grp == success_value).sum().item())
        out[str(uid)] = (num_correct, int(mask.sum()))
    return out


def is_prompt_sft(num_correct: int, group_size: int, gamma: float) -> bool:
    """HPT eq.10 gate: route to SFT iff P = num_correct/group_size <= gamma.

    gamma == 0 (paper Qwen) => SFT iff the prompt is fully unsolved (0/n).
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    success_prob = num_correct / group_size
    return success_prob <= gamma


def route_prompts(
    scores: torch.Tensor,
    uids: np.ndarray,
    *,
    gamma: float,
    success_value: float = 1.0,
) -> dict[str, bool]:
    """Convenience: uid -> is_sft decision for a whole generated batch."""
    counts = group_success_counts(scores, uids, success_value=success_value)
    return {uid: is_prompt_sft(nc, gs, gamma) for uid, (nc, gs) in counts.items()}
