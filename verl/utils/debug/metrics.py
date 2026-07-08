# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def _rollout_diff_stats(
    actor_probs: torch.Tensor, rollout_probs: torch.Tensor, mask_bool: torch.Tensor
) -> tuple[int, float, float, float, float]:
    """Return (valid, max, mean, std, pearson) of |actor_prob - rollout_prob| over mask_bool.

    valid=0 (and NaN stats) when the mask selects no token — the caller records this rather
    than crashing on an empty reduction.
    """
    if not mask_bool.any():
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    pearson = pearson_correlation_coefficient(actor_probs, rollout_probs, mask_bool)
    diff = calculate_log_prob_diff(actor_probs, rollout_probs, mask_bool)
    return (
        1,
        torch.max(diff).detach().item(),
        torch.mean(diff).detach().item(),
        torch.std(diff).detach().item(),
        pearson,
    )


def calculate_debug_metrics(data: DataProto, exclude_rows: torch.Tensor | None = None) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
        exclude_rows: optional bool tensor of shape (batch,). Rows marked True are dropped
            from an ADDITIONAL "_rl" set of metrics. This exists for HPT: SFT rows carry a
            placeholder rollout_log_probs (zeros) — they never went through the rollout engine —
            so including them makes the diff |actor_prob - exp(0)| meaningless. Pass the SFT-row
            mask (async_hpt hpt_is_sft) to get a clean rollout-vs-actor precision gauge over the
            RL rows only. The original (all-row) keys are still emitted unchanged for back-compat
            and cross-run comparability.
    Returns:
        dict: metrics. The base keys are over ALL response tokens (unchanged behavior):
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
        When exclude_rows is given, the same five statistics are ALSO emitted over the kept
        (RL) rows under "_rl"-suffixed keys, plus the excluded-row count:
            "training/rollout_probs_diff_rl_{valid,max,mean,std}",
            "training/rollout_actor_probs_pearson_corr_rl",
            "training/rollout_probs_diff_sft_rows_excluded"
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    valid, diff_max, diff_mean, diff_std, pearson_corrcoef = _rollout_diff_stats(
        actor_probs, rollout_probs, response_mask_bool
    )
    metrics = {
        "training/rollout_probs_diff_valid": valid,
        "training/rollout_probs_diff_max": diff_max,
        "training/rollout_probs_diff_mean": diff_mean,
        "training/rollout_probs_diff_std": diff_std,
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }

    if exclude_rows is not None:
        # Drop excluded rows (e.g. HPT SFT rows with placeholder rollout logprobs) to isolate
        # the genuine rollout-engine-vs-trainer precision mismatch on RL rows only.
        keep_rows = (~exclude_rows.bool()).to(device=response_mask_bool.device).view(-1, 1)
        rl_mask_bool = response_mask_bool & keep_rows
        rl_valid, rl_max, rl_mean, rl_std, rl_pearson = _rollout_diff_stats(
            actor_probs, rollout_probs, rl_mask_bool
        )
        metrics.update(
            {
                "training/rollout_probs_diff_rl_valid": rl_valid,
                "training/rollout_probs_diff_rl_max": rl_max,
                "training/rollout_probs_diff_rl_mean": rl_mean,
                "training/rollout_probs_diff_rl_std": rl_std,
                "training/rollout_actor_probs_pearson_corr_rl": rl_pearson,
                "training/rollout_probs_diff_sft_rows_excluded": int(exclude_rows.bool().sum().item()),
            }
        )

    return metrics
