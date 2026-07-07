# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_entropy_clip_diagnostics,
    compute_value_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding

_HPT_LOSS_FIELD = "hpt_is_sft"
_HPT_TRUNCATED_RL_FIELD = "hpt_is_truncated_rl"
_OBSOLETE_HPT_LOSS_FIELDS = ("hpt_seq_weight", "hpt_length_divisor", "hpt_loss_denominator")


def _has_hpt_loss_fields(data) -> bool:
    obsolete = [field for field in _OBSOLETE_HPT_LOSS_FIELDS if field in data]
    if obsolete:
        raise ValueError(f"HPT branch-blind policy loss no longer accepts obsolete fields: {obsolete}.")
    return _HPT_LOSS_FIELD in data


def _hpt_row_mask(
    hpt_row_field: torch.Tensor, response_mask: torch.Tensor, log_prob: torch.Tensor, *, field_name: str
) -> torch.Tensor:
    batch_size = response_mask.shape[0]
    if tuple(hpt_row_field.shape) != (batch_size,):
        raise ValueError(f"HPT field {field_name!r} must have shape ({batch_size},), got {tuple(hpt_row_field.shape)}.")
    return hpt_row_field.to(device=log_prob.device, dtype=torch.bool).unsqueeze(-1)


def _hpt_sft_mask(hpt_is_sft: torch.Tensor, response_mask: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    return _hpt_row_mask(hpt_is_sft, response_mask, log_prob, field_name=_HPT_LOSS_FIELD)


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    hpt_policy_loss = _has_hpt_loss_fields(data)
    hpt_sft_entropy_enabled = bool(tu.get_non_tensor_data(data, "hpt_sft_entropy_enabled", False))
    hpt_sft_kl_enabled = bool(tu.get_non_tensor_data(data, "hpt_sft_kl_enabled", False))

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    hpt_has_truncated_rl_field = hpt_policy_loss and _HPT_TRUNCATED_RL_FIELD in data
    if hpt_policy_loss:
        fields.append(_HPT_LOSS_FIELD)
    if hpt_has_truncated_rl_field:
        fields.append(_HPT_TRUNCATED_RL_FIELD)
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    hpt_sft_token_mask = None
    hpt_truncated_rl_token_mask = None
    if hpt_policy_loss:
        if loss_mode not in {"cispo", "vanilla"}:
            raise ValueError(
                "HPT branch-blind policy loss supports only vanilla or cispo as the base policy loss mode."
            )
        hpt_sft_token_mask = _hpt_sft_mask(data["hpt_is_sft"], response_mask, log_prob) & response_mask
        if hpt_has_truncated_rl_field:
            hpt_truncated_rl_token_mask = (
                _hpt_row_mask(
                    data[_HPT_TRUNCATED_RL_FIELD],
                    response_mask,
                    log_prob,
                    field_name=_HPT_TRUNCATED_RL_FIELD,
                )
                & response_mask
            )
        old_log_prob = torch.where(hpt_sft_token_mask, log_prob.detach(), old_log_prob)
        if rollout_is_weights is not None:
            rollout_is_weights = torch.where(
                hpt_sft_token_mask, torch.ones_like(rollout_is_weights), rollout_is_weights
            )

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    if hpt_sft_token_mask is not None:
        pg_metrics["hpt/sft_response_token_count"] = Metric(
            value=hpt_sft_token_mask.to(dtype=torch.float32).sum().detach(),
            aggregation=AggregationType.SUM,
        )
        if hpt_sft_token_mask.any():
            sft_nll = masked_sum(-log_prob, hpt_sft_token_mask) / hpt_sft_token_mask.sum().clamp_min(1)
        else:
            sft_nll = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
        pg_metrics["hpt/sft_nll"] = Metric(value=sft_nll.detach(), aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_mask = response_mask
        if hpt_sft_token_mask is not None and not hpt_sft_entropy_enabled:
            entropy_mask = response_mask & ~hpt_sft_token_mask
        if hpt_truncated_rl_token_mask is not None:
            entropy_mask = entropy_mask & ~hpt_truncated_rl_token_mask
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=entropy_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

        # Entropy-resolved clip diagnostics over RL (rollout-provenance) tokens only.
        # De-confounds the sum-normed entropy_loss (which rises with the RL-token count) and
        # tests whether the low aggregate clip fraction concentrates on the high-entropy
        # pivotal minority. Pure analysis metrics: detached, never affect policy_loss. Emitted
        # as token-weighted SUM components (always present, 0 on all-SFT microbatches) so they
        # reduce correctly across dynamic-batch microbatches / DP ranks with uniform value
        # counts; finalize_entropy_clip_diagnostics recovers the ratios post-reduction. SFT
        # tokens are always excluded regardless of sft_entropy_enabled (their ratio is 1 by
        # self-detach, and they carry no rollout provenance to clip).
        rl_diag_mask = response_mask
        if hpt_sft_token_mask is not None:
            rl_diag_mask = response_mask & ~hpt_sft_token_mask
        if hpt_truncated_rl_token_mask is not None:
            rl_diag_mask = rl_diag_mask & ~hpt_truncated_rl_token_mask
        _clip_ratio = config.clip_ratio
        _clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else _clip_ratio
        _clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else _clip_ratio
        entropy_clip_diag = compute_entropy_clip_diagnostics(
            entropy=entropy,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            rl_mask=rl_diag_mask,
            cliprange_low=_clip_low,
            cliprange_high=_clip_high,
        )
        metrics.update(Metric.from_dict(entropy_clip_diag, aggregation=AggregationType.SUM))

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_mask = response_mask
        if hpt_sft_token_mask is not None and not hpt_sft_kl_enabled:
            kl_mask = response_mask & ~hpt_sft_token_mask
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
