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

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding

_HPT_LOSS_FIELDS = ("hpt_is_sft", "hpt_seq_weight", "hpt_length_divisor", "hpt_loss_denominator")


def _has_hpt_loss_fields(data) -> bool:
    present = [field in data for field in _HPT_LOSS_FIELDS]
    if any(present) and not all(present):
        missing = [field for field, exists in zip(_HPT_LOSS_FIELDS, present, strict=True) if not exists]
        raise ValueError(f"HPT policy loss batch is missing fields: {missing}.")
    return all(present)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}.")
        return float(value.item())
    return float(value)


def _compute_vanilla_token_losses(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: ActorConfig,
    rollout_is_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    if clip_ratio_c <= 1.0:
        raise ValueError(f"clip_ratio_c must be greater than 1.0, got {clip_ratio_c}.")

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
    token_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        token_losses = token_losses * rollout_is_weights

    return token_losses, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }


def _compute_hpt_prompt_equal_policy_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_weights: torch.Tensor | None,
    hpt_is_sft: torch.Tensor,
    hpt_seq_weight: torch.Tensor,
    hpt_length_divisor: torch.Tensor,
    hpt_loss_denominator: torch.Tensor,
    config: ActorConfig,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("HPT policy loss currently supports only vanilla as the base policy loss mode.")

    batch_size = response_mask.shape[0]
    for name, tensor in {
        "hpt_is_sft": hpt_is_sft,
        "hpt_seq_weight": hpt_seq_weight,
        "hpt_length_divisor": hpt_length_divisor,
        "hpt_loss_denominator": hpt_loss_denominator,
    }.items():
        if tuple(tensor.shape) != (batch_size,):
            raise ValueError(f"HPT field {name!r} must have shape ({batch_size},), got {tuple(tensor.shape)}.")

    hpt_is_sft = hpt_is_sft.to(device=log_prob.device, dtype=torch.bool)
    hpt_seq_weight = hpt_seq_weight.to(device=log_prob.device, dtype=log_prob.dtype)
    hpt_length_divisor = hpt_length_divisor.to(device=log_prob.device, dtype=log_prob.dtype)
    hpt_loss_denominator = hpt_loss_denominator.to(device=log_prob.device, dtype=log_prob.dtype)

    if (hpt_length_divisor <= 0).any():
        raise ValueError("HPT length divisors must be positive.")
    if (hpt_loss_denominator <= 0).any():
        raise ValueError("HPT loss denominators must be positive.")

    denominator = hpt_loss_denominator.max()
    if not torch.allclose(hpt_loss_denominator, torch.full_like(hpt_loss_denominator, denominator)):
        raise ValueError("HPT loss denominator must be identical across rows in an actor microbatch.")

    sft_mask = hpt_is_sft.unsqueeze(-1)
    effective_old_log_prob = torch.where(sft_mask, log_prob.detach(), old_log_prob)
    if rollout_is_weights is not None:
        rollout_is_weights = torch.where(sft_mask, torch.ones_like(rollout_is_weights), rollout_is_weights)

    token_losses, pg_metrics = _compute_vanilla_token_losses(
        old_log_prob=effective_old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    response_mask_float = response_mask.to(dtype=token_losses.dtype)
    seq_losses = (token_losses * response_mask_float).sum(dim=-1) / hpt_length_divisor
    dp_size = _as_float(config.global_batch_info.get("dp_size", 1))
    weighted_seq_losses = hpt_seq_weight * seq_losses
    sft_loss_component = weighted_seq_losses.masked_select(hpt_is_sft).sum() / denominator * dp_size
    rl_loss_component = weighted_seq_losses.masked_select(~hpt_is_sft).sum() / denominator * dp_size
    pg_loss = sft_loss_component + rl_loss_component

    metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    metrics["hpt/b_eff"] = Metric(value=denominator.detach(), aggregation=AggregationType.MEAN)
    metrics["hpt/sft_loss_component"] = Metric(value=sft_loss_component.detach(), aggregation=AggregationType.SUM)
    metrics["hpt/rl_loss_component"] = Metric(value=rl_loss_component.detach(), aggregation=AggregationType.SUM)

    sft_response_mask = sft_mask & response_mask
    metrics["hpt/sft_response_token_count"] = Metric(
        value=sft_response_mask.to(dtype=torch.float32).sum().detach(),
        aggregation=AggregationType.SUM,
    )
    if sft_response_mask.any():
        sft_nll = masked_sum(-log_prob, sft_response_mask) / sft_response_mask.sum().clamp_min(1)
    else:
        sft_nll = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
    metrics["hpt/sft_nll"] = Metric(value=sft_nll.detach(), aggregation=AggregationType.MEAN)
    return pg_loss, metrics


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

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if hpt_policy_loss:
        fields.extend(_HPT_LOSS_FIELDS)
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

    if hpt_policy_loss:
        pg_loss, pg_metrics = _compute_hpt_prompt_equal_policy_loss(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            rollout_is_weights=rollout_is_weights,
            hpt_is_sft=data["hpt_is_sft"],
            hpt_seq_weight=data["hpt_seq_weight"],
            hpt_length_divisor=data["hpt_length_divisor"],
            hpt_loss_denominator=data["hpt_loss_denominator"],
            config=config,
        )
    else:
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

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
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
