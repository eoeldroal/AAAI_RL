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

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto


def is_hpt_training_batch(batch: DataProto) -> bool:
    return bool(getattr(batch, "batch", None) is not None and "hpt_is_sft" in batch.batch)


def should_use_hpt_rollout_logprob_anchor(config: Any, batch: DataProto) -> bool:
    if not is_hpt_training_batch(batch):
        return False
    if not bool(OmegaConf.select(config, "async_hpt.enabled", default=False)):
        return False
    source = OmegaConf.select(config, "async_hpt.rl_old_logprob_source", default="rollout")
    if source != "rollout":
        raise ValueError(f"HPT v1 requires async_hpt.rl_old_logprob_source=rollout, got {source!r}.")
    return True


def apply_hpt_rollout_logprob_anchor(batch: DataProto) -> dict[str, float]:
    if "rollout_log_probs" not in batch.batch:
        raise ValueError("HPT rollout old-logprob source requires rollout_log_probs in the training batch.")
    if "response_mask" not in batch.batch:
        raise ValueError("HPT rollout old-logprob source requires response_mask in the training batch.")

    rollout_log_probs = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    if tuple(rollout_log_probs.shape) != tuple(response_mask.shape):
        raise ValueError(
            "HPT rollout_log_probs shape must match response_mask shape: "
            f"{tuple(rollout_log_probs.shape)} != {tuple(response_mask.shape)}."
        )

    batch.batch["old_log_probs"] = rollout_log_probs.clone().to(torch.float32)
    return {
        "hpt/old_logprob_from_rollout": 1.0,
        "hpt/num_sft_rows": float(batch.batch["hpt_is_sft"].to(torch.bool).sum().item()),
    }


def collect_hpt_batch_monitoring_metrics(batch: DataProto) -> dict[str, float]:
    if not is_hpt_training_batch(batch):
        return {}

    hpt_is_sft = batch.batch["hpt_is_sft"]
    if hpt_is_sft.dim() != 1:
        raise ValueError(f"HPT monitoring expects hpt_is_sft to be rank 1, got rank {hpt_is_sft.dim()}.")
    batch_size = int(hpt_is_sft.shape[0])
    if batch_size <= 0:
        raise ValueError("HPT monitoring received an empty training batch.")

    active_mask = _active_hpt_row_mask(batch, batch_size)
    if not active_mask.any():
        raise ValueError("HPT monitoring received a batch with no active HPT rows.")

    group_uids = _require_hpt_non_tensor(batch, "hpt_group_uid", batch_size)
    route_is_sft = _require_hpt_non_tensor(batch, "hpt_route_is_sft", batch_size)
    missing_tau = _require_hpt_non_tensor(batch, "hpt_missing_tau", batch_size)
    success_probability = _require_hpt_non_tensor(batch, "hpt_success_probability", batch_size)

    tensor_is_sft = hpt_is_sft.detach().cpu().to(torch.bool).tolist()
    groups: dict[str, dict[str, Any]] = {}
    for row_idx, active in enumerate(active_mask.tolist()):
        if not active:
            continue
        group_uid = _coerce_group_uid(group_uids[row_idx], row_idx)
        row_route_is_sft = _coerce_bool(route_is_sft[row_idx], "hpt_route_is_sft", row_idx)
        row_missing_tau = _coerce_bool(missing_tau[row_idx], "hpt_missing_tau", row_idx)
        row_success_probability = _coerce_probability(success_probability[row_idx], row_idx)
        if bool(tensor_is_sft[row_idx]) != row_route_is_sft:
            raise ValueError(
                "HPT monitoring found inconsistent SFT route metadata at row "
                f"{row_idx}: hpt_is_sft={bool(tensor_is_sft[row_idx])} "
                f"hpt_route_is_sft={row_route_is_sft}."
            )

        row_record = {
            "is_sft": row_route_is_sft,
            "missing_tau": row_missing_tau,
            "success_probability": row_success_probability,
            "rows": 1,
        }
        if group_uid not in groups:
            groups[group_uid] = row_record
            continue

        existing = groups[group_uid]
        for key in ("is_sft", "missing_tau", "success_probability"):
            if existing[key] != row_record[key]:
                raise ValueError(
                    f"HPT monitoring found inconsistent {key} metadata for group {group_uid!r}: "
                    f"{existing[key]!r} != {row_record[key]!r}."
                )
        existing["rows"] += 1

    num_groups = len(groups)
    if num_groups <= 0:
        raise ValueError("HPT monitoring found no active HPT groups.")

    num_sft = sum(1 for group in groups.values() if group["is_sft"])
    num_rl_groups = num_groups - num_sft
    missing_tau_count = sum(1 for group in groups.values() if group["missing_tau"])
    p_success_zero_count = sum(1 for group in groups.values() if group["success_probability"] == 0.0)

    return {
        "hpt/offline_data_ratio": float(num_sft / num_groups),
        "hpt/p_success_zero_ratio": float(p_success_zero_count / num_groups),
        "hpt/num_sft": float(num_sft),
        "hpt/num_rl_groups": float(num_rl_groups),
        "hpt/missing_tau_count": float(missing_tau_count),
    }


def filter_hpt_stale_rollout_samples(
    rollout_samples: list[Any],
    *,
    config: Any,
    current_param_version: int | None,
) -> tuple[list[Any], dict[str, int]]:
    if not bool(OmegaConf.select(config, "async_hpt.enabled", default=False)):
        return rollout_samples, {"hpt/rl_stale_dropped": 0, "hpt/sft_staleness_exempt": 0}

    k_max = OmegaConf.select(config, "async_hpt.k_max", default=None)
    if k_max is None:
        return rollout_samples, {"hpt/rl_stale_dropped": 0, "hpt/sft_staleness_exempt": 0}
    if current_param_version is None:
        raise ValueError("async_hpt.k_max requires current_param_version for branch-aware staleness filtering.")
    k_max = int(k_max)
    if k_max < 0:
        raise ValueError(f"async_hpt.k_max must be non-negative, got {k_max}.")

    kept = []
    stale_dropped = 0
    sft_exempt = 0
    for rollout_sample in rollout_samples:
        route = _require_hpt_route(rollout_sample)
        if route.is_sft:
            sft_exempt += 1
            kept.append(rollout_sample)
            continue

        max_generation_step = _extract_max_global_step(rollout_sample.full_batch)
        if int(current_param_version) - max_generation_step > k_max:
            stale_dropped += 1
            continue
        kept.append(rollout_sample)

    return kept, {"hpt/rl_stale_dropped": stale_dropped, "hpt/sft_staleness_exempt": sft_exempt}


def _active_hpt_row_mask(batch: DataProto, batch_size: int) -> torch.Tensor:
    obsolete = [
        field for field in ("hpt_seq_weight", "hpt_length_divisor", "hpt_loss_denominator") if field in batch.batch
    ]
    if obsolete:
        raise ValueError(f"HPT monitoring no longer accepts obsolete HPT loss fields: {obsolete}.")
    return torch.ones(batch_size, dtype=torch.bool)


def _require_hpt_non_tensor(batch: DataProto, key: str, batch_size: int) -> list[Any]:
    if key not in batch.non_tensor_batch:
        raise ValueError(f"HPT monitoring requires non_tensor_batch[{key!r}].")
    values = batch.non_tensor_batch[key]
    if len(values) != batch_size:
        raise ValueError(
            f"HPT monitoring non_tensor_batch[{key!r}] length must match batch size: {len(values)} != {batch_size}."
        )
    return values.tolist() if isinstance(values, np.ndarray) else list(values)


def _coerce_group_uid(value: Any, row_idx: int) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str) or not value:
        raise ValueError(f"HPT monitoring requires a non-empty hpt_group_uid at row {row_idx}, got {value!r}.")
    return value


def _coerce_bool(value: Any, key: str, row_idx: int) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, bool):
        raise ValueError(f"HPT monitoring requires boolean {key} at row {row_idx}, got {value!r}.")
    return value


def _coerce_probability(value: Any, row_idx: int) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"HPT monitoring requires numeric hpt_success_probability at row {row_idx}, got {value!r}.")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"HPT monitoring hpt_success_probability must be in [0, 1], got {probability!r} at row {row_idx}."
        )
    return probability


def _require_hpt_route(rollout_sample: Any):
    route = getattr(rollout_sample, "hpt_route", None)
    if route is None:
        raise ValueError(f"RolloutSample {getattr(rollout_sample, 'sample_id', '<unknown>')!r} has no HPT route.")
    if hasattr(route, "is_sft"):
        return route
    from verl.experimental.fully_async_policy.hpt_gate import HptRouteMetadata

    return HptRouteMetadata.model_validate(route)


def _extract_max_global_step(payload: Any) -> int:
    non_tensor_batch = payload.non_tensor_batch if isinstance(payload, DataProto) else None
    if non_tensor_batch is None or "max_global_steps" not in non_tensor_batch:
        raise ValueError("HPT RL staleness filtering requires max_global_steps in RL payload metadata.")

    raw_values = non_tensor_batch["max_global_steps"]
    values = raw_values.tolist() if isinstance(raw_values, np.ndarray) else list(raw_values)
    if not values:
        raise ValueError("HPT RL staleness filtering received empty max_global_steps metadata.")
    return max(_coerce_int_step(value) for value in values)


def _coerce_int_step(value: Any) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"HPT generation step metadata must be integer, got {value!r}.")
    return int(value)
