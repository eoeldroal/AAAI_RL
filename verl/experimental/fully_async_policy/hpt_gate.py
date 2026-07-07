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

from numbers import Integral, Real
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from verl.experimental.fully_async_policy.hpt_config import AsyncHptConfig, validate_async_hpt_config
from verl.experimental.fully_async_policy.hpt_payload import HptSftPayload, HptTauStore
from verl.protocol import DataProto


class HptRouteMetadata(BaseModel):
    """Route decision attached to a prompt group before queue insertion."""

    model_config = ConfigDict(extra="forbid")

    is_sft: bool
    prompt_uid: str = Field(min_length=1)
    group_uid: str = Field(min_length=1)
    missing_tau: bool = False
    success_probability: float = Field(ge=0.0, le=1.0)
    success_count: int = Field(ge=0)
    total_count: int = Field(gt=0)
    generated_response_lengths: tuple[int, ...] = Field(default_factory=tuple)
    gamma: float = Field(ge=0.0, le=1.0)
    success_threshold: float
    success_score_key: str = Field(min_length=1)

    @field_validator("prompt_uid", "group_uid", "success_score_key")
    @classmethod
    def _strip_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("generated_response_lengths")
    @classmethod
    def _validate_generated_response_lengths(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            return values
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"generated_response_lengths must contain integers, got {value!r}")
            value = int(value)
            if value < 0:
                raise ValueError(f"generated_response_lengths must be non-negative, got {value}.")
            normalized.append(value)
        return tuple(normalized)


class HptRouteDecision(BaseModel):
    """Validated in-process HPT route decision."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    metadata: HptRouteMetadata
    sft_payload: HptSftPayload | None = None


def build_hpt_rollout_gate(config) -> HptRolloutGate | None:
    """Build the rollouter-side HPT gate from the fully async config."""

    hpt_config = validate_async_hpt_config(config)
    if not hpt_config.enabled:
        return None
    # validate_async_hpt_config guarantees tau_dataset_path is set when enabled.
    assert hpt_config.tau_dataset_path is not None
    tau_store = HptTauStore.from_parquet(hpt_config.tau_dataset_path, messages_key=hpt_config.tau_messages_key)
    return HptRolloutGate(config=hpt_config, tau_store=tau_store)


class HptRolloutGate:
    """Decide whether a generated prompt group should train as RL or tau SFT."""

    def __init__(self, *, config: AsyncHptConfig, tau_store: HptTauStore):
        self.config = config
        self.tau_store = tau_store
        self.num_sft_routed = 0
        self.num_rl_routed = 0
        self.missing_tau_count = 0

    def route_rollout_sample(
        self,
        rollout_sample: Any,
        *,
        source_batch: DataProto,
        generated_batch: DataProto,
        group_uid: str | None = None,
    ) -> HptRouteMetadata:
        """Attach an HPT route decision to a generated rollout sample before queue insertion."""

        if not isinstance(generated_batch, DataProto):
            raise TypeError(f"HPT rollout gate requires generated DataProto, got {type(generated_batch)!r}.")
        group_uid = group_uid or f"uid_{rollout_sample.sample_id}"
        prompt_uid = extract_prompt_uid_from_generation_batch(source_batch)
        if len(generated_batch) != len(source_batch):
            raise ValueError(
                "HPT rollout gate requires generated rows to match source rows: "
                f"source={len(source_batch)} generated={len(generated_batch)}."
            )
        assign_training_group_uid(generated_batch, group_uid)
        ensure_generated_prompt_uid(generated_batch, prompt_uid)

        decision = self.route(generated_batch, group_uid=group_uid)
        rollout_sample.hpt_route = decision.metadata
        if decision.metadata.is_sft:
            rollout_sample.full_batch = decision.sft_payload
            self.num_sft_routed += 1
        else:
            rollout_sample.full_batch = generated_batch
            self.num_rl_routed += 1
            if decision.metadata.missing_tau:
                self.missing_tau_count += 1
        return decision.metadata

    def route(self, payload: DataProto, *, group_uid: str | None = None) -> HptRouteDecision:
        prompt_uid = extract_prompt_uid(payload)
        group_uid = group_uid or extract_group_uid(payload, default=prompt_uid)
        success_count, total_count = count_successful_rollouts(
            payload,
            score_key=self.config.success_score_key,
            success_threshold=self.config.success_threshold,
        )
        success_probability = success_count / total_count

        tau_payload = self.tau_store.get(prompt_uid)
        missing_tau = tau_payload is None
        route_to_sft = success_probability <= self.config.gamma and not missing_tau
        if success_probability <= self.config.gamma and missing_tau and self.config.fail_on_missing_tau:
            raise ValueError(f"HPT selected SFT for prompt_uid={prompt_uid!r}, but no tau payload exists.")

        metadata = HptRouteMetadata(
            is_sft=route_to_sft,
            prompt_uid=prompt_uid,
            group_uid=group_uid,
            missing_tau=missing_tau,
            success_probability=success_probability,
            success_count=success_count,
            total_count=total_count,
            generated_response_lengths=extract_generated_response_lengths(payload),
            gamma=self.config.gamma,
            success_threshold=self.config.success_threshold,
            success_score_key=self.config.success_score_key,
        )
        return HptRouteDecision(metadata=metadata, sft_payload=tau_payload if route_to_sft else None)

    def statistics(self) -> dict[str, int]:
        return {
            "hpt/num_sft_routed": self.num_sft_routed,
            "hpt/num_rl_routed": self.num_rl_routed,
            "hpt/missing_tau_count": self.missing_tau_count,
        }


def assign_training_group_uid(payload: DataProto, group_uid: str) -> None:
    """Assign the runtime training group uid used by GRPO/PPO grouping."""

    group_uid = _coerce_string(group_uid, "group_uid")
    payload.non_tensor_batch["uid"] = np.array([group_uid] * len(payload), dtype=object)


def ensure_generated_prompt_uid(payload: DataProto, prompt_uid: str) -> None:
    """Ensure routed RL payload keeps the source prompt_uid for HPT assembly."""

    prompt_uid = _coerce_string(prompt_uid, "prompt_uid")
    _set_or_validate_repeated_non_tensor_value(
        payload.non_tensor_batch,
        key="prompt_uid",
        value=prompt_uid,
        expected_len=len(payload),
    )


def extract_prompt_uid_from_generation_batch(batch: DataProto) -> str:
    """Read the source prompt_uid before rollout output overwrites runtime uid."""

    if not isinstance(batch, DataProto):
        raise TypeError(f"source_batch must be DataProto, got {type(batch)!r}")
    if "prompt_uid" not in batch.non_tensor_batch:
        raise ValueError("HPT routing requires source batch non_tensor_batch['prompt_uid'].")
    values = _as_list(batch.non_tensor_batch["prompt_uid"])
    prompt_uids = {_coerce_string(value, "prompt_uid") for value in values}
    if len(prompt_uids) != 1:
        raise ValueError(f"HPT routing expects one prompt_uid per rollout sample, got {sorted(prompt_uids)!r}.")
    return next(iter(prompt_uids))


def extract_prompt_uid(payload: DataProto) -> str:
    return _extract_single_string(payload, "prompt_uid")


def extract_group_uid(payload: DataProto, *, default: str) -> str:
    if "uid" not in payload.non_tensor_batch:
        return default
    return _extract_single_string(payload, "uid")


def count_successful_rollouts(
    payload: DataProto,
    *,
    score_key: str,
    success_threshold: float,
) -> tuple[int, int]:
    scores = extract_score_values(payload, score_key=score_key)
    total_count = len(scores)
    if total_count <= 0:
        raise ValueError("HPT routing requires at least one rollout score.")
    success_count = sum(1 for score in scores if score > success_threshold)
    return success_count, total_count


def extract_score_values(payload: DataProto, *, score_key: str) -> list[float]:
    if score_key in payload.non_tensor_batch:
        return [_coerce_score(value, score_key) for value in _as_list(payload.non_tensor_batch[score_key])]
    if payload.batch is not None and score_key in payload.batch:
        tensor = payload.batch[score_key]
        if tensor.dim() == 0:
            return [_coerce_score(tensor.item(), score_key)]
        return [_coerce_score(value, score_key) for value in tensor.detach().cpu().reshape(-1).tolist()]
    if score_key == "reward_score" and payload.batch is not None and "rm_scores" in payload.batch:
        # rm_scores carries the outcome reward at the TERMINAL response token
        # (valid_response_length - 1), NOT at the last tensor index — every position after the
        # terminal token is right-padding (0). Reduce with sum(-1), the same sequence-reward
        # reduction verl uses everywhere (core_algos GRPO `token_level_rewards.sum(-1)`,
        # metric_utils `token_level_scores.sum(-1)`).
        #
        # BUGFIX (Improvement_RL.md §6): the previous `[-1]` read the LAST index, so it only saw
        # the reward for responses that filled the entire budget (terminal == last index, i.e.
        # truncated/non-terminating). Every early-terminating (clean) rollout was silently scored 0
        # and its group routed to SFT. Combined with zero_reward_if_truncated (which zeros the
        # truncated rewards that `[-1]` *could* see), this drove on-policy success to exactly 0 and
        # collapsed the run to pure SFT. sum(-1) sees the terminal reward regardless of position.
        rm_scores = payload.batch["rm_scores"]
        if rm_scores.dim() == 1:
            return [_coerce_score(rm_scores.sum().item(), "rm_scores")]
        return [_coerce_score(row.sum().item(), "rm_scores") for row in rm_scores.detach().cpu()]
    raise ValueError(f"HPT routing could not find success score key {score_key!r} in rollout payload.")


def extract_generated_response_lengths(payload: DataProto) -> tuple[int, ...]:
    if payload.batch is None or "response_mask" not in payload.batch:
        raise ValueError("HPT routing requires generated payload batch['response_mask'] for response length metrics.")
    response_mask = payload.batch["response_mask"]
    if response_mask.dim() != 2:
        raise ValueError(f"HPT routing response_mask must be rank 2, got rank {response_mask.dim()}.")
    lengths = response_mask.sum(dim=-1).detach().cpu().tolist()
    return tuple(int(length) for length in lengths)


def _extract_single_string(payload: DataProto, key: str) -> str:
    if key not in payload.non_tensor_batch:
        raise ValueError(f"HPT routing requires non_tensor_batch[{key!r}].")
    values = _as_list(payload.non_tensor_batch[key])
    unique_values = {_coerce_string(value, key) for value in values}
    if len(unique_values) != 1:
        raise ValueError(f"HPT routing requires one unique {key}, got {sorted(unique_values)!r}.")
    return next(iter(unique_values))


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _set_or_validate_repeated_non_tensor_value(
    non_tensor_batch: dict[str, np.ndarray],
    *,
    key: str,
    value: str,
    expected_len: int,
) -> None:
    existing = non_tensor_batch.get(key)
    if existing is None:
        non_tensor_batch[key] = np.array([value] * expected_len, dtype=object)
        return
    observed = [_coerce_string(item, key) for item in _as_list(existing)]
    if len(observed) != expected_len:
        raise ValueError(f"{key} length {len(observed)} does not match expected length {expected_len}.")
    if any(item != value for item in observed):
        raise ValueError(f"{key} values {sorted(set(observed))} do not match expected value {value!r}.")


def _coerce_string(value: Any, key: str) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError(f"HPT routing expects {key} values to be strings, got {value!r}.")
    value = value.strip()
    if not value:
        raise ValueError(f"HPT routing expects non-empty {key} values.")
    return value


def _coerce_score(value: Any, key: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"HPT routing expects numeric {key} values, got {value!r}.")
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"HPT routing expects finite {key} values, got {value!r}.")
    return score
