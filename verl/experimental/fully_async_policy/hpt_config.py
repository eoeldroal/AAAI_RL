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

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


SUPPORTED_HPT_BASE_POLICY_LOSS_MODES = frozenset({"vanilla"})


class AsyncHptTrajectorySchedulerConfig(BaseModel):
    """Validated in-process view of trajectory-attempt scheduling options."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class AsyncHptConfig(BaseModel):
    """Validated in-process view of the async HPT config block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    gamma: float = Field(default=0.0, ge=0.0, le=1.0)
    beta: float = Field(default=1.0, ge=0.0)
    alpha: float = Field(default=1.0, ge=0.0)
    loss_aggregation: Literal["branch_blind"] = "branch_blind"
    sft_beta_mode: Literal["constant", "length_inverse"] = "constant"
    sft_entropy_enabled: bool = False
    sft_kl_enabled: bool = False
    rl_old_logprob_source: Literal["rollout"] = "rollout"
    tau_dataset_path: str | None = None
    tau_messages_key: str = "tau_messages"
    success_score_key: str = "reward_score"
    success_threshold: float = 0.0
    k_max: int | None = Field(default=None, ge=0)
    fail_on_missing_tau: bool = False
    trajectory_scheduler: AsyncHptTrajectorySchedulerConfig = Field(
        default_factory=AsyncHptTrajectorySchedulerConfig
    )

    @field_validator("tau_dataset_path")
    @classmethod
    def _strip_empty_tau_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("tau_messages_key", "success_score_key")
    @classmethod
    def _strip_required_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("config key must not be empty")
        return value


def load_async_hpt_config(config: DictConfig) -> AsyncHptConfig:
    """Load and validate the local async_hpt block without cross-config checks."""

    raw_config = OmegaConf.select(config, "async_hpt", default={})
    if isinstance(raw_config, DictConfig):
        payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    else:
        payload = raw_config
    if payload is None:
        payload = {}

    try:
        return AsyncHptConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid async_hpt config: {exc}") from exc


def validate_async_hpt_config(config: DictConfig) -> AsyncHptConfig:
    """Validate the phase-1 async HPT config contract.

    This function is intentionally limited to config-level invariants. Runtime
    routing, assembly, and loss behavior live in separate modules so the base
    fully-async RL path remains unchanged when async_hpt.enabled=false.
    """

    hpt_config = load_async_hpt_config(config)
    if hpt_config.trajectory_scheduler.enabled and not hpt_config.enabled:
        raise ValueError("async_hpt.trajectory_scheduler.enabled requires async_hpt.enabled=true")
    if not hpt_config.enabled:
        return hpt_config

    if hpt_config.tau_dataset_path is None:
        raise ValueError("async_hpt.tau_dataset_path must be set when async_hpt.enabled=true")

    if hpt_config.alpha != 1.0:
        raise ValueError("async_hpt.alpha is deprecated by branch-blind reduction and must remain 1.0.")

    if hpt_config.sft_beta_mode == "length_inverse":
        loss_scale_factor = OmegaConf.select(config, "actor_rollout_ref.actor.loss_scale_factor", default=None)
        if (
            loss_scale_factor is None
            or isinstance(loss_scale_factor, bool)
            or not isinstance(loss_scale_factor, int)
            or loss_scale_factor <= 0
        ):
            raise ValueError(
                "async_hpt.sft_beta_mode=length_inverse requires "
                "actor_rollout_ref.actor.loss_scale_factor to be a positive integer."
            )

    adv_estimator = OmegaConf.select(config, "algorithm.adv_estimator", default=None)
    if adv_estimator != "grpo":
        raise ValueError(
            "async_hpt.enabled=true currently requires algorithm.adv_estimator=grpo; "
            f"got {adv_estimator!r}"
        )

    norm_adv_by_std = OmegaConf.select(config, "algorithm.norm_adv_by_std_in_grpo", default=True)
    if norm_adv_by_std is not False:
        raise ValueError(
            "async_hpt.enabled=true currently requires algorithm.norm_adv_by_std_in_grpo=False "
            "for the Dr.GRPO-style phase-1 contract"
        )

    loss_mode = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla")
    if loss_mode not in SUPPORTED_HPT_BASE_POLICY_LOSS_MODES:
        supported = ", ".join(sorted(SUPPORTED_HPT_BASE_POLICY_LOSS_MODES))
        raise ValueError(
            "async_hpt.enabled=true currently supports "
            f"actor_rollout_ref.actor.policy_loss.loss_mode in {{{supported}}}; got {loss_mode!r}"
        )

    calculate_log_probs = OmegaConf.select(config, "actor_rollout_ref.rollout.calculate_log_probs", default=False)
    if calculate_log_probs is not True:
        raise ValueError(
            "async_hpt.enabled=true requires actor_rollout_ref.rollout.calculate_log_probs=True "
            "so RL rows can use rollout logprobs as the old policy anchor"
        )

    if hpt_config.trajectory_scheduler.enabled:
        rollout_mode = OmegaConf.select(config, "actor_rollout_ref.rollout.mode", default=None)
        if rollout_mode != "async":
            raise ValueError(
                "async_hpt.trajectory_scheduler.enabled requires "
                f"actor_rollout_ref.rollout.mode=async; got {rollout_mode!r}"
            )
        rollout_n = int(OmegaConf.select(config, "actor_rollout_ref.rollout.n", default=0))
        if rollout_n <= 1:
            raise ValueError(
                "async_hpt.trajectory_scheduler.enabled requires actor_rollout_ref.rollout.n > 1; "
                f"got {rollout_n}"
            )

    return hpt_config
