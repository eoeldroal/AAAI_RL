# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import math
import os
import time
from datetime import datetime
from typing import Any

import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.hpt_assembler import HptBatchAssembler
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.training_dump import TrainingTensorDumper, load_training_dump_config
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import _compute_metric_response_length, _compute_response_info
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self._init_dump_executor()
        self.validation_generations_logger = None
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.hpt_assembler = None
        # Off-path loss-boundary tensor dump for offline ablation analysis
        # (docs/Ablation_RL.md). No-op unless training_dump.enable=true.
        self._training_dumper = TrainingTensorDumper(load_training_dump_config(config))

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        hpt_enabled = bool(self.config.get("async_hpt", {}).get("enabled", False))
        consumer_start = time.time()
        # HPT row-alignment can leave a small residue of already-materialized groups that do not
        # fit the trainable multiple. Carry them into the next batch (seeded here, consumed first,
        # bounded one-step deferral) instead of discarding trained data or over-collecting to chase
        # exact alignment. Non-HPT collection has no carryover. See _plan_row_alignment_deferral.
        carryover = list(getattr(self, "_hpt_carryover_samples", [])) if hpt_enabled else []
        if hpt_enabled:
            self._hpt_carryover_samples = []
        serialized_queue_samples = list(carryover)
        num_carryover = len(carryover)
        queue_len = 0
        while len(serialized_queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(serialized_queue_samples)}/{self.required_samples} samples"
                )
                break

            serialized_queue_samples.append(sample)

            if len(serialized_queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(serialized_queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        if not serialized_queue_samples or len(serialized_queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None

        # Assemble batch - now working directly with RolloutSample objects
        if hpt_enabled:
            if self.hpt_assembler is None:
                self.hpt_assembler = HptBatchAssembler(config=self.config, tokenizer=self.tokenizer)
            required_multiple = self._hpt_required_training_multiple()
            max_queue_samples = self._hpt_max_queue_samples_for_trainable_batch(required_multiple)
            queue_samples = []
            materialized_batches = []
            learner_rows = 0
            for serialized_sample in serialized_queue_samples:
                queue_sample = ray.cloudpickle.loads(serialized_sample)
                queue_samples.append(queue_sample)
                materialized_batch = self.hpt_assembler.materialize_training_batch(queue_sample)
                materialized_batches.append(materialized_batch)
                learner_rows += len(materialized_batch)

            # Grow ONLY until we have at least one trainable multiple; never grow past it to chase
            # exact alignment. Growing-to-align over-collects 2-3x and can fail to converge when the
            # residue mod rollout_n is unmovable by the arriving groups -- the crash that took the
            # bounded run down. Beyond one multiple we TRIM (below), not grow.
            while required_multiple > 1 and learner_rows < required_multiple:
                if len(serialized_queue_samples) >= max_queue_samples:
                    raise ValueError(
                        "HPT learner-row-aware collection could not reach one trainable multiple within the "
                        "bounded queue window: "
                        f"learner_rows={learner_rows} required_multiple={required_multiple} "
                        f"queue_samples={len(serialized_queue_samples)} max_queue_samples={max_queue_samples}."
                    )
                sample, queue_len = await self.message_queue_client.get_sample()
                if sample is None:
                    print(
                        "[FullyAsyncTrainer] HPT learner-row-aware collection received termination before a "
                        "trainable batch was formed. "
                        f"learner_rows={learner_rows} required_multiple={required_multiple} "
                        f"queue_samples={len(serialized_queue_samples)}"
                    )
                    return None, None
                serialized_queue_samples.append(sample)
                queue_sample = ray.cloudpickle.loads(sample)
                queue_samples.append(queue_sample)
                materialized_batch = self.hpt_assembler.materialize_training_batch(queue_sample)
                materialized_batches.append(materialized_batch)
                learner_rows += len(materialized_batch)

            # Trim DOWN to the largest aligned batch by DEFERRING a residue subset of groups to the
            # next step (carryover). Prefer to defer freshly pulled groups (protected_prefix skips
            # carried-over groups) so carryover always trains within one step. No trained data is
            # discarded; the batch stays bounded near required_samples and cannot fail to align.
            total_collected_groups = len(materialized_batches)
            row_counts = [len(mb) for mb in materialized_batches]
            defer_indices = self._plan_row_alignment_deferral(
                row_counts, required_multiple, protected_prefix=num_carryover
            )
            if defer_indices is None:
                # Fresh groups alone cannot absorb the residue; allow deferring carryover groups too.
                defer_indices = self._plan_row_alignment_deferral(row_counts, required_multiple, protected_prefix=0)
            if defer_indices is None:
                raise ValueError(
                    "HPT learner-row-aware collection could not trim to an aligned batch: "
                    f"learner_rows={learner_rows} required_multiple={required_multiple} "
                    f"total_rows={sum(row_counts)} (no subset of collected groups sums to the residue)."
                )
            deferred_rows = sum(row_counts[i] for i in defer_indices)
            if defer_indices:
                self._hpt_carryover_samples = [serialized_queue_samples[i] for i in sorted(defer_indices)]
                keep = [i for i in range(total_collected_groups) if i not in defer_indices]
                queue_samples = [queue_samples[i] for i in keep]
                materialized_batches = [materialized_batches[i] for i in keep]
                learner_rows -= deferred_rows

            batch = self.hpt_assembler.concat_training_batches(materialized_batches)
            consumer_end = time.time()
            total_wait_time = consumer_end - consumer_start
            print(
                f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} "
                f"samples, learner_rows={learner_rows}, required_multiple={required_multiple}, "
                f"carryover_in={num_carryover}, deferred={len(defer_indices)} groups/{deferred_rows} rows, "
                f"total wait time: {total_wait_time:.2f} seconds. "
                f"mq_len: {queue_len}"
            )
            # Every collection unit stays explicit and reconcilable (AGENTS.md training contract):
            #   fresh_pulled == collected_queue_samples(retained) + carryover_out - carryover_in
            batch.meta_info["fully_async/hpt_collected_queue_samples"] = len(queue_samples)
            batch.meta_info["fully_async/hpt_required_training_multiple"] = required_multiple
            batch.meta_info["fully_async/hpt_carryover_in_groups"] = num_carryover
            batch.meta_info["fully_async/hpt_carryover_out_groups"] = len(defer_indices)
            batch.meta_info["fully_async/hpt_row_alignment_deferred_rows"] = deferred_rows
            batch.meta_info["fully_async/hpt_fresh_pulled_groups"] = total_collected_groups - num_carryover
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics={})
            self._add_hpt_async_sample_meta(batch)
            if "attention_mask" in batch.batch:
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        else:
            consumer_end = time.time()
            total_wait_time = consumer_end - consumer_start
            print(
                f"[FullyAsyncTrainer] Loop collection completed: "
                f"{len(serialized_queue_samples)}/{self.required_samples} samples, "
                f"total wait time: {total_wait_time:.2f} seconds. "
                f"mq_len: {queue_len}"
            )
            queue_samples = [ray.cloudpickle.loads(x) for x in serialized_queue_samples]
        if not hpt_enabled and self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        elif not hpt_enabled:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/mq_len"] = queue_len  # observability: message-queue depth (was console-only)
        return 0, batch

    def _hpt_max_queue_samples_for_trainable_batch(self, required_multiple: int) -> int:
        """Bound HPT row-aware queue reads using the async completed-sample budget."""

        default_window = max(self.required_samples, required_multiple) * 2
        async_training = self.config.get("async_training", {})
        completed_budget = async_training.get("max_completed_prompt_groups", None)
        if completed_budget is None:
            return default_window
        completed_budget = int(completed_budget)
        if completed_budget <= 0:
            raise ValueError(
                "async_training.max_completed_prompt_groups must be a positive integer when set; "
                f"got {completed_budget}"
            )
        return max(default_window, completed_budget)

    @staticmethod
    def _add_hpt_async_sample_meta(batch: DataProto) -> None:
        """Populate the async trainer metrics contract after HPT learner-row materialization."""

        def normalize_versions(key: str) -> list[int]:
            if key not in batch.non_tensor_batch:
                raise ValueError(f"HPT async trainer batch is missing non_tensor_batch[{key!r}].")
            values = batch.non_tensor_batch[key]
            normalized = []
            for value in values.tolist():
                if hasattr(value, "item"):
                    value = value.item()
                normalized.append(int(value))
            return normalized

        param_version_start = normalize_versions("min_global_steps")
        trajectory_param_versions = normalize_versions("max_global_steps")
        if len(param_version_start) != len(trajectory_param_versions):
            raise ValueError(
                "HPT async trainer batch has inconsistent param-version metadata lengths: "
                f"min_global_steps={len(param_version_start)} max_global_steps={len(trajectory_param_versions)}."
            )

        param_version_diff = [
            abs(end - start) for start, end in zip(param_version_start, trajectory_param_versions, strict=False)
        ]
        partial_num = sum(1 for diff in param_version_diff if diff != 0)
        batch.meta_info.update(
            {
                "param_version_diversity": len(set(trajectory_param_versions)),
                "trajectory_param_versions": trajectory_param_versions,
                "fully_async/partial/total_partial_num": partial_num,
                "fully_async/partial/partial_ratio": partial_num / len(param_version_diff)
                if param_version_diff
                else 0.0,
                "fully_async/partial/max_partial_span": max(param_version_diff) if param_version_diff else 0,
            }
        )

    def _hpt_required_training_multiple(self) -> int:
        """Return the learner-row multiple required by downstream trainer dispatch."""

        required_multiple = 1
        if self.config.trainer.balance_batch:
            required_multiple = math.lcm(required_multiple, self._get_dp_size(self.actor_rollout_wg, "actor"))

        actor_mini_batch_size = OmegaConf.select(
            self.config, "actor_rollout_ref.actor.ppo_mini_batch_size", default=None
        )
        rollout_n = OmegaConf.select(self.config, "actor_rollout_ref.rollout.n", default=1)
        if actor_mini_batch_size is not None:
            required_multiple = math.lcm(required_multiple, int(actor_mini_batch_size) * int(rollout_n))

        if getattr(self, "use_critic", False):
            critic_mini_batch_size = OmegaConf.select(self.config, "critic.ppo_mini_batch_size", default=None)
            if critic_mini_batch_size is not None:
                required_multiple = math.lcm(required_multiple, int(critic_mini_batch_size) * int(rollout_n))
        return required_multiple

    @staticmethod
    def _plan_row_alignment_deferral(
        row_counts: list[int], required_multiple: int, protected_prefix: int = 0
    ) -> set[int] | None:
        """Choose group indices to DEFER so the retained learner-row count is the largest
        multiple of ``required_multiple`` not exceeding the total row count.

        HPT groups contribute an uneven number of learner rows (RL groups ``rollout_n``,
        SFT groups ``1``), so the running row count rarely lands exactly on a multiple.
        Growing the batch until it does over-collects 2-3x and can fail to converge (the
        residue mod ``rollout_n`` is only movable by SFT groups, which may not arrive) --
        the failure that crashed the bounded run. Instead we keep the intended
        ``required_samples`` groups and remove a residue subset, deferring those groups to
        the next step (carryover) so no trained data is discarded and the batch stays
        bounded and crash-free.

        Only indices ``>= protected_prefix`` are eligible to defer, so groups already
        carried over from a previous step are always trained (staleness stays within one
        step). Group sizes are small positive ints, so an exact 0/1 subset-sum to the
        residue is cheap.

        Returns a (possibly empty) set of indices to defer, or ``None`` if no eligible
        subset sums to the residue that must be removed.
        """
        total = sum(row_counts)
        if required_multiple <= 1:
            return set()
        residue = total % required_multiple
        if residue == 0:
            return set()
        reachable = [False] * (residue + 1)
        predecessor: list[tuple[int, int]] = [(-1, -1)] * (residue + 1)
        reachable[0] = True
        for idx in range(protected_prefix, len(row_counts)):
            size = row_counts[idx]
            if size <= 0 or size > residue:
                continue
            for target in range(residue, size - 1, -1):
                if reachable[target - size] and not reachable[target]:
                    reachable[target] = True
                    predecessor[target] = (target - size, idx)
        if not reachable[residue]:
            return None
        defer: set[int] = set()
        target = residue
        while target > 0:
            prev, idx = predecessor[target]
            defer.add(idx)
            target = prev
        return defer

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            weights_updated = await self._fit_update_weights()
            if weights_updated:
                self._fit_log_aggregated_training_metrics()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_filter_truncated_rl_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            weights_updated = await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        if weights_updated:
            self._fit_log_aggregated_training_metrics()
        self._fit_postprocess_step(batch)

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        async_hpt_enabled = bool(OmegaConf.select(self.config, "async_hpt.enabled", default=False))
        old_logprob_source = OmegaConf.select(self.config, "async_hpt.rl_old_logprob_source", default="rollout")
        entry_proximal = OmegaConf.select(self.config, "async_hpt.entry_proximal", default="recent")
        if async_hpt_enabled and old_logprob_source == "entry":
            if entry_proximal != "recent":
                raise ValueError(f"Unsupported async_hpt.entry_proximal={entry_proximal!r}.")
            return super()._compute_old_log_prob(batch)

        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_dump_data(self, batch: DataProto):
        # Preserve the base rollout-generation logging, then sample the
        # loss-boundary tensors for offline ablation analysis (read-only).
        super()._fit_dump_data(batch)
        self._training_dumper.maybe_dump(
            batch,
            step=self.global_steps,
            param_version=self.current_param_version,
            local_trigger_step=self.local_trigger_step,
        )

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return False

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        if steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        with marked_timer("param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        return True

    def _fit_log_aggregated_training_metrics(self):
        aggregated_metrics = self.metrics_aggregator.get_aggregated_metrics()
        if aggregated_metrics:
            self.logger.log(
                data=aggregated_metrics,
                step=self.current_param_version,
            )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self, batch: DataProto):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(metrics=self.metrics, sample_count=len(batch), timestamp=time.time())

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def _fit_filter_truncated_rl_advantage(self, batch: DataProto) -> DataProto:
        """P0-2 (Improvement_RL.md §5.1/§5.5): zero the advantage of truncated RL rollouts.

        A truncated (budget-exhausting, non-terminating) rollout is a length artifact, not a
        reasoning signal: leaving its typically-negative advantage in the loss floods the
        gradient with ~8k-token degenerate sequences and flattens the policy (the observed
        entropy blow-up). Zeroing its advantage makes it contribute no policy gradient.

        Ordering invariant (§5.5.2): this runs AFTER compute_advantage, so the GRPO baseline
        has already counted these rows as failures (giving the surviving clean rows their
        correct positive advantage); only then is the truncated row's own advantage removed.
        SFT rows (hpt_is_sft) are never touched. Under the fixed seq-mean-token-sum-norm
        denominator (global_batch_size = ppo_mini_batch_size * n) this is gradient-equivalent
        to physical row removal, with no dilution and no batch-shape/divisibility risk.
        """
        if not self.config.reward.get("reward_kwargs", {}).get("zero_truncated_rl_advantage", False):
            return batch
        if "advantages" not in batch.batch or "response_mask" not in batch.batch:
            return batch
        resp_len = batch.batch["response_mask"].sum(dim=-1)
        cap = int(self.config.data.max_response_length)
        truncated = resp_len >= cap
        if "hpt_is_sft" in batch.batch:
            is_sft = batch.batch["hpt_is_sft"].to(torch.bool)
            rl_truncated = truncated & (~is_sft)
            n_rl = int((~is_sft).sum().item())
        else:
            rl_truncated = truncated
            n_rl = int(len(batch))
        n_zeroed = int(rl_truncated.sum().item())
        if n_zeroed > 0:
            batch.batch["advantages"][rl_truncated] = 0.0
        batch.batch["hpt_is_truncated_rl"] = rl_truncated
        self.metrics["hpt/truncated_rl_rows_zeroed"] = n_zeroed
        self.metrics["hpt/truncated_rl_frac"] = (n_zeroed / n_rl) if n_rl > 0 else 0.0
        return batch

    def _fit_collect_metrics(self, batch: DataProto):
        super()._fit_collect_metrics(batch)
        self._collect_metric_aggregation_weights(batch)

    def _collect_metric_aggregation_weights(self, batch: DataProto):
        response_info = _compute_response_info(batch)
        response_length = response_info["response_length"]
        metric_response_length = _compute_metric_response_length(batch, response_length)
        response_mask = batch.batch["response_mask"].bool()

        row_count = len(batch)
        non_aborted_count = int((response_length != 0).sum().item())
        metric_response_count = int(metric_response_length.numel())
        metric_non_aborted_count = int((metric_response_length != 0).sum().item())
        response_token_count = int(response_mask.sum().item())
        sft_row_count = 0
        sft_token_count = response_token_count
        entropy_token_count = response_token_count
        group_count = row_count
        if "hpt_is_sft" in batch.batch:
            hpt_is_sft = batch.batch["hpt_is_sft"].to(dtype=torch.bool)
            sft_row_count = int(hpt_is_sft.sum().item())
            sft_token_count = int((response_mask & hpt_is_sft.unsqueeze(-1)).sum().item())
            if "hpt_is_truncated_rl" in batch.batch:
                hpt_is_truncated_rl = batch.batch["hpt_is_truncated_rl"].to(dtype=torch.bool)
                truncated_rl_token_count = int((response_mask & hpt_is_truncated_rl.unsqueeze(-1)).sum().item())
            else:
                truncated_rl_token_count = 0
            entropy_token_count = response_token_count - truncated_rl_token_count
            if not bool(batch.meta_info.get("hpt_sft_entropy_enabled", False)):
                entropy_token_count -= sft_token_count
        # Unique prompt-groups: the correct cross-window weight for the group-weighted
        # on-policy success rate (so it is not re-biased by row counts during aggregation).
        if "hpt_group_uid" in batch.non_tensor_batch:
            group_count = len(set(batch.non_tensor_batch["hpt_group_uid"].tolist()))

        metric_weights = {
            "critic/score/mean": non_aborted_count,
            "critic/rewards/mean": non_aborted_count,
            "critic/advantages/mean": response_token_count,
            "critic/returns/mean": response_token_count,
            "critic/vf_loss": response_token_count,
            "critic/vf_clipfrac": response_token_count,
            "critic/vpred_mean": response_token_count,
            "response_length/mean": metric_response_count,
            "response_length/clip_ratio": metric_response_count,
            "response_length_non_aborted/mean": metric_non_aborted_count,
            "response_length_non_aborted/clip_ratio": metric_non_aborted_count,
            "response/aborted_ratio": metric_response_count,
            "prompt_length/mean": row_count,
            "prompt_length/clip_ratio": row_count,
            "actor/pg_loss": response_token_count,
            "actor/pg_clipfrac": response_token_count,
            "actor/pg_clipfrac_lower": response_token_count,
            "actor/ppo_kl": response_token_count,
            "actor/entropy": entropy_token_count,
            "actor/entropy_loss": entropy_token_count,
            "actor/kl_loss": response_token_count,
            "actor/hpt/sft_nll": sft_token_count,
            "hpt/sft_pseudo_reward/mean": sft_row_count,
            "hpt/onpolicy_success_rate": group_count,
            "kl": response_token_count,
            "k3_kl": response_token_count,
            "chi2_token": response_token_count,
            "rollout_is_mean": response_token_count,
            "rollout_is_std": response_token_count,
            "rollout_is_oob_ratio": response_token_count,
            "rollout_is_ratio_fraction_high": response_token_count,
            "rollout_is_ratio_fraction_low": response_token_count,
            "training_ppl": row_count,
            "training_log_ppl": row_count,
            "rollout_ppl": row_count,
            "rollout_log_ppl": row_count,
            "log_ppl_diff": row_count,
            "log_ppl_abs_diff": row_count,
            "ppl_ratio": row_count,
            "chi2_seq": row_count,
            "rollout_is_eff_sample_size": row_count,
        }
        if self.use_critic:
            metric_weights["critic/values/mean"] = response_token_count

        if "__num_turns__" in batch.non_tensor_batch:
            metric_weights["num_turns/mean"] = int(len(batch.non_tensor_batch["__num_turns__"]))
        if "tool_call_counts" in batch.non_tensor_batch:
            metric_weights["tool_call_counts/mean"] = int(len(batch.non_tensor_batch["tool_call_counts"]))

        # Observability (Improvement_RL.md §5.2 #9): the true GRPO-centered advantage signal.
        # The stock critic/advantages/mean tracks raw reward (bit-alias of critic/score/mean), so it
        # shows no centering signal; log the actual masked mean / abs-mean of the advantage tensor
        # over RL response tokens (SFT rows carry the constant beta pseudo-reward and are excluded).
        if "advantages" in batch.batch:
            advantages = batch.batch["advantages"]
            adv_mask = response_mask
            if "hpt_is_sft" in batch.batch:
                adv_mask = response_mask & (~batch.batch["hpt_is_sft"].to(torch.bool).unsqueeze(-1))
            adv_tok = int(adv_mask.sum().item())
            if adv_tok > 0:
                self.metrics["critic/advantages/centered_mean"] = float((advantages * adv_mask).sum().item() / adv_tok)
                self.metrics["critic/advantages/centered_absmean"] = float(
                    (advantages.abs() * adv_mask).sum().item() / adv_tok
                )
                metric_weights["critic/advantages/centered_mean"] = adv_tok
                metric_weights["critic/advantages/centered_absmean"] = adv_tok

        for metric_name in self.metrics:
            if metric_name.startswith("rollout_is_seq_"):
                metric_weights[metric_name] = row_count
            elif metric_name.startswith("rollout_rs_"):
                metric_weights[metric_name] = row_count if "_seq_" in metric_name else response_token_count

        for metric_name, weight in metric_weights.items():
            if metric_name in self.metrics:
                self.metrics[f"_metric_weight/{metric_name}"] = weight
