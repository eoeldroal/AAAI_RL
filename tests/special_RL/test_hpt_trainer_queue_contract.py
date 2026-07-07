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
import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

pytest.importorskip("ray")


class _OneSampleQueue:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def get_sample(self):
        self.calls += 1
        return self.payload, 0


class _FixedHptAssembler:
    def __init__(self, batch):
        self.batch = batch
        self.seen_samples = None

    def assemble_rollout_samples(self, queue_samples):
        self.seen_samples = queue_samples
        return self.batch

    def materialize_training_batch(self, queue_sample):
        self.seen_samples = [queue_sample]
        return self.batch

    def concat_training_batches(self, batches):
        return self.batch


class _CountingIncrementalHptAssembler:
    def __init__(self, rollout_n=4):
        self.rollout_n = rollout_n
        self.full_assembly_calls = 0
        self.materialized_sample_ids = []
        self.concat_calls = 0

    def assemble_rollout_samples(self, queue_samples):
        self.full_assembly_calls += 1
        return self._concat_samples(queue_samples)

    def materialize_training_batch(self, queue_sample):
        self.materialized_sample_ids.append(queue_sample.sample_id)
        return self._batch_for_sample(queue_sample)

    def concat_training_batches(self, batches):
        self.concat_calls += 1
        from verl.protocol import DataProto

        return DataProto.concat(batches)

    def _concat_samples(self, queue_samples):
        from verl.protocol import DataProto

        return DataProto.concat([self._batch_for_sample(sample) for sample in queue_samples])

    def _batch_for_sample(self, queue_sample):
        route = queue_sample.hpt_route
        if not route.is_sft:
            return _make_rl_group_payload(
                group_uid=route.group_uid,
                prompt_uid=route.prompt_uid,
                rollout_n=self.rollout_n,
            )
        batch = _make_sft_payload(group_uid=route.group_uid, prompt_uid=route.prompt_uid)
        batch.non_tensor_batch["min_global_steps"] = np.array([0], dtype=object)
        batch.non_tensor_batch["max_global_steps"] = np.array([0], dtype=object)
        batch.batch["hpt_is_sft"] = torch.tensor([True], dtype=torch.bool)
        return batch


class _CollectingQueue:
    def __init__(self):
        self.payloads = []

    async def put_sample(self, *, sample):
        self.payloads.append(sample)
        return True


class _QueueReader:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def get_sample(self):
        self.calls += 1
        if not self.payloads:
            return None, 0
        return self.payloads.pop(0), len(self.payloads)


class _FakeActorRolloutWorkerGroup:
    def __init__(self, dp_size):
        self._dispatch_info = {"actor": list(range(dp_size))}

    def _query_dispatch_info(self, role):
        return self._dispatch_info[role]


class _CapturingActorRolloutWorkerGroup(_FakeActorRolloutWorkerGroup):
    def __init__(self, dp_size):
        super().__init__(dp_size)
        self.updated_batch = None

    def update_actor(self, batch_td):
        from verl.utils import tensordict_utils as tu

        self.updated_batch = batch_td
        for key in (
            "old_log_probs",
            "advantages",
            "response_mask",
            "hpt_is_sft",
        ):
            assert key in batch_td.keys(), f"missing update_actor field: {key}"
        for removed_key in ("hpt_seq_weight", "hpt_length_divisor", "hpt_loss_denominator"):
            assert removed_key not in batch_td.keys(), f"obsolete HPT field reached actor update: {removed_key}"
        batch_size = int(batch_td["hpt_is_sft"].shape[0])
        assert tuple(batch_td["old_log_probs"].shape) == tuple(batch_td["response_mask"].shape)
        assert tuple(batch_td["advantages"].shape) == tuple(batch_td["response_mask"].shape)
        assert tuple(batch_td["response_mask"].shape) == (batch_size, 4)
        assert tuple(batch_td["hpt_is_sft"].shape) == (batch_size,)
        assert tu.get(batch_td, "hpt_sft_entropy_enabled") is False
        assert tu.get(batch_td, "hpt_sft_kl_enabled") is False
        assert tu.get(batch_td, "compute_loss") is True
        assert tu.get(batch_td, "global_batch_size") == batch_size
        return tu.get_tensordict({}, non_tensor_dict={"metrics": {"mfu": [0.0], "loss": [1.25]}})


class _EmptyTauStore:
    def get(self, _prompt_uid):
        return None


class _SingleTauStore:
    def __init__(self, payload):
        self.payload = payload

    def get(self, _prompt_uid):
        return self.payload


class _FakeAsyncRolloutManager:
    def __init__(self, rollout_n):
        self.rollout_n = rollout_n
        self.seen_attempts = []

    async def generate_sequences_single(self, attempt_batch):
        rollout_index = int(attempt_batch.non_tensor_batch["hpt_rollout_index"][0])
        uid = str(attempt_batch.non_tensor_batch["uid"][0])
        prompt_uid = str(attempt_batch.non_tensor_batch["prompt_uid"][0])
        self.seen_attempts.append((rollout_index, uid, prompt_uid))

        # Complete out of order to verify that the accumulator restores rollout_index order.
        await asyncio.sleep(0.001 * (self.rollout_n - rollout_index))
        return _make_generated_attempt_payload(rollout_index, prompt_uid=prompt_uid)


def _make_async_hpt_config(*, rollout_n=4):
    return OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {}},
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": False,
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 4, "policy_loss": {"loss_mode": "vanilla"}},
                "rollout": {
                    "n": rollout_n,
                    "mode": "async",
                    "calculate_log_probs": True,
                    "prompt_length": 8,
                    "response_length": 4,
                },
            },
            "async_hpt": {
                "enabled": True,
                "gamma": 0.5,
                "alpha": 1.0,
                "beta": 1.0,
                "loss_aggregation": "branch_blind",
                "sft_beta_mode": "constant",
                "sft_entropy_enabled": False,
                "sft_kl_enabled": False,
                "tau_dataset_path": "unused-for-rl-route.parquet",
                "fail_on_missing_tau": False,
                "trajectory_scheduler": {"enabled": True},
            },
            "trainer": {"balance_batch": False},
        }
    )


def _make_hpt_route(*, is_sft, prompt_uid, group_uid, rollout_n=4):
    from verl.experimental.fully_async_policy.hpt_gate import HptRouteMetadata

    return HptRouteMetadata(
        is_sft=is_sft,
        prompt_uid=prompt_uid,
        group_uid=group_uid,
        missing_tau=not is_sft,
        success_probability=0.0 if is_sft else 1.0,
        success_count=0 if is_sft else rollout_n,
        total_count=rollout_n,
        gamma=0.5,
        success_threshold=0.0,
        success_score_key="reward_score",
    )


def _make_source_batch(*, rollout_n=4, prompt_uid="prompt-a"):
    from verl.protocol import DataProto

    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.arange(rollout_n).view(rollout_n, 1)},
        non_tensors={"prompt_uid": np.array([prompt_uid] * rollout_n, dtype=object)},
    )


def _make_rl_group_payload(*, group_uid, prompt_uid, rollout_n=4):
    from verl.protocol import DataProto

    payload = DataProto.concat(
        [_make_generated_attempt_payload(rollout_index, prompt_uid=prompt_uid) for rollout_index in range(rollout_n)]
    )
    payload.non_tensor_batch["uid"] = np.array([group_uid] * rollout_n, dtype=object)
    payload.non_tensor_batch["prompt_uid"] = np.array([prompt_uid] * rollout_n, dtype=object)
    payload.non_tensor_batch["min_global_steps"] = np.array(list(range(rollout_n)), dtype=object)
    payload.non_tensor_batch["max_global_steps"] = np.array(list(range(rollout_n)), dtype=object)
    payload.non_tensor_batch["extra_info"] = np.array(
        [{"prompt_uid": prompt_uid, "rollout_index": rollout_index} for rollout_index in range(rollout_n)],
        dtype=object,
    )
    return payload


def _make_generated_group_with_lengths(*, group_uid, prompt_uid, lengths):
    from verl.protocol import DataProto

    rows = []
    for rollout_index, length in enumerate(lengths):
        response_mask = torch.tensor([[1] * length + [0] * (4 - length)], dtype=torch.long)
        responses = torch.tensor([[10 + rollout_index, 20 + rollout_index, 30 + rollout_index, 40 + rollout_index]])
        prompts = torch.tensor([[1, 2]], dtype=torch.long)
        input_ids = torch.cat([prompts, responses], dim=-1)
        attention_mask = torch.cat([torch.ones((1, 2), dtype=torch.long), response_mask], dim=-1)
        rows.append(
            DataProto.from_dict(
                tensors={
                    "prompts": prompts,
                    "responses": responses,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": torch.arange(input_ids.shape[-1]).view(1, -1),
                    "response_mask": response_mask,
                    "rollout_log_probs": torch.zeros_like(response_mask, dtype=torch.float32),
                    "rm_scores": torch.zeros_like(response_mask, dtype=torch.float32),
                },
                non_tensors={
                    "uid": np.array([group_uid], dtype=object),
                    "prompt_uid": np.array([prompt_uid], dtype=object),
                    "min_global_steps": np.array([rollout_index], dtype=object),
                    "max_global_steps": np.array([rollout_index], dtype=object),
                    "extra_info": np.array([{"prompt_uid": prompt_uid, "rollout_index": rollout_index}], dtype=object),
                },
            )
        )
    return DataProto.concat(rows)


def _make_sft_payload(*, group_uid, prompt_uid):
    from verl.protocol import DataProto

    prompts = torch.tensor([[1, 2]], dtype=torch.long)
    responses = torch.tensor([[31, 32, 0, 0]], dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(input_ids.shape[-1]).view(1, -1),
            "response_mask": response_mask,
        },
        non_tensors={
            "uid": np.array([f"sft_{group_uid}"], dtype=object),
            "prompt_uid": np.array([prompt_uid], dtype=object),
            "hpt_group_uid": np.array([group_uid], dtype=object),
            "extra_info": np.array([{"prompt_uid": prompt_uid, "hpt_route": "sft"}], dtype=object),
        },
    )


def _make_hpt_queue_sample(*, route_kind, idx, rollout_n=4):
    from verl.experimental.fully_async_policy.detach_utils import RolloutSample

    prompt_uid = f"prompt-{idx}"
    group_uid = f"group-{idx}"
    is_sft = route_kind == "sft"
    payload = (
        _make_sft_payload(group_uid=group_uid, prompt_uid=prompt_uid)
        if is_sft
        else _make_rl_group_payload(group_uid=group_uid, prompt_uid=prompt_uid, rollout_n=rollout_n)
    )
    return RolloutSample(
        full_batch=payload,
        sample_id=f"sample-{idx}",
        epoch=0,
        rollout_status={},
        hpt_route=_make_hpt_route(
            is_sft=is_sft,
            prompt_uid=prompt_uid,
            group_uid=group_uid,
            rollout_n=rollout_n,
        ),
    )


def _make_mixed_hpt_queue_samples(*, rollout_n=4):
    return [
        _make_hpt_queue_sample(route_kind="rl", idx=0, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="rl", idx=1, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=2, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=3, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="rl", idx=4, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=5, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=6, rollout_n=rollout_n),
    ]


def test_hpt_gate_preserves_prerouting_generated_response_lengths_for_sft_route():
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
    from verl.experimental.fully_async_policy.hpt_gate import HptRolloutGate
    from verl.experimental.fully_async_policy.hpt_payload import HptSftPayload

    prompt_uid = "prompt-a"
    group_uid = "uid_sample_0_1"
    lengths = (2, 4, 0, 3)
    config = _make_async_hpt_config(rollout_n=len(lengths))
    hpt_config = validate_async_hpt_config(config)
    tau_payload = HptSftPayload(
        prompt_uid=prompt_uid,
        messages=[{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    )
    gate = HptRolloutGate(config=hpt_config, tau_store=_SingleTauStore(tau_payload))
    generated_batch = _make_generated_group_with_lengths(group_uid=group_uid, prompt_uid=prompt_uid, lengths=lengths)

    decision = gate.route(generated_batch, group_uid=group_uid)

    assert decision.metadata.is_sft is True
    assert decision.metadata.generated_response_lengths == lengths
    assert decision.metadata.success_probability == 0.0


def _make_actor_config(
    *,
    loss_agg_mode="seq-mean-token-sum-norm",
    loss_scale_factor=4,
    entropy_coeff=0.0,
    use_kl_loss=False,
    kl_loss_type="low_var_kl",
):
    from verl.workers.config import ActorConfig

    return ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size=4,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
        loss_agg_mode=loss_agg_mode,
        loss_scale_factor=loss_scale_factor,
        entropy_coeff=entropy_coeff,
        use_kl_loss=use_kl_loss,
        kl_loss_type=kl_loss_type,
        policy_loss={"loss_mode": "vanilla"},
        global_batch_info={"dp_size": 4},
    )


def _make_row_aware_hpt_trainer(queue_samples, *, rollout_n=4):
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=rollout_n)
    trainer.config.trainer.balance_batch = True
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=4)
    trainer.use_critic = False
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None
    trainer.current_param_version = 5
    trainer.stale_trajectory_processed = 0
    trainer.metrics = {}
    trainer.timing_raw = {}
    return trainer


def _prepare_hpt_batch_for_actor_update(trainer):
    async def prepare():
        _, batch = await trainer._get_samples_from_queue()
        trainer._collect_metrics_from_samples(batch, trainer.metrics)
        batch = trainer._fit_compute_reward(batch)
        batch = trainer._fit_compute_log_prob(batch)
        batch = trainer._fit_compute_ref_log_prob(batch)
        batch = trainer._fit_compute_critic(batch)
        return trainer._fit_compute_advantage(batch)

    return prepare()


def _make_generated_attempt_payload(rollout_index, *, prompt_uid):
    import numpy as np

    from verl.protocol import DataProto

    response_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    responses = torch.tensor([[10 + rollout_index, 20 + rollout_index, 0, 0]], dtype=torch.long)
    prompts = torch.tensor([[1, 2]], dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    rm_scores[:, -1] = 1.0
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(input_ids.shape[-1]).view(1, -1),
            "response_mask": response_mask,
            "rollout_log_probs": torch.full(response_mask.shape, -0.1 * (rollout_index + 1)),
            "rm_scores": rm_scores,
        },
        non_tensors={
            "uid": np.array([f"attempt-{rollout_index}"], dtype=object),
            "prompt_uid": np.array([prompt_uid], dtype=object),
            "min_global_steps": np.array([rollout_index], dtype=object),
            "max_global_steps": np.array([rollout_index], dtype=object),
            "extra_info": np.array([{"prompt_uid": prompt_uid, "rollout_index": rollout_index}], dtype=object),
        },
        meta_info={"metrics": [{"generate_sequences": float(rollout_index + 1), "tool_calls": 0.0}]},
    )


@pytest.mark.asyncio
async def test_async_hpt_trainer_sets_global_token_num_after_queue_assembly():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.protocol import DataProto

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)

    assembled_batch = DataProto.from_dict(
        tensors={
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                ],
                dtype=torch.long,
            )
        },
        non_tensors={
            "min_global_steps": np.array([0, 0], dtype=object),
            "max_global_steps": np.array([0, 0], dtype=object),
        },
    )
    sample = {"sample_id": "sample-0"}

    trainer.required_samples = 1
    trainer.message_queue_client = _OneSampleQueue(ray.cloudpickle.dumps(sample))
    trainer.config = OmegaConf.create({"async_hpt": {"enabled": True}, "trainer": {"balance_batch": False}})
    trainer.tokenizer = None
    trainer.hpt_assembler = _FixedHptAssembler(assembled_batch)

    epoch, batch = await trainer._get_samples_from_queue()

    assert epoch == 0
    assert batch is assembled_batch
    assert trainer.hpt_assembler.seen_samples == [sample]
    assert batch.meta_info["global_token_num"] == [3, 2]
    assert batch.meta_info["fully_async/total_wait_time"] >= 0


def test_async_hpt_config_accepts_branch_blind_controls_and_rejects_old_prompt_equal():
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config

    config = _make_async_hpt_config()
    hpt_config = validate_async_hpt_config(config)

    assert hpt_config.loss_aggregation == "branch_blind"
    assert hpt_config.beta == 1.0
    assert hpt_config.sft_beta_mode == "constant"
    assert hpt_config.sft_entropy_enabled is False
    assert hpt_config.sft_kl_enabled is False

    config.async_hpt.loss_aggregation = "prompt_equal"
    with pytest.raises(ValueError, match="loss_aggregation"):
        validate_async_hpt_config(config)


def test_async_hpt_config_rejects_non_unit_alpha_and_invalid_beta_mode():
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config

    config = _make_async_hpt_config()
    config.async_hpt.alpha = 0.5
    with pytest.raises(ValueError, match="alpha"):
        validate_async_hpt_config(config)

    config = _make_async_hpt_config()
    config.async_hpt.sft_beta_mode = "unsupported"
    with pytest.raises(ValueError, match="sft_beta_mode"):
        validate_async_hpt_config(config)


def test_async_hpt_length_inverse_beta_requires_positive_loss_scale_factor():
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config

    config = _make_async_hpt_config()
    config.async_hpt.sft_beta_mode = "length_inverse"
    with pytest.raises(ValueError, match="loss_scale_factor"):
        validate_async_hpt_config(config)

    config.actor_rollout_ref.actor.loss_scale_factor = 4.5
    with pytest.raises(ValueError, match="loss_scale_factor"):
        validate_async_hpt_config(config)

    config.actor_rollout_ref.actor.loss_scale_factor = 8
    hpt_config = validate_async_hpt_config(config)
    assert hpt_config.sft_beta_mode == "length_inverse"


@pytest.mark.asyncio
async def test_async_hpt_trajectory_scheduler_to_trainer_queue_rl_contract():
    import ray

    from verl.experimental.fully_async_policy.detach_utils import RolloutSample
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
    from verl.experimental.fully_async_policy.hpt_gate import HptRolloutGate
    from verl.experimental.fully_async_policy.hpt_rollout_accumulator import HptPromptGroupAccumulator

    rollout_n = 4
    config = _make_async_hpt_config(rollout_n=rollout_n)
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    hpt_config = validate_async_hpt_config(config)

    rollouter_cls = FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = object.__new__(rollouter_cls)
    rollouter.config = config
    rollouter.hpt_rollout_gate = HptRolloutGate(config=hpt_config, tau_store=_EmptyTauStore())
    rollouter.hpt_rollout_accumulator = HptPromptGroupAccumulator(rollout_n=rollout_n)
    rollouter.hpt_scheduler_groups = {}
    rollouter.hpt_closed_group_uids = set()
    rollouter.lock = asyncio.Lock()
    rollouter.active_tasks = set()
    rollouter.max_concurrent_trajectory_attempts = rollout_n
    rollouter.max_concurrent_samples = rollout_n
    rollouter.paused = False
    rollouter._resume_event = asyncio.Event()
    rollouter._resume_event.set()
    rollouter.async_rollout_manager = _FakeAsyncRolloutManager(rollout_n=rollout_n)
    rollouter.message_queue_client = _CollectingQueue()
    rollouter.processed_sample_count = 0
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0

    async def get_statistics():
        return {"queue_size": 0, "staleness_samples": 0}

    rollouter.get_statistics = get_statistics

    rollout_sample = RolloutSample(
        full_batch=_make_source_batch(rollout_n=rollout_n, prompt_uid="prompt-a"),
        sample_id="sample_0_1",
        epoch=0,
        rollout_status={},
    )

    await rollouter._submit_hpt_trajectory_attempts(rollout_sample)
    await asyncio.gather(*list(rollouter.active_tasks))

    assert rollouter.async_rollout_manager.seen_attempts == [
        (0, "uid_sample_0_1", "prompt-a"),
        (1, "uid_sample_0_1", "prompt-a"),
        (2, "uid_sample_0_1", "prompt-a"),
        (3, "uid_sample_0_1", "prompt-a"),
    ]
    assert rollouter.hpt_rollout_accumulator.open_group_count() == 0
    assert rollouter.hpt_rollout_accumulator.stored_attempt_count() == 0
    assert rollouter.processed_sample_count == 1
    assert rollouter.total_generated_samples == 1
    assert len(rollouter.message_queue_client.payloads) == 1

    queued_sample = ray.cloudpickle.loads(rollouter.message_queue_client.payloads[0])
    assert queued_sample.hpt_route.is_sft is False
    assert queued_sample.hpt_route.prompt_uid == "prompt-a"
    assert queued_sample.hpt_route.group_uid == "uid_sample_0_1"
    assert queued_sample.hpt_route.success_probability == 1.0
    assert queued_sample.hpt_route.missing_tau is True
    assert queued_sample.full_batch.batch["responses"][:, 0].tolist() == [10, 11, 12, 13]
    assert queued_sample.full_batch.non_tensor_batch["uid"].tolist() == ["uid_sample_0_1"] * rollout_n

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 1
    trainer.message_queue_client = _QueueReader(rollouter.message_queue_client.payloads)
    trainer.config = config
    trainer.tokenizer = None
    trainer.hpt_assembler = None
    trainer.use_critic = False
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.current_param_version = 5
    trainer.stale_trajectory_processed = 0
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer.global_steps = 1
    trainer.actor_rollout_wg = _CapturingActorRolloutWorkerGroup(dp_size=1)
    trainer.config.trainer.critic_warmup = 0
    trainer.config.actor_rollout_ref.actor.calculate_entropy = False
    trainer.config.actor_rollout_ref.actor.entropy_coeff = 0.0
    trainer.config.actor_rollout_ref.actor.ppo_epochs = 1
    trainer.config.actor_rollout_ref.actor.data_loader_seed = 0
    trainer.config.actor_rollout_ref.actor.shuffle = False
    trainer.config.actor_rollout_ref.rollout.multi_turn = {"enable": False}
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    trainer.config.distillation = {"enabled": False}

    epoch, batch = await trainer._get_samples_from_queue()

    assert epoch == 0
    assert len(batch) == rollout_n
    assert batch.batch["responses"][:, 0].tolist() == [10, 11, 12, 13]
    assert batch.batch["hpt_is_sft"].tolist() == [False] * rollout_n
    assert "hpt_seq_weight" not in batch.batch
    assert "hpt_length_divisor" not in batch.batch
    assert "hpt_loss_denominator" not in batch.batch
    assert batch.meta_info["global_token_num"] == [4, 4, 4, 4]
    assert batch.non_tensor_batch["uid"].tolist() == ["uid_sample_0_1"] * rollout_n
    assert batch.non_tensor_batch["prompt_uid"].tolist() == ["prompt-a"] * rollout_n
    assert batch.non_tensor_batch["hpt_group_uid"].tolist() == ["uid_sample_0_1"] * rollout_n
    assert batch.non_tensor_batch["hpt_route_is_sft"].tolist() == [False] * rollout_n
    assert batch.non_tensor_batch["hpt_missing_tau"].tolist() == [True] * rollout_n

    trainer._collect_metrics_from_samples(batch, trainer.metrics)
    batch = trainer._fit_compute_reward(batch)
    batch = trainer._fit_compute_log_prob(batch)
    batch = trainer._fit_compute_ref_log_prob(batch)
    batch = trainer._fit_compute_critic(batch)
    batch = trainer._fit_compute_advantage(batch)
    returned_batch = trainer._fit_update_actor(batch)

    assert returned_batch is batch
    assert trainer.actor_rollout_wg.updated_batch is not None
    assert trainer.metrics["hpt/old_logprob_from_rollout"] == 1.0
    assert trainer.metrics["hpt/num_rl_groups"] == 1.0
    assert trainer.metrics["actor/loss"] == 1.25
    assert trainer.metrics["perf/mfu/actor"] == 0.0


@pytest.mark.asyncio
async def test_async_hpt_rollouter_pause_gate_tracks_live_consumption_not_monotonic_production():
    """The staleness pause gate must reflect the live outstanding buffer (in-flight open
    groups + completed queued samples), not a monotonic produced-since-sync counter.

    When the trainer drains the queue, the outstanding buffer drops below
    ``max_required_samples`` and the rollouter must be eligible to resume within the same
    parameter version instead of stalling until the next weight sync. A monotonic counter
    would still read ``>= max_required_samples`` after consumption and keep the rollouter
    (and its idle rollout GPUs) paused. Guards the async-scheduling contract behind
    ``_should_pause_generation`` / ``_outstanding_sample_count``; see the circular-wait
    pitfall in ``docs/AsyncBudget_RL.md``.
    """
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
    from verl.experimental.fully_async_policy.hpt_gate import HptRolloutGate
    from verl.experimental.fully_async_policy.hpt_rollout_accumulator import HptPromptGroupAccumulator

    rollout_n = 4
    config = _make_async_hpt_config(rollout_n=rollout_n)
    hpt_config = validate_async_hpt_config(config)

    rollouter_cls = FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = object.__new__(rollouter_cls)
    rollouter.config = config
    rollouter.hpt_rollout_gate = HptRolloutGate(config=hpt_config, tau_store=_EmptyTauStore())
    rollouter.hpt_rollout_accumulator = HptPromptGroupAccumulator(rollout_n=rollout_n)
    rollouter.active_tasks = set()
    rollouter.paused = False
    rollouter.max_required_samples = 10
    # Keep the queue-full (#1) and HPT completed-attempt-storage (#3) gates far away so the
    # staleness gate (#2) is the only one under test.
    rollouter.max_queue_size = 10_000

    queue_state = {"queue_size": 0}

    class _StubQueueClient:
        async def get_statistics(self):
            return {"queue_size": queue_state["queue_size"]}

    rollouter.message_queue_client = _StubQueueClient()

    # Whole outstanding buffer lives in the completed queue (no in-flight groups).
    assert rollouter._hpt_trajectory_scheduler_enabled() is True
    assert rollouter.hpt_rollout_accumulator.open_group_count() == 0

    # Outstanding buffer at the ceiling -> pause.
    queue_state["queue_size"] = 10
    assert rollouter._outstanding_sample_count(10) == 10
    assert await rollouter._should_pause_generation() is True

    # Trainer consumes 6 samples -> outstanding 4 < ceiling 10 -> must not pause.
    # A monotonic produced-since-sync counter would still read >= 10 here and stall.
    queue_state["queue_size"] = 4
    assert rollouter._outstanding_sample_count(4) == 4
    assert await rollouter._should_pause_generation() is False


@pytest.mark.asyncio
async def test_async_hpt_trainer_collects_extra_queue_samples_until_learner_rows_are_trainable():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    rollout_n = 4
    config = _make_async_hpt_config(rollout_n=rollout_n)
    config.trainer.balance_batch = True
    # The first four queue samples produce 10 learner rows:
    #   2 RL groups * 4 rows + 2 SFT rows = 10.
    # The trainer must not close the batch there because DP=4 and actor mini=16.
    # It should continue until seven queue samples produce 16 learner rows.
    queue_samples = [
        _make_hpt_queue_sample(route_kind="rl", idx=0, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="rl", idx=1, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=2, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=3, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="rl", idx=4, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=5, rollout_n=rollout_n),
        _make_hpt_queue_sample(route_kind="sft", idx=6, rollout_n=rollout_n),
    ]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = config
    trainer.tokenizer = None
    trainer.hpt_assembler = None
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=4)
    trainer.use_prefix_grouper = False

    epoch, batch = await trainer._get_samples_from_queue()

    assert epoch == 0
    assert trainer.message_queue_client.calls == 7
    assert len(batch) == 16
    assert int(batch.batch["hpt_is_sft"].sum().item()) == 4
    assert batch.meta_info["global_token_num"] == [4] * 16
    assert batch.meta_info["fully_async/hpt_collected_queue_samples"] == 7
    assert batch.meta_info["fully_async/hpt_required_training_multiple"] == 16


@pytest.mark.asyncio
async def test_async_hpt_trainer_topup_materializes_each_queue_sample_once():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    rollout_n = 4
    queue_samples = [_make_hpt_queue_sample(route_kind="sft", idx=idx, rollout_n=rollout_n) for idx in range(8)]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=rollout_n)
    trainer.config.trainer.balance_batch = True
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    trainer.config.actor_rollout_ref.rollout.n = rollout_n
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = _CountingIncrementalHptAssembler(rollout_n=rollout_n)

    _, batch = await trainer._get_samples_from_queue()

    assert len(batch) == 8
    assert trainer.message_queue_client.calls == 8
    assert trainer.hpt_assembler.full_assembly_calls == 0
    assert trainer.hpt_assembler.concat_calls == 1
    assert trainer.hpt_assembler.materialized_sample_ids == [f"sample-{idx}" for idx in range(8)]
    assert batch.meta_info["fully_async/hpt_collected_queue_samples"] == 8
    assert batch.meta_info["fully_async/hpt_required_training_multiple"] == 8


def test_hpt_incremental_materialization_matches_full_assembly_contract():
    from verl.experimental.fully_async_policy.hpt_assembler import HptBatchAssembler

    config = _make_async_hpt_config(rollout_n=4)
    assembler = HptBatchAssembler(config=config, tokenizer=None)
    queue_samples = _make_mixed_hpt_queue_samples(rollout_n=4)

    full_batch = assembler.assemble_rollout_samples(queue_samples)
    materialized = [assembler.materialize_training_batch(sample) for sample in queue_samples]
    incremental_batch = assembler.concat_training_batches(materialized)

    assert len(incremental_batch) == len(full_batch)
    assert set(incremental_batch.batch.keys()) == set(full_batch.batch.keys())
    for key in full_batch.batch.keys():
        assert torch.equal(incremental_batch.batch[key], full_batch.batch[key]), key
    assert set(incremental_batch.non_tensor_batch.keys()) == set(full_batch.non_tensor_batch.keys())
    for key in full_batch.non_tensor_batch.keys():
        assert incremental_batch.non_tensor_batch[key].tolist() == full_batch.non_tensor_batch[key].tolist(), key
    assert incremental_batch.meta_info["hpt_sft_entropy_enabled"] is False
    assert incremental_batch.meta_info["hpt_sft_kl_enabled"] is False


@pytest.mark.asyncio
async def test_async_hpt_trainer_returns_none_when_topup_terminates_before_trainable_rows():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    queue_samples = [_make_hpt_queue_sample(route_kind="sft", idx=idx, rollout_n=4) for idx in range(4)]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 1
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.trainer.balance_batch = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.use_critic = False

    epoch, batch = await trainer._get_samples_from_queue()

    assert epoch is None
    assert batch is None
    assert trainer.message_queue_client.calls == 5


@pytest.mark.asyncio
async def test_async_hpt_trainer_trims_overshoot_to_aligned_batch_and_defers_residue():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    # 14 SFT then 1 RL: the batch grows only to the first multiple (14 SFT rows < 16, one RL
    # group crosses to 18 >= 16), then TRIMS the 2-row residue back to 16 by DEFERRING two SFT
    # groups to the next step (carryover). This is the regression guard for the old grow-to-align
    # loop, which over-collected until the running row count happened to land on a multiple
    # (here it would consume many more groups to reach 96 rows).
    route_kinds = ["sft"] * 14 + ["rl"] * 18 + ["sft"] * 10
    queue_samples = [
        _make_hpt_queue_sample(route_kind=route_kind, idx=idx, rollout_n=4)
        for idx, route_kind in enumerate(route_kinds)
    ]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.trainer.balance_batch = True
    trainer.config.async_training = {"max_completed_prompt_groups": 64}
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=4)
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None

    _, batch = await trainer._get_samples_from_queue()

    # Bounded: grew to 18 rows (15 groups) then trimmed to 16, NOT grown to 96 rows (42 groups).
    assert trainer.message_queue_client.calls == 15
    assert len(batch) == 16
    assert len(batch) % 16 == 0
    assert batch.meta_info["fully_async/hpt_required_training_multiple"] == 16
    # Retained (trained) group count, and the residue deferred to the next step.
    assert batch.meta_info["fully_async/hpt_collected_queue_samples"] == 13
    assert batch.meta_info["fully_async/hpt_carryover_in_groups"] == 0
    assert batch.meta_info["fully_async/hpt_carryover_out_groups"] == 2
    assert batch.meta_info["fully_async/hpt_row_alignment_deferred_rows"] == 2
    assert batch.meta_info["fully_async/hpt_fresh_pulled_groups"] == 15
    assert batch.meta_info["fully_async/hpt_carryover_discarded_groups"] == 0
    # Explicit reconciliation identity (AGENTS.md: every collection unit stays accountable).
    m = batch.meta_info
    assert m["fully_async/hpt_fresh_pulled_groups"] == (
        m["fully_async/hpt_collected_queue_samples"]
        + m["fully_async/hpt_carryover_out_groups"]
        + m["fully_async/hpt_carryover_discarded_groups"]
        - m["fully_async/hpt_carryover_in_groups"]
    )
    # The 2 deferred groups are held for the next step, not discarded.
    assert len(trainer._hpt_carryover_samples) == 2


def test_plan_row_alignment_deferral_is_empty_when_already_aligned():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral
    assert plan([8, 8, 8, 8], 16) == set()  # 32 % 16 == 0
    assert plan([1, 1, 8], 1) == set()  # multiple of 1: nothing to remove


def test_plan_row_alignment_deferral_removes_exact_residue_via_whole_groups():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral
    row_counts = [1] * 14 + [4]  # 18 rows, multiple 16 -> defer 2 rows
    defer = plan(row_counts, 16)
    assert defer is not None
    retained_rows = sum(row_counts) - sum(row_counts[i] for i in defer)
    assert retained_rows == 16 and retained_rows % 16 == 0
    assert all(row_counts[i] == 1 for i in defer)  # residue 2 removed via two size-1 groups


def test_plan_row_alignment_deferral_aligns_pure_rl_by_dropping_whole_groups():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral
    defer = plan([8, 8, 8, 8, 8], 16)  # 40 rows -> retained 32 -> defer one size-8 group
    assert defer is not None and len(defer) == 1


def test_plan_row_alignment_deferral_protects_carryover_prefix():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral
    # residue 2 is only reachable via the two leading size-1 (carried-over) groups; protecting
    # them makes the residue unreachable, so a carried-over group is never re-deferred here.
    assert plan([1, 1, 8], 8, protected_prefix=2) is None
    assert plan([1, 1, 8], 8, protected_prefix=0) == {0, 1}


def test_plan_row_alignment_deferral_returns_none_when_residue_unreachable():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral
    # size-8 groups only, residue 3 (mod 7): no subset of {8, 16, 24} sums to 3.
    assert plan([8, 8, 8], 7) is None


@pytest.mark.asyncio
async def test_async_hpt_collection_trims_the_composition_that_crashed_the_grow_loop():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    # rollout_n=8, mini=1, dp=1 -> required_multiple 8. Two SFT (2 rows) then RL groups (8 rows):
    # the residue mod 8 is 2, and RL groups (each +8) can NEVER move it to 0 -- the grow-to-align
    # loop diverged and raised ValueError here (the crash that took the bounded run down at
    # learner_rows=3095). Trimming defers the two SFT and returns an aligned 8-row batch.
    route_kinds = ["sft", "sft"] + ["rl"] * 6
    queue_samples = [
        _make_hpt_queue_sample(route_kind=route_kind, idx=idx, rollout_n=8)
        for idx, route_kind in enumerate(route_kinds)
    ]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 2
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=8)
    trainer.config.trainer.balance_batch = True
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    trainer.config.async_training = {"max_completed_prompt_groups": 64}
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None

    _, batch = await trainer._get_samples_from_queue()  # must not raise

    assert batch.meta_info["fully_async/hpt_required_training_multiple"] == 8
    assert len(batch) == 8 and len(batch) % 8 == 0
    assert trainer.message_queue_client.calls == 3  # 2 initial + 1 grow, then trim (no more pulls)
    assert batch.meta_info["fully_async/hpt_carryover_out_groups"] == 2
    assert batch.meta_info["fully_async/hpt_row_alignment_deferred_rows"] == 2


@pytest.mark.asyncio
async def test_async_hpt_carryover_round_trips_and_is_consumed_within_one_step():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    # Step 1 overshoots (13 rows -> grow one RL -> 17) and defers the 1-row residue (an SFT group)
    # to carryover. Step 2 seeds that carried SFT, protects it (protected_prefix), and lands on an
    # aligned 16 rows so the carried group TRAINS this step and carryover empties. No data is lost.
    route_kinds = ["rl", "rl", "rl", "sft", "rl", "sft", "sft", "sft", "rl", "rl", "rl"]
    queue_samples = [
        _make_hpt_queue_sample(route_kind=route_kind, idx=idx, rollout_n=4)
        for idx, route_kind in enumerate(route_kinds)
    ]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.trainer.balance_batch = True
    trainer.config.async_training = {"max_completed_prompt_groups": 64}
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=4)
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None

    _, batch1 = await trainer._get_samples_from_queue()
    assert len(batch1) == 16
    assert batch1.meta_info["fully_async/hpt_carryover_in_groups"] == 0
    assert batch1.meta_info["fully_async/hpt_carryover_out_groups"] == 1
    assert len(trainer._hpt_carryover_samples) == 1  # deferred, held for next step

    _, batch2 = await trainer._get_samples_from_queue()
    assert len(batch2) == 16
    assert batch2.meta_info["fully_async/hpt_carryover_in_groups"] == 1  # carried SFT seeded first
    assert batch2.meta_info["fully_async/hpt_carryover_out_groups"] == 0  # landed aligned
    assert len(trainer._hpt_carryover_samples) == 0  # carried group trained, carryover drained
    assert batch2.meta_info["fully_async/hpt_carryover_discarded_groups"] == 0  # nothing dropped
    # Reconciliation identity holds across the carry (fresh = retained + out + discarded - in).
    m2 = batch2.meta_info
    assert m2["fully_async/hpt_fresh_pulled_groups"] == (
        m2["fully_async/hpt_collected_queue_samples"]
        + m2["fully_async/hpt_carryover_out_groups"]
        + m2["fully_async/hpt_carryover_discarded_groups"]
        - m2["fully_async/hpt_carryover_in_groups"]
    )


@pytest.mark.asyncio
async def test_async_hpt_carryover_is_discarded_not_re_deferred_when_fresh_cannot_absorb_residue():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    # Safety valve: a carried-over group's staleness must never exceed one step. Here the carried
    # groups are two SFT (2 rows) and every fresh group is RL (8 rows, rollout_n=8), so the residue
    # (2 rows mod 8) can ONLY be absorbed by the carried SFT groups -- the protected plan fails and
    # the fallback would re-defer them. Instead they are DISCARDED (dropped, not re-carried), so the
    # batch aligns to 8 rows and carryover empties within one step.
    carried = [_make_hpt_queue_sample(route_kind="sft", idx=100 + j, rollout_n=8) for j in range(2)]
    fresh = [_make_hpt_queue_sample(route_kind="rl", idx=j, rollout_n=8) for j in range(3)]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 2
    trainer._hpt_carryover_samples = [ray.cloudpickle.dumps(s) for s in carried]  # seeded from a prior step
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(s) for s in fresh])
    trainer.config = _make_async_hpt_config(rollout_n=8)
    trainer.config.trainer.balance_batch = True
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    trainer.config.async_training = {"max_completed_prompt_groups": 64}
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None

    _, batch = await trainer._get_samples_from_queue()

    assert batch.meta_info["fully_async/hpt_required_training_multiple"] == 8
    assert len(batch) == 8 and len(batch) % 8 == 0
    assert batch.meta_info["fully_async/hpt_carryover_in_groups"] == 2
    assert batch.meta_info["fully_async/hpt_carryover_out_groups"] == 0  # nothing re-carried
    assert batch.meta_info["fully_async/hpt_carryover_discarded_groups"] == 2  # the stale SFT dropped
    assert len(trainer._hpt_carryover_samples) == 0  # staleness bounded to one step
    m = batch.meta_info
    assert m["fully_async/hpt_fresh_pulled_groups"] == (
        m["fully_async/hpt_collected_queue_samples"]
        + m["fully_async/hpt_carryover_out_groups"]
        + m["fully_async/hpt_carryover_discarded_groups"]
        - m["fully_async/hpt_carryover_in_groups"]
    )


@pytest.mark.asyncio
async def test_async_hpt_trainer_batch_preserves_async_param_version_metrics():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)

    queue_samples = [
        _make_hpt_queue_sample(route_kind="rl", idx=0),
        _make_hpt_queue_sample(route_kind="rl", idx=1),
        _make_hpt_queue_sample(route_kind="sft", idx=2),
        _make_hpt_queue_sample(route_kind="sft", idx=3),
        _make_hpt_queue_sample(route_kind="rl", idx=4),
        _make_hpt_queue_sample(route_kind="sft", idx=5),
        _make_hpt_queue_sample(route_kind="sft", idx=6),
    ]
    trainer.required_samples = 4
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.trainer.balance_batch = True
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=4)
    trainer.use_critic = False
    trainer.tokenizer = None
    trainer.hpt_assembler = None
    trainer.current_param_version = 5
    trainer.stale_trajectory_processed = 0

    _, batch = await trainer._get_samples_from_queue()
    metrics = {}

    trainer._collect_metrics_from_samples(batch, metrics)

    assert len(batch.meta_info["trajectory_param_versions"]) == 16
    assert batch.meta_info["param_version_diversity"] == 4
    assert batch.meta_info["fully_async/partial/total_partial_num"] == 0
    assert metrics["fully_async/count/stale_trajectory_processed"] == 16
    assert metrics["fully_async/count/current_param_version"] == 5


@pytest.mark.asyncio
async def test_async_hpt_trainer_requires_param_version_metadata_after_hpt_assembly():
    import ray

    from verl.experimental.fully_async_policy.detach_utils import RolloutSample
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    assembled_batch = _make_generated_attempt_payload(0, prompt_uid="prompt-a")
    assembled_batch.non_tensor_batch.pop("min_global_steps")
    sample = RolloutSample(
        full_batch=assembled_batch,
        sample_id="sample-0",
        epoch=0,
        rollout_status={},
        hpt_route=_make_hpt_route(is_sft=False, prompt_uid="prompt-a", group_uid="uid_sample_0_1"),
    )

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 1
    trainer.message_queue_client = _OneSampleQueue(ray.cloudpickle.dumps(sample))
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    trainer.config.actor_rollout_ref.rollout.n = 1
    trainer.config.trainer.balance_batch = False
    trainer.tokenizer = None
    trainer.hpt_assembler = _FixedHptAssembler(assembled_batch)
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.use_critic = False

    with pytest.raises(ValueError, match="min_global_steps"):
        await trainer._get_samples_from_queue()


@pytest.mark.asyncio
async def test_async_hpt_batch_reaches_reward_advantage_and_actor_loss_contract():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    trainer = _make_row_aware_hpt_trainer(_make_mixed_hpt_queue_samples())

    _, batch = await trainer._get_samples_from_queue()
    trainer._collect_metrics_from_samples(batch, trainer.metrics)
    batch = trainer._fit_compute_reward(batch)
    batch = trainer._fit_compute_log_prob(batch)
    batch = trainer._fit_compute_ref_log_prob(batch)
    batch = trainer._fit_compute_critic(batch)
    batch = trainer._fit_compute_advantage(batch)

    assert len(batch) == 16
    for key in (
        "old_log_probs",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "returns",
        "hpt_is_sft",
    ):
        assert key in batch.batch
    for removed_key in ("hpt_seq_weight", "hpt_length_divisor", "hpt_loss_denominator"):
        assert removed_key not in batch.batch
    assert tuple(batch.batch["old_log_probs"].shape) == tuple(batch.batch["response_mask"].shape)
    assert tuple(batch.batch["advantages"].shape) == tuple(batch.batch["response_mask"].shape)
    assert tuple(batch.batch["returns"].shape) == tuple(batch.batch["response_mask"].shape)
    assert torch.isfinite(batch.batch["advantages"]).all()
    assert torch.isfinite(batch.batch["returns"]).all()
    assert trainer.metrics["hpt/old_logprob_from_rollout"] == 1.0
    assert trainer.metrics["hpt/num_sft_rows"] == 4.0
    assert trainer.metrics["hpt/num_rl_groups"] == 3.0

    actor_config = _make_actor_config()
    actor_data = batch.batch.clone()
    tu.assign_non_tensor(
        actor_data,
        dp_size=4,
        batch_num_tokens=int(batch.batch["response_mask"].sum().item()),
        global_batch_size=len(batch),
    )
    full_sequence_token_count = int(batch.batch["attention_mask"].sum().item())
    model_log_probs = torch.zeros(full_sequence_token_count, dtype=torch.float32, requires_grad=True)

    policy_loss, loss_metrics = ppo_loss(
        config=actor_config,
        model_output={"log_probs": model_log_probs},
        data=actor_data,
    )
    policy_loss.backward()

    assert torch.isfinite(policy_loss).item()
    assert model_log_probs.grad is not None
    assert torch.isfinite(model_log_probs.grad).all()
    assert "hpt/b_eff" not in loss_metrics
    assert "hpt/sft_loss_component" not in loss_metrics
    assert "hpt/rl_loss_component" not in loss_metrics


@pytest.mark.asyncio
async def test_async_hpt_entry_anchor_still_emits_routing_composition_metrics(monkeypatch):
    # Regression: the HPT routing-composition metrics (offline_data_ratio, num_sft,
    # num_rl_groups, missing_tau_count, p_success_zero_ratio) describe the assembled
    # batch, not the old-logprob anchor, so they must be emitted in BOTH anchor modes.
    # A version that collected them only inside the rollout-anchor branch dropped every
    # one of these for entry-anchor (decoupled) mode -- i.e. for all M-family runs.
    from omegaconf import open_dict

    from verl.protocol import DataProto

    trainer = _make_row_aware_hpt_trainer(_make_mixed_hpt_queue_samples())
    with open_dict(trainer.config):
        trainer.config.async_hpt.rl_old_logprob_source = "entry"
        trainer.config.async_hpt.entry_proximal = "recent"
        # entry anchor requires the decoupling w-slot contract (token TIS, no rejection).
        trainer.config.algorithm.rollout_correction = {
            "rollout_is": "token",
            "rollout_is_threshold": 2.0,
            "rollout_rs": None,
            "bypass_mode": False,
        }
        trainer.config.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-sum-norm"
        trainer.config.actor_rollout_ref.actor.loss_scale_factor = 8192

    # Entry mode recomputes old_log_probs via the actor forward (option B, current
    # weights). Stub that expensive boundary with a zero proximal so this stays a
    # CPU-only contract test focused on metric emission, not the forward itself.
    def _fake_recompute_old_log_prob(batch):
        response_mask = batch.batch["response_mask"]
        return (
            DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros_like(response_mask, dtype=torch.float32),
                    "entropys": torch.zeros_like(response_mask, dtype=torch.float32),
                }
            ),
            0.0,
        )

    monkeypatch.setattr(trainer, "_compute_old_log_prob", _fake_recompute_old_log_prob)

    _, batch = await trainer._get_samples_from_queue()
    trainer._fit_compute_log_prob(batch)

    # Entry anchor skips the rollout-anchor branch, so its rollout-only flag must be absent...
    assert "hpt/old_logprob_from_rollout" not in trainer.metrics
    # ...but the routing-composition contract must hold identically to the rollout-anchor path.
    assert trainer.metrics["hpt/num_sft"] == 4.0
    assert trainer.metrics["hpt/num_rl_groups"] == 3.0
    assert trainer.metrics["hpt/offline_data_ratio"] == pytest.approx(4.0 / 7.0)
    assert "hpt/missing_tau_count" in trainer.metrics
    assert "hpt/p_success_zero_ratio" in trainer.metrics


def test_hpt_all_rl_policy_loss_matches_vanilla_policy_loss():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    hpt_config = _make_actor_config()
    vanilla_config = _make_actor_config()
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]),
            "response_mask": response_mask,
            "old_log_probs": torch.tensor([[-0.2, -0.3, 0.0, 0.0], [-0.1, -0.2, -0.4, 0.0]]),
            "advantages": torch.tensor([[1.0, 0.5, 0.0, 0.0], [0.2, -0.3, 0.4, 0.0]]),
            "hpt_is_sft": torch.tensor([False, False]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum().item()), global_batch_size=2)
    vanilla_data = data.exclude("hpt_is_sft")
    log_probs = torch.tensor(
        [0.0, -0.15, -0.25, 0.0, 0.0, -0.05, -0.15, -0.35, 0.0],
        requires_grad=True,
    )

    hpt_loss, hpt_metrics = ppo_loss(hpt_config, {"log_probs": log_probs}, data)
    vanilla_loss, vanilla_metrics = ppo_loss(vanilla_config, {"log_probs": log_probs}, vanilla_data)

    assert hpt_loss.item() == pytest.approx(vanilla_loss.item(), rel=1e-6, abs=1e-6)
    assert hpt_metrics["actor/pg_clipfrac"].aggregate() == pytest.approx(
        vanilla_metrics["actor/pg_clipfrac"].aggregate()
    )
    assert "hpt/b_eff" not in hpt_metrics


def test_hpt_sft_self_detach_keeps_ratio_one_and_masks_auxiliary_terms():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    config = _make_actor_config(entropy_coeff=0.5, use_kl_loss=True, kl_loss_type="mse")
    response_mask = torch.ones((2, 4), dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": torch.tensor([[-10.0, -10.0, -10.0, -10.0], [-0.3, -0.4, -0.5, -0.6]]),
            "ref_log_prob": torch.tensor([[10.0, 10.0, 10.0, 10.0], [1.0, 1.0, 1.0, 1.0]]),
            "advantages": torch.ones((2, 4)),
            "hpt_is_sft": torch.tensor([True, False]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=8, global_batch_size=2)
    log_probs = torch.full((12,), -0.2, requires_grad=True)
    entropy = torch.cat([torch.full((6,), 100.0), torch.ones(6)])

    _, metrics = ppo_loss(config, {"log_probs": log_probs, "entropy": entropy}, data)

    assert metrics["actor/ppo_kl"].aggregate() == pytest.approx(-0.125, abs=1e-6)
    assert metrics["actor/entropy"].aggregate() == pytest.approx(0.5, abs=1e-6)
    assert metrics["actor/entropy_loss"].aggregate() == pytest.approx(0.5, abs=1e-6)
    assert metrics["kl_loss"].aggregate() == pytest.approx(0.36, abs=1e-6)


def test_hpt_truncated_rl_rows_are_dead_for_entropy_terms():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    config = _make_actor_config(entropy_coeff=0.5)
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],  # clean RL row: entropy should survive
            [1, 1, 1, 1],  # truncated RL row: dead for actor-update entropy
            [1, 1, 1, 1],  # SFT row: excluded while hpt_sft_entropy_enabled=False
        ],
        dtype=torch.long,
    )
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2], [1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 0, 0], [5, 6, 7, 8], [9, 10, 11, 12]]),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros((3, 4)),
            "advantages": torch.ones((3, 4)),
            "hpt_is_sft": torch.tensor([False, False, True]),
            "hpt_is_truncated_rl": torch.tensor([False, True, False]),
        },
        batch_size=[3],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum().item()), global_batch_size=3)
    log_probs = torch.zeros(16, requires_grad=True)
    entropy = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,  # row 0: clean RL response tokens contribute
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,  # row 1: truncated RL response tokens must be excluded
            0.0,
            0.0,
            200.0,
            200.0,  # row 2: SFT response tokens must stay excluded
        ]
    )

    _, metrics = ppo_loss(config, {"log_probs": log_probs, "entropy": entropy}, data)

    assert metrics["actor/entropy_loss"].aggregate() == pytest.approx(1 / 6, abs=1e-6)
    assert metrics["actor/_entropy_rl_count"].aggregate() == pytest.approx(2.0, abs=1e-6)
    assert metrics["actor/_entropy_rl_sum"].aggregate() == pytest.approx(2.0, abs=1e-6)


def test_hpt_sft_auxiliary_terms_can_be_enabled_explicitly():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    config = _make_actor_config(entropy_coeff=0.5, use_kl_loss=True, kl_loss_type="mse")
    response_mask = torch.ones((2, 4), dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2], [1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros((2, 4)),
            "ref_log_prob": torch.tensor([[10.0, 10.0, 10.0, 10.0], [1.0, 1.0, 1.0, 1.0]]),
            "advantages": torch.ones((2, 4)),
            "hpt_is_sft": torch.tensor([True, False]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=8,
        global_batch_size=2,
        hpt_sft_entropy_enabled=True,
        hpt_sft_kl_enabled=True,
    )
    log_probs = torch.zeros(12, requires_grad=True)
    entropy = torch.cat([torch.full((6,), 100.0), torch.ones(6)])

    _, metrics = ppo_loss(config, {"log_probs": log_probs, "entropy": entropy}, data)

    assert metrics["actor/entropy"].aggregate() == pytest.approx(50.5, abs=1e-6)
    assert metrics["actor/entropy_loss"].aggregate() == pytest.approx(50.5, abs=1e-6)
    assert metrics["kl_loss"].aggregate() == pytest.approx(25.25, abs=1e-6)


def test_hpt_monitoring_rejects_obsolete_loss_weight_fields():
    from verl.experimental.fully_async_policy.hpt_training import collect_hpt_batch_monitoring_metrics

    batch = _make_generated_attempt_payload(0, prompt_uid="prompt-a")
    batch.batch["hpt_is_sft"] = torch.tensor([False], dtype=torch.bool)
    batch.batch["hpt_seq_weight"] = torch.tensor([0.0])
    batch.non_tensor_batch["hpt_group_uid"] = np.array(["uid_sample_0_1"], dtype=object)
    batch.non_tensor_batch["hpt_route_is_sft"] = np.array([False], dtype=object)
    batch.non_tensor_batch["hpt_missing_tau"] = np.array([True], dtype=object)
    batch.non_tensor_batch["hpt_success_probability"] = np.array([1.0], dtype=object)

    with pytest.raises(ValueError, match="obsolete HPT loss fields"):
        collect_hpt_batch_monitoring_metrics(batch)


@pytest.mark.asyncio
async def test_async_hpt_batch_reaches_actor_update_handoff_with_hpt_fields():
    trainer = _make_row_aware_hpt_trainer(_make_mixed_hpt_queue_samples())
    trainer.actor_rollout_wg = _CapturingActorRolloutWorkerGroup(dp_size=4)
    trainer.global_steps = 1
    trainer.config.trainer.critic_warmup = 0
    trainer.config.actor_rollout_ref.actor.calculate_entropy = False
    trainer.config.actor_rollout_ref.actor.entropy_coeff = 0.0
    trainer.config.actor_rollout_ref.actor.ppo_epochs = 1
    trainer.config.actor_rollout_ref.actor.data_loader_seed = 0
    trainer.config.actor_rollout_ref.actor.shuffle = False
    trainer.config.actor_rollout_ref.rollout.multi_turn = {"enable": False}
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    trainer.config.distillation = {"enabled": False}

    batch = await _prepare_hpt_batch_for_actor_update(trainer)
    returned_batch = trainer._fit_update_actor(batch)

    assert returned_batch is batch
    assert trainer.actor_rollout_wg.updated_batch is not None
    assert trainer.metrics["actor/loss"] == 1.25
    assert trainer.metrics["perf/mfu/actor"] == 0.0
    assert batch.meta_info["multi_turn"] is False
    assert batch.meta_info["temperature"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU HPT learner contract smoke.")
def test_GPU_hpt_rollout_logprob_anchor_and_loss_fields_stay_aligned_on_cuda():
    from verl.experimental.fully_async_policy.hpt_training import (
        apply_hpt_rollout_logprob_anchor,
        collect_hpt_batch_monitoring_metrics,
        should_use_hpt_rollout_logprob_anchor,
    )

    config = _make_async_hpt_config(rollout_n=4)
    batch = _make_generated_attempt_payload(0, prompt_uid="prompt-a")
    for key in list(batch.batch.keys()):
        batch.batch[key] = batch.batch[key].to("cuda")

    batch.batch["hpt_is_sft"] = torch.tensor([False], dtype=torch.bool, device="cuda")
    batch.non_tensor_batch["hpt_group_uid"] = np.array(["uid_sample_0_1"], dtype=object)
    batch.non_tensor_batch["hpt_route_is_sft"] = np.array([False], dtype=object)
    batch.non_tensor_batch["hpt_missing_tau"] = np.array([True], dtype=object)
    batch.non_tensor_batch["hpt_success_probability"] = np.array([1.0], dtype=object)

    assert should_use_hpt_rollout_logprob_anchor(config, batch) is True
    anchor_metrics = apply_hpt_rollout_logprob_anchor(batch)
    monitoring_metrics = collect_hpt_batch_monitoring_metrics(batch)

    current_log_probs = torch.zeros_like(batch.batch["old_log_probs"], requires_grad=True)
    response_mask = batch.batch["response_mask"].to(torch.float32)
    surrogate_loss = ((current_log_probs - batch.batch["old_log_probs"].detach()) * response_mask).sum()
    surrogate_loss.backward()

    assert batch.batch["old_log_probs"].is_cuda
    assert tuple(batch.batch["old_log_probs"].shape) == tuple(batch.batch["response_mask"].shape)
    assert torch.isfinite(surrogate_loss).item()
    assert current_log_probs.grad is not None
    assert current_log_probs.grad.is_cuda
    assert anchor_metrics == {"hpt/old_logprob_from_rollout": 1.0, "hpt/num_sft_rows": 0.0}
    assert monitoring_metrics["hpt/num_rl_groups"] == 1.0
    assert monitoring_metrics["hpt/missing_tau_count"] == 1.0
