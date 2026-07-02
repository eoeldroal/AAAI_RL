import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

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
            "hpt_seq_weight",
            "hpt_length_divisor",
            "hpt_loss_denominator",
        ):
            assert key in batch_td.keys(), f"missing update_actor field: {key}"
        batch_size = int(batch_td["hpt_is_sft"].shape[0])
        assert tuple(batch_td["old_log_probs"].shape) == tuple(batch_td["response_mask"].shape)
        assert tuple(batch_td["advantages"].shape) == tuple(batch_td["response_mask"].shape)
        assert tuple(batch_td["response_mask"].shape) == (batch_size, 4)
        assert tuple(batch_td["hpt_is_sft"].shape) == (batch_size,)
        assert tuple(batch_td["hpt_loss_denominator"].shape) == (batch_size,)
        assert tu.get(batch_td, "compute_loss") is True
        assert tu.get(batch_td, "global_batch_size") == batch_size
        return tu.get_tensordict({}, non_tensor_dict={"metrics": {"mfu": [0.0], "loss": [1.25]}})


class _EmptyTauStore:
    def get(self, _prompt_uid):
        return None


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
    assert batch.batch["hpt_seq_weight"].tolist() == pytest.approx([0.25] * rollout_n)
    assert batch.batch["hpt_loss_denominator"].tolist() == pytest.approx([1.0] * rollout_n)
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
async def test_async_hpt_trainer_fails_closed_when_queue_window_cannot_form_trainable_rows():
    import ray

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    assembled_batch = _make_generated_attempt_payload(0, prompt_uid="prompt-a")
    queue_samples = [_make_hpt_queue_sample(route_kind="rl", idx=idx, rollout_n=4) for idx in range(32)]

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.required_samples = 1
    trainer.message_queue_client = _QueueReader([ray.cloudpickle.dumps(sample) for sample in queue_samples])
    trainer.config = _make_async_hpt_config(rollout_n=4)
    trainer.config.trainer.balance_batch = False
    trainer.tokenizer = None
    trainer.hpt_assembler = _FixedHptAssembler(assembled_batch)
    trainer.actor_rollout_wg = _FakeActorRolloutWorkerGroup(dp_size=1)
    trainer.use_critic = False

    with pytest.raises(ValueError, match="could not form a trainable batch"):
        await trainer._get_samples_from_queue()

    assert trainer.message_queue_client.calls == 32


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
    from verl.workers.config import ActorConfig
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
        "hpt_seq_weight",
        "hpt_length_divisor",
        "hpt_loss_denominator",
    ):
        assert key in batch.batch
    assert tuple(batch.batch["old_log_probs"].shape) == tuple(batch.batch["response_mask"].shape)
    assert tuple(batch.batch["advantages"].shape) == tuple(batch.batch["response_mask"].shape)
    assert tuple(batch.batch["returns"].shape) == tuple(batch.batch["response_mask"].shape)
    assert torch.isfinite(batch.batch["advantages"]).all()
    assert torch.isfinite(batch.batch["returns"]).all()
    assert trainer.metrics["hpt/old_logprob_from_rollout"] == 1.0
    assert trainer.metrics["hpt/num_sft_rows"] == 4.0
    assert trainer.metrics["hpt/num_rl_groups"] == 3.0

    actor_config = ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size=4,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
        loss_agg_mode="token-mean",
        policy_loss={"loss_mode": "vanilla"},
        global_batch_info={"dp_size": 4},
    )
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
    assert "hpt/b_eff" in loss_metrics
    assert "hpt/sft_loss_component" in loss_metrics
    assert "hpt/rl_loss_component" in loss_metrics


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
    batch.batch["hpt_seq_weight"] = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    batch.batch["hpt_length_divisor"] = torch.tensor([4.0], dtype=torch.float32, device="cuda")
    batch.batch["hpt_loss_denominator"] = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    batch.non_tensor_batch["hpt_group_uid"] = np.array(["uid_sample_0_1"], dtype=object)
    batch.non_tensor_batch["hpt_route_is_sft"] = np.array([False], dtype=object)
    batch.non_tensor_batch["hpt_missing_tau"] = np.array([True], dtype=object)
    batch.non_tensor_batch["hpt_success_probability"] = np.array([1.0], dtype=object)

    assert should_use_hpt_rollout_logprob_anchor(config, batch) is True
    anchor_metrics = apply_hpt_rollout_logprob_anchor(batch)
    monitoring_metrics = collect_hpt_batch_monitoring_metrics(batch)

    current_log_probs = torch.zeros_like(batch.batch["old_log_probs"], requires_grad=True)
    response_mask = batch.batch["response_mask"].to(torch.float32)
    surrogate_loss = (
        (current_log_probs - batch.batch["old_log_probs"].detach())
        * response_mask
        * batch.batch["hpt_seq_weight"].view(-1, 1)
    ).sum() / batch.batch["hpt_loss_denominator"].sum()
    surrogate_loss.backward()

    assert batch.batch["old_log_probs"].is_cuda
    assert tuple(batch.batch["old_log_probs"].shape) == tuple(batch.batch["response_mask"].shape)
    assert torch.isfinite(surrogate_loss).item()
    assert current_log_probs.grad is not None
    assert current_log_probs.grad.is_cuda
    assert anchor_metrics == {"hpt/old_logprob_from_rollout": 1.0, "hpt/num_sft_rows": 0.0}
    assert monitoring_metrics["hpt/num_rl_groups"] == 1.0
    assert monitoring_metrics["hpt/missing_tau_count"] == 1.0
