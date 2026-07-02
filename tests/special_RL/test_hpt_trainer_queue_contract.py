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

    async def get_sample(self):
        if not self.payloads:
            return None, 0
        return self.payloads.pop(0), len(self.payloads)


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
            "algorithm": {"adv_estimator": "grpo", "norm_adv_by_std_in_grpo": False},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "vanilla"}},
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


def _make_source_batch(*, rollout_n=4, prompt_uid="prompt-a"):
    from verl.protocol import DataProto

    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.arange(rollout_n).view(rollout_n, 1)},
        non_tensors={"prompt_uid": np.array([prompt_uid] * rollout_n, dtype=object)},
    )


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
        }
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
