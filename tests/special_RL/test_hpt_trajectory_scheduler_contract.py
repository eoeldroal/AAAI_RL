import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("ray")


@pytest.mark.asyncio
async def test_hpt_trajectory_scheduler_attempt_batches_keep_async_rollout_uid(monkeypatch):
    from verl.experimental.fully_async_policy.detach_utils import RolloutSample
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from verl.experimental.fully_async_policy.hpt_rollout_accumulator import HptPromptGroupAccumulator
    from verl.protocol import DataProto

    rollouter_cls = FullyAsyncRollouter.__ray_metadata__.modified_class
    seen_attempts = []

    async def record_attempt(self, *, group_uid, prompt_uid, rollout_index, attempt_batch):
        seen_attempts.append(
            {
                "group_uid": group_uid,
                "prompt_uid": prompt_uid,
                "rollout_index": rollout_index,
                "uid": str(attempt_batch.non_tensor_batch["uid"][0]),
                "hpt_rollout_index": int(attempt_batch.non_tensor_batch["hpt_rollout_index"][0]),
            }
        )

    monkeypatch.setattr(
        rollouter_cls,
        "_process_hpt_trajectory_attempt_streaming",
        record_attempt,
    )

    rollouter = object.__new__(rollouter_cls)
    rollouter.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 4}},
            "async_hpt": {"trajectory_scheduler": {"enabled": True}},
        }
    )
    rollouter.hpt_rollout_gate = object()
    rollouter.hpt_rollout_accumulator = HptPromptGroupAccumulator(rollout_n=4)
    rollouter.hpt_scheduler_groups = {}
    rollouter.hpt_closed_group_uids = set()
    rollouter.lock = asyncio.Lock()
    rollouter.active_tasks = set()
    rollouter.max_concurrent_trajectory_attempts = 64
    rollouter.paused = False
    rollouter._resume_event = asyncio.Event()
    rollouter._resume_event.set()

    source_batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(4, 1, dtype=torch.uint8)},
        non_tensors={"prompt_uid": np.array(["prompt-a"] * 4, dtype=object)},
    )
    rollout_sample = RolloutSample(
        full_batch=source_batch,
        sample_id="sample_0_1",
        epoch=0,
        rollout_status={},
    )

    await rollouter._submit_hpt_trajectory_attempts(rollout_sample)
    await asyncio.gather(*list(rollouter.active_tasks))

    assert seen_attempts == [
        {
            "group_uid": "uid_sample_0_1",
            "prompt_uid": "prompt-a",
            "rollout_index": 0,
            "uid": "uid_sample_0_1",
            "hpt_rollout_index": 0,
        },
        {
            "group_uid": "uid_sample_0_1",
            "prompt_uid": "prompt-a",
            "rollout_index": 1,
            "uid": "uid_sample_0_1",
            "hpt_rollout_index": 1,
        },
        {
            "group_uid": "uid_sample_0_1",
            "prompt_uid": "prompt-a",
            "rollout_index": 2,
            "uid": "uid_sample_0_1",
            "hpt_rollout_index": 2,
        },
        {
            "group_uid": "uid_sample_0_1",
            "prompt_uid": "prompt-a",
            "rollout_index": 3,
            "uid": "uid_sample_0_1",
            "hpt_rollout_index": 3,
        },
    ]


@pytest.mark.asyncio
async def test_hpt_trajectory_attempt_exception_drops_group_without_crash():
    # Regression guard: a single attempt's generation raising must NOT crash the
    # rollouter and must NOT strand the group's already-completed sibling attempts
    # in the accumulator. It must fail closed by dropping the whole prompt group.
    from verl.experimental.fully_async_policy.detach_utils import RolloutSample
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from verl.experimental.fully_async_policy.hpt_rollout_accumulator import (
        HptPromptGroupAccumulator,
        HptTrajectoryAttemptResult,
    )
    from verl.protocol import DataProto

    rollouter_cls = FullyAsyncRollouter.__ray_metadata__.modified_class

    group_uid = "uid_sample_0_1"

    class _BoomManager:
        async def generate_sequences_single(self, attempt_batch):
            raise RuntimeError("sglang boom")

    rollouter = object.__new__(rollouter_cls)
    rollouter.async_rollout_manager = _BoomManager()
    rollouter.hpt_rollout_gate = object()
    rollouter.hpt_rollout_accumulator = HptPromptGroupAccumulator(rollout_n=4)
    rollouter.hpt_scheduler_groups = {}
    rollouter.hpt_closed_group_uids = set()
    rollouter.lock = asyncio.Lock()
    rollouter.processed_sample_count = 0

    # A sibling attempt of the same group already completed and is buffered.
    sibling_payload = DataProto.from_dict(tensors={"x": torch.zeros(1, 1)})
    rollouter.hpt_rollout_accumulator.add(
        HptTrajectoryAttemptResult(group_uid=group_uid, prompt_uid="prompt-a", rollout_index=0, payload=sibling_payload)
    )
    source_batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(4, 1, dtype=torch.uint8)},
        non_tensors={"prompt_uid": np.array(["prompt-a"] * 4, dtype=object)},
    )
    rollouter.hpt_scheduler_groups[group_uid] = {
        "rollout_sample": RolloutSample(
            full_batch=source_batch, sample_id="sample_0_1", epoch=0, rollout_status={}
        ),
        "source_batch": source_batch,
        "prompt_uid": "prompt-a",
    }
    assert rollouter.hpt_rollout_accumulator.stored_attempt_count() == 1

    attempt_batch = DataProto.from_dict(
        tensors={"x": torch.zeros(1, 1)},
        non_tensors={
            "uid": np.array([group_uid], dtype=object),
            "hpt_rollout_index": np.array([1], dtype=object),
        },
    )

    # Must not raise (no crash).
    await rollouter._process_hpt_trajectory_attempt_streaming(
        group_uid=group_uid, prompt_uid="prompt-a", rollout_index=1, attempt_batch=attempt_batch
    )

    # Group dropped, siblings not stranded, sample accounted as processed.
    assert group_uid in rollouter.hpt_closed_group_uids
    assert group_uid not in rollouter.hpt_scheduler_groups
    assert rollouter.hpt_rollout_accumulator.open_group_count() == 0
    assert rollouter.hpt_rollout_accumulator.stored_attempt_count() == 0
    assert rollouter.processed_sample_count == 1
