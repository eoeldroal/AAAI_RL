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
import os
import signal
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager as RewardLoopDAPORewardManager
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.reward_score.upt_v6_adapter import compute_score as compute_upt_v6_score
from verl.workers.reward_manager.dapo import DAPORewardManager

REPO_ROOT = Path(__file__).resolve().parents[3]
UPT_ENTROPY_MATH_PATH = Path(
    REPO_ROOT / "../Unify-Post-Training/hpt/verl/verl/mix_src/entropy_math/__init__.py"
).resolve()
UPT_V6_ADAPTER_PATH = (REPO_ROOT / "verl/utils/reward_score/upt_v6_adapter.py").resolve()


class _DummyTokenizer:
    eos_token = "</s>"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class _BoxedAnswerTokenizer:
    eos_token = "</s>"

    def decode(self, token_ids, skip_special_tokens=True):
        if int(token_ids[0]) == 7:
            return "The answer is \\boxed{180}."
        if int(token_ids[0]) == 9:
            return "The answer is \\boxed{\\sqrt{4}}."
        return "prompt"


def _constant_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.5


def _signal_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise TimeoutError("unit test alarm")

    signal.signal(signal.SIGALRM, _handler)
    signal.signal(signal.SIGALRM, old_handler)
    return 1.0


def _overlong_buffer_cfg(enable: bool, length: int = 128):
    return OmegaConf.create({"enable": enable, "len": length, "penalty_factor": 1.0, "log": False})


def _make_data(batch_size: int = 2, seq_len: int = 4) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(batch_size, seq_len, dtype=torch.long),
            "responses": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 2 * seq_len, dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "1"}] * batch_size, dtype=object),
            "data_source": np.array(["unit_test"] * batch_size, dtype=object),
        },
    )


def _make_upt_v6_custom_reward_config(entropy_math_path: str):
    return OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": str(UPT_V6_ADAPTER_PATH),
                    "name": "compute_score",
                    "reward_kwargs": {
                        "entropy_math_path": entropy_math_path,
                        "use_process_pool": True,
                        "process_timeout": 30.0,
                    },
                },
                "reward_kwargs": {
                    "compute_score_in_executor": True,
                    "overlong_buffer_cfg": {"enable": False, "len": 128, "penalty_factor": 1.0, "log": False},
                    "max_resp_len": None,
                },
            }
        }
    )


def test_construct_with_overlong_buffer_disabled():
    """max_resp_len is not required when the overlong penalty is disabled. See issue #5858."""
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
        max_resp_len=None,
        overlong_buffer_cfg=_overlong_buffer_cfg(enable=False),
    )
    assert reward_manager.max_resp_len is None


def test_construct_with_overlong_buffer_enabled_requires_max_resp_len():
    with pytest.raises(AssertionError, match="max_resp_len must be provided"):
        DAPORewardManager(
            tokenizer=_DummyTokenizer(),
            num_examine=0,
            compute_score=_constant_compute_score,
            max_resp_len=None,
            overlong_buffer_cfg=_overlong_buffer_cfg(enable=True),
        )


def test_construct_with_overlong_buffer_enabled_rejects_short_max_resp_len():
    with pytest.raises(AssertionError, match="max_resp_len must be larger"):
        DAPORewardManager(
            tokenizer=_DummyTokenizer(),
            num_examine=0,
            compute_score=_constant_compute_score,
            max_resp_len=64,
            overlong_buffer_cfg=_overlong_buffer_cfg(enable=True, length=128),
        )


def test_call_without_overlong_buffer_cfg():
    """The default overlong_buffer_cfg=None must not crash at __call__ time."""
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
    )
    reward_tensor = reward_manager(_make_data())
    assert torch.all(reward_tensor[:, -1] == 0.5)


def test_call_with_overlong_buffer_enabled_applies_penalty():
    reward_manager = DAPORewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_constant_compute_score,
        max_resp_len=4,
        overlong_buffer_cfg=_overlong_buffer_cfg(enable=True, length=2),
    )
    reward_tensor = reward_manager(_make_data(seq_len=4))
    # exceed_len = 4 - (4 - 2) = 2, so overlong_reward = -2 / 2 * 1.0 = -1.0
    assert torch.all(reward_tensor[:, -1] == 0.5 - 1.0)


def test_call_can_use_upt_v6_adapter_as_custom_compute_score(tmp_path):
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        "\n".join(
            [
                "def compute_score(model_response, gt_answer, fast=False):",
                "    assert model_response == '<think>\\nThe answer is \\\\boxed{180}.'",
                "    return gt_answer == '180'",
            ]
        )
    )
    reward_manager = DAPORewardManager(
        tokenizer=_BoxedAnswerTokenizer(),
        num_examine=0,
        compute_score=partial(compute_upt_v6_score, entropy_math_path=str(scorer)),
        overlong_buffer_cfg=_overlong_buffer_cfg(enable=False),
    )
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 2, dtype=torch.long),
            "responses": torch.tensor([[7, 8]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "180"}], dtype=object),
            "data_source": np.array(["numina_olympiads"], dtype=object),
        },
    )

    reward_tensor = reward_manager(data)

    assert reward_tensor.tolist() == [[0.0, 1.0]]


def test_reward_loop_construct_with_overlong_buffer_disabled():
    """The experimental reward loop manager accepts a disabled overlong buffer without max_resp_len."""
    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "overlong_buffer_cfg": {"enable": False, "len": 128, "penalty_factor": 1.0, "log": False},
                    "max_resp_len": None,
                }
            }
        }
    )
    reward_manager = RewardLoopDAPORewardManager(
        config=config, tokenizer=_DummyTokenizer(), compute_score=_constant_compute_score
    )
    assert reward_manager.max_resp_len is None


def test_reward_loop_can_run_signal_based_sync_scorer_without_executor():
    async def _run():
        config = OmegaConf.create(
            {
                "reward": {
                    "reward_kwargs": {
                        "compute_score_in_executor": False,
                        "overlong_buffer_cfg": {"enable": False, "len": 128, "penalty_factor": 1.0, "log": False},
                        "max_resp_len": None,
                    }
                }
            }
        )
        reward_manager = RewardLoopDAPORewardManager(
            config=config, tokenizer=_DummyTokenizer(), compute_score=_signal_compute_score
        )
        return await reward_manager.run_single(_make_data(batch_size=1))

    result = asyncio.run(_run())

    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["acc"] == 1.0


def test_reward_loop_can_use_upt_v6_adapter_without_executor_on_cpu():
    if not UPT_ENTROPY_MATH_PATH.exists():
        pytest.skip(f"UPT entropy_math scorer is not available: {UPT_ENTROPY_MATH_PATH}")

    async def _run():
        config = OmegaConf.create(
            {
                "reward": {
                    "reward_kwargs": {
                        "compute_score_in_executor": False,
                        "overlong_buffer_cfg": {"enable": False, "len": 128, "penalty_factor": 1.0, "log": False},
                        "max_resp_len": None,
                    }
                }
            }
        )
        reward_manager = RewardLoopDAPORewardManager(
            config=config,
            tokenizer=_BoxedAnswerTokenizer(),
            compute_score=partial(compute_upt_v6_score, entropy_math_path=str(UPT_ENTROPY_MATH_PATH)),
        )
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.ones(1, 2, dtype=torch.long),
                "responses": torch.tensor([[9, 8]], dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "2"}], dtype=object),
                "data_source": np.array(["numina_olympiads"], dtype=object),
            },
        )
        return await reward_manager.run_single(data)

    result = asyncio.run(_run())

    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["acc"] == 1.0


@ray.remote(num_cpus=1)
class _RewardLoopDAPORayActor:
    async def run_upt_v6_dapo(self, entropy_math_path: str) -> dict:
        config = _make_upt_v6_custom_reward_config(entropy_math_path)
        compute_score = get_custom_reward_fn(config)
        assert compute_score is not None
        reward_manager = RewardLoopDAPORewardManager(
            config=config,
            tokenizer=_BoxedAnswerTokenizer(),
            compute_score=compute_score,
        )
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.ones(1, 2, dtype=torch.long),
                "responses": torch.tensor([[9, 8]], dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": "2"}], dtype=object),
                "data_source": np.array(["numina_olympiads"], dtype=object),
            },
        )
        return await reward_manager.run_single(data)


def test_reward_loop_can_use_upt_v6_adapter_inside_ray_actor_on_cpu():
    if not UPT_ENTROPY_MATH_PATH.exists():
        pytest.skip(f"UPT entropy_math scorer is not available: {UPT_ENTROPY_MATH_PATH}")

    pythonpath = os.pathsep.join(
        [str(REPO_ROOT), str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    )
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
    )
    try:
        actor = _RewardLoopDAPORayActor.remote()
        result = ray.get(actor.run_upt_v6_dapo.remote(str(UPT_ENTROPY_MATH_PATH)), timeout=60)
    finally:
        ray.shutdown()

    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["acc"] == 1.0
