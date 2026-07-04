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

"""CPU contract tests for the training-side per-token dump (docs/Ablation_RL.md).

No GPU/Ray: exercises the real config loader, extraction, sampling, offload, and
round-trip serialization on plain DataProto batches. The load-bearing test is
that dumping is *read-only* -- it must never move, cast, or mutate the live batch.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.training_dump import (
    TrainingDumpConfig,
    TrainingTensorDumper,
    load_training_dump_config,
)

_FLOAT_KEYS = ("log_probs", "old_log_probs", "rollout_log_probs", "advantages")


def _make_batch(rows: int = 4, resp_len: int = 6, with_hpt: bool = True, with_scores: bool = True) -> DataProto:
    gen = torch.Generator().manual_seed(1234)
    tensors = {
        "response_mask": torch.ones((rows, resp_len), dtype=torch.long),
        "log_probs": torch.randn((rows, resp_len), generator=gen),
        "old_log_probs": torch.randn((rows, resp_len), generator=gen),
        "rollout_log_probs": torch.randn((rows, resp_len), generator=gen),
        "advantages": torch.randn((rows, resp_len), generator=gen),
    }
    if with_hpt:
        # per-row route flag, shape (rows,) -- matches losses._hpt_sft_mask expectation
        tensors["hpt_is_sft"] = torch.tensor([i % 2 for i in range(rows)], dtype=torch.long)
    if with_scores:
        tensors["token_level_scores"] = torch.randn((rows, resp_len), generator=gen)
    non_tensors = {
        "uid": np.array([f"sample_0_{i}" for i in range(rows)], dtype=object),
        "prompt_uid": np.array([f"p{i}" for i in range(rows)], dtype=object),
        "min_global_steps": np.array([1] * rows, dtype=np.int64),
        "max_global_steps": np.array([2] * rows, dtype=np.int64),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


# --------------------------------------------------------------------------- #
# Config loader
# --------------------------------------------------------------------------- #
def test_config_defaults_disabled():
    cfg = load_training_dump_config(OmegaConf.create({}))
    assert cfg.enable is False
    assert cfg.dir is None
    assert cfg.sample_every_n_steps == 20
    assert cfg.offload is True


def test_config_enable_requires_dir():
    with pytest.raises(ValueError, match="requires training_dump.dir"):
        load_training_dump_config(OmegaConf.create({"training_dump": {"enable": True}}))


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="Invalid training_dump config"):
        load_training_dump_config(OmegaConf.create({"training_dump": {"enable": False, "bogus": 1}}))


def test_config_valid_roundtrip():
    cfg = load_training_dump_config(
        OmegaConf.create(
            {
                "training_dump": {
                    "enable": True,
                    "dir": "/tmp/x",
                    "dtype": "bf16",
                    "sample_every_n_steps": 3,
                    "max_rows": 8,
                }
            }
        )
    )
    assert cfg.enable is True
    assert cfg.dtype == "bf16"
    assert cfg.sample_every_n_steps == 3
    assert cfg.max_rows == 8


# --------------------------------------------------------------------------- #
# Sampling / disabled behavior
# --------------------------------------------------------------------------- #
def test_disabled_dumper_is_noop():
    dumper = TrainingTensorDumper(load_training_dump_config(OmegaConf.create({})))
    assert dumper.should_dump(0) is False
    assert dumper.maybe_dump(_make_batch(), step=0, param_version=0) is None


def test_sampling_cadence(tmp_path):
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), sample_every_n_steps=5, offload=False)
    dumper = TrainingTensorDumper(cfg)
    assert dumper.should_dump(0) and dumper.should_dump(5) and dumper.should_dump(10)
    assert not dumper.should_dump(1)
    assert not dumper.should_dump(4)
    assert not dumper.should_dump(-5)


# --------------------------------------------------------------------------- #
# Read-only invariant (the load-bearing test)
# --------------------------------------------------------------------------- #
def test_dump_never_mutates_live_batch(tmp_path):
    batch = _make_batch(rows=4, resp_len=6)
    before = {k: batch.batch[k].clone() for k in batch.batch.keys()}
    dtypes = {k: batch.batch[k].dtype for k in batch.batch.keys()}

    # bf16 storage must NOT cast the live fp32 tensors.
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="bf16", offload=False, sample_every_n_steps=1)
    TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=3, local_trigger_step=1)

    for key in before:
        assert batch.batch[key].dtype == dtypes[key], f"{key} dtype changed"
        assert batch.batch[key].device.type == "cpu", f"{key} device changed"
        assert torch.equal(batch.batch[key], before[key]), f"{key} values changed"


def test_dumped_payload_shares_no_storage_with_live_batch(tmp_path):
    batch = _make_batch(rows=3, resp_len=4)
    original = batch.batch["log_probs"].clone()
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=False, sample_every_n_steps=1)
    path = TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=1)

    loaded = DataProto.load_from_disk(path)
    loaded.batch["log_probs"] += 999.0  # mutate the reconstructed copy
    assert torch.equal(batch.batch["log_probs"], original), "live batch aliased the dumped payload"


# --------------------------------------------------------------------------- #
# Round-trip fidelity / provenance
# --------------------------------------------------------------------------- #
def test_roundtrip_fidelity_and_meta(tmp_path):
    batch = _make_batch(rows=3, resp_len=5)
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=False, sample_every_n_steps=1)
    path = TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=2, local_trigger_step=4)
    assert path is not None

    loaded = DataProto.load_from_disk(path)
    for key in _FLOAT_KEYS + ("response_mask", "hpt_is_sft", "token_level_scores"):
        assert torch.equal(loaded.batch[key], batch.batch[key]), f"{key} not faithful"
    # offline join key + staleness provenance survive
    assert list(loaded.non_tensor_batch["prompt_uid"]) == list(batch.non_tensor_batch["prompt_uid"])
    assert list(loaded.non_tensor_batch["max_global_steps"]) == [2, 2, 2]
    assert loaded.meta_info["training_dump/global_step"] == 0
    assert loaded.meta_info["training_dump/param_version"] == 2
    assert loaded.meta_info["training_dump/local_trigger_step"] == 4
    assert loaded.meta_info["training_dump/total_rows"] == 3
    assert loaded.meta_info["training_dump/dumped_rows"] == 3


def test_atomic_write_leaves_no_tmp(tmp_path):
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=False, sample_every_n_steps=1)
    TrainingTensorDumper(cfg).maybe_dump(_make_batch(), step=0, param_version=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_00000000.dp"]


# --------------------------------------------------------------------------- #
# Row cap / dtype cast / optional keys
# --------------------------------------------------------------------------- #
def test_row_cap_takes_first_n_and_records_total(tmp_path):
    batch = _make_batch(rows=10, resp_len=4)
    cfg = TrainingDumpConfig(
        enable=True, dir=str(tmp_path), dtype="fp32", offload=False, sample_every_n_steps=1, max_rows=4
    )
    path = TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=1)

    loaded = DataProto.load_from_disk(path)
    assert loaded.batch["log_probs"].shape[0] == 4
    assert torch.equal(loaded.batch["log_probs"], batch.batch["log_probs"][:4])
    assert list(loaded.non_tensor_batch["prompt_uid"]) == [f"p{i}" for i in range(4)]
    assert loaded.meta_info["training_dump/total_rows"] == 10
    assert loaded.meta_info["training_dump/dumped_rows"] == 4


def test_dtype_cast_bf16_preserves_masks(tmp_path):
    batch = _make_batch(rows=2, resp_len=4)
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="bf16", offload=False, sample_every_n_steps=1)
    path = TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=1)

    loaded = DataProto.load_from_disk(path)
    assert loaded.batch["log_probs"].dtype == torch.bfloat16
    assert loaded.batch["advantages"].dtype == torch.bfloat16
    assert loaded.batch["response_mask"].dtype == torch.long  # mask dtype preserved
    assert loaded.batch["hpt_is_sft"].dtype == torch.long  # per-row flag preserved
    assert torch.allclose(loaded.batch["log_probs"].float(), batch.batch["log_probs"], atol=3e-2)


def test_base_rl_path_without_hpt_or_score_keys(tmp_path):
    batch = _make_batch(rows=2, resp_len=4, with_hpt=False, with_scores=False)
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=False, sample_every_n_steps=1)
    path = TrainingTensorDumper(cfg).maybe_dump(batch, step=0, param_version=1)

    loaded = DataProto.load_from_disk(path)
    assert "hpt_is_sft" not in loaded.batch.keys()
    assert "token_level_scores" not in loaded.batch.keys()
    assert "log_probs" in loaded.batch.keys()
    assert "old_log_probs" in loaded.batch.keys()


# --------------------------------------------------------------------------- #
# Offload path
# --------------------------------------------------------------------------- #
def test_offload_writes_after_close(tmp_path):
    batch = _make_batch(rows=3, resp_len=4)
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=True, sample_every_n_steps=1)
    dumper = TrainingTensorDumper(cfg)
    assert dumper.maybe_dump(batch, step=0, param_version=1) is None  # offloaded -> path not returned
    dumper.close()  # flush background write

    path = tmp_path / "step_00000000.dp"
    assert path.exists()
    loaded = DataProto.load_from_disk(str(path))
    assert torch.equal(loaded.batch["log_probs"], batch.batch["log_probs"])


def test_offload_skips_when_previous_write_in_flight(tmp_path):
    cfg = TrainingDumpConfig(enable=True, dir=str(tmp_path), dtype="fp32", offload=True, sample_every_n_steps=1)
    dumper = TrainingTensorDumper(cfg)

    class _NotDone:
        def done(self):
            return False

    dumper._pending = _NotDone()
    assert dumper.maybe_dump(_make_batch(rows=2, resp_len=3), step=0, param_version=1) is None
    assert dumper._dropped_busy == 1
    assert not (tmp_path / "step_00000000.dp").exists()

    dumper._pending = None
    dumper.close()


# --------------------------------------------------------------------------- #
# Trainer wiring (real _fit_dump_data entry point)
# --------------------------------------------------------------------------- #
def test_fit_dump_data_forwards_to_dumper_and_calls_super():
    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    # Minimal attrs read by the base _fit_dump_data (rollout_data_dir=None -> no-op).
    trainer.timing_raw = {}
    trainer.reward_extra_infos_dict = {}
    trainer.config = OmegaConf.create({"trainer": {"rollout_data_dir": None}})
    trainer.global_steps = 5
    trainer.current_param_version = 2
    trainer.local_trigger_step = 3

    seen = {}

    class _RecordingDumper:
        def maybe_dump(self, batch, *, step, param_version, local_trigger_step):
            seen.update(step=step, param_version=param_version, local_trigger_step=local_trigger_step)

    trainer._training_dumper = _RecordingDumper()
    trainer._fit_dump_data(_make_batch(rows=2, resp_len=3))  # real override

    assert seen == {"step": 5, "param_version": 2, "local_trigger_step": 3}
