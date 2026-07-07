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

"""Training-side per-token dump for offline ablation analysis.

The generation-side rollout dump (``skip.async_rollout``) captures the rollout
output only, *before* reward scoring and *before* the loss boundary. The tensors
that make the DR-004/005 ablations analyzable -- the entry/current/rollout
log-probs, the advantage, and the per-token clip inputs -- exist only inside the
trainer after ``_fit_compute_log_prob`` + ``_fit_compute_advantage``. This module
samples those loss-boundary tensors to disk so quantities like the decoupled
``w``/``r`` decomposition (A5) and the clip death density (A6c) can be
reconstructed offline without pre-committing to a wandb scalar for each cut.

Design constraints (see ``docs/Ablation_RL.md`` for the full rationale):
  - Read-only: the live training batch is never moved, cast, or mutated.
  - Sampled: only every ``sample_every_n_steps`` steps, capped at ``max_rows``.
  - Offloaded: serialization + disk write run on a background thread; if a prior
    write is still in flight on a dump step the step is skipped, so the trainer
    hot path never blocks (mirrors the generation dump's executor offload).
  - Disabled by default: the base fully-async RL/HPT path is untouched.
"""

from __future__ import annotations

import atexit
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from verl import DataProto

logger = logging.getLogger(__name__)

# Loss-boundary per-token tensors absent from the generation-side rollout dump.
# Missing keys are silently skipped so the base RL path (no HPT fields) still dumps.
DUMP_TENSOR_KEYS: tuple[str, ...] = (
    "response_mask",
    "log_probs",
    "old_log_probs",
    "rollout_log_probs",
    "advantages",
    "hpt_is_sft",
    "hpt_is_truncated_rl",
    "token_level_scores",
    "token_level_rewards",
)
# Per-row provenance: offline join key + staleness reconstruction.
DUMP_NON_TENSOR_KEYS: tuple[str, ...] = (
    "uid",
    "prompt_uid",
    "min_global_steps",
    "max_global_steps",
)
# Float tensors are stored in the configured dtype; masks/ids keep their dtype.
_FLOAT_CAST_KEYS = frozenset(
    {
        "log_probs",
        "old_log_probs",
        "rollout_log_probs",
        "advantages",
        "token_level_scores",
        "token_level_rewards",
    }
)
_STORAGE_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class TrainingDumpConfig(BaseModel):
    """Validated in-process view of the ``training_dump`` config block."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = False
    dir: str | None = None
    sample_every_n_steps: int = Field(default=20, ge=1)
    max_rows: int = Field(default=256, ge=1)
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    offload: bool = True

    @field_validator("dir")
    @classmethod
    def _strip_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


def load_training_dump_config(config: DictConfig) -> TrainingDumpConfig:
    """Load and validate the local ``training_dump`` block (absent block -> disabled)."""

    raw_config = OmegaConf.select(config, "training_dump", default={})
    if isinstance(raw_config, DictConfig):
        payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    else:
        payload = raw_config
    if payload is None:
        payload = {}

    try:
        dump_config = TrainingDumpConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid training_dump config: {exc}") from exc

    if dump_config.enable and dump_config.dir is None:
        raise ValueError("training_dump.enable=true requires training_dump.dir to be set")
    return dump_config


def _write_dataproto(payload: DataProto, path: str) -> None:
    """Serialize atomically so an interrupted write never leaves a half file on disk."""
    tmp_path = f"{path}.tmp"
    payload.save_to_disk(tmp_path)
    os.replace(tmp_path, path)


def _safe_write_dataproto(payload: DataProto, path: str) -> None:
    """Best-effort write: a diagnostic dump must never crash a training run (I/O edge).

    Runs on both the sync and offloaded paths, so a failure is logged rather than
    propagated into ``fit_step`` (sync) or silently swallowed by the Future (offload).
    """
    try:
        _write_dataproto(payload, path)
    except Exception as exc:  # noqa: BLE001 - I/O edge, log and continue (see rollout_skip)
        logger.warning("training_dump: failed to write %s: %s", path, exc)


class TrainingTensorDumper:
    """Samples loss-boundary per-token tensors to disk for offline ablation analysis.

    Never mutates the live batch: each dumped tensor is an independent CPU clone.
    """

    def __init__(self, config: TrainingDumpConfig):
        self.config = config
        self._dir: Optional[Path] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Optional[Future] = None
        self._dropped_busy = 0
        if config.enable:
            self._dir = Path(config.dir).expanduser().resolve()
            self._dir.mkdir(parents=True, exist_ok=True)
            if config.offload:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-dump")
                atexit.register(self.close)

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def should_dump(self, step: int) -> bool:
        return self.config.enable and step >= 0 and (step % self.config.sample_every_n_steps == 0)

    def _extract(self, batch: DataProto, step: int, param_version: int, local_trigger_step: int) -> Optional[DataProto]:
        """Build an independent CPU DataProto of the loss-boundary tensors (read-only)."""
        if batch is None or batch.batch is None:
            return None
        total_rows = int(batch.batch.batch_size[0])
        if total_rows == 0:
            return None
        n = min(total_rows, self.config.max_rows)
        cast_dtype = _STORAGE_DTYPES[self.config.dtype]

        tensors: dict[str, torch.Tensor] = {}
        for key in DUMP_TENSOR_KEYS:
            if key not in batch.batch:
                continue
            tensor = batch.batch[key][:n].detach().to("cpu")
            if key in _FLOAT_CAST_KEYS and tensor.is_floating_point():
                tensor = tensor.to(cast_dtype)
            # clone() guarantees the payload shares no storage with the live batch,
            # so the background writer is safe even when the batch is already on CPU.
            tensors[key] = tensor.clone()
        if not tensors:
            return None

        non_tensors: dict[str, np.ndarray] = {}
        for key in DUMP_NON_TENSOR_KEYS:
            if key in batch.non_tensor_batch:
                non_tensors[key] = np.array(batch.non_tensor_batch[key][:n], copy=True)

        meta_info = {
            "training_dump/global_step": int(step),
            "training_dump/param_version": int(param_version),
            "training_dump/local_trigger_step": int(local_trigger_step),
            "training_dump/total_rows": total_rows,
            "training_dump/dumped_rows": n,
        }
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors or None,
            meta_info=meta_info,
        )

    def maybe_dump(
        self, batch: DataProto, *, step: int, param_version: int, local_trigger_step: int = 0
    ) -> Optional[str]:
        """Dump this step's loss-boundary tensors if sampling selects it.

        Returns the written path (sync mode) or ``None`` (skipped / offloaded).
        """
        if not self.should_dump(step):
            return None
        payload = self._extract(batch, step, param_version, local_trigger_step)
        if payload is None:
            return None
        path = str(self._dir / f"step_{int(step):08d}.dp")

        if self._executor is None:
            _safe_write_dataproto(payload, path)
            return path

        # Never block the trainer: if the previous write has not finished, skip.
        if self._pending is not None and not self._pending.done():
            self._dropped_busy += 1
            logger.warning(
                "training_dump: previous write still in flight; skipping step %d (dropped_busy=%d)",
                step,
                self._dropped_busy,
            )
            return None
        self._pending = self._executor.submit(_safe_write_dataproto, payload, path)
        return None

    def close(self) -> None:
        """Flush pending writes and release the executor (idempotent)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
