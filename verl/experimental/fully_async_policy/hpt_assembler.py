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

from numbers import Integral
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.hpt_config import AsyncHptConfig, validate_async_hpt_config
from verl.experimental.fully_async_policy.hpt_gate import HptRouteMetadata
from verl.experimental.fully_async_policy.hpt_payload import HptSftPayload, HptTauToAgentLoopOutputAdapter
from verl.utils.model import compute_position_id_with_mask


class HptBatchAssembler:
    """Materialize routed HPT samples into learner-consumable DataProto rows."""

    def __init__(self, *, config, tokenizer, processor=None):
        self.config = config
        self.hpt_config: AsyncHptConfig = validate_async_hpt_config(config)
        if not self.hpt_config.enabled:
            raise ValueError("HptBatchAssembler requires async_hpt.enabled=true.")
        self.tokenizer = tokenizer
        self.processor = processor
        self.adapter = HptTauToAgentLoopOutputAdapter(
            tokenizer=tokenizer,
            processor=processor,
            apply_chat_template_kwargs=dict(config.data.get("apply_chat_template_kwargs", {})),
        )

    def assemble_rollout_samples(self, rollout_samples: list[Any]) -> DataProto:
        if not rollout_samples:
            raise ValueError("HPT assembly requires at least one rollout sample.")

        return self.concat_training_batches(
            [self.materialize_training_batch(rollout_sample) for rollout_sample in rollout_samples]
        )

    def materialize_training_batch(self, rollout_sample: Any) -> DataProto:
        route = self.require_route(rollout_sample)
        batch = self.materialize_rollout_sample(rollout_sample, route)
        return self.normalize_hpt_training_batch(batch, route)

    def concat_training_batches(self, batches: list[DataProto]) -> DataProto:
        if not batches:
            raise ValueError("HPT assembly requires at least one materialized batch.")

        self.normalize_mixed_schema(batches)
        batch = DataProto.concat(batches)
        batch.meta_info["hpt_sft_entropy_enabled"] = bool(self.hpt_config.sft_entropy_enabled)
        batch.meta_info["hpt_sft_kl_enabled"] = bool(self.hpt_config.sft_kl_enabled)
        return batch

    def materialize_rollout_sample(self, rollout_sample: Any, route: HptRouteMetadata | None = None) -> DataProto:
        route = route or self.require_route(rollout_sample)
        payload = rollout_sample.full_batch
        if route.is_sft:
            if isinstance(payload, HptSftPayload):
                return self.materialize_sft_payload(payload, route)
            if isinstance(payload, DataProto):
                return payload
            raise TypeError(f"HPT SFT route requires HptSftPayload or DataProto, got {type(payload)!r}.")
        if not isinstance(payload, DataProto):
            raise TypeError(f"HPT RL route requires DataProto, got {type(payload)!r}.")
        return payload

    def materialize_sft_payload(self, payload: HptSftPayload, route: HptRouteMetadata) -> DataProto:
        if payload.prompt_uid != route.prompt_uid:
            raise ValueError(
                "HPT SFT payload prompt_uid does not match route metadata: "
                f"payload={payload.prompt_uid!r} route={route.prompt_uid!r}."
            )

        output = self.adapter.to_agent_loop_output(payload)
        prompt_ids = self._pad_token_ids(
            output.prompt_ids,
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            padding_side="left",
            return_attention_mask=True,
        )
        response_ids = self._pad_token_ids(
            output.response_ids,
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            padding_side="right",
            return_attention_mask=True,
        )
        response_mask = self._pad_token_ids(
            output.response_mask,
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            padding_side="right",
            return_attention_mask=False,
            pad_id=0,
        )["input_ids"]

        attention_mask = torch.cat([prompt_ids["attention_mask"], response_ids["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_ids["input_ids"], response_ids["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        tensors = {
            "prompts": prompt_ids["input_ids"],
            "responses": response_ids["input_ids"],
            "response_mask": response_mask * response_ids["attention_mask"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        non_tensors = {
            "__num_turns__": np.array([output.num_turns], dtype=np.int32),
            "raw_prompt": _object_array([payload.messages]),
            "uid": np.array([self._training_uid(route)], dtype=object),
            "prompt_uid": np.array([route.prompt_uid], dtype=object),
            "hpt_group_uid": np.array([route.group_uid], dtype=object),
            "min_global_steps": np.array([0], dtype=object),
            "max_global_steps": np.array([0], dtype=object),
            "extra_info": np.array([{"prompt_uid": route.prompt_uid, "hpt_route": "sft"}], dtype=object),
        }
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info={"metrics": [output.metrics.model_dump()]},
        )

    def normalize_hpt_training_batch(self, batch: DataProto, route: HptRouteMetadata) -> DataProto:
        if not isinstance(batch, DataProto):
            raise TypeError(f"HPT assembly expected DataProto, got {type(batch)!r}.")
        if batch.batch is None:
            raise ValueError("HPT assembly requires tensor batch data.")

        response_mask = self._require_response_mask(batch)
        batch_size = int(response_mask.shape[0])

        self._ensure_repeated_non_tensor(batch, "uid", self._training_uid(route))
        self._ensure_repeated_non_tensor(batch, "prompt_uid", route.prompt_uid)
        self._ensure_repeated_non_tensor(batch, "hpt_group_uid", route.group_uid)
        self._ensure_repeated_non_tensor(batch, "hpt_route_is_sft", bool(route.is_sft))
        self._ensure_repeated_non_tensor(batch, "hpt_missing_tau", bool(route.missing_tau))
        self._ensure_repeated_non_tensor(batch, "hpt_success_probability", float(route.success_probability))
        generated_response_lengths = tuple(route.generated_response_lengths)
        if not generated_response_lengths:
            generated_response_lengths = tuple(
                int(length) for length in response_mask.sum(dim=-1).detach().cpu().tolist()
            )
        self._ensure_repeated_non_tensor(batch, "hpt_generated_response_lengths", generated_response_lengths)
        self._ensure_extra_info_prompt_uid(batch, route.prompt_uid)
        self._normalize_non_tensor_arrays(batch)

        if route.is_sft:
            if batch_size != 1:
                raise ValueError(f"HPT SFT route expects one materialized row, got {batch_size}.")
            self._normalize_sft_tensors(batch, response_mask)
        else:
            self._normalize_rl_tensors(batch, response_mask, route)

        supervised_lengths = response_mask.sum(dim=-1)
        if (supervised_lengths <= 0).any():
            bad_rows = (supervised_lengths <= 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"HPT supervised response length must be positive; bad rows={bad_rows}.")

        batch.batch["hpt_is_sft"] = torch.full(
            (batch_size,), bool(route.is_sft), dtype=torch.bool, device=response_mask.device
        )
        return batch

    def normalize_mixed_schema(self, batches: list[DataProto]) -> None:
        if not batches:
            return

        tensor_keys = set().union(*(set(batch.batch.keys()) for batch in batches if batch.batch is not None))
        for batch in batches:
            missing_tensor_keys = tensor_keys - set(batch.batch.keys())
            if missing_tensor_keys:
                raise ValueError(
                    "HPT mixed tensor schema mismatch before concat. "
                    f"Missing tensor keys {sorted(missing_tensor_keys)} cannot be safely defaulted."
                )

        non_tensor_keys = set().union(*(set(batch.non_tensor_batch.keys()) for batch in batches))
        for batch in batches:
            batch_size = len(batch)
            for key in non_tensor_keys - set(batch.non_tensor_batch.keys()):
                batch.non_tensor_batch[key] = self._default_non_tensor_array(key, batch_size)
            self._normalize_non_tensor_arrays(batch)

    @staticmethod
    def require_route(rollout_sample: Any) -> HptRouteMetadata:
        route = getattr(rollout_sample, "hpt_route", None)
        if route is None:
            sample_id = getattr(rollout_sample, "sample_id", "<unknown>")
            raise ValueError(f"RolloutSample {sample_id!r} has no hpt_route.")
        if isinstance(route, HptRouteMetadata):
            return route
        return HptRouteMetadata.model_validate(route)

    def _pad_token_ids(
        self,
        tokens: list[int],
        *,
        max_length: int,
        padding_side: str,
        return_attention_mask: bool,
        pad_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(tokens) > max_length:
            raise ValueError(f"HPT token sequence length {len(tokens)} exceeds configured max_length={max_length}.")
        pad_id = self.tokenizer.pad_token_id if pad_id is None else pad_id
        pad_id = 0 if pad_id is None else int(pad_id)
        padded = list(tokens)
        pad_len = max_length - len(padded)
        attention = [1] * len(padded)
        if padding_side == "left":
            padded = [pad_id] * pad_len + padded
            attention = [0] * pad_len + attention
        elif padding_side == "right":
            padded = padded + [pad_id] * pad_len
            attention = attention + [0] * pad_len
        else:
            raise ValueError(f"Unsupported padding_side={padding_side!r}.")

        output = {"input_ids": torch.tensor([padded], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention], dtype=torch.long)
        return output

    def _normalize_sft_tensors(self, batch: DataProto, response_mask: torch.Tensor) -> None:
        batch.batch["rollout_log_probs"] = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        supervised = response_mask > 0
        for row_idx in range(response_mask.shape[0]):
            supervised_positions = supervised[row_idx].nonzero(as_tuple=True)[0]
            if supervised_positions.numel() <= 0:
                raise ValueError(f"HPT SFT row {row_idx} has no supervised response tokens.")
            rm_scores[row_idx, supervised_positions[-1]] = self._sft_terminal_reward(
                supervised_token_count=int(supervised_positions.numel())
            )
        batch.batch["rm_scores"] = rm_scores

    def _sft_terminal_reward(self, *, supervised_token_count: int) -> float:
        if supervised_token_count <= 0:
            raise ValueError(f"HPT SFT supervised_token_count must be positive, got {supervised_token_count}.")
        if self.hpt_config.sft_beta_mode == "constant":
            return float(self.hpt_config.beta)
        loss_scale_factor = int(self.config.actor_rollout_ref.actor.loss_scale_factor)
        return float(self.hpt_config.beta) * float(loss_scale_factor) / float(supervised_token_count)

    @staticmethod
    def _normalize_rl_tensors(batch: DataProto, response_mask: torch.Tensor, route: HptRouteMetadata) -> None:
        for key in ("rollout_log_probs", "rm_scores"):
            if key not in batch.batch:
                raise ValueError(f"HPT RL route requires batch tensor field {key!r}.")
            if tuple(batch.batch[key].shape) != tuple(response_mask.shape):
                raise ValueError(
                    f"HPT RL field {key!r} shape {tuple(batch.batch[key].shape)} "
                    f"does not match response_mask shape {tuple(response_mask.shape)}."
                )

        HptBatchAssembler._validate_repeated_non_tensor(batch, "uid", route.group_uid)
        HptBatchAssembler._validate_repeated_non_tensor(batch, "prompt_uid", route.prompt_uid)
        HptBatchAssembler._validate_integer_non_tensor(batch, "min_global_steps")
        HptBatchAssembler._validate_integer_non_tensor(batch, "max_global_steps")

    @staticmethod
    def _require_response_mask(batch: DataProto) -> torch.Tensor:
        if "response_mask" not in batch.batch:
            raise ValueError("HPT batch is missing response_mask.")
        response_mask = batch.batch["response_mask"]
        if response_mask.dim() != 2:
            raise ValueError(f"HPT response_mask must be rank 2, got rank {response_mask.dim()}.")
        if (response_mask.sum(dim=-1) <= 0).any():
            bad_rows = (response_mask.sum(dim=-1) <= 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"HPT batch has rows with no supervised response tokens: {bad_rows}.")
        return response_mask

    @staticmethod
    def _training_uid(route: HptRouteMetadata) -> str:
        return f"sft_{route.group_uid}" if route.is_sft else route.group_uid

    @staticmethod
    def _ensure_repeated_non_tensor(batch: DataProto, key: str, value: Any) -> None:
        if key in batch.non_tensor_batch:
            HptBatchAssembler._validate_repeated_non_tensor(batch, key, value)
            return
        values = np.empty(len(batch), dtype=object)
        values[:] = [value] * len(batch)
        batch.non_tensor_batch[key] = values

    @staticmethod
    def _validate_repeated_non_tensor(batch: DataProto, key: str, value: Any) -> None:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            raise ValueError(f"HPT batch is missing non_tensor_batch[{key!r}].")
        for row_idx, row_value in enumerate(values.tolist()):
            normalized = row_value.item() if isinstance(row_value, np.generic) else row_value
            if normalized != value:
                raise ValueError(
                    f"HPT non_tensor_batch[{key!r}] row {row_idx}={normalized!r} does not match expected {value!r}."
                )

    @staticmethod
    def _validate_integer_non_tensor(batch: DataProto, key: str) -> None:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            raise ValueError(f"HPT RL batch is missing non_tensor_batch[{key!r}].")
        for row_idx, value in enumerate(values.tolist()):
            value = value.item() if isinstance(value, np.generic) else value
            if not isinstance(value, Integral):
                raise ValueError(f"HPT non_tensor_batch[{key!r}] row {row_idx} must be an integer, got {value!r}.")

    @staticmethod
    def _ensure_extra_info_prompt_uid(batch: DataProto, prompt_uid: str) -> None:
        if "extra_info" not in batch.non_tensor_batch:
            batch.non_tensor_batch["extra_info"] = np.array([{"prompt_uid": prompt_uid} for _ in range(len(batch))])
            return

        updated = []
        for value in batch.non_tensor_batch["extra_info"].tolist():
            item = dict(value) if isinstance(value, dict) else {}
            existing_prompt_uid = item.get("prompt_uid")
            if existing_prompt_uid is not None and existing_prompt_uid != prompt_uid:
                raise ValueError(
                    "HPT extra_info prompt_uid does not match route metadata: "
                    f"extra_info={existing_prompt_uid!r} route={prompt_uid!r}."
                )
            item["prompt_uid"] = prompt_uid
            updated.append(item)
        batch.non_tensor_batch["extra_info"] = np.array(updated, dtype=object)

    @staticmethod
    def _normalize_non_tensor_arrays(batch: DataProto) -> None:
        for key, value in list(batch.non_tensor_batch.items()):
            if isinstance(value, np.ndarray):
                if value.shape[0] != len(batch):
                    raise ValueError(
                        f"HPT non_tensor_batch[{key!r}] length {value.shape[0]} does not match batch size {len(batch)}."
                    )
                continue
            array = np.array(value, dtype=object)
            if array.shape[0] != len(batch):
                raise ValueError(
                    f"HPT non_tensor_batch[{key!r}] length {array.shape[0]} does not match batch size {len(batch)}."
                )
            batch.non_tensor_batch[key] = array

    @staticmethod
    def _default_non_tensor_array(key: str, batch_size: int) -> np.ndarray:
        if key in {"processing_times", "tool_calls_times"}:
            return np.zeros(batch_size, dtype=float)
        if key in {"min_global_steps", "max_global_steps"}:
            return np.zeros(batch_size, dtype=object)
        if key == "extra_info":
            return np.array([{} for _ in range(batch_size)], dtype=object)
        return np.array([None] * batch_size, dtype=object)


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array
