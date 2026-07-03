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

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.utils.tokenizer.tokenizer import normalize_token_ids

_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class HptSftPayload(BaseModel):
    """Validated in-process payload for one tau supervised trajectory."""

    model_config = ConfigDict(extra="forbid")

    prompt_uid: str = Field(min_length=1)
    messages: list[dict[str, Any]]

    @field_validator("prompt_uid")
    @classmethod
    def _strip_prompt_uid(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt_uid must not be empty")
        return value

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_messages(cls, value: Any) -> Any:
        return convert_nested_value_to_list_recursive(value)

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must not be empty")
        for index, message in enumerate(value):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be a dict")
            role = message.get("role")
            if role not in _ALLOWED_MESSAGE_ROLES:
                raise ValueError(f"messages[{index}].role must be one of {sorted(_ALLOWED_MESSAGE_ROLES)}")
            if "content" not in message:
                raise ValueError(f"messages[{index}] is missing content")
        return value

    @model_validator(mode="after")
    def _require_assistant_message(self) -> HptSftPayload:
        if not any(message.get("role") == "assistant" for message in self.messages):
            raise ValueError("tau messages must contain at least one assistant message")
        return self


class HptTauStore:
    """Prompt_uid keyed lookup table for validated tau payloads."""

    def __init__(self, payloads: dict[str, HptSftPayload]):
        self._payloads = dict(payloads)

    @classmethod
    def from_parquet(cls, path: str | Path, *, messages_key: str = "tau_messages") -> HptTauStore:
        messages_key = messages_key.strip()
        if not messages_key:
            raise ValueError("messages_key must not be empty")

        dataframe = pd.read_parquet(path)
        required_columns = {"prompt_uid", messages_key}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"tau parquet is missing required columns: {sorted(missing_columns)}")

        duplicate_mask = dataframe["prompt_uid"].duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_values = dataframe.loc[duplicate_mask, "prompt_uid"].unique()
            duplicate_list = ", ".join(sorted(str(value) for value in duplicate_values))
            raise ValueError(f"Duplicate tau prompt_uid values found: {duplicate_list}")

        payloads: dict[str, HptSftPayload] = {}
        for row in dataframe[["prompt_uid", messages_key]].to_dict(orient="records"):
            messages = decode_tau_messages(row[messages_key], prompt_uid=str(row["prompt_uid"]))
            if is_missing_tau_messages(messages):
                continue
            payload = HptSftPayload.model_validate({"prompt_uid": row["prompt_uid"], "messages": messages})
            payloads[payload.prompt_uid] = payload
        return cls(payloads)

    def get(self, prompt_uid: str) -> HptSftPayload | None:
        return self._payloads.get(prompt_uid)

    def __contains__(self, prompt_uid: object) -> bool:
        return prompt_uid in self._payloads

    def __getitem__(self, prompt_uid: str) -> HptSftPayload:
        return self._payloads[prompt_uid]

    def __len__(self) -> int:
        return len(self._payloads)


def is_missing_tau_messages(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, list | tuple) and len(value) == 0:
        return True
    if hasattr(value, "tolist") and not isinstance(value, dict | str | bytes):
        listed = value.tolist()
        if isinstance(listed, list) and len(listed) == 0:
            return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def decode_tau_messages(value: Any, *, prompt_uid: str) -> Any:
    if not isinstance(value, str):
        return value
    if not value.strip():
        return value

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON tau messages for prompt_uid={prompt_uid!r}: {exc.msg}") from exc

    if not isinstance(decoded, list):
        raise ValueError(
            "JSON tau messages must decode to a list of message dicts "
            f"for prompt_uid={prompt_uid!r}, got {type(decoded).__name__}."
        )
    return decoded


class HptTauToAgentLoopOutputAdapter:
    """Convert text-only tau transcripts into the existing AgentLoopOutput contract."""

    def __init__(
        self,
        tokenizer,
        *,
        processor=None,
        tools: list[dict[str, Any]] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.tools = tools
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

    def to_agent_loop_output(self, payload: HptSftPayload) -> AgentLoopOutput:
        self._reject_multimodal_tau(payload)
        messages = payload.messages
        first_assistant_idx = self._first_assistant_index(messages)
        if first_assistant_idx <= 0:
            raise ValueError("tau messages must contain context before the first assistant target")

        prompt_ids = self._tokenize(messages[:first_assistant_idx], add_generation_prompt=True)
        running_ids = self._tokenize(messages[: first_assistant_idx + 1], add_generation_prompt=False)
        if running_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("tau tokenization is not prefix-stable at the first assistant target")

        response_ids = running_ids[len(prompt_ids) :]
        response_mask = [1] * len(response_ids)

        for message_index in range(first_assistant_idx + 1, len(messages)):
            next_running_ids = self._tokenize(messages[: message_index + 1], add_generation_prompt=False)
            if next_running_ids[: len(running_ids)] != running_ids:
                raise ValueError(f"tau tokenization is not prefix-stable at message index {message_index}")
            segment = next_running_ids[len(running_ids) :]
            response_ids.extend(segment)
            response_mask.extend([1 if messages[message_index].get("role") == "assistant" else 0] * len(segment))
            running_ids = next_running_ids

        if len(response_ids) != len(response_mask):
            raise ValueError("tau adapter produced misaligned response ids and response mask")
        if sum(response_mask) <= 0:
            raise ValueError("tau adapter produced no supervised assistant tokens")

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=None,
            num_turns=len(messages),
            metrics=AgentLoopMetrics(),
            extra_fields={"hpt_prompt_uid": payload.prompt_uid},
        )

    def _tokenize(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        processing_class = self.processor if self.processor is not None else self.tokenizer
        tokenized = apply_chat_template(
            processing_class,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=self.tools,
            **self.apply_chat_template_kwargs,
        )
        return normalize_token_ids(tokenized)

    @staticmethod
    def _first_assistant_index(messages: list[dict[str, Any]]) -> int:
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                return index
        raise ValueError("tau messages must contain at least one assistant message")

    @staticmethod
    def _reject_multimodal_tau(payload: HptSftPayload) -> None:
        for message_index, message in enumerate(payload.messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type", "text") != "text":
                    raise ValueError(
                        "phase-1 HPT tau adapter only supports text content; "
                        f"prompt_uid={payload.prompt_uid!r}, message_index={message_index}"
                    )
