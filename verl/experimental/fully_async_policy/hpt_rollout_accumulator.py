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

from dataclasses import dataclass, field

from verl.protocol import DataProto


@dataclass(frozen=True)
class HptTrajectoryAttemptResult:
    """One completed HPT rollout attempt at trajectory-attempt granularity."""

    group_uid: str
    prompt_uid: str
    rollout_index: int
    payload: DataProto


@dataclass(frozen=True)
class CompletedHptPromptGroup:
    """Prompt-group payload reconstructed from rollout.n trajectory attempts."""

    group_uid: str
    prompt_uid: str
    payload: DataProto


@dataclass
class _OpenHptPromptGroup:
    group_uid: str
    prompt_uid: str
    rollout_n: int
    attempts: dict[int, DataProto] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return len(self.attempts) == self.rollout_n


class HptPromptGroupAccumulator:
    """Collect one-row HPT trajectory attempts into prompt-level rollout groups."""

    def __init__(self, rollout_n: int) -> None:
        rollout_n = int(rollout_n)
        if rollout_n <= 0:
            raise ValueError(f"rollout_n must be positive, got {rollout_n}")
        self.rollout_n = rollout_n
        self._groups: dict[str, _OpenHptPromptGroup] = {}

    def add(self, result: HptTrajectoryAttemptResult) -> None:
        group_uid = _normalize_non_empty_string(result.group_uid, "group_uid")
        prompt_uid = _normalize_non_empty_string(result.prompt_uid, "prompt_uid")
        rollout_index = int(result.rollout_index)
        if rollout_index < 0 or rollout_index >= self.rollout_n:
            raise ValueError(
                f"rollout_index must be in [0, {self.rollout_n}), got {rollout_index} for group_uid={group_uid!r}"
            )
        _validate_one_row_payload(result.payload, group_uid=group_uid, rollout_index=rollout_index)

        group = self._groups.get(group_uid)
        if group is None:
            group = _OpenHptPromptGroup(group_uid=group_uid, prompt_uid=prompt_uid, rollout_n=self.rollout_n)
            self._groups[group_uid] = group
        elif group.prompt_uid != prompt_uid:
            raise ValueError(
                f"Mixed prompt_uid in group_uid={group_uid!r}: existing={group.prompt_uid!r}, got={prompt_uid!r}"
            )

        if rollout_index in group.attempts:
            raise ValueError(f"duplicate rollout_index={rollout_index} for group_uid={group_uid!r}")
        group.attempts[rollout_index] = result.payload

    def pop_ready(self) -> list[CompletedHptPromptGroup]:
        ready_group_uids = [group_uid for group_uid, group in self._groups.items() if group.ready]
        completed: list[CompletedHptPromptGroup] = []
        for group_uid in ready_group_uids:
            group = self._groups.pop(group_uid)
            ordered_payloads = [group.attempts[index] for index in range(self.rollout_n)]
            payload = DataProto.concat(ordered_payloads)
            payload.meta_info["metrics"] = _collect_agent_loop_metrics(ordered_payloads)
            completed.append(
                CompletedHptPromptGroup(
                    group_uid=group.group_uid,
                    prompt_uid=group.prompt_uid,
                    payload=payload,
                )
            )
        return completed

    def discard(self, group_uid: str) -> None:
        group_uid = _normalize_non_empty_string(group_uid, "group_uid")
        self._groups.pop(group_uid, None)

    def stranded_count(self) -> int:
        return sum(len(group.attempts) for group in self._groups.values() if not group.ready)

    def open_group_count(self) -> int:
        return len(self._groups)

    def stored_attempt_count(self) -> int:
        return sum(len(group.attempts) for group in self._groups.values())

    def completed_attempt_storage_count(self, *, queue_size: int) -> int:
        queue_size = int(queue_size)
        if queue_size < 0:
            raise ValueError(f"queue_size must be non-negative, got {queue_size}")
        return queue_size * self.rollout_n + self.stored_attempt_count()


def _normalize_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value)!r}")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_one_row_payload(payload: DataProto, *, group_uid: str, rollout_index: int) -> None:
    if not isinstance(payload, DataProto):
        raise TypeError(f"HPT trajectory attempt payload must be DataProto, got {type(payload)!r}")
    if len(payload) != 1:
        raise ValueError(
            f"HPT trajectory attempt payload must contain exactly one row for group_uid={group_uid!r}, "
            f"rollout_index={rollout_index}, got {len(payload)}"
        )


def _collect_agent_loop_metrics(payloads: list[DataProto]) -> list[dict]:
    merged_metrics: list[dict] = []
    for payload in payloads:
        metrics = payload.meta_info.get("metrics")
        if metrics is None:
            raise ValueError("DataProto HPT trajectory attempt is missing metrics.")
        payload_row_count = len(payload)
        if isinstance(metrics, list):
            if len(metrics) != payload_row_count:
                raise ValueError(
                    "DataProto metrics row count must match payload row count: "
                    f"metrics={len(metrics)} rows={payload_row_count}"
                )
            for item in metrics:
                if not isinstance(item, dict):
                    raise TypeError(f"DataProto metrics list items must be dicts, got {type(item)!r}")
                merged_metrics.append(dict(item))
            continue
        if isinstance(metrics, dict):
            rows = _metrics_dict_to_rows(metrics)
            if len(rows) != payload_row_count:
                raise ValueError(
                    "DataProto metrics row count must match payload row count: "
                    f"metrics={len(rows)} rows={payload_row_count}"
                )
            merged_metrics.extend(rows)
            continue
        raise TypeError(f"DataProto metrics must be a list of dicts or a dict, got {type(metrics)!r}")
    return merged_metrics


def _metrics_dict_to_rows(metrics: dict) -> list[dict]:
    if not metrics:
        return []
    values = list(metrics.values())
    if not all(isinstance(value, list) for value in values):
        return [dict(metrics)]

    row_count = len(values[0])
    if any(len(value) != row_count for value in values):
        raise ValueError("DataProto metrics dict-of-lists must have equal-length values.")
    return [{key: value[index] for key, value in metrics.items()} for index in range(row_count)]
