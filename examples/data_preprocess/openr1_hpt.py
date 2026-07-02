# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Prepare a small OpenR1 text dataset that exercises async HPT end to end.

The output keeps the ordinary RL train/val parquet contract and writes a
separate tau parquet keyed by ``prompt_uid``. The smoke path intentionally keeps
tau for only part of the rows so HPT sees both SFT-routed and RL-routed prompt
groups without changing the model or distributed setup.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from typing import Any


OPENR1_DATASET = "Elliott/Openr1-Math-46k-8192"
DEFAULT_REWARD_DATA_SOURCE = "numina_olympiads"


def build_hpt_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    split: str,
    prompt_uid_prefix: str,
    normalize_data_source: str,
    strip_system_prompt: bool,
    tau_keep_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tau_keep_every <= 0:
        raise ValueError(f"tau_keep_every must be positive, got {tau_keep_every}.")

    train_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_rows):
        prompt_uid = f"{prompt_uid_prefix}_{split}_{index:08d}"
        prompt = _normalize_messages(item["prompt"])
        target = _normalize_messages(item["target"])
        if strip_system_prompt:
            prompt = _strip_leading_system_prompt(prompt)

        reward_model = dict(item.get("reward_model") or {})
        extra_info = dict(item.get("extra_info") or {})
        extra_info.update(
            {
                "split": split,
                "index": index,
                "prompt_uid": prompt_uid,
                "source_data_source": item.get("data_source"),
            }
        )

        train_rows.append(
            {
                "data_source": normalize_data_source,
                "prompt": prompt,
                "ability": item.get("ability") or "math",
                "reward_model": reward_model,
                "extra_info": extra_info,
                "prompt_uid": prompt_uid,
            }
        )

        if index % tau_keep_every == 0:
            tau_rows.append(
                {
                    "prompt_uid": prompt_uid,
                    "tau_messages": json.dumps(prompt + target, ensure_ascii=False),
                }
            )

    return train_rows, tau_rows


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("OpenR1 messages must be a non-empty list.")

    normalized = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"OpenR1 message {idx} must be a dict.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"OpenR1 message {idx} has unsupported role={role!r}.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"OpenR1 message {idx} must contain non-empty text content.")
        normalized.append({"role": role, "content": content})
    return normalized


def _strip_leading_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages[1:]
    return messages


def _select_short_target_rows(
    dataset,
    *,
    total_rows: int,
    max_target_chars: int,
    reward_data_source: str,
    tokenizer_path: str | None,
    max_prompt_tokens: int,
    max_response_tokens: int,
    strip_system_prompt: bool,
) -> list[dict[str, Any]]:
    from verl.utils.reward_score import default_compute_score

    token_counter = _build_token_counter(tokenizer_path)
    selected = []
    for row in dataset:
        target_text = _target_text(row)
        if len(target_text) > max_target_chars:
            continue
        try:
            score = default_compute_score(
                reward_data_source,
                target_text,
                row["reward_model"]["ground_truth"],
                dict(row.get("extra_info") or {}),
            )
        except Exception:
            continue
        score_value = score.get("score", 0.0) if isinstance(score, dict) else float(score)
        if score_value <= 0:
            continue
        if token_counter is not None and not _within_token_limits(
            row,
            token_counter=token_counter,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            strip_system_prompt=strip_system_prompt,
        ):
            continue
        selected.append(row)
        if len(selected) >= total_rows:
            break
    if len(selected) < total_rows:
        raise RuntimeError(
            f"Only found {len(selected)} OpenR1 rows with target <= {max_target_chars} chars; "
            f"need {total_rows}."
        )
    return selected


def _target_text(row: dict[str, Any]) -> str:
    return "\n".join(message.get("content", "") for message in row.get("target", []))


def _build_token_counter(tokenizer_path: str | None):
    if not tokenizer_path:
        return None

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _within_token_limits(
    row: dict[str, Any],
    *,
    token_counter,
    max_prompt_tokens: int,
    max_response_tokens: int,
    strip_system_prompt: bool,
) -> bool:
    prompt = _normalize_messages(row["prompt"])
    target = _normalize_messages(row["target"])
    if strip_system_prompt:
        prompt = _strip_leading_system_prompt(prompt)

    prompt_ids = _chat_token_ids(token_counter, prompt, add_generation_prompt=True)
    if len(prompt_ids) > max_prompt_tokens:
        return False

    messages = prompt + target
    first_assistant_idx = next((idx for idx, message in enumerate(messages) if message["role"] == "assistant"), None)
    if first_assistant_idx is None:
        return False

    prompt_part_ids = _chat_token_ids(
        token_counter,
        messages[:first_assistant_idx],
        add_generation_prompt=True,
    )
    full_ids = _chat_token_ids(token_counter, messages, add_generation_prompt=False)
    if full_ids[: len(prompt_part_ids)] != prompt_part_ids:
        return False

    response_len = len(full_ids) - len(prompt_part_ids)
    return response_len <= max_response_tokens


def _chat_token_ids(tokenizer, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    try:
        from verl.utils.tokenizer.chat_template import apply_chat_template
        from verl.utils.tokenizer.tokenizer import normalize_token_ids

        token_ids = apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return normalize_token_ids(token_ids)
    except ModuleNotFoundError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("chat template returned multiple tokenized rows for one message list")
            token_ids = token_ids[0]
        return list(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/openr1_hpt_smoke")
    parser.add_argument("--dataset", default=OPENR1_DATASET)
    parser.add_argument("--train_size", type=int, default=12)
    parser.add_argument("--val_size", type=int, default=4)
    parser.add_argument("--max_target_chars", type=int, default=6000)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_response_tokens", type=int, default=2048)
    parser.add_argument("--prompt_uid_prefix", default="openr1_hpt_smoke")
    parser.add_argument("--normalize_data_source", default=DEFAULT_REWARD_DATA_SOURCE)
    parser.add_argument("--tau_keep_every", type=int, default=2)
    parser.add_argument("--keep_system_prompt", action="store_true")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    import datasets
    import pandas as pd

    required_rows = args.train_size + args.val_size
    raw_dataset = datasets.load_dataset(args.dataset, split="train")
    selected_rows = _select_short_target_rows(
        raw_dataset,
        total_rows=required_rows,
        max_target_chars=args.max_target_chars,
        reward_data_source=args.normalize_data_source,
        tokenizer_path=args.tokenizer_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        strip_system_prompt=not args.keep_system_prompt,
    )
    train_source = selected_rows[: args.train_size]
    val_source = selected_rows[args.train_size :]

    train_rows, train_tau_rows = build_hpt_rows(
        train_source,
        split="train",
        prompt_uid_prefix=args.prompt_uid_prefix,
        normalize_data_source=args.normalize_data_source,
        strip_system_prompt=not args.keep_system_prompt,
        tau_keep_every=args.tau_keep_every,
    )
    val_rows, _ = build_hpt_rows(
        val_source,
        split="val",
        prompt_uid_prefix=args.prompt_uid_prefix,
        normalize_data_source=args.normalize_data_source,
        strip_system_prompt=not args.keep_system_prompt,
        tau_keep_every=args.tau_keep_every,
    )

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_save_dir, "train.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(local_save_dir, "test.parquet"))
    pd.DataFrame(train_tau_rows).to_parquet(os.path.join(local_save_dir, "tau.parquet"))
    with open(os.path.join(local_save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "max_target_chars": args.max_target_chars,
                "tokenizer_path": args.tokenizer_path,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_response_tokens": args.max_response_tokens,
                "prompt_uid_prefix": args.prompt_uid_prefix,
                "normalize_data_source": args.normalize_data_source,
                "tau_keep_every": args.tau_keep_every,
                "keep_system_prompt": args.keep_system_prompt,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        "Prepared OpenR1 HPT smoke dataset: "
        f"train={len(train_rows)}, val={len(val_rows)}, tau={len(train_tau_rows)} "
        f"under {local_save_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
