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

The output keeps the prompt-level HPT dataset contract: every train row has a
stable ``prompt_uid`` and an in-row ``tau_messages`` transcript. The same train
parquet can therefore feed both the RL dataloader and the tau lookup path.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
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
) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
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
                "tau_messages": json.dumps(prompt + target, ensure_ascii=False),
            }
        )

    return train_rows


def build_unify_eval_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    split: str,
    data_source: str,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build eval rows.

    ``system_prompt`` MUST match the training prompt: the OpenR1 train rows carry a
    leading system message that defines the <think>/Solution/\\boxed{} answer format,
    and every RL rollout is conditioned on it. If eval omits that system message, the
    model is evaluated under a different (weaker) instruction than it was trained on,
    which depresses accuracy and breaks train/val comparability. So when the train
    split keeps its system prompt, the eval split must inject the identical one.
    """
    eval_rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_rows):
        question = item.get("prompt")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{data_source} eval row {index} must contain a non-empty prompt.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{data_source} eval row {index} must contain a non-empty answer.")

        item_id = item.get("id")
        eval_index = f"{data_source}-{item_id}" if item_id is not None else f"{data_source}-{index}"
        extra_info = {"split": split, "index": eval_index}
        if "source" in item:
            extra_info["source"] = item["source"]

        prompt_messages: list[dict[str, Any]] = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": question})

        eval_rows.append(
            {
                "data_source": data_source,
                "prompt": prompt_messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": extra_info,
            }
        )
    return eval_rows


def _row_within_limits(
    row: dict[str, Any],
    row_idx: int,
    *,
    max_target_chars: int,
    token_counter,
    max_prompt_tokens: int,
    max_response_tokens: int,
    strip_system_prompt: bool,
    token_limit_action: str,
) -> bool:
    """Whether a row fits the char/token budget.

    Returns False to skip the row under ``filter``; raises ValueError under ``fail``.
    """
    if max_target_chars > 0 and len(_target_text(row)) > max_target_chars:
        if token_limit_action == "fail":
            raise ValueError(f"OpenR1 row {row_idx} exceeds max_target_chars={max_target_chars}.")
        return False
    if token_counter is not None and not _within_token_limits(
        row,
        token_counter=token_counter,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
        strip_system_prompt=strip_system_prompt,
    ):
        if token_limit_action == "fail":
            raise ValueError(
                f"OpenR1 row {row_idx} exceeds prompt/response token limits "
                f"({max_prompt_tokens}, {max_response_tokens})."
            )
        return False
    return True


def select_openr1_rows_for_hpt(
    dataset,
    *,
    total_rows: int,
    max_target_chars: int,
    reward_data_source: str,
    tokenizer_path: str | None,
    max_prompt_tokens: int,
    max_response_tokens: int,
    strip_system_prompt: bool,
    token_limit_action: str = "filter",
) -> list[dict[str, Any]]:
    if token_limit_action not in {"filter", "fail", "ignore"}:
        raise ValueError(f"Unsupported token_limit_action={token_limit_action!r}.")
    if total_rows == 0:
        return []
    if total_rows > 0:
        return _select_short_target_rows(
            dataset,
            total_rows=total_rows,
            max_target_chars=max_target_chars,
            reward_data_source=reward_data_source,
            tokenizer_path=tokenizer_path,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            strip_system_prompt=strip_system_prompt,
            token_limit_action=token_limit_action,
        )

    token_counter = _build_token_counter(tokenizer_path) if token_limit_action != "ignore" else None
    selected = []
    for row_idx, row in enumerate(dataset):
        if _row_within_limits(
            row,
            row_idx,
            max_target_chars=max_target_chars,
            token_counter=token_counter,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            strip_system_prompt=strip_system_prompt,
            token_limit_action=token_limit_action,
        ):
            selected.append(row)
    return selected


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


def _leading_system_content(rows: Iterable[dict[str, Any]]) -> str | None:
    """Return the (assumed uniform) leading system-prompt text from source rows.

    OpenR1 rows all carry the same system message; we lift it verbatim so the eval
    split can be conditioned on the identical instruction as the train split.
    """
    for row in rows:
        messages = row.get("prompt")
        if isinstance(messages, list) and messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


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
    token_limit_action: str = "filter",
) -> list[dict[str, Any]]:
    from verl.utils.reward_score import default_compute_score

    token_counter = _build_token_counter(tokenizer_path) if token_limit_action != "ignore" else None
    selected = []
    for row_idx, row in enumerate(dataset):
        if not _row_within_limits(
            row,
            row_idx,
            max_target_chars=max_target_chars,
            token_counter=token_counter,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            strip_system_prompt=strip_system_prompt,
            token_limit_action=token_limit_action,
        ):
            continue
        try:
            score = default_compute_score(
                reward_data_source,
                _target_text(row),
                row["reward_model"]["ground_truth"],
                dict(row.get("extra_info") or {}),
            )
        except Exception:
            continue
        score_value = score.get("score", 0.0) if isinstance(score, dict) else float(score)
        if score_value <= 0:
            continue
        selected.append(row)
        if len(selected) >= total_rows:
            break
    if len(selected) < total_rows:
        raise RuntimeError(
            f"Only found {len(selected)} OpenR1 rows with target <= {max_target_chars} chars; need {total_rows}."
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
                raise ValueError("chat template returned multiple tokenized rows for one message list") from None
            token_ids = token_ids[0]
        return list(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="datas/openr1_hpt_smoke")
    parser.add_argument("--dataset", default=OPENR1_DATASET)
    parser.add_argument("--eval_json_dir", default=None)
    parser.add_argument("--eval_data_sources", nargs="*", default=("AIME24", "AMC23", "MATH-500"))
    parser.add_argument("--train_size", type=int, default=12)
    parser.add_argument("--val_size", type=int, default=4)
    parser.add_argument("--max_target_chars", type=int, default=6000)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_response_tokens", type=int, default=2048)
    parser.add_argument("--token_limit_action", choices=("filter", "fail", "ignore"), default="filter")
    parser.add_argument("--prompt_uid_prefix", default="openr1_hpt_smoke")
    parser.add_argument("--normalize_data_source", default=DEFAULT_REWARD_DATA_SOURCE)
    parser.add_argument("--keep_system_prompt", action="store_true")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    import datasets
    import pandas as pd

    raw_dataset = datasets.load_dataset(args.dataset, split="train")
    required_rows = args.train_size + args.val_size if args.train_size >= 0 else -1
    selected_rows = select_openr1_rows_for_hpt(
        raw_dataset,
        total_rows=required_rows,
        max_target_chars=args.max_target_chars,
        reward_data_source=args.normalize_data_source,
        tokenizer_path=args.tokenizer_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        strip_system_prompt=not args.keep_system_prompt,
        token_limit_action=args.token_limit_action,
    )
    if args.train_size < 0:
        train_source = selected_rows
        val_source = []
    else:
        train_source = selected_rows[: args.train_size]
        val_source = selected_rows[args.train_size :]

    train_rows = build_hpt_rows(
        train_source,
        split="train",
        prompt_uid_prefix=args.prompt_uid_prefix,
        normalize_data_source=args.normalize_data_source,
        strip_system_prompt=not args.keep_system_prompt,
    )
    val_rows = build_hpt_rows(
        val_source,
        split="val",
        prompt_uid_prefix=args.prompt_uid_prefix,
        normalize_data_source=args.normalize_data_source,
        strip_system_prompt=not args.keep_system_prompt,
    )

    # Eval must be conditioned on the SAME system prompt as train. When we keep the
    # system prompt on train, lift the (uniform) OpenR1 system message from the source
    # and inject the identical one into every eval row; otherwise leave eval bare so it
    # matches the stripped train split. This keeps train/val prompt distributions equal.
    eval_system_prompt = _leading_system_content(train_source) if args.keep_system_prompt else None
    if args.keep_system_prompt and not eval_system_prompt:
        raise RuntimeError(
            "keep_system_prompt=True but no leading system prompt found in source rows; "
            "cannot guarantee train/val prompt parity."
        )

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_save_dir, "train.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(local_save_dir, "test.parquet"))
    eval_counts = _write_unify_eval_parquets(
        eval_json_dir=args.eval_json_dir,
        eval_data_sources=args.eval_data_sources,
        local_save_dir=local_save_dir,
        system_prompt=eval_system_prompt,
    )
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
                "token_limit_action": args.token_limit_action,
                "prompt_uid_prefix": args.prompt_uid_prefix,
                "normalize_data_source": args.normalize_data_source,
                "keep_system_prompt": args.keep_system_prompt,
                "eval_system_prompt_injected": bool(eval_system_prompt),
                "hpt_dataset_format": "unified_prompt_rows",
                "eval_json_dir": args.eval_json_dir,
                "eval_counts": eval_counts,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        "Prepared OpenR1 HPT smoke dataset: "
        f"train={len(train_rows)}, val={len(val_rows)}, tau={len(train_rows)} "
        f"under {local_save_dir}",
        flush=True,
    )


def _write_unify_eval_parquets(
    *,
    eval_json_dir: str | None,
    eval_data_sources: Iterable[str],
    local_save_dir: str,
    system_prompt: str | None = None,
) -> dict[str, int]:
    if eval_json_dir is None:
        return {}

    import pandas as pd

    eval_root = Path(eval_json_dir).expanduser()
    eval_counts: dict[str, int] = {}
    for data_source in eval_data_sources:
        source_path = eval_root / data_source / "test.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Unify eval json not found: {source_path}")
        raw_rows = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError(f"Unify eval json must contain a list: {source_path}")
        rows = build_unify_eval_rows(raw_rows, split="test", data_source=data_source, system_prompt=system_prompt)
        output_dir = Path(local_save_dir) / data_source
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output_dir / "test.parquet")
        eval_counts[data_source] = len(rows)
    return eval_counts


if __name__ == "__main__":
    main()
