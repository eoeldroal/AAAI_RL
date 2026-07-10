# Copyright 2026
#
# Tau (LUFFY demonstration) loading + tokenization for the synchronous paper HPT.
#
# The paper routes unsolved prompts to an SFT step on their demonstration. We build
# a `prompt_uid -> demo_response_token_ids` lookup ONCE from the train parquet's
# `tau_messages` (the [user, assistant] transcript): the demonstration RESPONSE is
# the assistant turn, tokenized + EOS. At routing time an unsolved prompt's SFT row
# is built by cloning one of its rollout rows and overwriting the response with this
# demo (paper_hpt_routing.build_sft_row_from_template) — so the prompt stays and the
# schema matches the RL rows automatically.

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _decode_messages(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, np.ndarray):
        return list(raw)
    if isinstance(raw, list):
        return raw
    raise TypeError(f"unsupported tau_messages type: {type(raw)!r}")


def _assistant_text(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content")
    return None


def load_demo_response_ids(
    train_parquet: str,
    tokenizer,
    *,
    messages_key: str = "tau_messages",
    uid_key: str = "prompt_uid",
    max_response_length: int,
) -> dict[str, list[int]]:
    """Return {prompt_uid: demo_response_token_ids} (assistant turn + EOS, truncated).

    Raises on missing columns so a mis-specified dataset fails fast rather than
    silently producing empty SFT supervision.
    """
    df = pd.read_parquet(train_parquet, columns=[uid_key, messages_key])
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id; cannot terminate demo responses.")
    cap = max(1, int(max_response_length) - 1)  # leave room for EOS
    out: dict[str, list[int]] = {}
    for uid, raw in zip(df[uid_key].tolist(), df[messages_key].tolist(), strict=False):
        text = _assistant_text(_decode_messages(raw))
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:cap]
        if not ids:
            continue
        out[str(uid)] = list(ids) + [eos]
    if not out:
        raise ValueError(f"no usable tau demonstrations parsed from {train_parquet}.")
    return out
