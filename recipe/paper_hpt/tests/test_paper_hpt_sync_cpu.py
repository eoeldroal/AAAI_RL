# Copyright 2026
#
# CPU tests for the PRODUCTION sync-HPT path (fit-hook design):
#   - tau loading + tokenization (paper_hpt_tau.load_demo_response_ids)
#   - SFT-row-from-template cloning (paper_hpt_routing.build_sft_row_from_template)
#   - synchronous routing (paper_hpt_routing.route_generated_batch_synchronous)

import numpy as np
import pytest
import torch

from recipe.paper_hpt import paper_hpt_routing as routing
from recipe.paper_hpt.paper_hpt_tau import load_demo_response_ids
from verl.protocol import DataProto


# --------------------------------------------------------------------------- #
# tau loading
# --------------------------------------------------------------------------- #
class _FakeTok:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        # deterministic: one id per word, offset by length
        return {"input_ids": [10 + len(w) for w in text.split()]}


def test_load_demo_response_ids(tmp_path):
    import pandas as pd

    p = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "prompt_uid": ["u0", "u1"],
            "tau_messages": [
                [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "aa bbb"}],
                [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "c"}],
            ],
        }
    ).to_parquet(p)
    demos = load_demo_response_ids(str(p), _FakeTok(), max_response_length=8192)
    assert demos["u0"] == [12, 13, 99]  # 'aa'->12,'bbb'->13, +EOS
    assert demos["u1"] == [11, 99]


def test_load_demo_response_ids_truncates_and_keeps_eos(tmp_path):
    import pandas as pd

    p = tmp_path / "t.parquet"
    pd.DataFrame({"prompt_uid": ["u0"],
                  "tau_messages": [[{"role": "assistant", "content": "a b c d"}]]}).to_parquet(p)
    demos = load_demo_response_ids(str(p), _FakeTok(), max_response_length=3)  # cap=2 + EOS
    assert len(demos["u0"]) == 3 and demos["u0"][-1] == 99


# --------------------------------------------------------------------------- #
# full-schema RL batch fixture (what the batch looks like at routing time)
# --------------------------------------------------------------------------- #
def _rl_batch(scores, uids, puids, P=2, R=4):
    n = len(uids)
    prompts = torch.arange(n * P).reshape(n, P) + 1
    responses = torch.arange(n * R).reshape(n, R) + 100
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.ones(n, P + R, dtype=torch.long)
    position_ids = torch.arange(P + R).unsqueeze(0).repeat(n, 1)
    return DataProto.from_single_dict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": torch.ones(n, R),
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
            "old_log_probs": torch.randn(n, R),
            "uid": np.array(uids, dtype=object),
            "prompt_uid": np.array(puids, dtype=object),
        }
    )


# --------------------------------------------------------------------------- #
# build_sft_row_from_template
# --------------------------------------------------------------------------- #
def test_build_sft_row_from_template():
    b = _rl_batch([[1.0, 0, 0, 0]], ["p0"], ["u0"], P=2, R=4)
    template = b.select_idxs([0])
    sft = routing.build_sft_row_from_template(template, demo_ids=[7, 8, 9], pad_id=0)
    assert len(sft) == 1
    # demo placed in response, rest padded
    assert sft.batch["responses"][0].tolist() == [7, 8, 9, 0]
    assert sft.batch["response_mask"][0].tolist() == [1, 1, 1, 0]
    # prompt kept; input_ids = prompt + new response
    assert sft.batch["input_ids"][0].tolist() == [1, 2, 7, 8, 9, 0]
    # flagged SFT; rewards + old_log_probs zeroed
    assert bool(sft.batch["hpt_is_sft"].all())
    assert sft.batch["token_level_scores"].abs().sum().item() == 0.0
    assert sft.batch["old_log_probs"].abs().sum().item() == 0.0
    # attention over prompt(2)+demo(3), position_ids recomputed
    assert sft.batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0]


def test_build_sft_row_truncates_long_demo():
    b = _rl_batch([[1.0, 0, 0, 0]], ["p0"], ["u0"], P=2, R=4)
    sft = routing.build_sft_row_from_template(b.select_idxs([0]), demo_ids=[7, 8, 9, 10, 11], pad_id=0)
    assert sft.batch["responses"][0].tolist() == [7, 8, 9, 10]  # truncated to R=4
    assert sft.batch["response_mask"][0].tolist() == [1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# route_generated_batch_synchronous
# --------------------------------------------------------------------------- #
def test_route_synchronous_keeps_solved_injects_unsolved():
    # p0 solved (row0 correct), p1 unsolved. 2 rollouts each.
    b = _rl_batch(
        [[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ["p0", "p0", "p1", "p1"], ["u0", "u0", "u1", "u1"],
    )
    routed, m = routing.route_generated_batch_synchronous(
        b, {"u1": [7, 8, 9]}, gamma=0.0, pad_id=0
    )
    assert len(routed) == 3  # p0: 2 RL + p1: 1 SFT
    assert routed.batch["hpt_is_sft"].tolist() == [False, False, True]
    assert routed.batch["responses"][2].tolist() == [7, 8, 9, 0]  # SFT demo
    assert m["hpt/num_sft"] == 1.0
    assert m["hpt/num_rl_groups"] == 1.0
    assert m["hpt/offline_data_ratio"] == pytest.approx(0.5)
    assert m["hpt/p_success_zero_ratio"] == pytest.approx(0.5)  # p1 is 0/2
    assert m["hpt/missing_tau_count"] == 0.0


def test_route_synchronous_missing_tau_falls_back_to_rl():
    b = _rl_batch(
        [[0, 0, 0, 0], [0, 0, 0, 0]], ["p1", "p1"], ["u1", "u1"],
    )
    routed, m = routing.route_generated_batch_synchronous(b, {}, gamma=0.0, pad_id=0)  # no tau for u1
    assert len(routed) == 2  # kept as RL (no crash)
    assert routed.batch["hpt_is_sft"].tolist() == [False, False]
    assert m["hpt/missing_tau_count"] == 1.0
    assert m["hpt/num_sft"] == 0.0


def test_route_synchronous_all_solved_no_sft():
    b = _rl_batch([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], ["p0", "p0"], ["u0", "u0"])
    routed, m = routing.route_generated_batch_synchronous(b, {}, gamma=0.0, pad_id=0)
    assert len(routed) == 2
    assert routed.batch["hpt_is_sft"].tolist() == [False, False]
    assert m["hpt/offline_data_ratio"] == 0.0


def test_route_synchronous_pads_to_dp_multiple_loss_neutral():
    # p0 solved(2 RL) + p1 unsolved(1 SFT) = 3 rows; pad to multiple of 4 -> +1 neutral row.
    b = _rl_batch(
        [[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ["p0", "p0", "p1", "p1"], ["u0", "u0", "u1", "u1"],
    )
    routed, m = routing.route_generated_batch_synchronous(
        b, {"u1": [7, 8, 9]}, gamma=0.0, pad_id=0, pad_to_multiple=4
    )
    assert len(routed) == 4  # 3 real + 1 pad
    assert m["hpt/pad_rows"] == 1.0
    # pad row (last) is loss-neutral: response_mask all 0, dummy uid, not SFT
    assert routed.batch["response_mask"][-1].abs().sum().item() == 0.0
    assert routed.batch["hpt_is_sft"][-1].item() is False or routed.batch["hpt_is_sft"][-1].item() == False  # noqa: E712
    assert str(routed.non_tensor_batch["uid"][-1]).startswith("__pad_")


def test_route_synchronous_no_pad_when_already_divisible():
    b = _rl_batch([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], ["p0", "p0"], ["u0", "u0"])
    routed, m = routing.route_generated_batch_synchronous(b, {}, gamma=0.0, pad_id=0, pad_to_multiple=2)
    assert len(routed) == 2 and m["hpt/pad_rows"] == 0.0
