# Copyright 2026
#
# Routing tests (CPU): DataProto batch reconstruction — keep solved prompts' RL
# rows, replace unsolved prompts with their SFT demonstration row, with schema
# alignment (tau dropped, hpt_is_sft set) for a clean concat.

import numpy as np
import pytest
import torch

from recipe.paper_hpt import paper_hpt_routing as routing
from verl.protocol import DataProto


def _rl_batch(scores=None):
    # 2 prompts x 2 rollouts. default: p0 solved (row0 correct), p1 unsolved.
    if scores is None:
        scores = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    return DataProto.from_single_dict(
        {
            "input_ids": torch.arange(4 * 4).reshape(4, 4),
            "token_level_scores": torch.tensor(scores),
            "uid": np.array(["p0", "p0", "p1", "p1"], dtype=object),
        }
    )


def _sft_row(uid):
    return DataProto.from_single_dict(
        {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "token_level_scores": torch.zeros(1, 2),
            "uid": np.array([uid], dtype=object),
            "hpt_is_sft": torch.ones(1, dtype=torch.bool),
        }
    )


def test_keep_solved_replace_unsolved():
    routed = routing.route_generated_batch(_rl_batch(), {"p1": _sft_row("p1")}, gamma=0.0)
    assert len(routed) == 3
    assert routed.batch["hpt_is_sft"].tolist() == [False, False, True]
    assert routed.batch["input_ids"][0].tolist() == list(range(0, 4))  # p0 RL row kept verbatim


def test_all_solved_no_sft_rows_needed():
    routed = routing.route_generated_batch(_rl_batch([[1.0, 0.0]] * 4), {}, gamma=0.0)
    assert len(routed) == 4
    assert routed.batch["hpt_is_sft"].tolist() == [False] * 4


def test_all_unsolved_all_sft():
    routed = routing.route_generated_batch(
        _rl_batch([[0.0, 0.0]] * 4), {"p0": _sft_row("p0"), "p1": _sft_row("p1")}, gamma=0.0
    )
    assert len(routed) == 2
    assert routed.batch["hpt_is_sft"].tolist() == [True, True]
    assert routed.non_tensor_batch["uid"].tolist() == ["p0", "p1"]


def test_missing_sft_row_raises():
    with pytest.raises(KeyError):
        routing.route_generated_batch(_rl_batch(), {}, gamma=0.0)


def test_drops_tau_aux_keys_before_concat():
    b = _rl_batch()
    b.batch["tgt_input_ids"] = torch.zeros(4, 4, dtype=torch.long)
    routed = routing.route_generated_batch(b, {"p1": _sft_row("p1")}, gamma=0.0)
    assert "tgt_input_ids" not in routed.batch
    assert routed.batch["hpt_is_sft"].tolist() == [False, False, True]


def test_schema_mismatch_raises_clear_error():
    b = _rl_batch()
    b.batch["extra_train_key"] = torch.zeros(4, 2)  # not a tau/aux key; SFT row lacks it
    with pytest.raises(ValueError, match="mismatched schemas"):
        routing.route_generated_batch(b, {"p1": _sft_row("p1")}, gamma=0.0)


def test_three_prompts_preserve_order():
    b = DataProto.from_single_dict(
        {
            "input_ids": torch.arange(6 * 4).reshape(6, 4),
            "token_level_scores": torch.tensor(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
            ),
            "uid": np.array(["p0", "p0", "p1", "p1", "p2", "p2"], dtype=object),
        }
    )
    routed = routing.route_generated_batch(b, {"p1": _sft_row("p1")}, gamma=0.0)
    assert len(routed) == 5
    assert routed.batch["hpt_is_sft"].tolist() == [False, False, True, False, False]
    assert routed.non_tensor_batch["uid"].tolist() == ["p0", "p0", "p1", "p2", "p2"]


def test_gamma_fraction_routes_low_success():
    # p0: 1/2=0.5, p1: 0/2. gamma=0.5 -> both <=0.5 -> both SFT.
    routed = routing.route_generated_batch(
        _rl_batch(), {"p0": _sft_row("p0"), "p1": _sft_row("p1")}, gamma=0.5
    )
    assert len(routed) == 2
    assert routed.batch["hpt_is_sft"].tolist() == [True, True]


def test_sft_flag_required_on_sft_rows():
    bad = DataProto.from_single_dict(
        {"input_ids": torch.zeros(1, 4, dtype=torch.long),
         "token_level_scores": torch.zeros(1, 2),
         "uid": np.array(["p1"], dtype=object)}  # missing hpt_is_sft
    )
    with pytest.raises(KeyError):
        routing.route_generated_batch(_rl_batch(), {"p1": bad}, gamma=0.0)
