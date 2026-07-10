# Copyright 2026
#
# Gate tests (CPU): success counting + the eq.10 SFT/RL decision.

import numpy as np
import pytest
import torch

from recipe.paper_hpt import paper_hpt_gate as gate


def test_group_success_counts():
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    uids = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    counts = gate.group_success_counts(scores, uids)
    assert counts["p0"] == (1, 2)
    assert counts["p1"] == (0, 2)


def test_group_success_counts_partial_credit_not_counted():
    # only exact success_value counts; partial-credit scores do not.
    scores = torch.tensor([0.5, 1.0, 0.3])
    uids = np.array(["p0", "p0", "p0"], dtype=object)
    assert gate.group_success_counts(scores, uids)["p0"] == (1, 3)


def test_group_success_counts_shape_guards():
    with pytest.raises(ValueError):
        gate.group_success_counts(torch.zeros(2, 2), np.array(["a", "b"], dtype=object))
    with pytest.raises(ValueError):
        gate.group_success_counts(torch.zeros(3), np.array(["a", "b"], dtype=object))


def test_is_prompt_sft_gamma_zero_binary():
    assert gate.is_prompt_sft(0, 8, gamma=0.0) is True
    assert gate.is_prompt_sft(1, 8, gamma=0.0) is False
    assert gate.is_prompt_sft(8, 8, gamma=0.0) is False


def test_is_prompt_sft_gamma_fraction_and_boundary():
    assert gate.is_prompt_sft(2, 8, gamma=2 / 8) is True   # P == gamma -> SFT (<=)
    assert gate.is_prompt_sft(3, 8, gamma=2 / 8) is False
    with pytest.raises(ValueError):
        gate.is_prompt_sft(0, 0, gamma=0.0)


def test_route_prompts_end_to_end():
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    uids = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    assert gate.route_prompts(scores, uids, gamma=0.0) == {"p0": False, "p1": True}
