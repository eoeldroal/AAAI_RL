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

"""Fail-closed tests for the Math-Verify custom reward adapter.

The adapter feeds the reward loop (train AND val). The distortion risks covered:
  - a ground truth that is already ``\\boxed{...}`` must not be double-boxed;
  - the adapter must return a float in {0.0, 1.0} (the HPT gate keys on >0 vs 0);
  - grading must never raise into the async reward loop (a parse failure degrades
    to 0.0), matching the never-crash contract;
  - the reward manager calls it as ``compute_score(data_source=, solution_str=,
    ground_truth=, extra_info=, <reward_kwargs>)`` -- that signature must hold.
"""

import pytest

from verl.utils.reward_score import math_verify_adapter


def test_normalize_ground_truth_unwraps_single_box_only():
    normalize = math_verify_adapter._normalize_ground_truth

    # A ground truth that is entirely one box is unwrapped (math_verify re-wraps it).
    assert normalize("\\boxed{204}") == "204"
    assert normalize("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    # Bare / LaTeX answers are passed through untouched.
    assert normalize("27") == "27"
    assert normalize("\\left( 3, \\frac{\\pi}{2} \\right)") == "\\left( 3, \\frac{\\pi}{2} \\right)"
    # Multi-box ground truths keep their inner boxes for math_verify to handle
    # (unwrapping the outer alone would corrupt them).
    multi = "\\boxed{1} and \\boxed{2}"
    assert normalize(multi) == multi


def test_adapter_normalizes_gt_forwards_timeout_and_returns_float(monkeypatch):
    import verl.utils.reward_score.math_verify as mv

    seen = {}

    def fake_compute_score(model_output, ground_truth, timeout_score=0, timeout=30.0):
        seen.update(model_output=model_output, ground_truth=ground_truth, timeout=timeout)
        return 1  # int -> adapter must coerce to float

    monkeypatch.setattr(mv, "compute_score", fake_compute_score)

    out = math_verify_adapter.compute_score(
        data_source="numina_olympiads",
        solution_str="answer is \\boxed{204}",
        ground_truth="\\boxed{204}",
        extra_info={"split": "train"},
        timeout=12.0,
    )

    # Ground truth double box unwrapped before delegation; full response passed through.
    assert seen["ground_truth"] == "204"
    assert seen["model_output"] == "answer is \\boxed{204}"
    # reward_kwargs (timeout) reaches the underlying grader.
    assert seen["timeout"] == 12.0
    # Reward-manager contract: a float score, not an int/dict.
    assert isinstance(out, float)
    assert out == 1.0


def test_adapter_scores_unboxed_and_boxed_correct_answers():
    # math_verify's key advantage over a boxed-only grader: an answer stated without
    # \boxed{} is still recovered. Exercises the real grader (the semantic core).
    assert math_verify_adapter.compute_score(solution_str="The answer is 27.", ground_truth="27", timeout=15.0) == 1.0
    assert (
        math_verify_adapter.compute_score(solution_str="so \\boxed{204}", ground_truth="\\boxed{204}", timeout=15.0)
        == 1.0
    )


def test_adapter_rejects_wrong_answer():
    assert math_verify_adapter.compute_score(solution_str="It is 5.", ground_truth="27", timeout=15.0) == 0.0


def test_adapter_degrades_to_zero_and_never_raises_on_garbage():
    # Never-crash contract: unparseable input scores 0.0 rather than raising into the
    # async reward loop (the failure class that previously crashed validation).
    out = math_verify_adapter.compute_score(solution_str="((( not math \x00\x00 ]]]", ground_truth="27", timeout=15.0)
    assert isinstance(out, float)
    assert out == 0.0


@pytest.mark.parametrize("bad_gt", ["", None])
def test_adapter_handles_degenerate_ground_truth_without_raising(bad_gt):
    out = math_verify_adapter.compute_score(solution_str="\\boxed{1}", ground_truth=bad_gt, timeout=15.0)
    assert isinstance(out, float)
    assert out in (0.0, 1.0)
