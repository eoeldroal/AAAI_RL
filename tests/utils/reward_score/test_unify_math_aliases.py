import pytest

from verl.utils.reward_score import default_compute_score


@pytest.mark.parametrize(
    ("data_source", "solution", "ground_truth"),
    [
        ("MATH-500", "We get the final answer \\boxed{1}.", "1"),
        ("AIME24", "The answer is \\boxed{204}.", "204"),
        ("AMC23", "The answer is \\boxed{27}.", "27"),
    ],
)
def test_default_compute_score_accepts_unify_math_eval_aliases(data_source, solution, ground_truth):
    score = default_compute_score(
        data_source=data_source,
        solution_str=solution,
        ground_truth=ground_truth,
    )

    if isinstance(score, dict):
        assert score["score"] > 0
    else:
        assert score > 0
