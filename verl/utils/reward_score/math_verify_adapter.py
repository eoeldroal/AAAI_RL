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

"""Custom reward adapter that scores with HF Math-Verify, applied UNIFORMLY to
train and validation via ``reward.custom_reward_function``.

Why this adapter (rather than ``default_compute_score``):
    ``default_compute_score`` routes by ``data_source`` -- our train rows
    (``numina_olympiads``) would hit ``prime_math`` while the eval benchmarks
    (``MATH-500``/``AIME24``/...) would hit ``math_reward``. That makes train and
    val use *different* graders. Pinning a single custom function keeps the grader
    identical for every split.

Why Math-Verify:
    It extracts answers both as LaTeX/boxed and as bare expressions (no ``\\boxed{}``
    required) and checks symbolic equivalence. On our rollouts it has the highest
    recall of the candidate graders (entropy_math / prime_math / math_reward /
    math_dapo) while never over-crediting relative to their union. It also never
    raises -- a timeout or parse failure degrades to ``timeout_score`` (0.0) inside
    ``verl.utils.reward_score.math_verify`` -- which matches the reward loop's
    never-crash contract (the crash class we hit previously came from a grader that
    could raise into the async reward loop).
"""

import re

# Matches a ground truth that is *entirely* a single ``\boxed{...}`` wrapper.
_BOXED_ONLY = re.compile(r"^\s*\\boxed\s*\{(?P<inner>.*)\}\s*$", re.DOTALL)


def _normalize_ground_truth(ground_truth) -> str:
    """Unwrap a ground truth that is itself a single ``\\boxed{...}``.

    ``math_verify.compute_score`` wraps the ground truth in ``\\boxed{gt}`` before
    parsing. Some dataset answers are already boxed (e.g. AIME "\\boxed{204}"), which
    would otherwise become a confusing double box. Unwrap only when the whole answer
    is a single box and the inner text has no further ``\\boxed`` (so multi-answer
    ground truths are left untouched for math_verify to handle).
    """
    gt = str(ground_truth)
    match = _BOXED_ONLY.match(gt)
    if match is not None and "\\boxed" not in match.group("inner"):
        return match.group("inner").strip()
    return gt


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    *,
    timeout: float = 30.0,
    timeout_score: float = 0.0,
    **kwargs,
) -> float:
    """verl custom-reward entrypoint. Returns 1.0 for a correct answer else 0.0.

    Args:
        data_source: Unused (kept for the custom_reward_function contract).
        solution_str: Decoded model response from the reward manager.
        ground_truth: Dataset answer.
        extra_info: Unused.
        timeout: Hard wall-clock bound (s) for a single Math-Verify call; on timeout
            the score degrades to ``timeout_score`` rather than raising.
        timeout_score: Score returned on timeout (default 0.0).
    """
    # Imported lazily so importing this module never drags in math_verify at parse time.
    from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

    del data_source, extra_info, kwargs

    gt = _normalize_ground_truth(ground_truth)
    score = math_verify_compute_score(
        str(solution_str),
        gt,
        timeout_score=timeout_score,
        timeout=timeout,
    )
    return float(score)
