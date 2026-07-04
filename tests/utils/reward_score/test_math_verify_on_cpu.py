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

"""Regression tests for verl.utils.reward_score.math_verify.

math_verify's own grader/parser modules log a full traceback (via logging.error)
whenever a comparison is pathological -- a huge power expression, an integral
needing numeric quadrature, or garbled unicode from a degenerate rollout -- even
though they still return a valid score. On a live async-HPT run this flooded the
console/log file with noise for every hard/degenerate response. The fix silences
those two loggers for the duration of grading, inside the worker process where
parse()/verify() actually run (grading is dispatched to a spawned subprocess, so
suppressing the loggers in the caller has no effect on the child's own logging
state).
"""

import logging

import pytest

from verl.utils.reward_score.math_verify import (
    _NOISY_MATH_VERIFY_LOGGERS,
    _suppress_math_verify_tracebacks,
    _verify_in_subprocess,
)


def test_suppress_math_verify_tracebacks_disables_and_restores_loggers():
    loggers = [logging.getLogger(name) for name in _NOISY_MATH_VERIFY_LOGGERS]
    for logger in loggers:
        logger.disabled = False

    with _suppress_math_verify_tracebacks():
        assert all(logger.disabled for logger in loggers)

    assert all(not logger.disabled for logger in loggers)


def test_suppress_math_verify_tracebacks_restores_prior_disabled_state():
    logger = logging.getLogger(_NOISY_MATH_VERIFY_LOGGERS[0])
    logger.disabled = True
    try:
        with _suppress_math_verify_tracebacks():
            assert logger.disabled
        assert logger.disabled
    finally:
        logger.disabled = False


def test_verify_in_subprocess_scores_correct_and_incorrect_answers():
    assert _verify_in_subprocess("\\boxed{4}", "The answer is \\boxed{4}.") == 1.0
    assert _verify_in_subprocess("\\boxed{4}", "The answer is \\boxed{5}.") == 0.0


def test_verify_in_subprocess_silences_noisy_loggers_on_garbled_input(caplog):
    # Reproduces the exact failure class observed live: a non-ASCII/garbled model
    # response (e.g. from a degenerate rollout) makes math_verify's own parser raise
    # internally and log the traceback via logging.error before recovering.
    garbled = "final answer \\boxed{᭑}"

    with caplog.at_level(logging.DEBUG):
        score = _verify_in_subprocess("\\boxed{1}", garbled)

    assert isinstance(score, float)
    noisy_records = [r for r in caplog.records if r.name in _NOISY_MATH_VERIFY_LOGGERS]
    assert noisy_records == [], f"expected no log records from {_NOISY_MATH_VERIFY_LOGGERS}, got {noisy_records}"


@pytest.mark.parametrize(
    "gold,pred",
    [
        # Astronomically large exponent: numeric equality-checking can time out
        # internally inside math_verify without raising out to us.
        ("\\boxed{1}", "\\boxed{1997^{1996^{1997}}}"),
        # An integral requiring numeric quadrature to resolve.
        ("\\boxed{0.747}", "\\boxed{\\int_0^1 x^9 (1-x)^{90}\\,dx}"),
    ],
)
def test_verify_in_subprocess_degrades_on_pathological_comparisons_without_noise(caplog, gold, pred):
    with caplog.at_level(logging.DEBUG):
        score = _verify_in_subprocess(gold, pred)

    assert isinstance(score, float)
    noisy_records = [r for r in caplog.records if r.name in _NOISY_MATH_VERIFY_LOGGERS]
    assert noisy_records == [], f"expected no log records from {_NOISY_MATH_VERIFY_LOGGERS}, got {noisy_records}"
