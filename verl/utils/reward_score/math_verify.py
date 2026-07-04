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

import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# math_verify's own grader/parser log a full traceback (via logging.error) for every
# pathological comparison it internally times out or fails to parse -- e.g. a huge
# power expression, an integral needing numeric quadrature, or garbled unicode from a
# degenerate rollout. It still returns a valid (usually 0) score in these cases; the
# logging is just noisy. Silence it inside the worker process where grading runs.
_NOISY_MATH_VERIFY_LOGGERS = ("math_verify.grader", "math_verify.parser")


@contextmanager
def _suppress_math_verify_tracebacks():
    states = []
    for logger_name in _NOISY_MATH_VERIFY_LOGGERS:
        logger = logging.getLogger(logger_name)
        states.append((logger, logger.disabled))
        logger.disabled = True
    try:
        yield
    finally:
        for logger, disabled in states:
            logger.disabled = disabled


_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    with _suppress_math_verify_tracebacks():
        extracted_gold = parse(ground_truth_boxed, gold_targets)
        extracted_pred = parse(model_output, pred_targets)
        if extracted_gold and extracted_pred:
            return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
