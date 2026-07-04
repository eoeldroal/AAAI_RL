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

"""Adapter for using UPT reward_impl_version=6 through verl custom rewards.

The UPT scorer is intentionally loaded from an explicit file path rather than
vendored here. That keeps the comparison tied to the exact UPT implementation
while preserving verl's custom_reward_function contract.
"""

import concurrent.futures
import importlib
import importlib.util
import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

UPT_V6_THOUGHT_PREFIX = "<think>\n"
_NOISY_MATH_VERIFY_LOGGERS = ("math_verify.grader", "math_verify.parser")
_EXECUTOR_LOCK = threading.Lock()
_EXECUTORS: dict[str, ProcessPoolExecutor] = {}


def _resolve_entropy_math_path(entropy_math_path: str) -> Path:
    path = Path(entropy_math_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"UPT v6 entropy_math_path does not exist: {path}")
    return path


@lru_cache(maxsize=8)
def _load_upt_v6_compute_score(entropy_math_path: str) -> Callable:
    path = _resolve_entropy_math_path(entropy_math_path)

    spec = importlib.util.spec_from_file_location(f"upt_v6_entropy_math_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load UPT v6 entropy_math module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute_score = getattr(module, "compute_score", None)
    if not callable(compute_score):
        raise AttributeError(f"UPT v6 entropy_math module has no callable compute_score: {path}")
    return compute_score


def _compute_score_in_process(entropy_math_path: str, response: str, ground_truth: str) -> float:
    upt_compute_score = _load_upt_v6_compute_score(entropy_math_path)
    with _suppress_math_verify_tracebacks():
        return float(upt_compute_score(response, ground_truth))


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


def _get_process_target() -> Callable[[str, str, str], float]:
    module = importlib.import_module("verl.utils.reward_score.upt_v6_adapter")
    return module._compute_score_in_process


def _get_executor(entropy_math_path: str) -> ProcessPoolExecutor:
    path = str(_resolve_entropy_math_path(entropy_math_path))
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(path)
        if executor is None:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
            _EXECUTORS[path] = executor
        return executor


def _recycle_executor(path: str) -> None:
    """Hard-kill and discard the executor cached under ``path``.

    A single-worker pool that is blocked on a signal-deaf computation (e.g. sympy
    expanding a huge expression, which ignores the scorer's SIGALRM) would stall
    every subsequent scoring. ``future.result(timeout=...)`` only stops *waiting*;
    it does not stop the child. So we SIGKILL the worker (uncatchable, unlike the
    in-scorer SIGALRM) and drop the executor; a fresh one is created lazily next call.
    """
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.pop(path, None)
    if executor is None:
        return
    for proc in list(getattr(executor, "_processes", {}).values()):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    executor.shutdown(wait=False, cancel_futures=True)


def _score_via_process(entropy_math_path: str, response: str, ground_truth: str, timeout: float | None) -> float:
    """Run the UPT scorer in a worker process under a HARD wall-clock bound.

    UPT runs the scorer synchronously on the main thread, where its SIGALRM timeout
    fires and grading therefore *never raises* — an unverifiable answer simply scores
    0. We run in an async, non-main-thread host (SIGALRM unavailable), so we must
    reproduce that contract at this boundary: any timeout / dead worker / unexpected
    failure degrades to 0.0 and never propagates an exception into verl (which would
    crash training or, as observed, validation). The offending worker is killed so it
    cannot stall later scorings.
    """
    path = str(_resolve_entropy_math_path(entropy_math_path))
    try:
        future = _get_executor(entropy_math_path).submit(
            _get_process_target(), entropy_math_path, response, ground_truth
        )
    except BrokenProcessPool:
        _recycle_executor(path)
        future = _get_executor(entropy_math_path).submit(
            _get_process_target(), entropy_math_path, response, ground_truth
        )
    try:
        return float(future.result(timeout=timeout))
    except concurrent.futures.TimeoutError:
        logger.warning("UPT v6 reward exceeded %ss; killing worker and scoring 0.0.", timeout)
        _recycle_executor(path)
        return 0.0
    except BrokenProcessPool:
        logger.warning("UPT v6 reward worker died (BrokenProcessPool); scoring 0.0 and recycling.")
        _recycle_executor(path)
        return 0.0
    except Exception:  # noqa: BLE001 - grading must never crash the training/validation loop
        logger.exception("UPT v6 reward worker failed unexpectedly; scoring 0.0 and recycling.")
        _recycle_executor(path)
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    entropy_math_path: str | None = None,
    add_thought_prefix: bool = True,
    use_process_pool: bool = False,
    process_timeout: float | None = 30.0,
    **kwargs,
) -> float:
    """Compute reward with the exact checked-out UPT v6 scorer.

    Args:
        data_source: Included for verl custom reward compatibility.
        solution_str: Decoded model response from the reward manager.
        ground_truth: Dataset answer.
        extra_info: Included for verl custom reward compatibility.
        entropy_math_path: Path to UPT's ``entropy_math/__init__.py``.
        add_thought_prefix: Reproduce UPT RewardManager v6 preprocessing.
        use_process_pool: Run the scorer in a child process. This is required
            when the caller is not Python's main thread because UPT v6 uses
            ``signal.signal`` for timeout handling.
        process_timeout: Hard wall-clock bound (seconds) for a single scoring. On
            timeout the worker is SIGKILLed and the score degrades to 0.0 rather
            than raising -- matching UPT's contract that grading never crashes the
            training/validation loop. ``None`` disables the outer bound.
    """

    del data_source, extra_info, kwargs
    if not entropy_math_path:
        raise ValueError("entropy_math_path is required for UPT v6 reward parity.")

    response = str(solution_str)
    if add_thought_prefix:
        response = f"{UPT_V6_THOUGHT_PREFIX}{response}"

    if use_process_pool:
        return _score_via_process(entropy_math_path, response, ground_truth, process_timeout)

    upt_compute_score = _load_upt_v6_compute_score(entropy_math_path)
    return float(upt_compute_score(response, ground_truth))
