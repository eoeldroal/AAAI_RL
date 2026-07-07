# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import importlib.util
import textwrap
from pathlib import Path

import pytest

from verl.utils.import_utils import load_extern_object

REPO_ROOT = Path(__file__).resolve().parents[3]
UPT_V6_ENTROPY_MATH = (REPO_ROOT / "../Unify-Post-Training/hpt/verl/verl/mix_src/entropy_math/__init__.py").resolve()
UPT_V6_ADAPTER_PATH = REPO_ROOT / "verl/utils/reward_score/upt_v6_adapter.py"


def _load_reference_compute_score(path: Path):
    spec = importlib.util.spec_from_file_location("upt_v6_reference_entropy_math", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def test_upt_v6_adapter_requires_explicit_reference_path():
    from verl.utils.reward_score.upt_v6_adapter import compute_score

    with pytest.raises(ValueError, match="entropy_math_path"):
        compute_score(
            data_source="numina_olympiads",
            solution_str=r"The answer is \boxed{1}.",
            ground_truth="1",
        )


def test_upt_v6_adapter_matches_verl_reward_function_signature(tmp_path):
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        "\n".join(
            [
                "def compute_score(model_response, gt_answer, fast=False):",
                "    assert model_response.startswith('<think>\\n')",
                "    return gt_answer in model_response",
            ]
        )
    )

    from verl.utils.reward_score.upt_v6_adapter import compute_score

    assert (
        compute_score(
            data_source="numina_olympiads",
            solution_str=r"The answer is \boxed{180}.",
            ground_truth="180",
            entropy_math_path=str(scorer),
        )
        == 1.0
    )
    assert (
        compute_score(
            data_source="numina_olympiads",
            solution_str=r"The answer is \boxed{0}.",
            ground_truth="180",
            entropy_math_path=str(scorer),
        )
        == 0.0
    )


def test_upt_v6_adapter_process_pool_survives_custom_module_loader(tmp_path):
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        "\n".join(
            [
                "def compute_score(model_response, gt_answer, fast=False):",
                "    assert model_response == '<think>\\nThe answer is \\\\boxed{180}.'",
                "    return gt_answer == '180'",
            ]
        )
    )
    compute_score = load_extern_object(str(UPT_V6_ADAPTER_PATH), "compute_score")

    assert (
        compute_score(
            data_source="numina_olympiads",
            solution_str=r"The answer is \boxed{180}.",
            ground_truth="180",
            entropy_math_path=str(scorer),
            use_process_pool=True,
        )
        == 1.0
    )


def test_upt_v6_adapter_process_pool_suppresses_math_verify_tracebacks(tmp_path, capfd):
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        textwrap.dedent(
            """
            import logging

            def compute_score(model_response, gt_answer, fast=False):
                try:
                    raise TimeoutError("Operation timed out!")
                except TimeoutError:
                    logging.getLogger("math_verify.grader").exception("Error comparing bad expression")
                return False
            """
        )
    )
    compute_score = load_extern_object(str(UPT_V6_ADAPTER_PATH), "compute_score")

    assert (
        compute_score(
            data_source="numina_olympiads",
            solution_str=r"The answer is \boxed{(-x**3 + x**2 + 1)**1000}.",
            ground_truth="60",
            entropy_math_path=str(scorer),
            use_process_pool=True,
        )
        == 0.0
    )

    captured = capfd.readouterr()
    assert "Operation timed out" not in captured.err
    assert "Error comparing bad expression" not in captured.err


def test_upt_v6_adapter_process_pool_timeout_degrades_to_zero_without_raising(tmp_path):
    """A signal-deaf scorer that exceeds ``process_timeout`` must score 0.0 rather
    than raise -- matching UPT's contract that grading never crashes the loop.

    Regression for the validation crash: an unhandled reward ``TimeoutError`` from
    ``future.result(timeout=...)`` propagated through _validate and killed the run.
    """
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        textwrap.dedent(
            """
            import time

            def compute_score(model_response, gt_answer, fast=False):
                time.sleep(30)  # emulate sympy work that ignores the in-scorer SIGALRM
                return True
            """
        )
    )
    compute_score = load_extern_object(str(UPT_V6_ADAPTER_PATH), "compute_score")

    result = compute_score(
        data_source="numina_olympiads",
        solution_str=r"The answer is \boxed{1}.",
        ground_truth="1",
        entropy_math_path=str(scorer),
        use_process_pool=True,
        process_timeout=2.0,
    )
    assert result == 0.0


def test_upt_v6_adapter_recycles_worker_so_single_pool_does_not_stall(tmp_path):
    """After a timeout the worker is SIGKILLed and the pool recycled. Two sequential
    timeouts on the SAME path must both return 0.0 quickly: without kill+recycle the
    second would queue forever behind the still-running first child (max_workers=1).
    """
    scorer = tmp_path / "entropy_math.py"
    scorer.write_text(
        textwrap.dedent(
            """
            import time

            def compute_score(model_response, gt_answer, fast=False):
                time.sleep(30)
                return True
            """
        )
    )
    compute_score = load_extern_object(str(UPT_V6_ADAPTER_PATH), "compute_score")

    for _ in range(2):
        assert (
            compute_score(
                data_source="numina_olympiads",
                solution_str=r"The answer is \boxed{1}.",
                ground_truth="1",
                entropy_math_path=str(scorer),
                use_process_pool=True,
                process_timeout=2.0,
            )
            == 0.0
        )


@pytest.mark.skipif(not UPT_V6_ENTROPY_MATH.exists(), reason="UPT v6 entropy_math checkout is not available")
def test_upt_v6_adapter_matches_checked_out_upt_v6_reference():
    reference_compute_score = _load_reference_compute_score(UPT_V6_ENTROPY_MATH)

    from verl.utils.reward_score.upt_v6_adapter import compute_score

    cases = [
        (r"The answer is \boxed{180}.", "180"),
        (r"The answer is \boxed{\frac{1}{4}}.", r"\frac{1}{4}"),
        (r"The answer is \boxed{D}.", "D"),
        (r"The answer is D.", "D"),
        (r"The answer is \boxed{0}.", "180"),
    ]
    for solution_str, ground_truth in cases:
        expected = float(reference_compute_score(f"<think>\n{solution_str}", ground_truth))
        actual = compute_score(
            data_source="numina_olympiads",
            solution_str=solution_str,
            ground_truth=ground_truth,
            entropy_math_path=str(UPT_V6_ENTROPY_MATH),
        )
        assert actual == expected
