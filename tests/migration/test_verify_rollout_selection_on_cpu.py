from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migration" / "verify_rollout_selection.py"


def _load_module():
    assert SCRIPT.is_file(), f"missing verifier: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("verify_rollout_selection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_validation_rejects_duplicate_groups() -> None:
    module = _load_module()
    manifest = pd.DataFrame(
        {
            "run": ["M5", "M5"],
            "feed_step": [1, 1],
            "attempt_count": [8, 8],
            "archive_name": ["a.tar.zst", "a.tar.zst"],
            "prompt_hash": ["a", "a"],
            "reward_scores": [[0.0] * 8, [0.0] * 8],
            "response_hashes": ["a", "a"],
            "actual_route": ["sft", "sft"],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        module._validate_manifest(manifest, expected_groups=2)


def test_manifest_validation_requires_eight_attempts() -> None:
    module = _load_module()
    manifest = pd.DataFrame(
        {
            "run": ["M5"],
            "feed_step": [1],
            "attempt_count": [7],
            "archive_name": ["a.tar.zst"],
            "prompt_hash": ["a"],
            "reward_scores": [[0.0] * 7],
            "response_hashes": ["a"],
            "actual_route": ["sft"],
        }
    )

    with pytest.raises(ValueError, match="attempt_count"):
        module._validate_manifest(manifest, expected_groups=1)


def test_restore_validation_requires_every_run_root() -> None:
    module = _load_module()
    manifest = pd.DataFrame({"run": ["M5", "nocispo", "RLonly"]})

    with pytest.raises(ValueError, match="exactly match"):
        module._validate_run_roots(manifest, {"M5": Path("/tmp")}, archives_only=False)
    with pytest.raises(ValueError, match="exactly match"):
        module._validate_run_roots(manifest, {}, archives_only=False)
    module._validate_run_roots(manifest, {}, archives_only=True)
