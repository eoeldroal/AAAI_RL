from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migration" / "select_rollout_groups.py"
SPEC = importlib.util.spec_from_file_location("select_rollout_groups", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_route_class_distinguishes_rlonly_k0_from_sft() -> None:
    assert MODULE._route_class("M5", 0) == "sft"
    assert MODULE._route_class("nocispo", 0) == "sft"
    assert MODULE._route_class("RLonly", 0) == "zero_variance_rl"
    assert MODULE._route_class("RLonly", 4) == "informative_rl"
    assert MODULE._route_class("RLonly", 8) == "zero_variance_rl"


def test_group_classification_uses_reward_scores_and_marks_incomplete() -> None:
    assert MODULE._classify_group("M5", [0.0] * 8, attempt_count=8) == (0, "k0", "sft")
    assert MODULE._classify_group("M5", [0.0] * 7, attempt_count=7) == (
        0,
        "incomplete",
        "incomplete",
    )
    assert MODULE._classify_group("RLonly", [1.0] * 8, attempt_count=8) == (
        8,
        "k8",
        "zero_variance_rl",
    )


def test_target_budget_keeps_events_and_stratum_coverage() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5"] * 6,
            "feed_step": [1, 2, 3, 4, 5, 6],
            "run_id": ["run"] * 6,
            "prompt_hash": [f"prompt-{index}" for index in range(6)],
        }
    )
    selected = {
        ("M5", 1): {"paired_core"},
        ("M5", 2): {"partial"},
        ("M5", 3): {"stratum_pv000_004_k0"},
        ("M5", 4): {"stratum_pv000_004_k8"},
        ("M5", 5): {"stratum_pv000_004_k0"},
        ("M5", 6): {"stratum_pv000_004_k8"},
    }

    capped = MODULE._apply_target_budget(frame, selected, target_groups=4)

    assert len(capped) == 4
    assert ("M5", 1) in capped
    assert ("M5", 2) in capped
    reasons = {reason for values in capped.values() for reason in values}
    assert "stratum_pv000_004_k0" in reasons
    assert "stratum_pv000_004_k8" in reasons


def test_target_budget_fills_from_census_when_candidates_are_short() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5"] * 5,
            "feed_step": [1, 2, 3, 4, 5],
            "run_id": ["run"] * 5,
            "prompt_hash": [f"prompt-{index}" for index in range(5)],
        }
    )
    selected = {("M5", 1): {"partial"}}

    capped = MODULE._apply_target_budget(frame, selected, target_groups=4)

    assert len(capped) == 4
    assert capped[("M5", 1)] == {"partial"}
    assert sum("target_fill" in reasons for reasons in capped.values()) == 3


def test_target_budget_round_robins_strata() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5"] * 8,
            "feed_step": list(range(1, 9)),
            "run_id": ["run"] * 8,
            "prompt_hash": list("abcdefgh"),
        }
    )
    selected = {("M5", feed_step): {"stratum_A" if feed_step <= 4 else "stratum_B"} for feed_step in range(1, 9)}

    capped = MODULE._apply_target_budget(frame, selected, target_groups=6)

    counts = {reason: sum(reason in reasons for reasons in capped.values()) for reason in ("stratum_A", "stratum_B")}
    assert counts == {"stratum_A": 3, "stratum_B": 3}


def test_stratum_reason_keeps_run_identity() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5", "RLonly"],
            "feed_step": [1, 1],
            "run_id": ["m5", "rl"],
            "prompt_hash": ["a", "b"],
            "max_param_version": [2, 2],
            "k_class": ["k0", "k0"],
        }
    )
    selected: dict[tuple[str, int], set[str]] = {}

    MODULE._select_strata(frame, selected, quota=1)

    assert selected[("M5", 1)] == {"stratum_M5_pv000_004_k0"}
    assert selected[("RLonly", 1)] == {"stratum_RLonly_pv000_004_k0"}


def test_prompt_hash_normalizes_numpy_and_python_containers() -> None:
    numpy_prompt = np.array([{"role": np.str_("user"), "content": np.str_("question")}], dtype=object)
    python_prompt = [{"content": "question", "role": "user"}]

    assert MODULE._canonical_prompt_hash(numpy_prompt) == MODULE._canonical_prompt_hash(python_prompt)


def test_unique_prompt_sampling_is_not_weighted_by_repeat_count() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5", "M5", "M5"],
            "feed_step": [1, 2, 3],
            "run_id": ["run", "run", "run"],
            "prompt_hash": ["repeated", "repeated", "single"],
        }
    )
    namespace = "unit"
    expected = min(
        {"repeated", "single"},
        key=lambda prompt_hash: MODULE._priority(namespace, "run", prompt_hash),
    )

    chosen = MODULE._choose_unique_prompt_rows(frame, 1, namespace)

    assert chosen.iloc[0].prompt_hash == expected


def test_archive_members_include_all_eight_attempts(tmp_path: Path) -> None:
    group = tmp_path / "42"
    for attempt_index in range(8):
        attempt = group / f"attempt_{attempt_index}"
        attempt.mkdir(parents=True)
        (attempt / "gen_batch.dp").write_bytes(b"dp")
        (attempt / "meta.json").write_text("{}", encoding="utf-8")

    members = MODULE._group_archive_members(group, tmp_path)

    assert len(members) == 16
    assert members[0] == "42/attempt_0/gen_batch.dp"
    assert members[-1] == "42/attempt_7/meta.json"


def test_archive_shards_use_deterministic_size_balancing() -> None:
    manifest = pd.DataFrame(
        {
            "run": ["M5"] * 4,
            "feed_step": [1, 2, 3, 4],
            "run_id": ["run"] * 4,
            "prompt_hash": ["a", "b", "c", "d"],
            "raw_bytes": [10, 9, 8, 7],
        }
    )

    assigned = MODULE._assign_archive_shards(manifest, archive_shards=2)

    assert assigned.groupby("archive_shard").raw_bytes.sum().sort_values().tolist() == [17, 17]
    assert assigned.archive_name.nunique() == 2
    assert assigned.archive_member.tolist() == ["1", "2", "3", "4"]


def test_trajectory_class_covers_improvement_and_regression() -> None:
    improvement = pd.DataFrame(
        {
            "max_param_version": [1, 6, 11],
            "feed_step": [1, 2, 3],
            "k_class": ["k0", "k1_7", "k8"],
        }
    )
    regression = improvement.assign(k_class=["k8", "k1_7", "k0"])

    assert MODULE._trajectory_class(improvement) == "improvement"
    assert MODULE._trajectory_class(regression) == "regression"


def test_matched_event_selection_adds_case_and_control() -> None:
    frame = pd.DataFrame(
        {
            "run": ["M5", "M5"],
            "run_id": ["run", "run"],
            "feed_step": [1, 2],
            "prompt_hash": ["case", "control"],
            "max_param_version": [10, 10],
            "k_class": ["k0", "k0"],
            "source_subtype": ["olympiads", "olympiads"],
            "prompt_token_length": [128, 128],
            "tau_char_length": [8000, 8000],
            "response_length_max": [100, 100],
            "generation_time_max": [1.0, 1.0],
            "partial_attempts": [1, 0],
            "partial_span": [0, 0],
            "truncated_attempts": [0, 0],
            "truncated_correct_attempts": [0, 0],
            "raw_correct_attempts": [0, 0],
            "correct_attempts": [0, 0],
        }
    )
    selected: dict[tuple[str, int], set[str]] = {}

    MODULE._select_matched_event_pairs(frame, selected)

    reasons = {reason for values in selected.values() for reason in values}
    assert any(reason.startswith("event_partial_case_") for reason in reasons)
    assert any(reason.startswith("event_partial_control_") for reason in reasons)
