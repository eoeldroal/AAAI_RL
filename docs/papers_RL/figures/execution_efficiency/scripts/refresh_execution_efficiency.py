#!/usr/bin/env python3
"""Export W&B cycle histories and rebuild StreamWeave efficiency assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


BUNDLE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BUNDLE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
VERIFIED_PATH = DATA_DIR / "verified_snapshot.json"
FIGURE_INPUT_PATH = DATA_DIR / "figure3_execution_efficiency.json"

RAW_COLUMNS = [
    "run_alias",
    "_step",
    "_timestamp",
    "_runtime",
    "groups",
    "training_time_s",
    "generation_or_interface_time_s",
    "parameter_sync_time_s",
    "rl_groups",
    "expert_groups",
    "carryover_discarded_groups",
    "queue_size",
    "partial_rollouts",
    "validation_time_s",
]

GPU_COUNT = 8
SYSTEM_RAW_COLUMNS = [
    "run_alias",
    "_timestamp",
    "_runtime",
    *[f"gpu_{index}_sm_active" for index in range(GPU_COUNT)],
    *[f"gpu_{index}_power_watts" for index in range(GPU_COUNT)],
]
GPU_ACTIVITY_THRESHOLDS = [10, 20, 30, 40, 50]
FIGURE_GPU_ACTIVITY_THRESHOLD = 20


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def finite_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_run_history(manifest: dict[str, Any], alias: str) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as error:
        raise SystemExit(
            "wandb is required for --fetch. Use the uv command documented in "
            "figures/execution_efficiency/data/README.md."
        ) from error

    run_spec = manifest["runs"][alias]
    run_path = "/".join(
        [manifest["entity"], manifest["project"], run_spec["run_id"]]
    )
    run = wandb.Api(timeout=120).run(run_path)
    key_map = run_spec["metric_keys"]
    rows: list[dict[str, Any]] = []

    for history_row in run.scan_history(page_size=1000):
        normalized: dict[str, Any] = {
            "run_alias": alias,
            "_step": history_row.get("_step"),
            "_timestamp": history_row.get("_timestamp"),
            "_runtime": history_row.get("_runtime"),
        }
        for normalized_key in RAW_COLUMNS[4:]:
            source_key = key_map.get(normalized_key)
            normalized[normalized_key] = (
                history_row.get(source_key) if source_key else None
            )

        groups = finite_number(normalized["groups"])
        training_time = finite_number(normalized["training_time_s"])
        if groups is None or groups <= 0 or training_time is None or training_time <= 0:
            continue
        rows.append(normalized)

    expected_cycles = int(run_spec["expected_cycles"])
    if len(rows) != expected_cycles:
        raise SystemExit(
            f"{alias}: expected {expected_cycles} cycle rows, received {len(rows)}. "
            "Refuse to write a potentially sampled or incomplete history."
        )
    return rows


def fetch_run_system_history(
    manifest: dict[str, Any], alias: str
) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as error:
        raise SystemExit(
            "wandb is required for --fetch. Use the uv command documented in "
            "figures/execution_efficiency/data/README.md."
        ) from error

    run_spec = manifest["runs"][alias]
    run_path = "/".join(
        [manifest["entity"], manifest["project"], run_spec["run_id"]]
    )
    run = wandb.Api(timeout=120).run(run_path)
    expected_rows = int(run_spec["expected_system_rows"])
    history = run.history(
        samples=max(expected_rows * 2, 10_000),
        pandas=False,
        stream="system",
    )
    rows = []
    for history_row in history:
        normalized = {
            "run_alias": alias,
            "_timestamp": history_row.get("_timestamp"),
            "_runtime": history_row.get("_runtime"),
        }
        for index in range(GPU_COUNT):
            normalized[f"gpu_{index}_sm_active"] = history_row.get(
                f"system.gpu.{index}.smActive"
            )
            normalized[f"gpu_{index}_power_watts"] = history_row.get(
                f"system.gpu.{index}.powerWatts"
            )
        rows.append(normalized)

    if len(rows) != expected_rows:
        raise SystemExit(
            f"{alias}: expected {expected_rows} system rows, received {len(rows)}. "
            "Refuse to write a sampled or incomplete telemetry history."
        )
    return rows


def read_raw_history(alias: str) -> list[dict[str, Any]]:
    path = RAW_DIR / f"{alias}_history.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run with --fetch in an authenticated W&B environment."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_raw_system_history(alias: str) -> list[dict[str, Any]]:
    path = RAW_DIR / f"{alias}_system.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run with --fetch in an authenticated W&B environment."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in rows:
        groups = finite_number(row.get("groups"))
        training_time = finite_number(row.get("training_time_s"))
        if groups is None or groups <= 0 or training_time is None or training_time <= 0:
            continue
        normalized = {"groups": groups, "training_time_s": training_time}
        for key in RAW_COLUMNS[6:]:
            value = finite_number(row.get(key))
            if value is not None:
                normalized[key] = value
        timestamp = finite_number(row.get("_timestamp"))
        if timestamp is not None:
            normalized["_timestamp"] = timestamp
        result.append(normalized)
    return result


def numeric_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    result = []
    for row in rows:
        timestamp = finite_number(row.get("_timestamp"))
        sm_values = [
            finite_number(row.get(f"gpu_{index}_sm_active"))
            for index in range(GPU_COUNT)
        ]
        power_values = [
            finite_number(row.get(f"gpu_{index}_power_watts"))
            for index in range(GPU_COUNT)
        ]
        has_sm = all(value is not None for value in sm_values)
        has_power = all(value is not None for value in power_values)
        if timestamp is None or not (has_sm or has_power):
            continue
        normalized = {"_timestamp": timestamp}
        if has_sm:
            normalized.update(
                {
                    f"gpu_{index}_sm_active": float(sm_values[index])
                    for index in range(GPU_COUNT)
                }
            )
        if has_power:
            normalized.update(
                {
                    f"gpu_{index}_power_watts": float(power_values[index])
                    for index in range(GPU_COUNT)
                }
            )
            normalized["total_gpu_power_watts"] = sum(
                float(value) for value in power_values
            )
        result.append(normalized)
    return result


def aggregate(alias: str, run_id: str, rows: list[dict[str, float]]) -> dict[str, Any]:
    groups = sum(row["groups"] for row in rows)
    training_time = sum(row["training_time_s"] for row in rows)
    throughput = groups / training_time
    return {
        "run_alias": alias,
        "run_id": run_id,
        "cycles": len(rows),
        "groups": groups,
        "training_time_s": training_time,
        "throughput_groups_per_s": throughput,
        "seconds_per_128_groups": 128.0 / throughput,
    }


def training_intervals(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    intervals = []
    for row in rows:
        # W&B logs a cycle after optional validation. End-anchoring on the row
        # timestamp is safe only for cycles without a validation timer.
        if "validation_time_s" in row:
            continue
        timestamp = row.get("_timestamp")
        duration = row.get("training_time_s")
        if timestamp is None or duration is None:
            continue
        intervals.append(
            {
                "start": timestamp - duration,
                "end": timestamp,
            }
        )
    return intervals


def telemetry_in_intervals(
    rows: list[dict[str, float]], intervals: list[dict[str, float]]
) -> list[dict[str, float]]:
    ordered_rows = sorted(rows, key=lambda row: row["_timestamp"])
    ordered_intervals = sorted(intervals, key=lambda interval: interval["start"])
    selected = []
    interval_index = 0

    for row in ordered_rows:
        timestamp = row["_timestamp"]
        while (
            interval_index < len(ordered_intervals)
            and timestamp > ordered_intervals[interval_index]["end"]
        ):
            interval_index += 1
        if interval_index >= len(ordered_intervals):
            break
        interval = ordered_intervals[interval_index]
        if interval["start"] <= timestamp <= interval["end"]:
            selected.append(row)
    return selected


def gpu_activity_summary(
    rows: list[dict[str, float]], threshold: int
) -> dict[str, Any]:
    counts = [
        sum(
            row[f"gpu_{index}_sm_active"] > threshold
            for index in range(GPU_COUNT)
        )
        for row in rows
    ]
    if not counts:
        raise ValueError("No complete GPU telemetry rows were selected.")
    histogram = [
        {
            "active_gpu_count": active_count,
            "samples": counts.count(active_count),
            "share": counts.count(active_count) / len(counts),
        }
        for active_count in range(GPU_COUNT + 1)
    ]
    return {
        "threshold_percent": threshold,
        "samples": len(counts),
        "zero_active_share": histogram[0]["share"],
        "mean_active_gpus": sum(counts) / len(counts),
        "histogram": histogram,
    }


def group_window_intervals(
    rows: list[dict[str, float]],
    start_group: float,
    budget_groups: float,
) -> list[dict[str, float]]:
    """Map a prompt-group work window to linearly allocated cycle intervals."""
    end_group = start_group + budget_groups
    cursor = 0.0
    selected_groups = 0.0
    intervals = []

    for row in rows:
        if "validation_time_s" in row:
            continue
        groups = row.get("groups")
        timestamp = row.get("_timestamp")
        duration = row.get("training_time_s")
        if groups is None or timestamp is None or duration is None:
            continue

        row_start = cursor
        row_end = cursor + groups
        overlap_start = max(row_start, start_group)
        overlap_end = min(row_end, end_group)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap:
            cycle_start = timestamp - duration
            start_fraction = (overlap_start - row_start) / groups
            end_fraction = (overlap_end - row_start) / groups
            intervals.append(
                {
                    "start": cycle_start + duration * start_fraction,
                    "end": cycle_start + duration * end_fraction,
                }
            )
            selected_groups += overlap
        cursor = row_end
        if cursor >= end_group:
            break

    if not math.isclose(
        selected_groups,
        budget_groups,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Requested {budget_groups} non-validation groups at offset "
            f"{start_group}, but only selected {selected_groups}."
        )
    return intervals


def derive_gpu_activity(
    histories: dict[str, list[dict[str, float]]],
    system_histories: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    selected = {}
    startup_excluded_selected = {}
    sm_rows_by_alias = {}
    interval_counts = {}
    excluded_validation_cycles = {}
    for alias in ("sync", "streamweave"):
        intervals = training_intervals(histories[alias])
        sm_rows = [
            row
            for row in system_histories[alias]
            if all(
                f"gpu_{index}_sm_active" in row
                for index in range(GPU_COUNT)
            )
        ]
        sm_rows_by_alias[alias] = sm_rows
        selected[alias] = telemetry_in_intervals(sm_rows, intervals)
        startup_excluded_intervals = training_intervals(histories[alias][1:])
        startup_excluded_selected[alias] = telemetry_in_intervals(
            sm_rows, startup_excluded_intervals
        )
        interval_counts[alias] = len(intervals)
        excluded_validation_cycles[alias] = (
            len(histories[alias]) - len(intervals)
        )

    sensitivity = []
    summaries = {}
    for threshold in GPU_ACTIVITY_THRESHOLDS:
        threshold_summaries = {
            alias: gpu_activity_summary(selected[alias], threshold)
            for alias in ("sync", "streamweave")
        }
        sensitivity.append(
            {
                "threshold_percent": threshold,
                "sync_zero_active_share": threshold_summaries["sync"][
                    "zero_active_share"
                ],
                "streamweave_zero_active_share": threshold_summaries[
                    "streamweave"
                ]["zero_active_share"],
                "sync_mean_active_gpus": threshold_summaries["sync"][
                    "mean_active_gpus"
                ],
                "streamweave_mean_active_gpus": threshold_summaries[
                    "streamweave"
                ]["mean_active_gpus"],
            }
        )
        if threshold == FIGURE_GPU_ACTIVITY_THRESHOLD:
            summaries = threshold_summaries

    non_validation_group_totals = {
        alias: sum(
            row["groups"]
            for row in histories[alias]
            if "validation_time_s" not in row
        )
        for alias in ("sync", "streamweave")
    }
    equal_work_budget = min(non_validation_group_totals.values())
    streamweave_total = non_validation_group_totals["streamweave"]
    equal_work_intervals = {
        "sync_full": group_window_intervals(
            histories["sync"],
            start_group=0.0,
            budget_groups=equal_work_budget,
        ),
        "streamweave_first": group_window_intervals(
            histories["streamweave"],
            start_group=0.0,
            budget_groups=equal_work_budget,
        ),
        "streamweave_last": group_window_intervals(
            histories["streamweave"],
            start_group=streamweave_total - equal_work_budget,
            budget_groups=equal_work_budget,
        ),
    }
    equal_work_summaries = {
        label: gpu_activity_summary(
            telemetry_in_intervals(
                sm_rows_by_alias[
                    "sync" if label == "sync_full" else "streamweave"
                ],
                intervals,
            ),
            FIGURE_GPU_ACTIVITY_THRESHOLD,
        )
        for label, intervals in equal_work_intervals.items()
    }

    return {
        "metric": "count of GPUs with system.gpu.<i>.smActive above threshold",
        "sample_interval_seconds": 15,
        "interval_definition": (
            "[history._timestamp - timing_s/step, history._timestamp], "
            "excluding cycles with a validation timer"
        ),
        "training_cycles": interval_counts,
        "excluded_validation_cycles": excluded_validation_cycles,
        "figure_threshold_percent": FIGURE_GPU_ACTIVITY_THRESHOLD,
        "runs": summaries,
        "startup_excluded": {
            alias: gpu_activity_summary(
                startup_excluded_selected[alias],
                FIGURE_GPU_ACTIVITY_THRESHOLD,
            )
            for alias in ("sync", "streamweave")
        },
        "equal_non_validation_work": {
            "budget_groups": equal_work_budget,
            **equal_work_summaries,
        },
        "threshold_sensitivity": sensitivity,
    }


def non_validation_cycles(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    cycles = []
    for history_index, row in enumerate(rows, start=1):
        if "validation_time_s" in row:
            continue
        timestamp = row.get("_timestamp")
        duration = row.get("training_time_s")
        groups = row.get("groups")
        if timestamp is None or duration is None or groups is None:
            continue
        cycles.append(
            {
                "history_index": history_index,
                "start": timestamp - duration,
                "end": timestamp,
                "duration_s": duration,
                "groups": groups,
            }
        )
    return cycles


def validation_shifted_cycles(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    cycles = []
    for history_index, row in enumerate(rows, start=1):
        timestamp = row.get("_timestamp")
        duration = row.get("training_time_s")
        groups = row.get("groups")
        if timestamp is None or duration is None or groups is None:
            continue
        validation_duration = row.get("validation_time_s", 0.0)
        end = timestamp - validation_duration
        cycles.append(
            {
                "history_index": history_index,
                "start": end - duration,
                "end": end,
                "duration_s": duration,
                "groups": groups,
            }
        )
    return cycles


def power_rows_by_cycle(
    telemetry_rows: list[dict[str, float]],
    cycles: list[dict[str, float]],
) -> list[dict[str, Any]]:
    ordered_cycles = sorted(cycles, key=lambda cycle: cycle["start"])
    ordered_rows = sorted(telemetry_rows, key=lambda row: row["_timestamp"])
    assigned = [[] for _ in ordered_cycles]
    cycle_index = 0

    # Assign every 15-second telemetry row to at most one training cycle.
    # This avoids double counting when W&B's aggregated async timers produce
    # slightly overlapping reconstructed intervals.
    for row in ordered_rows:
        timestamp = row["_timestamp"]
        while (
            cycle_index < len(ordered_cycles)
            and timestamp > ordered_cycles[cycle_index]["end"]
        ):
            cycle_index += 1
        if cycle_index >= len(ordered_cycles):
            break
        cycle = ordered_cycles[cycle_index]
        if cycle["start"] <= timestamp <= cycle["end"]:
            assigned[cycle_index].append(row["total_gpu_power_watts"])

    result = []
    for cycle, samples in zip(ordered_cycles, assigned, strict=True):
        if not samples:
            continue
        mean_power = sum(samples) / len(samples)
        trimmed_samples = samples[1:-1] if len(samples) > 2 else samples
        trimmed_mean_power = sum(trimmed_samples) / len(trimmed_samples)
        duration = cycle["duration_s"]
        groups = cycle["groups"]
        result.append(
            {
                **cycle,
                "power_samples": len(samples),
                "mean_power_watts": mean_power,
                "trimmed_mean_power_watts": trimmed_mean_power,
                "throughput_groups_per_s": groups / duration,
                "energy_per_group_joules": mean_power * duration / groups,
                "trimmed_energy_per_group_joules": (
                    trimmed_mean_power * duration / groups
                ),
            }
        )
    return result


def weighted_energy_slice(
    points: list[dict[str, float]],
    start_group: float,
    budget_groups: float,
) -> dict[str, float]:
    end_group = start_group + budget_groups
    cursor = 0.0
    selected_groups = 0.0
    selected_energy = 0.0

    for point in points:
        point_start = cursor
        point_end = cursor + point["groups"]
        overlap = max(
            0.0,
            min(point_end, end_group) - max(point_start, start_group),
        )
        if overlap:
            selected_groups += overlap
            selected_energy += overlap * point["energy_per_group_joules"]
        cursor = point_end
        if cursor >= end_group:
            break

    if not math.isclose(
        selected_groups,
        budget_groups,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Requested {budget_groups} energy groups at offset {start_group}, "
            f"but only selected {selected_groups}."
        )
    return {
        "groups": selected_groups,
        "energy_per_group_joules": selected_energy / selected_groups,
    }


def derive_gpu_energy(
    histories: dict[str, list[dict[str, float]]],
    system_histories: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    cycle_rows = {}
    summaries = {}
    coverage_summaries = {}
    for alias in ("sync", "streamweave"):
        power_rows = [
            row
            for row in system_histories[alias]
            if "total_gpu_power_watts" in row
        ]
        cycles = non_validation_cycles(histories[alias])
        points = power_rows_by_cycle(power_rows, cycles)
        if len(points) != len(cycles):
            raise ValueError(
                f"{alias}: power telemetry covers {len(points)} of "
                f"{len(cycles)} non-validation training cycles."
            )

        total_groups = sum(point["groups"] for point in points)
        total_duration = sum(point["duration_s"] for point in points)
        all_power_samples = [
            row["total_gpu_power_watts"]
            for row in telemetry_in_intervals(
                power_rows,
                [
                    {"start": cycle["start"], "end": cycle["end"]}
                    for cycle in cycles
                ],
            )
        ]
        pooled_mean_power = sum(all_power_samples) / len(all_power_samples)
        pooled_energy = pooled_mean_power * total_duration / total_groups
        cycle_weighted_energy = (
            sum(
                point["mean_power_watts"] * point["duration_s"]
                for point in points
            )
            / total_groups
        )
        edge_trimmed_energy = (
            sum(
                point["trimmed_mean_power_watts"] * point["duration_s"]
                for point in points
            )
            / total_groups
        )
        startup_excluded_points = points[1:]
        startup_excluded_groups = sum(
            point["groups"] for point in startup_excluded_points
        )
        startup_excluded_energy = (
            sum(
                point["mean_power_watts"] * point["duration_s"]
                for point in startup_excluded_points
            )
            / startup_excluded_groups
        )

        full_cycles = validation_shifted_cycles(histories[alias])
        full_points = power_rows_by_cycle(power_rows, full_cycles)
        if len(full_points) != len(full_cycles):
            raise ValueError(
                f"{alias}: validation-shifted power telemetry covers "
                f"{len(full_points)} of {len(full_cycles)} training cycles."
            )
        full_groups = sum(point["groups"] for point in full_points)
        full_energy = (
            sum(
                point["mean_power_watts"] * point["duration_s"]
                for point in full_points
            )
            / full_groups
        )
        cycle_rows[alias] = points
        summaries[alias] = {
            "cycles": len(points),
            "telemetry_samples": len(all_power_samples),
            "groups": total_groups,
            "training_time_s": total_duration,
            "pooled_mean_power_watts": pooled_mean_power,
            "pooled_energy_per_group_joules": pooled_energy,
            "cycle_weighted_energy_per_group_joules": cycle_weighted_energy,
            "edge_trimmed_energy_per_group_joules": edge_trimmed_energy,
            "startup_excluded_energy_per_group_joules": (
                startup_excluded_energy
            ),
            "groups_per_kwh_cycle_weighted": (
                3_600_000.0 / cycle_weighted_energy
            ),
        }
        coverage_summaries[alias] = {
            "cycles": len(full_points),
            "telemetry_samples": sum(
                point["power_samples"] for point in full_points
            ),
            "groups": full_groups,
            "validation_shifted_energy_per_group_joules": full_energy,
        }

    reductions = {}
    for estimator in (
        "pooled_energy_per_group_joules",
        "cycle_weighted_energy_per_group_joules",
        "edge_trimmed_energy_per_group_joules",
        "startup_excluded_energy_per_group_joules",
    ):
        reductions[estimator] = 1.0 - (
            summaries["streamweave"][estimator]
            / summaries["sync"][estimator]
        )
    full_coverage_reduction = 1.0 - (
        coverage_summaries["streamweave"][
            "validation_shifted_energy_per_group_joules"
        ]
        / coverage_summaries["sync"][
            "validation_shifted_energy_per_group_joules"
        ]
    )
    non_validation_group_totals = {
        alias: sum(point["groups"] for point in cycle_rows[alias])
        for alias in ("sync", "streamweave")
    }
    equal_work_budget = min(non_validation_group_totals.values())
    streamweave_total = non_validation_group_totals["streamweave"]
    equal_work = {
        "budget_groups": equal_work_budget,
        "sync_full": weighted_energy_slice(
            cycle_rows["sync"],
            start_group=0.0,
            budget_groups=equal_work_budget,
        ),
        "streamweave_first": weighted_energy_slice(
            cycle_rows["streamweave"],
            start_group=0.0,
            budget_groups=equal_work_budget,
        ),
        "streamweave_last": weighted_energy_slice(
            cycle_rows["streamweave"],
            start_group=streamweave_total - equal_work_budget,
            budget_groups=equal_work_budget,
        ),
    }
    sync_equal_energy = equal_work["sync_full"][
        "energy_per_group_joules"
    ]
    for label in ("streamweave_first", "streamweave_last"):
        equal_work[label]["relative_reduction"] = 1.0 - (
            equal_work[label]["energy_per_group_joules"]
            / sync_equal_energy
        )

    return {
        "metric": "sample-based eight-GPU energy estimate per consumed prompt group",
        "sample_interval_seconds": 15,
        "population": "fully observed non-validation training cycles",
        "estimator": (
            "cycle mean of total gpu power multiplied by cycle training "
            "duration, divided by consumed prompt groups"
        ),
        "runs": summaries,
        "validation_shifted_full_coverage": coverage_summaries,
        "cycle_rows": cycle_rows,
        "relative_reduction": reductions,
        "validation_shifted_full_coverage_reduction": (
            full_coverage_reduction
        ),
        "equal_non_validation_work": equal_work,
        "reduction_range": [
            min([*reductions.values(), full_coverage_reduction]),
            max([*reductions.values(), full_coverage_reduction]),
        ],
    }


def work_weighted_energy_ecdf(
    points: list[dict[str, float]],
    cap_kilojoules: float,
) -> list[dict[str, float]]:
    total_groups = sum(point["groups"] for point in points)
    cumulative_groups = 0.0
    result = []
    for point in sorted(
        points,
        key=lambda item: item["energy_per_group_joules"],
    ):
        cumulative_groups += point["groups"]
        energy_kilojoules = point["energy_per_group_joules"] / 1000.0
        result.append(
            {
                "energy_kilojoules": min(
                    energy_kilojoules,
                    cap_kilojoules,
                ),
                "cumulative_group_share_percent": (
                    cumulative_groups / total_groups * 100.0
                ),
                "right_censored": energy_kilojoules > cap_kilojoules,
            }
        )
    return result


def validate_against_verified(
    sync: dict[str, Any],
    streamweave: dict[str, Any],
    gpu_activity: dict[str, Any],
    gpu_energy: dict[str, Any],
    allow_drift: bool,
) -> None:
    if allow_drift or not VERIFIED_PATH.exists():
        return

    verified_snapshot = read_json(VERIFIED_PATH)
    verified = verified_snapshot["headline"]
    checks = [
        ("sync.groups", sync["groups"], float(verified["sync"]["groups"]), 1e-6),
        (
            "sync.training_time_s",
            sync["training_time_s"],
            float(verified["sync"]["training_time_s"]),
            1e-3,
        ),
        (
            "streamweave.groups",
            streamweave["groups"],
            float(verified["streamweave"]["groups"]),
            1e-6,
        ),
        (
            "streamweave.training_time_s",
            streamweave["training_time_s"],
            float(verified["streamweave"]["training_time_s"]),
            1e-3,
        ),
    ]
    verified_gpu = verified_snapshot.get("gpu_telemetry", {})
    if "active_gpu_distribution" in verified_gpu:
        expected = verified_gpu["active_gpu_distribution"]
        checks.extend(
            [
                (
                    "gpu.sync.samples",
                    gpu_activity["runs"]["sync"]["samples"],
                    float(expected["sync"]["samples"]),
                    0.0,
                ),
                (
                    "gpu.streamweave.samples",
                    gpu_activity["runs"]["streamweave"]["samples"],
                    float(expected["streamweave"]["samples"]),
                    0.0,
                ),
                (
                    "gpu.sync.zero_active_share",
                    gpu_activity["runs"]["sync"]["zero_active_share"],
                    float(expected["sync"]["zero_active_share"]),
                    1e-9,
                ),
                (
                    "gpu.streamweave.zero_active_share",
                    gpu_activity["runs"]["streamweave"]["zero_active_share"],
                    float(expected["streamweave"]["zero_active_share"]),
                    1e-9,
                ),
                (
                    "gpu.sync.mean_active_gpus",
                    gpu_activity["runs"]["sync"]["mean_active_gpus"],
                    float(expected["sync"]["mean_active_gpus"]),
                    1e-9,
                ),
                (
                    "gpu.streamweave.mean_active_gpus",
                    gpu_activity["runs"]["streamweave"]["mean_active_gpus"],
                    float(expected["streamweave"]["mean_active_gpus"]),
                    1e-9,
                ),
            ]
        )
        for alias in ("sync", "streamweave"):
            expected_histogram = expected[alias][
                "histogram_samples_for_0_to_8_active_gpus"
            ]
            computed_histogram = [
                item["samples"]
                for item in gpu_activity["runs"][alias]["histogram"]
            ]
            if len(computed_histogram) != len(expected_histogram):
                raise SystemExit(
                    f"gpu.{alias}.histogram length drift: "
                    f"computed={len(computed_histogram)}, "
                    f"verified={len(expected_histogram)}"
                )
            for index, (computed_count, expected_count) in enumerate(
                zip(computed_histogram, expected_histogram)
            ):
                checks.append(
                    (
                        f"gpu.{alias}.histogram[{index}]",
                        computed_count,
                        float(expected_count),
                        0.0,
                    )
                )

            expected_startup = expected["first_training_cycle_excluded"][alias]
            computed_startup = gpu_activity["startup_excluded"][alias]
            checks.extend(
                [
                    (
                        f"gpu.{alias}.startup_excluded.samples",
                        computed_startup["samples"],
                        float(expected_startup["samples"]),
                        0.0,
                    ),
                    (
                        f"gpu.{alias}.startup_excluded.zero_active_share",
                        computed_startup["zero_active_share"],
                        float(expected_startup["zero_active_share"]),
                        1e-9,
                    ),
                    (
                        f"gpu.{alias}.startup_excluded.mean_active_gpus",
                        computed_startup["mean_active_gpus"],
                        float(expected_startup["mean_active_gpus"]),
                        1e-9,
                    ),
                ]
            )

        sensitivity_expected = expected["threshold_sensitivity"]
        sensitivity_computed = gpu_activity["threshold_sensitivity"]
        for alias in ("sync", "streamweave"):
            zero_values = [
                item[f"{alias}_zero_active_share"]
                for item in sensitivity_computed
            ]
            mean_values = [
                item[f"{alias}_mean_active_gpus"]
                for item in sensitivity_computed
            ]
            for label, computed_range, expected_range in (
                (
                    "zero_active_share_range",
                    [min(zero_values), max(zero_values)],
                    sensitivity_expected[f"{alias}_zero_active_share_range"],
                ),
                (
                    "mean_active_gpus_range",
                    [min(mean_values), max(mean_values)],
                    sensitivity_expected[f"{alias}_mean_active_gpus_range"],
                ),
            ):
                if len(computed_range) != len(expected_range):
                    raise SystemExit(
                        f"gpu.{alias}.{label} length drift: "
                        f"computed={len(computed_range)}, "
                        f"verified={len(expected_range)}"
                    )
                for index, (computed_value, expected_value) in enumerate(
                    zip(computed_range, expected_range)
                ):
                    checks.append(
                        (
                            f"gpu.{alias}.{label}[{index}]",
                            computed_value,
                            float(expected_value),
                            1e-9,
                        )
                    )

    if "estimated_energy_per_prompt_group" in verified_gpu:
        expected = verified_gpu["estimated_energy_per_prompt_group"]
        for alias in ("sync", "streamweave"):
            computed = gpu_energy["runs"][alias]
            expected_run = expected[alias]
            checks.extend(
                [
                    (
                        f"gpu_energy.{alias}.cycles",
                        computed["cycles"],
                        float(expected_run["cycles"]),
                        0.0,
                    ),
                    (
                        f"gpu_energy.{alias}.telemetry_samples",
                        computed["telemetry_samples"],
                        float(expected_run["telemetry_samples"]),
                        0.0,
                    ),
                    (
                        f"gpu_energy.{alias}.cycle_weighted",
                        computed[
                            "cycle_weighted_energy_per_group_joules"
                        ],
                        float(
                            expected_run[
                                "cycle_weighted_energy_per_group_joules"
                            ]
                        ),
                        1e-6,
                    ),
                ]
            )
        expected_range = expected["relative_reduction_range"]
        for index, (computed_value, expected_value) in enumerate(
            zip(
                gpu_energy["reduction_range"],
                expected_range,
                strict=True,
            )
        ):
            checks.append(
                (
                    f"gpu_energy.reduction_range[{index}]",
                    computed_value,
                    float(expected_value),
                    1e-9,
                )
            )

    drift = [
        f"{name}: computed={computed_value}, verified={verified_value}"
        for name, computed_value, verified_value, tolerance in checks
        if not math.isclose(
            computed_value, verified_value, rel_tol=0.0, abs_tol=tolerance
        )
    ]
    if drift:
        raise SystemExit(
            "Computed history differs from verified_snapshot.json:\n"
            + "\n".join(drift)
            + "\nRe-run with --allow-drift only after reviewing the run lineage."
        )


def work_slice(
    rows: list[dict[str, float]], start_group: float, budget_groups: float
) -> dict[str, float]:
    end_group = start_group + budget_groups
    cursor = 0.0
    selected_groups = 0.0
    selected_time = 0.0

    for row in rows:
        row_start = cursor
        row_end = cursor + row["groups"]
        overlap = max(0.0, min(row_end, end_group) - max(row_start, start_group))
        if overlap:
            selected_groups += overlap
            selected_time += row["training_time_s"] * overlap / row["groups"]
        cursor = row_end
        if cursor >= end_group:
            break

    if not math.isclose(selected_groups, budget_groups, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Requested {budget_groups} groups at offset {start_group}, "
            f"but only selected {selected_groups}."
        )
    return {
        "groups": selected_groups,
        "time_s": selected_time,
        "throughput_groups_per_s": selected_groups / selected_time,
    }


def derive_assets(
    manifest: dict[str, Any],
    histories: dict[str, list[dict[str, float]]],
    system_histories: dict[str, list[dict[str, float]]],
    allow_drift: bool,
) -> None:
    aggregates = {
        alias: aggregate(alias, spec["run_id"], histories[alias])
        for alias, spec in manifest["runs"].items()
    }
    sync = aggregates["sync"]
    streamweave = aggregates["streamweave"]
    speedup = (
        streamweave["throughput_groups_per_s"] / sync["throughput_groups_per_s"]
    )
    gpu_activity = derive_gpu_activity(histories, system_histories)
    gpu_energy = derive_gpu_energy(histories, system_histories)
    validate_against_verified(
        sync,
        streamweave,
        gpu_activity,
        gpu_energy,
        allow_drift=allow_drift,
    )
    for item in aggregates.values():
        item["relative_throughput"] = (
            item["throughput_groups_per_s"] / sync["throughput_groups_per_s"]
        )

    budget = float(manifest["aggregation_rules"]["equal_work_budget_groups"])
    streamweave_total_groups = sum(row["groups"] for row in histories["streamweave"])
    equal_work = {
        "sync": work_slice(histories["sync"], 0.0, budget),
        "streamweave_first": work_slice(histories["streamweave"], 0.0, budget),
        "streamweave_last": work_slice(
            histories["streamweave"], streamweave_total_groups - budget, budget
        ),
    }
    sync_equal_throughput = equal_work["sync"]["throughput_groups_per_s"]
    for value in equal_work.values():
        value["relative_to_sync"] = (
            value["throughput_groups_per_s"] / sync_equal_throughput
        )

    full_blocks = int(streamweave_total_groups // budget)
    blocks = []
    for index in range(full_blocks):
        block = work_slice(histories["streamweave"], index * budget, budget)
        block.update(
            {
                "run_alias": "streamweave",
                "block_index": index + 1,
                "block_size_groups": int(budget),
                "relative_to_sync": (
                    block["throughput_groups_per_s"] / sync_equal_throughput
                ),
            }
        )
        blocks.append(block)

    sync_generation_time = sum(
        row.get("generation_or_interface_time_s", 0.0)
        for row in histories["sync"]
    )
    sync_generation_share = sync_generation_time / sync["training_time_s"]
    sync_generation_per_128 = 128.0 * sync_generation_time / sync["groups"]

    computed = {
        "schema_version": 1,
        "manifest": "manifest.json",
        "headline": {
            "sync": sync,
            "streamweave": streamweave,
            "throughput_ratio": speedup,
        },
        "critical_path": {
            "sync_generation_time_s": sync_generation_time,
            "sync_generation_share": sync_generation_share,
            "sync_generation_seconds_per_128_groups": sync_generation_per_128,
            "streamweave_total_seconds_per_128_groups": streamweave[
                "seconds_per_128_groups"
            ],
            "difference_from_sync_generation_only_s": (
                streamweave["seconds_per_128_groups"] - sync_generation_per_128
            ),
        },
        "equal_work": equal_work,
        "block_stability": blocks,
        "gpu_activity": gpu_activity,
        "gpu_energy": {
            key: value
            for key, value in gpu_energy.items()
            if key != "cycle_rows"
        },
    }
    write_json(DATA_DIR / "computed_snapshot.json", computed)

    aggregate_rows = []
    for alias in ("sync", "streamweave"):
        item = aggregates[alias]
        aggregate_rows.append(
            {
                "run_alias": alias,
                "run_id": item["run_id"],
                "cycles": item["cycles"],
                "groups": f'{item["groups"]:.0f}',
                "training_time_s": f'{item["training_time_s"]:.6f}',
                "throughput_groups_per_s": f'{item["throughput_groups_per_s"]:.6f}',
                "seconds_per_128_groups": f'{item["seconds_per_128_groups"]:.6f}',
                "relative_throughput": f'{item["relative_throughput"]:.6f}',
            }
        )
    write_csv(
        DATA_DIR / "run_aggregates.csv",
        [
            "run_alias",
            "run_id",
            "cycles",
            "groups",
            "training_time_s",
            "throughput_groups_per_s",
            "seconds_per_128_groups",
            "relative_throughput",
        ],
        aggregate_rows,
    )

    equal_rows = []
    for label, item in equal_work.items():
        alias, window = (
            ("sync", "full")
            if label == "sync"
            else ("streamweave", label.removeprefix("streamweave_"))
        )
        equal_rows.append(
            {
                "run_alias": alias,
                "window": window,
                "budget_groups": int(budget),
                "time_s": f'{item["time_s"]:.6f}',
                "throughput_groups_per_s": f'{item["throughput_groups_per_s"]:.6f}',
                "relative_to_sync": f'{item["relative_to_sync"]:.6f}',
                "boundary_allocation": (
                    "exact" if label == "sync" else "linear_in_group_fraction"
                ),
            }
        )
    write_csv(
        DATA_DIR / "equal_work_windows.csv",
        [
            "run_alias",
            "window",
            "budget_groups",
            "time_s",
            "throughput_groups_per_s",
            "relative_to_sync",
            "boundary_allocation",
        ],
        equal_rows,
    )

    write_csv(
        DATA_DIR / "block_throughput.csv",
        [
            "run_alias",
            "block_index",
            "block_size_groups",
            "throughput_groups_per_s",
            "relative_to_sync",
        ],
        [
            {
                "run_alias": item["run_alias"],
                "block_index": item["block_index"],
                "block_size_groups": item["block_size_groups"],
                "throughput_groups_per_s": f'{item["throughput_groups_per_s"]:.6f}',
                "relative_to_sync": f'{item["relative_to_sync"]:.6f}',
            }
            for item in blocks
        ],
    )

    figure_gpu = gpu_activity["runs"]
    distribution_rows = []
    for active_count in range(GPU_COUNT + 1):
        sync_bin = figure_gpu["sync"]["histogram"][active_count]
        streamweave_bin = figure_gpu["streamweave"]["histogram"][active_count]
        distribution_rows.append(
            {
                "active_gpu_count": active_count,
                "sync_samples": sync_bin["samples"],
                "sync_share_percent": f'{sync_bin["share"] * 100:.6f}',
                "streamweave_samples": streamweave_bin["samples"],
                "streamweave_share_percent": (
                    f'{streamweave_bin["share"] * 100:.6f}'
                ),
            }
        )
    write_csv(
        DATA_DIR / "gpu_activity_distribution.csv",
        [
            "active_gpu_count",
            "sync_samples",
            "sync_share_percent",
            "streamweave_samples",
            "streamweave_share_percent",
        ],
        distribution_rows,
    )

    write_csv(
        DATA_DIR / "gpu_activity_threshold_sensitivity.csv",
        [
            "threshold_percent",
            "sync_zero_active_share_percent",
            "streamweave_zero_active_share_percent",
            "sync_mean_active_gpus",
            "streamweave_mean_active_gpus",
        ],
        [
            {
                "threshold_percent": item["threshold_percent"],
                "sync_zero_active_share_percent": (
                    f'{item["sync_zero_active_share"] * 100:.6f}'
                ),
                "streamweave_zero_active_share_percent": (
                    f'{item["streamweave_zero_active_share"] * 100:.6f}'
                ),
                "sync_mean_active_gpus": f'{item["sync_mean_active_gpus"]:.6f}',
                "streamweave_mean_active_gpus": (
                    f'{item["streamweave_mean_active_gpus"]:.6f}'
                ),
            }
            for item in gpu_activity["threshold_sensitivity"]
        ],
    )

    write_csv(
        DATA_DIR / "gpu_activity_population_sensitivity.csv",
        [
            "population",
            "sync_samples",
            "streamweave_samples",
            "sync_zero_active_share_percent",
            "streamweave_zero_active_share_percent",
            "sync_mean_active_gpus",
            "streamweave_mean_active_gpus",
        ],
        [
            {
                "population": "all_non_validation_training_intervals",
                "sync_samples": figure_gpu["sync"]["samples"],
                "streamweave_samples": figure_gpu["streamweave"]["samples"],
                "sync_zero_active_share_percent": (
                    f'{figure_gpu["sync"]["zero_active_share"] * 100:.6f}'
                ),
                "streamweave_zero_active_share_percent": (
                    f'{figure_gpu["streamweave"]["zero_active_share"] * 100:.6f}'
                ),
                "sync_mean_active_gpus": (
                    f'{figure_gpu["sync"]["mean_active_gpus"]:.6f}'
                ),
                "streamweave_mean_active_gpus": (
                    f'{figure_gpu["streamweave"]["mean_active_gpus"]:.6f}'
                ),
            },
            {
                "population": "first_training_cycle_excluded",
                "sync_samples": gpu_activity["startup_excluded"]["sync"][
                    "samples"
                ],
                "streamweave_samples": gpu_activity["startup_excluded"][
                    "streamweave"
                ]["samples"],
                "sync_zero_active_share_percent": (
                    f'{gpu_activity["startup_excluded"]["sync"]["zero_active_share"] * 100:.6f}'
                ),
                "streamweave_zero_active_share_percent": (
                    f'{gpu_activity["startup_excluded"]["streamweave"]["zero_active_share"] * 100:.6f}'
                ),
                "sync_mean_active_gpus": (
                    f'{gpu_activity["startup_excluded"]["sync"]["mean_active_gpus"]:.6f}'
                ),
                "streamweave_mean_active_gpus": (
                    f'{gpu_activity["startup_excluded"]["streamweave"]["mean_active_gpus"]:.6f}'
                ),
            },
            {
                "population": "equal_non_validation_work_first",
                "sync_samples": gpu_activity["equal_non_validation_work"][
                    "sync_full"
                ]["samples"],
                "streamweave_samples": gpu_activity[
                    "equal_non_validation_work"
                ]["streamweave_first"]["samples"],
                "sync_zero_active_share_percent": (
                    f'{gpu_activity["equal_non_validation_work"]["sync_full"]["zero_active_share"] * 100:.6f}'
                ),
                "streamweave_zero_active_share_percent": (
                    f'{gpu_activity["equal_non_validation_work"]["streamweave_first"]["zero_active_share"] * 100:.6f}'
                ),
                "sync_mean_active_gpus": (
                    f'{gpu_activity["equal_non_validation_work"]["sync_full"]["mean_active_gpus"]:.6f}'
                ),
                "streamweave_mean_active_gpus": (
                    f'{gpu_activity["equal_non_validation_work"]["streamweave_first"]["mean_active_gpus"]:.6f}'
                ),
            },
            {
                "population": "equal_non_validation_work_last",
                "sync_samples": gpu_activity["equal_non_validation_work"][
                    "sync_full"
                ]["samples"],
                "streamweave_samples": gpu_activity[
                    "equal_non_validation_work"
                ]["streamweave_last"]["samples"],
                "sync_zero_active_share_percent": (
                    f'{gpu_activity["equal_non_validation_work"]["sync_full"]["zero_active_share"] * 100:.6f}'
                ),
                "streamweave_zero_active_share_percent": (
                    f'{gpu_activity["equal_non_validation_work"]["streamweave_last"]["zero_active_share"] * 100:.6f}'
                ),
                "sync_mean_active_gpus": (
                    f'{gpu_activity["equal_non_validation_work"]["sync_full"]["mean_active_gpus"]:.6f}'
                ),
                "streamweave_mean_active_gpus": (
                    f'{gpu_activity["equal_non_validation_work"]["streamweave_last"]["mean_active_gpus"]:.6f}'
                ),
            },
        ],
    )

    write_csv(
        DATA_DIR / "gpu_energy_summary.csv",
        [
            "run_alias",
            "cycles",
            "telemetry_samples",
            "groups",
            "training_time_s",
            "pooled_mean_power_watts",
            "pooled_energy_per_group_joules",
            "cycle_weighted_energy_per_group_joules",
            "edge_trimmed_energy_per_group_joules",
            "startup_excluded_energy_per_group_joules",
            "groups_per_kwh_cycle_weighted",
        ],
        [
            {
                "run_alias": alias,
                **{
                    key: (
                        f"{value:.6f}"
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in gpu_energy["runs"][alias].items()
                },
            }
            for alias in ("sync", "streamweave")
        ],
    )

    energy_cycle_rows = []
    for alias in ("sync", "streamweave"):
        for point in gpu_energy["cycle_rows"][alias]:
            energy_cycle_rows.append(
                {
                    "run_alias": alias,
                    "history_index": point["history_index"],
                    "groups": f'{point["groups"]:.0f}',
                    "training_time_s": f'{point["duration_s"]:.6f}',
                    "power_samples": point["power_samples"],
                    "mean_power_watts": (
                        f'{point["mean_power_watts"]:.6f}'
                    ),
                    "throughput_groups_per_s": (
                        f'{point["throughput_groups_per_s"]:.6f}'
                    ),
                    "energy_per_group_joules": (
                        f'{point["energy_per_group_joules"]:.6f}'
                    ),
                    "trimmed_energy_per_group_joules": (
                        f'{point["trimmed_energy_per_group_joules"]:.6f}'
                    ),
                }
            )
    write_csv(
        DATA_DIR / "gpu_energy_cycle_points.csv",
        [
            "run_alias",
            "history_index",
            "groups",
            "training_time_s",
            "power_samples",
            "mean_power_watts",
            "throughput_groups_per_s",
            "energy_per_group_joules",
            "trimmed_energy_per_group_joules",
        ],
        energy_cycle_rows,
    )

    write_csv(
        DATA_DIR / "gpu_energy_sensitivity.csv",
        [
            "estimator",
            "sync_energy_per_group_joules",
            "streamweave_energy_per_group_joules",
            "relative_reduction_percent",
        ],
        [
            {
                "estimator": estimator.removesuffix(
                    "_energy_per_group_joules"
                ),
                "sync_energy_per_group_joules": (
                    f'{gpu_energy["runs"]["sync"][estimator]:.6f}'
                ),
                "streamweave_energy_per_group_joules": (
                    f'{gpu_energy["runs"]["streamweave"][estimator]:.6f}'
                ),
                "relative_reduction_percent": (
                    f'{gpu_energy["relative_reduction"][estimator] * 100:.6f}'
                ),
            }
            for estimator in (
                "pooled_energy_per_group_joules",
                "cycle_weighted_energy_per_group_joules",
                "edge_trimmed_energy_per_group_joules",
                "startup_excluded_energy_per_group_joules",
            )
        ]
        + [
            {
                "estimator": "validation_shifted_full_coverage",
                "sync_energy_per_group_joules": (
                    f'{gpu_energy["validation_shifted_full_coverage"]["sync"]["validation_shifted_energy_per_group_joules"]:.6f}'
                ),
                "streamweave_energy_per_group_joules": (
                    f'{gpu_energy["validation_shifted_full_coverage"]["streamweave"]["validation_shifted_energy_per_group_joules"]:.6f}'
                ),
                "relative_reduction_percent": (
                    f'{gpu_energy["validation_shifted_full_coverage_reduction"] * 100:.6f}'
                ),
            }
        ],
    )

    write_csv(
        DATA_DIR / "gpu_energy_equal_work_sensitivity.csv",
        [
            "run_alias",
            "window",
            "budget_groups",
            "energy_per_group_joules",
            "relative_reduction_vs_sync_percent",
            "boundary_allocation",
        ],
        [
            {
                "run_alias": "sync",
                "window": "full_non_validation",
                "budget_groups": int(
                    gpu_energy["equal_non_validation_work"]["budget_groups"]
                ),
                "energy_per_group_joules": (
                    f'{gpu_energy["equal_non_validation_work"]["sync_full"]["energy_per_group_joules"]:.6f}'
                ),
                "relative_reduction_vs_sync_percent": "0.000000",
                "boundary_allocation": "exact",
            },
            *[
                {
                    "run_alias": "streamweave",
                    "window": window,
                    "budget_groups": int(
                        gpu_energy["equal_non_validation_work"][
                            "budget_groups"
                        ]
                    ),
                    "energy_per_group_joules": (
                        f'{gpu_energy["equal_non_validation_work"][label]["energy_per_group_joules"]:.6f}'
                    ),
                    "relative_reduction_vs_sync_percent": (
                        f'{gpu_energy["equal_non_validation_work"][label]["relative_reduction"] * 100:.6f}'
                    ),
                    "boundary_allocation": "linear_in_group_fraction",
                }
                for label, window in (
                    ("streamweave_first", "first"),
                    ("streamweave_last", "last"),
                )
            ],
        ],
    )

    write_csv(
        DATA_DIR / "execution_summary_table.csv",
        [
            "system",
            "seconds_per_128_groups",
            "throughput_groups_per_s",
            "relative_throughput",
        ],
        [
            {
                "system": "Synchronous",
                "seconds_per_128_groups": f'{sync["seconds_per_128_groups"]:.6f}',
                "throughput_groups_per_s": (
                    f'{sync["throughput_groups_per_s"]:.6f}'
                ),
                "relative_throughput": "1.000000",
            },
            {
                "system": "StreamWeave",
                "seconds_per_128_groups": (
                    f'{streamweave["seconds_per_128_groups"]:.6f}'
                ),
                "throughput_groups_per_s": (
                    f'{streamweave["throughput_groups_per_s"]:.6f}'
                ),
                "relative_throughput": f"{speedup:.6f}",
            },
        ],
    )

    figure_input = {
        "figure_id": "figure3_execution_efficiency",
        "status": "locked_main_full_history_and_system_telemetry",
        "source_snapshot": "execution_efficiency/computed_snapshot.json",
        "work_unit": "prompt_group",
        "equivalent_groups": 128,
        "runs": {
            "sync": manifest["runs"]["sync"]["run_id"],
            "streamweave": manifest["runs"]["streamweave"]["run_id"],
        },
        "gpu_activity": {
            "threshold_percent": FIGURE_GPU_ACTIVITY_THRESHOLD,
            "sample_interval_seconds": gpu_activity["sample_interval_seconds"],
            "sync_samples": figure_gpu["sync"]["samples"],
            "streamweave_samples": figure_gpu["streamweave"]["samples"],
            "distribution": [
                {
                    "active_gpu_count": row["active_gpu_count"],
                    "sync_share_percent": float(row["sync_share_percent"]),
                    "streamweave_share_percent": float(
                        row["streamweave_share_percent"]
                    ),
                }
                for row in distribution_rows
            ],
            "zero_active_share_percent": {
                "sync": figure_gpu["sync"]["zero_active_share"] * 100,
                "streamweave": (
                    figure_gpu["streamweave"]["zero_active_share"] * 100
                ),
            },
            "mean_active_gpus": {
                "sync": figure_gpu["sync"]["mean_active_gpus"],
                "streamweave": figure_gpu["streamweave"]["mean_active_gpus"],
            },
        },
        "gpu_energy": {
            "sample_interval_seconds": gpu_energy[
                "sample_interval_seconds"
            ],
            "population": gpu_energy["population"],
            "estimator": gpu_energy["estimator"],
            "sync_energy_per_group_joules": gpu_energy["runs"]["sync"][
                "cycle_weighted_energy_per_group_joules"
            ],
            "streamweave_energy_per_group_joules": gpu_energy["runs"][
                "streamweave"
            ]["cycle_weighted_energy_per_group_joules"],
            "relative_reduction_percent": gpu_energy[
                "relative_reduction"
            ]["cycle_weighted_energy_per_group_joules"]
            * 100,
            "sensitivity_reduction_percent": [
                value * 100 for value in gpu_energy["reduction_range"]
            ],
            "ecdf": {
                "x_cap_kilojoules": 2.5,
                "sync": work_weighted_energy_ecdf(
                    gpu_energy["cycle_rows"]["sync"],
                    cap_kilojoules=2.5,
                ),
                "streamweave": work_weighted_energy_ecdf(
                    gpu_energy["cycle_rows"]["streamweave"],
                    cap_kilojoules=2.5,
                ),
            },
            "cycle_points_csv": "execution_efficiency/gpu_energy_cycle_points.csv",
        },
        "execution_summary": {
            "sync": {
                "seconds_per_128_groups": sync["seconds_per_128_groups"],
                "throughput_groups_per_s": sync["throughput_groups_per_s"],
                "relative_throughput": 1.0,
            },
            "streamweave": {
                "seconds_per_128_groups": streamweave[
                    "seconds_per_128_groups"
                ],
                "throughput_groups_per_s": streamweave[
                    "throughput_groups_per_s"
                ],
                "relative_throughput": speedup,
            },
        },
        "restrictions": [
            "GPU bins count telemetry intervals, not independent training trials.",
            "Zero active means no GPU exceeded the stated SM threshold; do not relabel it as idle or stall.",
            "The distribution is mechanism evidence, not a causal decomposition of the 1.64x result.",
            "Do not add stall, tail-wait, MFU, or time-to-quality headlines.",
            "Exact accelerator model and trainer/rollouter topology remain Appendix-only.",
            "GPU energy is a sample-based eight-GPU estimate, not node-total energy or an external power-meter reading.",
            "Prompt-group normalization is not token-normalized energy.",
        ],
    }
    write_json(FIGURE_INPUT_PATH, figure_input)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch full histories from W&B before deriving local assets.",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Write derived assets even if headline totals differ from the verified snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(MANIFEST_PATH)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_histories: dict[str, list[dict[str, Any]]] = {}
    raw_system_histories: dict[str, list[dict[str, Any]]] = {}
    for alias in manifest["runs"]:
        if args.fetch:
            rows = fetch_run_history(manifest, alias)
            write_csv(RAW_DIR / f"{alias}_history.csv", RAW_COLUMNS, rows)
            raw_histories[alias] = rows
            system_rows = fetch_run_system_history(manifest, alias)
            write_csv(
                RAW_DIR / f"{alias}_system.csv",
                SYSTEM_RAW_COLUMNS,
                system_rows,
            )
            raw_system_histories[alias] = system_rows
        else:
            raw_histories[alias] = read_raw_history(alias)
            raw_system_histories[alias] = read_raw_system_history(alias)

    histories = {
        alias: numeric_rows(rows) for alias, rows in raw_histories.items()
    }
    system_histories = {
        alias: numeric_system_rows(rows)
        for alias, rows in raw_system_histories.items()
    }
    derive_assets(
        manifest,
        histories,
        system_histories,
        allow_drift=args.allow_drift,
    )


if __name__ == "__main__":
    main()
