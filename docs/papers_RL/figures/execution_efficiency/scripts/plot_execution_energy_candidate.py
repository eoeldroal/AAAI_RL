#!/usr/bin/env python3
"""Render a work-weighted GPU-energy candidate from local W&B binaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
SLATE = "#7F8D9F"
SLATE_LIGHT = "#EDF1F5"
TEAL = "#0A8B7B"
TEAL_LIGHT = "#DDF1ED"
WHITE = "#FFFFFF"
GPU_COUNT = 8


def scan_wandb(path: Path) -> tuple[list[dict], list[dict]]:
    store = datastore.DataStore()
    store.open_for_scan(str(path))
    history_rows = []
    power_rows = []

    while True:
        try:
            data = store.scan_data()
        except AssertionError:
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        record_type = record.WhichOneof("record_type")

        if record_type == "history":
            row = {}
            for item in record.history.item:
                key = (
                    "/".join(item.nested_key)
                    if item.nested_key
                    else item.key
                )
                try:
                    row[key] = json.loads(item.value_json)
                except json.JSONDecodeError:
                    continue
            history_rows.append(row)
        elif record_type == "stats":
            row = {
                "_timestamp": (
                    record.stats.timestamp.seconds
                    + record.stats.timestamp.nanos / 1e9
                )
            }
            for item in record.stats.item:
                try:
                    row[item.key] = json.loads(item.value_json)
                except json.JSONDecodeError:
                    continue
            keys = [f"gpu.{index}.powerWatts" for index in range(GPU_COUNT)]
            if all(key in row for key in keys):
                row["total_gpu_power_w"] = sum(float(row[key]) for key in keys)
                power_rows.append(row)

    return history_rows, power_rows


def cycle_energy_rows(
    history_rows: list[dict],
    power_rows: list[dict],
) -> list[dict[str, float]]:
    result = []
    for row in history_rows:
        if (
            "timing_s/step" not in row
            or "hpt/onpolicy_num_groups" not in row
        ):
            continue
        if "timing_s/testing" in row or "rollouter/validate_time" in row:
            continue
        end = float(row["_timestamp"])
        duration = float(row["timing_s/step"])
        groups = float(row["hpt/onpolicy_num_groups"])
        samples = [
            float(power_row["total_gpu_power_w"])
            for power_row in power_rows
            if end - duration <= power_row["_timestamp"] <= end
        ]
        if not samples:
            continue
        mean_power = float(np.mean(samples))
        result.append(
            {
                "groups": groups,
                "duration_s": duration,
                "mean_power_w": mean_power,
                "energy_per_group_j": mean_power * duration / groups,
                "sample_count": float(len(samples)),
            }
        )
    return result


def weighted_ecdf(
    rows: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([row["energy_per_group_j"] for row in rows])
    weights = np.asarray([row["groups"] for row in rows])
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return sorted_values / 1000.0, cumulative * 100.0


def aggregate_energy(rows: list[dict[str, float]]) -> dict[str, float]:
    groups = sum(row["groups"] for row in rows)
    duration = sum(row["duration_s"] for row in rows)
    energy = sum(
        row["mean_power_w"] * row["duration_s"]
        for row in rows
    )
    return {
        "groups": groups,
        "duration_s": duration,
        "mean_power_w": energy / duration,
        "energy_per_group_j": energy / groups,
        "throughput": groups / duration,
    }


def render(
    sync_rows: list[dict[str, float]],
    streamweave_rows: list[dict[str, float]],
    output_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    sync_x, sync_y = weighted_ecdf(sync_rows)
    streamweave_x, streamweave_y = weighted_ecdf(streamweave_rows)
    sync = aggregate_energy(sync_rows)
    streamweave = aggregate_energy(streamweave_rows)

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(4.15, 3.15),
        facecolor=WHITE,
    )
    axis.fill_betweenx(
        sync_y,
        sync_x,
        2.5,
        color=SLATE_LIGHT,
        alpha=0.75,
        linewidth=0,
    )
    axis.fill_betweenx(
        streamweave_y,
        streamweave_x,
        0,
        color=TEAL_LIGHT,
        alpha=0.8,
        linewidth=0,
    )
    axis.step(
        sync_x,
        sync_y,
        where="post",
        color=SLATE,
        linewidth=2.0,
        linestyle=(0, (5, 3)),
        label="Synchronous",
    )
    axis.step(
        streamweave_x,
        streamweave_y,
        where="post",
        color=TEAL,
        linewidth=2.4,
        label="StreamWeave",
    )
    axis.axvline(
        sync["energy_per_group_j"] / 1000.0,
        color=SLATE,
        linewidth=1.0,
        linestyle=(0, (2, 2)),
    )
    axis.axvline(
        streamweave["energy_per_group_j"] / 1000.0,
        color=TEAL,
        linewidth=1.0,
        linestyle=(0, (2, 2)),
    )
    axis.set_xlim(0.5, 2.45)
    axis.set_ylim(0, 100)
    axis.set_xticks((0.5, 1.0, 1.5, 2.0))
    axis.set_yticks((0, 20, 40, 60, 80, 100))
    axis.set_xlabel("Estimated GPU energy per prompt group (kJ)")
    axis.set_ylabel("Share of processed prompt groups (%)")
    axis.set_title(
        "Work-weighted energy distribution",
        loc="left",
        color=INK,
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    axis.grid(axis="both", color=GRID, linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.6,
    )
    axis.text(
        streamweave["energy_per_group_j"] / 1000.0 - 0.03,
        13,
        f"{streamweave['energy_per_group_j'] / 1000.0:.2f} kJ",
        color=TEAL,
        ha="right",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    axis.text(
        sync["energy_per_group_j"] / 1000.0 + 0.03,
        13,
        f"{sync['energy_per_group_j'] / 1000.0:.2f} kJ",
        color=SLATE,
        ha="left",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    axis.text(
        0.97,
        0.93,
        "lower is better",
        transform=axis.transAxes,
        color=MUTED,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.88)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "StreamWeave GPU energy candidate",
        "Description": (
            "Work-weighted empirical distribution of cycle-level GPU energy "
            "per consumed prompt group."
        ),
    }
    figure.savefig(
        output_base.with_suffix(".svg"),
        facecolor=WHITE,
        metadata=metadata,
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        facecolor=WHITE,
        bbox_inches="tight",
        metadata={
            "Title": metadata["Title"],
            "Subject": metadata["Description"],
        },
    )
    figure.savefig(
        output_base.with_suffix(".png"),
        facecolor=WHITE,
        bbox_inches="tight",
        dpi=240,
        metadata={"Title": metadata["Title"]},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-wandb", type=Path, required=True)
    parser.add_argument("--streamweave-wandb", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    arguments = parser.parse_args()

    sync_history, sync_power = scan_wandb(arguments.sync_wandb)
    streamweave_history, streamweave_power = scan_wandb(
        arguments.streamweave_wandb
    )
    sync_rows = cycle_energy_rows(sync_history, sync_power)
    streamweave_rows = cycle_energy_rows(
        streamweave_history,
        streamweave_power,
    )
    if len(sync_rows) != 94 or len(streamweave_rows) != 152:
        raise ValueError(
            "Unexpected energy population: "
            f"sync={len(sync_rows)}, StreamWeave={len(streamweave_rows)}"
        )
    render(sync_rows, streamweave_rows, arguments.output_base)


if __name__ == "__main__":
    main()
