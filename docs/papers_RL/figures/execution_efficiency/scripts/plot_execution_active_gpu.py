#!/usr/bin/env python3
"""Render the active-GPU distribution from frozen execution telemetry."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
SLATE = "#8A98AA"
SLATE_LIGHT = "#F1F4F7"
TEAL = "#0A8B7B"
TEAL_LIGHT = "#DDF1ED"
WHITE = "#FFFFFF"
THRESHOLD = 20.0
GPU_COUNT = 8


def finite(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def training_intervals(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    result = []
    for row in rows:
        if row.get("validation_time_s") not in (None, ""):
            continue
        end = finite(row.get("_timestamp"))
        duration = finite(row.get("training_time_s"))
        if end is None or duration is None:
            continue
        result.append((end - duration, end))
    return result


def selected_system_rows(
    rows: list[dict[str, str]],
    intervals: list[tuple[float, float]],
) -> list[list[float]]:
    selected = []
    interval_index = 0
    ordered_intervals = sorted(intervals)
    for row in sorted(rows, key=lambda item: float(item["_timestamp"])):
        timestamp = finite(row.get("_timestamp"))
        values = [
            finite(row.get(f"gpu_{index}_sm_active"))
            for index in range(GPU_COUNT)
        ]
        if timestamp is None or any(value is None for value in values):
            continue
        while (
            interval_index < len(ordered_intervals)
            and timestamp > ordered_intervals[interval_index][1]
        ):
            interval_index += 1
        if interval_index >= len(ordered_intervals):
            break
        start, end = ordered_intervals[interval_index]
        if start <= timestamp <= end:
            selected.append([float(value) for value in values])
    return selected


def active_gpu_distribution(rows: list[list[float]]) -> np.ndarray:
    counts = [
        sum(value > THRESHOLD for value in row)
        for row in rows
    ]
    return np.asarray(
        [100.0 * counts.count(index) / len(counts) for index in range(9)]
    )


def render(
    sync_rows: list[list[float]],
    streamweave_rows: list[list[float]],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    sync_distribution = active_gpu_distribution(sync_rows)
    streamweave_distribution = active_gpu_distribution(streamweave_rows)

    count_figure, count_axis = plt.subplots(
        1,
        1,
        figsize=(4.15, 3.2),
        facecolor=WHITE,
    )

    positions = np.arange(9)
    width = 0.36
    count_axis.bar(
        positions - width / 2,
        sync_distribution,
        width=width,
        color=SLATE_LIGHT,
        edgecolor=SLATE,
        linewidth=1.0,
        label="Synchronous",
        zorder=3,
    )
    count_axis.bar(
        positions + width / 2,
        streamweave_distribution,
        width=width,
        color=TEAL_LIGHT,
        edgecolor=TEAL,
        linewidth=1.0,
        label="StreamWeave",
        zorder=3,
    )
    count_axis.set_xlim(-0.55, 8.55)
    count_axis.set_ylim(0, 75)
    count_axis.set_xticks(positions)
    count_axis.set_yticks((0, 20, 40, 60))
    count_axis.set_xlabel("GPUs above 20% SM activity")
    count_axis.set_ylabel("Share of training intervals (%)")
    count_axis.set_title(
        "Active-GPU count distribution",
        loc="left",
        color=INK,
        fontsize=9.3,
        fontweight="bold",
        pad=8,
    )
    count_axis.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
    count_axis.spines["top"].set_visible(False)
    count_axis.spines["right"].set_visible(False)
    count_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
        fontsize=7.5,
    )
    count_axis.annotate(
        "27.9% -> 4.7%",
        xy=(0.18, streamweave_distribution[0] + 0.5),
        xytext=(0.9, 45),
        color=INK,
        fontsize=8.0,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED,
            "linewidth": 0.9,
        },
    )
    count_figure.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.86)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            count_figure,
            output_dir / "execution_activity_active_gpu",
            "StreamWeave active-GPU count distribution",
            "Distribution of GPUs above the SM-activity threshold.",
        ),
    )
    for figure, path, title, description in outputs:
        figure.savefig(
            path.with_suffix(".svg"),
            facecolor=WHITE,
            metadata={"Title": title, "Description": description},
        )
        figure.savefig(
            path.with_suffix(".pdf"),
            facecolor=WHITE,
            bbox_inches="tight",
            metadata={"Title": title, "Subject": description},
        )
        figure.savefig(
            path.with_suffix(".png"),
            facecolor=WHITE,
            bbox_inches="tight",
            dpi=240,
            metadata={"Title": title},
        )
        plt.close(figure)


def main() -> None:
    bundle_dir = Path(__file__).resolve().parent.parent
    data_dir = bundle_dir / "data"
    raw_dir = data_dir / "raw"

    sync_rows = selected_system_rows(
        read_csv(raw_dir / "sync_system.csv"),
        training_intervals(read_csv(raw_dir / "sync_history.csv")),
    )
    streamweave_rows = selected_system_rows(
        read_csv(raw_dir / "streamweave_system.csv"),
        training_intervals(read_csv(raw_dir / "streamweave_history.csv")),
    )
    if len(sync_rows) != 287 or len(streamweave_rows) != 974:
        raise ValueError(
            "Frozen telemetry population changed: "
            f"sync={len(sync_rows)}, StreamWeave={len(streamweave_rows)}"
        )
    render(
        sync_rows,
        streamweave_rows,
        bundle_dir / "outputs",
    )


if __name__ == "__main__":
    main()
