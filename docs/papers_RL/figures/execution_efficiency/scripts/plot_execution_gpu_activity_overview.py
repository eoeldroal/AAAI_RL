#!/usr/bin/env python3
"""Render full-history GPU activity and matched-wall-clock work."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_execution_active_gpu import (
    GPU_COUNT,
    THRESHOLD,
    read_csv,
    selected_system_rows,
    training_intervals,
)


INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
SLATE = "#8795A7"
TEAL = "#078B7B"
TEAL_DARK = "#076C61"
TEAL_LIGHT = "#D9F0EC"
WHITE = "#FFFFFF"


def active_gpu_counts(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(
        [sum(value > THRESHOLD for value in row) for row in rows],
        dtype=float,
    )


def trim_to_horizon(points: list[dict], horizon_s: float) -> list[dict]:
    result = []
    for index, point in enumerate(points):
        if point["cumulative_time_s"] <= horizon_s:
            result.append(point)
            continue
        previous = points[index - 1]
        span = point["cumulative_time_s"] - previous["cumulative_time_s"]
        fraction = (horizon_s - previous["cumulative_time_s"]) / span
        result.append(
            {
                "cumulative_time_s": horizon_s,
                "cumulative_groups": previous["cumulative_groups"]
                + fraction
                * (
                    point["cumulative_groups"]
                    - previous["cumulative_groups"]
                ),
            }
        )
        break
    return result


def render_heatmap(
    axis: plt.Axes,
    rows: list[list[float]],
    title: str,
    cmap: LinearSegmentedColormap,
    show_xaxis: bool,
) -> None:
    values = np.asarray(rows, dtype=float).T
    axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, 100, GPU_COUNT - 0.5, -0.5),
        cmap=cmap,
        vmin=0,
        vmax=100,
        rasterized=True,
    )
    axis.set_xlim(0, 100)
    axis.set_ylim(GPU_COUNT - 0.5, -0.5)
    axis.set_yticks(np.arange(GPU_COUNT))
    axis.set_yticklabels([f"G{index}" for index in range(GPU_COUNT)])
    axis.set_xticks((0, 20, 40, 60, 80, 100))
    axis.tick_params(
        axis="both",
        colors=INK,
        labelsize=5.8,
        length=2.0,
        width=0.65,
    )
    if show_xaxis:
        axis.set_xlabel(
            "Training progress (%)",
            color=INK,
            fontsize=6.7,
            labelpad=2,
        )
    else:
        axis.tick_params(axis="x", labelbottom=False)
    axis.set_title(
        title,
        loc="left",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
        pad=2.5,
    )
    for spine in axis.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)


def render(
    sync_rows: list[list[float]],
    streamweave_rows: list[list[float]],
    cumulative_work: dict,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    cmap = LinearSegmentedColormap.from_list(
        "streamweave_activity",
        [WHITE, "#E5F2F0", "#A7D8D1", "#3CA99B", TEAL_DARK],
    )
    figure = plt.figure(figsize=(7.0, 2.25), facecolor=WHITE)
    outer_grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.62, 0.88, 1.02),
        left=0.055,
        right=0.985,
        bottom=0.235,
        top=0.81,
        wspace=0.42,
    )
    heatmap_grid = outer_grid[0, 0].subgridspec(
        2,
        2,
        width_ratios=(1.0, 0.035),
        hspace=0.34,
        wspace=0.055,
    )
    sync_axis = figure.add_subplot(heatmap_grid[0, 0])
    streamweave_axis = figure.add_subplot(heatmap_grid[1, 0])
    color_axis = figure.add_subplot(heatmap_grid[:, 1])
    distribution_axis = figure.add_subplot(outer_grid[0, 1])
    work_axis = figure.add_subplot(outer_grid[0, 2])

    render_heatmap(
        sync_axis,
        sync_rows,
        "Synchronous",
        cmap,
        show_xaxis=False,
    )
    render_heatmap(
        streamweave_axis,
        streamweave_rows,
        "StreamWeave",
        cmap,
        show_xaxis=True,
    )
    streamweave_axis.axhline(1.5, color=INK, linewidth=0.9)
    streamweave_axis.text(
        99.0,
        0.5,
        "trainer",
        color=INK,
        ha="right",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.14",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )
    streamweave_axis.text(
        99.0,
        4.5,
        "rollouter",
        color=INK,
        ha="right",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.14",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )

    scalar_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0, vmax=100),
        cmap=cmap,
    )
    colorbar = figure.colorbar(scalar_mappable, cax=color_axis)
    colorbar.set_ticks((0, 20, 40, 60, 80, 100))
    colorbar.ax.tick_params(labelsize=5.4, colors=MUTED, length=2.0)
    colorbar.ax.set_title(
        "SM (%)",
        color=INK,
        fontsize=5.7,
        pad=3,
    )
    colorbar.outline.set_edgecolor(INK)
    colorbar.outline.set_linewidth(0.7)

    sync_counts = active_gpu_counts(sync_rows)
    streamweave_counts = active_gpu_counts(streamweave_rows)
    thresholds = np.arange(1, GPU_COUNT + 1)
    sync_coverage = np.asarray(
        [100.0 * np.mean(sync_counts >= value) for value in thresholds]
    )
    streamweave_coverage = np.asarray(
        [100.0 * np.mean(streamweave_counts >= value) for value in thresholds]
    )

    distribution_axis.fill_between(
        thresholds,
        sync_coverage,
        streamweave_coverage,
        color=TEAL_LIGHT,
        alpha=0.55,
        zorder=1,
    )
    distribution_axis.plot(
        thresholds,
        sync_coverage,
        color=SLATE,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        marker="o",
        markersize=3.7,
        markerfacecolor=WHITE,
        markeredgecolor=SLATE,
        markeredgewidth=1.0,
        label="Synchronous",
        zorder=3,
    )
    distribution_axis.plot(
        thresholds,
        streamweave_coverage,
        color=TEAL,
        linewidth=2.0,
        marker="o",
        markersize=3.7,
        markerfacecolor=TEAL,
        markeredgecolor=WHITE,
        markeredgewidth=0.7,
        label="StreamWeave",
        zorder=4,
    )
    distribution_axis.set_xlim(0.75, 8.25)
    distribution_axis.set_ylim(50, 100)
    distribution_axis.set_xticks(thresholds)
    distribution_axis.set_yticks((50, 60, 70, 80, 90, 100))
    distribution_axis.tick_params(
        axis="both",
        labelsize=6.1,
        length=2.5,
        width=0.7,
    )
    distribution_axis.set_xlabel(
        "At least k active GPUs",
        color=INK,
        fontsize=6.7,
        labelpad=4,
    )
    distribution_axis.set_ylabel(
        "Telemetry intervals (%)",
        color=INK,
        fontsize=6.7,
        labelpad=4,
    )
    distribution_axis.grid(
        axis="y",
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )
    distribution_axis.spines["top"].set_visible(False)
    distribution_axis.spines["right"].set_visible(False)
    distribution_axis.set_title(
        "(b) Concurrent GPU activity",
        loc="left",
        color=INK,
        fontsize=8.0,
        fontweight="bold",
        pad=5,
    )
    distribution_axis.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 0.01),
        ncol=1,
        frameon=False,
        fontsize=5.9,
        labelspacing=0.25,
        handlelength=1.5,
    )

    common_horizon_s = float(cumulative_work["common_horizon_s"])
    sync_work = trim_to_horizon(
        cumulative_work["sync"],
        common_horizon_s,
    )
    streamweave_work = trim_to_horizon(
        cumulative_work["streamweave"],
        common_horizon_s,
    )

    def work_series(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        wall_clock = np.asarray(
            [
                point["cumulative_time_s"] / 60.0
                for point in points
            ],
            dtype=float,
        )
        cumulative_groups = np.asarray(
            [point["cumulative_groups"] for point in points],
            dtype=float,
        )
        return wall_clock, cumulative_groups

    sync_wall_clock, sync_groups = work_series(sync_work)
    streamweave_wall_clock, streamweave_groups = work_series(
        streamweave_work
    )
    sync_reference = sync_groups[-1]
    if sync_reference <= 0:
        raise ValueError("Synchronous cumulative work must be positive")
    sync_groups = sync_groups / sync_reference
    streamweave_groups = streamweave_groups / sync_reference
    common_horizon_min = common_horizon_s / 60.0
    shared_grid = np.linspace(0.0, common_horizon_min, 401)
    sync_interpolated = np.interp(
        shared_grid,
        sync_wall_clock,
        sync_groups,
    )
    streamweave_interpolated = np.interp(
        shared_grid,
        streamweave_wall_clock,
        streamweave_groups,
    )
    work_axis.fill_between(
        shared_grid,
        sync_interpolated,
        streamweave_interpolated,
        color=TEAL_LIGHT,
        alpha=0.55,
        zorder=1,
    )
    work_axis.plot(
        sync_wall_clock,
        sync_groups,
        color=SLATE,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        zorder=3,
    )
    work_axis.plot(
        streamweave_wall_clock,
        streamweave_groups,
        color=TEAL,
        linewidth=2.3,
        zorder=4,
    )
    work_axis.scatter(
        [common_horizon_min, common_horizon_min],
        [sync_groups[-1], streamweave_groups[-1]],
        s=22,
        color=[SLATE, TEAL],
        edgecolors=WHITE,
        linewidths=0.7,
        zorder=5,
    )
    work_axis.set_xlim(0, common_horizon_min)
    work_axis.set_ylim(0, 1.82)
    work_axis.set_xticks((0, 20, 40, 60, 80))
    work_axis.set_yticks((0, 0.4, 0.8, 1.2, 1.6))
    work_axis.tick_params(
        axis="both",
        labelsize=6.1,
        length=2.5,
        width=0.7,
    )
    work_axis.set_xlabel(
        "Elapsed wall-clock (min)",
        color=INK,
        fontsize=6.7,
        labelpad=4,
    )
    work_axis.set_ylabel(
        "Relative cumulative work",
        color=INK,
        fontsize=6.7,
        labelpad=4,
    )
    work_axis.grid(
        axis="both",
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )
    work_axis.spines["top"].set_visible(False)
    work_axis.spines["right"].set_visible(False)
    work_axis.set_title(
        "(c) Cumulative work",
        loc="left",
        color=INK,
        fontsize=8.0,
        fontweight="bold",
        pad=5,
    )
    work_axis.text(
        common_horizon_min * 0.96,
        streamweave_groups[-1] + 0.055,
        f"SW  {streamweave_groups[-1]:.2f}x",
        color=TEAL,
        fontsize=6.2,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    work_axis.text(
        common_horizon_min * 0.96,
        sync_groups[-1] - 0.075,
        f"Sync  {sync_groups[-1]:.2f}x",
        color=MUTED,
        fontsize=6.2,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

    figure.suptitle(
        "(a) Full-history GPU activity",
        x=0.055,
        y=0.955,
        ha="left",
        color=INK,
        fontsize=8.0,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "execution_gpu_activity_overview"
    svg_path = output_base.with_suffix(".svg")
    metadata = {
        "Title": "GPU activity and relative cumulative work",
        "Description": (
            "Full-history per-GPU SM activity, active-GPU coverage, and "
            "relative cumulative work under a matched wall-clock horizon."
        ),
    }
    figure.savefig(
        svg_path,
        facecolor=WHITE,
        metadata=metadata,
    )
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
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
    bundle_dir = Path(__file__).resolve().parent.parent
    data_dir = bundle_dir / "data"
    raw_dir = bundle_dir / "data" / "raw"
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
    evidence = json.loads(
        (data_dir / "figure_evidence_draft.json").read_text(encoding="utf-8")
    )
    render(
        sync_rows,
        streamweave_rows,
        evidence["cumulative_work"],
        bundle_dir / "outputs",
    )


if __name__ == "__main__":
    main()
