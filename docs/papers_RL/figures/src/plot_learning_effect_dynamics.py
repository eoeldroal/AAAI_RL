#!/usr/bin/env python3
"""Render the canonical StreamWeave learning-dynamics figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
TEAL = "#098B7A"
TEAL_LIGHT = "#D7F0EC"
SLATE = "#566477"
AMBER = "#D98710"
WHITE = "#FFFFFF"

COMPARISON_START = 0
COMPARISON_STOP = 160
EARLY_WINDOW = (20, 50)
LATE_WINDOW = (130, 160)


def centered_mean(values: np.ndarray, width: int = 3) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    radius = width // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        result[index] = values[start:stop].mean()
    if len(values):
        result[0] = values[0]
        result[-1] = values[-1]
    return result


def centered_weighted_rate(
    rates: np.ndarray, weights: np.ndarray, width: int = 7
) -> np.ndarray:
    result = np.empty_like(rates, dtype=float)
    radius = width // 2
    for index in range(len(rates)):
        start = max(0, index - radius)
        stop = min(len(rates), index + radius + 1)
        result[index] = np.average(rates[start:stop], weights=weights[start:stop])
    return result


def window_mean(points: list[dict], start: int, stop: int) -> float:
    values = [point["score"] for point in points if start <= point["cycle"] <= stop]
    return mean(values)


def normalized_progress(cycles: np.ndarray | float) -> np.ndarray | float:
    span = COMPARISON_STOP - COMPARISON_START
    return 100.0 * (cycles - COMPARISON_START) / span


def visible_quality(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.asarray([point["cycle"] for point in points], dtype=float)
    scores = np.asarray([point["score"] for point in points], dtype=float)
    visible = (cycles >= COMPARISON_START) & (cycles <= COMPARISON_STOP)
    return normalized_progress(cycles[visible]), scores[visible]


def render(snapshot: dict, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.72), facecolor=WHITE)
    quality_ax = fig.add_axes((0.095, 0.16, 0.81, 0.70))
    routing_ax = quality_ax.twinx()

    main_points = snapshot["main"]["quality"]
    off_points = snapshot["expert_off"]["quality"]
    main_x, main_y = visible_quality(main_points)
    off_x, off_y = visible_quality(off_points)

    early_start, early_stop = map(normalized_progress, EARLY_WINDOW)
    late_start, late_stop = map(normalized_progress, LATE_WINDOW)
    quality_ax.axvspan(
        early_start,
        early_stop,
        color="#F3F6F9",
        alpha=0.8,
        zorder=0,
    )
    quality_ax.axvspan(
        late_start,
        late_stop,
        color=TEAL_LIGHT,
        alpha=0.72,
        zorder=0,
    )

    quality_ax.plot(
        main_x,
        main_y,
        color=TEAL,
        alpha=0.25,
        linewidth=0.9,
        marker="o",
        markersize=2.25,
        zorder=2,
    )
    quality_ax.plot(
        off_x,
        off_y,
        color=SLATE,
        alpha=0.22,
        linewidth=0.9,
        marker="o",
        markersize=2.1,
        zorder=2,
    )
    quality_ax.plot(
        main_x,
        centered_mean(main_y),
        color=TEAL,
        linewidth=2.5,
        zorder=3,
    )
    quality_ax.plot(
        off_x,
        centered_mean(off_y),
        color=SLATE,
        linewidth=2.2,
        linestyle=(0, (5, 3)),
        zorder=3,
    )

    quality_ax.set_xlim(0, 100)
    quality_ax.set_ylim(15, 42)
    quality_ax.set_xlabel("Normalized training progress (%)")
    quality_ax.set_ylabel("Interim mean@8")
    quality_ax.set_xticks((0, 20, 40, 60, 80, 100))
    quality_ax.set_yticks((16, 20, 24, 28, 32, 36, 40))
    quality_ax.grid(axis="y", color=GRID, linewidth=0.8)
    quality_ax.spines["top"].set_visible(False)

    early_main = window_mean(main_points, *EARLY_WINDOW)
    early_off = window_mean(off_points, *EARLY_WINDOW)
    late_main = window_mean(main_points, *LATE_WINDOW)
    late_off = window_mean(off_points, *LATE_WINDOW)
    quality_ax.text(
        (early_start + early_stop) / 2,
        41.1,
        f"early  {early_main:.1f} / {early_off:.1f}",
        color=MUTED,
        ha="center",
        va="center",
        fontsize=8.1,
    )
    quality_ax.text(
        (late_start + late_stop) / 2,
        41.1,
        f"late  {late_main:.1f} / {late_off:.1f}",
        color=INK,
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
    )
    quality_ax.text(
        98.5,
        39.25,
        "StreamWeave",
        color=TEAL,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    quality_ax.text(
        98.5,
        34.0,
        "Async RL (expert-off)",
        color=SLATE,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    routing = [
        point
        for point in snapshot["main"]["routing"]
        if COMPARISON_START < point["cycle"] <= COMPARISON_STOP
    ]
    route_cycles = np.asarray([point["cycle"] for point in routing], dtype=float)
    route_x = normalized_progress(route_cycles)
    route_y = 100.0 * np.asarray([point["expert_rate"] for point in routing])
    route_groups = np.asarray([point["groups"] for point in routing], dtype=float)
    routing_ax.plot(
        route_x,
        route_y,
        color=AMBER,
        alpha=0.22,
        linewidth=0.65,
        zorder=1,
    )
    routing_ax.plot(
        route_x,
        centered_weighted_rate(route_y, route_groups),
        color=AMBER,
        linewidth=1.65,
        zorder=2,
    )
    routing_ax.set_ylim(0, 70)
    routing_ax.set_yticks((0, 20, 40, 60))
    routing_ax.set_ylabel("Expert routing (%)", color=AMBER, labelpad=10)
    routing_ax.tick_params(axis="y", colors=AMBER)
    routing_ax.spines["top"].set_visible(False)
    routing_ax.spines["left"].set_visible(False)
    routing_ax.spines["right"].set_color(AMBER)
    routing_ax.text(
        98.0,
        23.3,
        "late routing  20.2%",
        color=AMBER,
        ha="right",
        va="bottom",
        fontsize=7.9,
        fontweight="bold",
    )
    routing_ax.text(
        4.0,
        55.5,
        "initial routing  50.0%",
        color=AMBER,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )

    legend = [
        Line2D([0], [0], color=TEAL, linewidth=2.5, label="StreamWeave"),
        Line2D(
            [0],
            [0],
            color=SLATE,
            linewidth=2.2,
            linestyle=(0, (5, 3)),
            label="Async RL (expert-off)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.088, 0.975),
        frameon=False,
        ncol=2,
        handlelength=2.5,
        columnspacing=1.7,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=WHITE)
    fig.savefig(output_base.with_suffix(".pdf"), facecolor=WHITE, bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".png"),
        facecolor=WHITE,
        bbox_inches="tight",
        dpi=240,
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-snapshot", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.input_snapshot.read_text(encoding="utf-8"))
    render(snapshot, args.output_base)


if __name__ == "__main__":
    main()
