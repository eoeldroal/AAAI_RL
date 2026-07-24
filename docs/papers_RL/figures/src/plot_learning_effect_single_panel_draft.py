#!/usr/bin/env python3
"""Render a single-panel draft for StreamWeave learning-effect evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BENCHMARKS = (
    "AIME24",
    "AIME25",
    "AMC23",
    "MATH-500",
    "Minerva",
    "Olympiad-Bench",
)
QUALITY_KEYS = tuple(f"val-aux/{name}/reward/mean@8" for name in BENCHMARKS)

INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
TEAL = "#098B7A"
TEAL_LIGHT = "#D7F0EC"
SLATE = "#566477"
AMBER = "#D98710"
AMBER_LIGHT = "#F8E8C8"
WHITE = "#FFFFFF"


def parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def history_rows(path: Path) -> list[dict]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    datastore = DataStore()
    datastore.open_for_scan(str(path))
    rows: list[dict] = []
    while True:
        data = datastore.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item.key or ".".join(item.nested_key)
            row[key] = parse_value(item.value_json)
        rows.append(row)
    return rows


def extract_run(path: Path, include_routing: bool) -> dict:
    quality = []
    routing = []
    cumulative_groups = 0
    for row in history_rows(path):
        cycle = row.get("_step")
        if cycle is None:
            continue
        cycle = int(cycle)
        if all(key in row for key in QUALITY_KEYS):
            quality.append(
                {
                    "cycle": cycle,
                    "score": 100.0 * mean(float(row[key]) for key in QUALITY_KEYS),
                }
            )
        if include_routing and "hpt/onpolicy_num_groups" in row:
            groups = int(row["hpt/onpolicy_num_groups"])
            cumulative_groups += groups
            expert = int(row["hpt/num_sft"])
            routing.append(
                {
                    "cycle": cycle,
                    "groups": groups,
                    "cumulative_groups": cumulative_groups,
                    "expert_rate": expert / groups,
                }
            )
    return {"quality": quality, "routing": routing}


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


def render(snapshot: dict, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.9), facecolor=WHITE)
    quality_ax = fig.add_axes((0.095, 0.16, 0.81, 0.70))
    routing_ax = quality_ax.twinx()

    main = snapshot["main"]["quality"]
    off = snapshot["expert_off"]["quality"]
    main_x = np.asarray([point["cycle"] for point in main], dtype=float)
    main_y = np.asarray([point["score"] for point in main], dtype=float)
    off_x = np.asarray([point["cycle"] for point in off], dtype=float)
    off_y = np.asarray([point["score"] for point in off], dtype=float)

    visible_main = (main_x >= 0) & (main_x <= 160)
    visible_off = (off_x >= 0) & (off_x <= 160)
    main_x, main_y = main_x[visible_main], main_y[visible_main]
    off_x, off_y = off_x[visible_off], off_y[visible_off]

    for start, stop, color in ((20, 50, "#F3F6F9"), (130, 160, TEAL_LIGHT)):
        quality_ax.axvspan(start, stop, color=color, alpha=0.7, zorder=0)

    quality_ax.plot(
        main_x,
        main_y,
        color=TEAL,
        alpha=0.28,
        linewidth=1.0,
        marker="o",
        markersize=2.4,
        zorder=2,
    )
    quality_ax.plot(
        off_x,
        off_y,
        color=SLATE,
        alpha=0.24,
        linewidth=1.0,
        marker="o",
        markersize=2.2,
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

    quality_ax.set_xlim(0, 160)
    quality_ax.set_ylim(15, 42)
    quality_ax.set_ylabel("Interim mean@8")
    quality_ax.set_yticks((16, 20, 24, 28, 32, 36, 40))
    quality_ax.grid(axis="y", color=GRID, linewidth=0.8)
    quality_ax.spines["top"].set_visible(False)
    quality_ax.set_xlabel("Training cycle (0 = pre-training)")
    quality_ax.set_xticks((0, 20, 40, 60, 80, 100, 120, 140, 160))

    early_main = window_mean(snapshot["main"]["quality"], 20, 50)
    early_off = window_mean(snapshot["expert_off"]["quality"], 20, 50)
    late_main = window_mean(snapshot["main"]["quality"], 130, 160)
    late_off = window_mean(snapshot["expert_off"]["quality"], 130, 160)
    quality_ax.text(
        35,
        41.15,
        f"early  {early_main:.1f} / {early_off:.1f}",
        color=MUTED,
        ha="center",
        va="center",
        fontsize=8.2,
    )
    quality_ax.text(
        145,
        41.15,
        f"late  {late_main:.1f} / {late_off:.1f}",
        color=INK,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
    )
    quality_ax.text(
        158.5,
        39.25,
        "StreamWeave",
        color=TEAL,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    quality_ax.text(
        158.5,
        34.0,
        "expert-off",
        color=SLATE,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    quality_ax.text(
        157.5,
        15.35,
        "late all-failure: 32.4% / 37.0%",
        color=MUTED,
        ha="right",
        va="bottom",
        fontsize=7.8,
    )

    routing = [
        point
        for point in snapshot["main"]["routing"]
        if 1 <= point["cycle"] <= 160
    ]
    route_x = np.asarray([point["cycle"] for point in routing], dtype=float)
    route_y = 100.0 * np.asarray([point["expert_rate"] for point in routing])
    route_groups = np.asarray([point["groups"] for point in routing], dtype=float)
    routing_ax.plot(
        route_x,
        route_y,
        color=AMBER,
        alpha=0.24,
        linewidth=0.7,
        zorder=1,
    )
    routing_ax.plot(
        route_x,
        centered_weighted_rate(route_y, route_groups),
        color=AMBER,
        linewidth=1.6,
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
        157.5,
        23.3,
        "cycles 130–160  20.2%",
        color=AMBER,
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    routing_ax.text(
        7,
        55.5,
        "cycles 1–10  50.0%",
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
            label="expert-off",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.088, 0.98),
        frameon=False,
        ncol=2,
        handlelength=2.5,
        columnspacing=1.7,
    )
    fig.text(
        0.95,
        0.955,
        "0/8 census   22.4% groups · 26.8% tokens · 1.27× length",
        ha="right",
        va="center",
        color=INK,
        fontsize=8.3,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), facecolor=WHITE)
    fig.savefig(output_base.with_suffix(".pdf"), facecolor=WHITE, bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".png"),
        facecolor=WHITE,
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-wandb", type=Path)
    parser.add_argument("--expert-off-wandb", type=Path)
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()

    if args.input_snapshot is not None:
        snapshot = json.loads(args.input_snapshot.read_text(encoding="utf-8"))
    else:
        if args.main_wandb is None or args.expert_off_wandb is None:
            parser.error(
                "provide --input-snapshot or both --main-wandb and --expert-off-wandb"
            )
        snapshot = {
            "status": "draft",
            "metric": "six-benchmark macro mean@8",
            "x_axis": "training cycle",
            "main": extract_run(args.main_wandb, include_routing=True),
            "expert_off": extract_run(args.expert_off_wandb, include_routing=False),
            "all_failure_census": {
                "group_share": 0.2238,
                "response_token_share": 0.2683,
                "mean_length_ratio": 1.272,
                "source": "/private/tmp/streamweave_census_nocispo.parquet",
            },
            "sources": {
                "main": str(args.main_wandb),
                "expert_off": str(args.expert_off_wandb),
            },
        }
        if args.snapshot is None:
            parser.error("--snapshot is required when extracting W&B histories")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    render(snapshot, args.output_base)


if __name__ == "__main__":
    main()
