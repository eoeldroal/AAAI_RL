#!/usr/bin/env python3
"""Render cycle-level evidence for StreamWeave execution efficiency."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
SLATE_LIGHT = "#E7EBF0"
AMBER = "#D98710"
WHITE = "#FFFFFF"

EXPECTED = {
    "sync": {
        "cycles": 104,
        "groups": 13312,
        "time_s": 4780.190194,
    },
    "streamweave": {
        "cycles": 190,
        "groups": 86174,
        "time_s": 18828.415748,
    },
}


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


def extract_cycles(path: Path, alias: str) -> list[dict]:
    result = []
    for row in history_rows(path):
        if "hpt/onpolicy_num_groups" not in row or "timing_s/step" not in row:
            continue
        point = {
            "run": alias,
            "cycle": int(row["_step"]),
            "groups": int(row["hpt/onpolicy_num_groups"]),
            "time_s": float(row["timing_s/step"]),
        }
        if alias == "sync":
            point.update(
                {
                    "request_mean_s": float(
                        row["timing_s/agent_loop/generate_sequences/mean"]
                    ),
                    "request_max_s": float(
                        row["timing_s/agent_loop/generate_sequences/max"]
                    ),
                    "generation_phase_s": float(row["timing_s/gen"]),
                }
            )
        result.append(point)
    result.sort(key=lambda item: item["cycle"])
    return result


def validate_cycles(alias: str, cycles: list[dict]) -> None:
    expected = EXPECTED[alias]
    groups = sum(point["groups"] for point in cycles)
    time_s = sum(point["time_s"] for point in cycles)
    checks = [
        (len(cycles), expected["cycles"], 0.0, "cycles"),
        (groups, expected["groups"], 0.0, "groups"),
        (time_s, expected["time_s"], 1e-3, "time_s"),
    ]
    for observed, target, tolerance, label in checks:
        if not math.isclose(
            float(observed), float(target), rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"{alias} {label}: observed {observed}, expected {target}"
            )


def cumulative_points(cycles: list[dict]) -> list[dict]:
    result = [
        {
            "run": cycles[0]["run"],
            "cycle": 0,
            "cumulative_time_s": 0.0,
            "cumulative_groups": 0.0,
        }
    ]
    cumulative_time = 0.0
    cumulative_groups = 0.0
    for point in cycles:
        cumulative_time += point["time_s"]
        cumulative_groups += point["groups"]
        result.append(
            {
                "run": point["run"],
                "cycle": point["cycle"],
                "cumulative_time_s": cumulative_time,
                "cumulative_groups": cumulative_groups,
            }
        )
    return result


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
                "run": point["run"],
                "cycle": point["cycle"],
                "cumulative_time_s": horizon_s,
                "cumulative_groups": previous["cumulative_groups"]
                + fraction
                * (point["cumulative_groups"] - previous["cumulative_groups"]),
            }
        )
        break
    return result


def centered_mean(values: np.ndarray, width: int = 7) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    radius = width // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        result[index] = values[start:stop].mean()
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_snapshot(sync_path: Path, streamweave_path: Path) -> dict:
    sync = extract_cycles(sync_path, "sync")
    streamweave = extract_cycles(streamweave_path, "streamweave")
    validate_cycles("sync", sync)
    validate_cycles("streamweave", streamweave)

    sync_cumulative = cumulative_points(sync)
    streamweave_cumulative = cumulative_points(streamweave)
    common_horizon_s = sync_cumulative[-1]["cumulative_time_s"]
    streamweave_common = trim_to_horizon(streamweave_cumulative, common_horizon_s)

    sync_groups = sync_cumulative[-1]["cumulative_groups"]
    streamweave_common_groups = streamweave_common[-1]["cumulative_groups"]
    sync_throughput = sync_groups / common_horizon_s
    streamweave_throughput = (
        streamweave_cumulative[-1]["cumulative_groups"]
        / streamweave_cumulative[-1]["cumulative_time_s"]
    )

    return {
        "status": "draft",
        "figure": "execution_efficiency_evidence",
        "sync_timing": sync,
        "cumulative_work": {
            "sync": sync_cumulative,
            "streamweave": streamweave_cumulative,
            "common_horizon_s": common_horizon_s,
            "streamweave_at_common_horizon": streamweave_common_groups,
        },
        "aggregates": {
            "request_mean_s": mean(
                point["request_mean_s"] for point in sync
            ),
            "request_max_s": mean(point["request_max_s"] for point in sync),
            "generation_phase_s": mean(
                point["generation_phase_s"] for point in sync
            ),
            "sync_throughput": sync_throughput,
            "streamweave_throughput": streamweave_throughput,
            "throughput_ratio": streamweave_throughput / sync_throughput,
            "same_horizon_work_ratio": streamweave_common_groups / sync_groups,
        },
        "sources": {
            "sync": str(sync_path),
            "streamweave": str(streamweave_path),
        },
        "restrictions": [
            "The completion band is request spread, not measured GPU idle.",
            "The two mechanisms are not an additive speedup decomposition.",
            "The common-horizon work ratio is distinct from full-history throughput.",
        ],
    }


def export_snapshot_tables(snapshot: dict, data_dir: Path) -> None:
    timing_rows = snapshot["sync_timing"]
    write_csv(
        data_dir / "sync_cycle_timing.csv",
        [
            "run",
            "cycle",
            "groups",
            "time_s",
            "request_mean_s",
            "request_max_s",
            "generation_phase_s",
        ],
        timing_rows,
    )

    cumulative_rows = []
    for alias, points in snapshot["cumulative_work"].items():
        if not isinstance(points, list):
            continue
        for point in points:
            cumulative_rows.append(point)
    write_csv(
        data_dir / "cumulative_work.csv",
        [
            "run",
            "cycle",
            "cumulative_time_s",
            "cumulative_groups",
        ],
        cumulative_rows,
    )


def render(snapshot: dict, output_dir: Path) -> None:
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

    tail_fig, tail_ax = plt.subplots(
        1,
        1,
        figsize=(4.15, 3.15),
        facecolor=WHITE,
    )
    work_fig, work_ax = plt.subplots(
        1,
        1,
        figsize=(4.15, 3.15),
        facecolor=WHITE,
    )

    sync = snapshot["sync_timing"]
    x = np.asarray([point["cycle"] for point in sync], dtype=float)
    request_mean = np.asarray(
        [point["request_mean_s"] for point in sync], dtype=float
    )
    request_max = np.asarray(
        [point["request_max_s"] for point in sync], dtype=float
    )
    generation_phase = np.asarray(
        [point["generation_phase_s"] for point in sync], dtype=float
    )
    mean_smooth = centered_mean(request_mean)
    max_smooth = centered_mean(request_max)
    phase_smooth = centered_mean(generation_phase)

    tail_ax.fill_between(
        x,
        mean_smooth,
        max_smooth,
        color=SLATE_LIGHT,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )
    tail_ax.plot(
        x,
        request_mean,
        color=AMBER,
        alpha=0.16,
        linewidth=0.7,
        zorder=2,
    )
    tail_ax.plot(
        x,
        request_max,
        color=SLATE,
        alpha=0.18,
        linewidth=0.7,
        zorder=2,
    )
    tail_ax.plot(
        x,
        mean_smooth,
        color=AMBER,
        linewidth=1.8,
        zorder=3,
    )
    tail_ax.plot(
        x,
        max_smooth,
        color=SLATE,
        linewidth=2.0,
        zorder=3,
    )
    tail_ax.plot(
        x,
        phase_smooth,
        color=INK,
        linewidth=1.5,
        linestyle=(0, (4, 2)),
        zorder=4,
    )

    tail_ax.set_xlim(1, 104)
    tail_ax.set_ylim(0, 43)
    tail_ax.set_xticks((1, 20, 40, 60, 80, 104))
    tail_ax.set_yticks((0, 10, 20, 30, 40))
    tail_ax.set_xlabel("Synchronous training cycle")
    tail_ax.set_ylabel("Completion time (s)")
    tail_ax.grid(axis="y", color=GRID, linewidth=0.75)
    tail_ax.spines["top"].set_visible(False)
    tail_ax.spines["right"].set_visible(False)
    tail_ax.set_title(
        "Group barrier exposes the completion tail",
        loc="left",
        color=INK,
        fontsize=9.2,
        fontweight="bold",
        pad=9,
    )
    tail_ax.text(
        4,
        17.1,
        "request-completion\nspread",
        color=MUTED,
        ha="left",
        va="center",
        fontsize=7.4,
    )

    common_horizon_s = snapshot["cumulative_work"]["common_horizon_s"]
    sync_work = snapshot["cumulative_work"]["sync"]
    streamweave_work = trim_to_horizon(
        snapshot["cumulative_work"]["streamweave"], common_horizon_s
    )
    sync_minutes = np.asarray(
        [point["cumulative_time_s"] / 60.0 for point in sync_work]
    )
    sync_groups = np.asarray(
        [point["cumulative_groups"] / 1000.0 for point in sync_work]
    )
    streamweave_minutes = np.asarray(
        [point["cumulative_time_s"] / 60.0 for point in streamweave_work]
    )
    streamweave_groups = np.asarray(
        [point["cumulative_groups"] / 1000.0 for point in streamweave_work]
    )

    work_ax.plot(
        sync_minutes,
        sync_groups,
        color=SLATE,
        linewidth=2.2,
        linestyle=(0, (5, 3)),
        zorder=2,
    )
    work_ax.plot(
        streamweave_minutes,
        streamweave_groups,
        color=TEAL,
        linewidth=2.6,
        zorder=3,
    )
    work_ax.scatter(
        [sync_minutes[-1], streamweave_minutes[-1]],
        [sync_groups[-1], streamweave_groups[-1]],
        s=22,
        color=[SLATE, TEAL],
        edgecolors=WHITE,
        linewidths=0.7,
        zorder=4,
    )
    work_ax.set_xlim(0, math.ceil(common_horizon_s / 600.0) * 10)
    work_ax.set_ylim(0, 24)
    work_ax.set_xticks((0, 20, 40, 60, 80))
    work_ax.set_yticks((0, 5, 10, 15, 20))
    work_ax.set_xlabel("Training wall-clock (min)")
    work_ax.set_ylabel("Consumed prompt groups (thousands)")
    work_ax.grid(axis="both", color=GRID, linewidth=0.75)
    work_ax.spines["top"].set_visible(False)
    work_ax.spines["right"].set_visible(False)
    work_ax.set_title(
        "More work under the same wall-clock",
        loc="left",
        color=INK,
        fontsize=9.2,
        fontweight="bold",
        pad=9,
    )

    work_ax.text(
        77.5,
        streamweave_groups[-1] + 0.8,
        f"StreamWeave  {streamweave_groups[-1]:.1f}k",
        color=TEAL,
        ha="right",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )
    work_ax.text(
        77.5,
        sync_groups[-1] - 0.7,
        f"synchronous  {sync_groups[-1]:.1f}k",
        color=SLATE,
        ha="right",
        va="top",
        fontsize=7.8,
    )
    aggregates = snapshot["aggregates"]
    work_ax.text(
        4,
        22.6,
        "full history",
        color=MUTED,
        ha="left",
        va="top",
        fontsize=7.2,
    )
    work_ax.text(
        4,
        20.7,
        (
            f"{aggregates['sync_throughput']:.2f} "
            r"$\rightarrow$ "
            f"{aggregates['streamweave_throughput']:.2f} groups/s"
        ),
        color=INK,
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
    )
    work_ax.text(
        4,
        18.7,
        f"{aggregates['throughput_ratio']:.2f}x throughput",
        color=TEAL,
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="bold",
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=AMBER,
            linewidth=1.8,
            label=f"mean ({aggregates['request_mean_s']:.1f} s)",
        ),
        Line2D(
            [0],
            [0],
            color=SLATE,
            linewidth=2.0,
            label=f"slowest ({aggregates['request_max_s']:.1f} s)",
        ),
        Line2D(
            [0],
            [0],
            color=INK,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label=f"phase ({aggregates['generation_phase_s']:.1f} s)",
        ),
    ]
    tail_ax.legend(
        handles=legend,
        loc="upper left",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(-0.01, 1.01),
        handlelength=2.1,
        columnspacing=0.8,
        fontsize=7.0,
    )

    tail_fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.84)
    work_fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.84)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            tail_fig,
            output_dir / "execution_efficiency_completion_tail",
            "Synchronous group-completion tail",
            "Request-completion spread exposed by the synchronous group barrier.",
        ),
        (
            work_fig,
            output_dir / "execution_efficiency_cumulative_work",
            "Prompt-group work under a common wall-clock",
            "Cumulative consumed prompt groups under the same wall-clock horizon.",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-wandb", type=Path)
    parser.add_argument("--streamweave-wandb", type=Path)
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.input_snapshot is not None:
        snapshot = json.loads(args.input_snapshot.read_text(encoding="utf-8"))
    else:
        if args.sync_wandb is None or args.streamweave_wandb is None:
            parser.error(
                "provide --input-snapshot or both --sync-wandb and "
                "--streamweave-wandb"
            )
        if args.snapshot is None or args.data_dir is None:
            parser.error(
                "--snapshot and --data-dir are required when extracting histories"
            )
        snapshot = build_snapshot(args.sync_wandb, args.streamweave_wandb)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        export_snapshot_tables(snapshot, args.data_dir)

    render(snapshot, args.output_dir)


if __name__ == "__main__":
    main()
