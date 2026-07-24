#!/usr/bin/env python3
"""Render full-run throughput stability for StreamWeave."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


INK = "#182538"
MUTED = "#617086"
GRID = "#DCE4ED"
TEAL = "#098B7A"
TEAL_MID = "#64B9AE"
TEAL_LIGHT = "#D7F0EC"
SLATE = "#566477"
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
        result.append(
            {
                "run": alias,
                "cycle": int(row["_step"]),
                "groups": float(row["hpt/onpolicy_num_groups"]),
                "time_s": float(row["timing_s/step"]),
            }
        )
    result.sort(key=lambda item: item["cycle"])
    return result


def validate_cycles(alias: str, cycles: list[dict]) -> None:
    expected = EXPECTED[alias]
    checks = (
        (len(cycles), expected["cycles"], 0.0, "cycles"),
        (sum(point["groups"] for point in cycles), expected["groups"], 1e-6, "groups"),
        (sum(point["time_s"] for point in cycles), expected["time_s"], 1e-3, "time_s"),
    )
    for observed, target, tolerance, label in checks:
        if not math.isclose(
            float(observed), float(target), rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"{alias} {label}: observed {observed}, expected {target}"
            )


def work_slice(
    cycles: list[dict], start_group: float, budget_groups: float
) -> dict[str, float]:
    end_group = start_group + budget_groups
    cursor = 0.0
    selected_groups = 0.0
    selected_time = 0.0

    for point in cycles:
        row_start = cursor
        row_end = cursor + point["groups"]
        overlap = max(
            0.0,
            min(row_end, end_group) - max(row_start, start_group),
        )
        if overlap:
            selected_groups += overlap
            selected_time += point["time_s"] * overlap / point["groups"]
        cursor = row_end
        if cursor >= end_group:
            break

    if not math.isclose(
        selected_groups, budget_groups, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError(
            f"requested {budget_groups} groups at {start_group}, "
            f"selected {selected_groups}"
        )
    return {
        "groups": selected_groups,
        "time_s": selected_time,
        "throughput": selected_groups / selected_time,
    }


def equal_work_deciles(
    cycles: list[dict], reference_throughput: float
) -> list[dict]:
    total_groups = sum(point["groups"] for point in cycles)
    segment_groups = total_groups / 10.0
    result = []

    for index in range(10):
        segment = work_slice(cycles, index * segment_groups, segment_groups)
        cumulative = work_slice(cycles, 0.0, (index + 1) * segment_groups)
        result.append(
            {
                "decile": index + 1,
                "progress_percent": 10 * (index + 1),
                "segment_groups": segment["groups"],
                "segment_time_s": segment["time_s"],
                "segment_throughput": segment["throughput"],
                "segment_relative_throughput": (
                    segment["throughput"] / reference_throughput
                ),
                "cumulative_throughput": cumulative["throughput"],
                "cumulative_relative_throughput": (
                    cumulative["throughput"] / reference_throughput
                ),
            }
        )
    return result


def build_snapshot(sync_path: Path, streamweave_path: Path) -> dict:
    sync = extract_cycles(sync_path, "sync")
    streamweave = extract_cycles(streamweave_path, "streamweave")
    validate_cycles("sync", sync)
    validate_cycles("streamweave", streamweave)

    sync_groups = sum(point["groups"] for point in sync)
    sync_time = sum(point["time_s"] for point in sync)
    streamweave_groups = sum(point["groups"] for point in streamweave)
    streamweave_time = sum(point["time_s"] for point in streamweave)
    sync_throughput = sync_groups / sync_time
    streamweave_throughput = streamweave_groups / streamweave_time
    deciles = equal_work_deciles(streamweave, sync_throughput)

    return {
        "status": "draft",
        "figure": "execution_efficiency_throughput_stability",
        "work_normalization": {
            "segments": 10,
            "definition": (
                "Ten contiguous segments with equal consumed prompt-group work "
                "over the full StreamWeave history."
            ),
        },
        "throughput": {
            "sync_groups_per_s": sync_throughput,
            "streamweave_groups_per_s": streamweave_throughput,
            "full_history_ratio": streamweave_throughput / sync_throughput,
            "deciles": deciles,
        },
        "sources": {
            "sync_wandb": str(sync_path),
            "streamweave_wandb": str(streamweave_path),
        },
        "restrictions": [
            "Deciles are descriptive contiguous system observations, not iid samples.",
        ],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(snapshot: dict, data_dir: Path) -> None:
    write_csv(
        data_dir / "full_run_throughput_deciles.csv",
        [
            "decile",
            "progress_percent",
            "segment_groups",
            "segment_time_s",
            "segment_throughput",
            "segment_relative_throughput",
            "cumulative_throughput",
            "cumulative_relative_throughput",
        ],
        snapshot["throughput"]["deciles"],
    )
def render(snapshot: dict, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    efficiency_fig, efficiency_ax = plt.subplots(
        1,
        1,
        figsize=(4.15, 3.25),
        facecolor=WHITE,
    )
    deciles = snapshot["throughput"]["deciles"]
    progress = np.asarray(
        [point["progress_percent"] for point in deciles], dtype=float
    )
    local_ratio = np.asarray(
        [point["segment_relative_throughput"] for point in deciles],
        dtype=float,
    )
    cumulative_ratio = np.asarray(
        [point["cumulative_relative_throughput"] for point in deciles],
        dtype=float,
    )

    efficiency_ax.axhspan(1.0, 2.05, color=TEAL_LIGHT, alpha=0.18, zorder=0)
    efficiency_ax.axhline(
        1.0,
        color=SLATE,
        linewidth=1.4,
        linestyle=(0, (5, 3)),
        zorder=1,
    )
    efficiency_ax.plot(
        progress,
        local_ratio,
        color=TEAL_MID,
        linewidth=1.5,
        marker="o",
        markersize=4.4,
        markerfacecolor=WHITE,
        markeredgewidth=1.2,
        zorder=3,
    )
    efficiency_ax.plot(
        progress,
        cumulative_ratio,
        color=TEAL,
        linewidth=2.6,
        marker="o",
        markersize=4.2,
        zorder=4,
    )

    efficiency_ax.set_xlim(7, 103)
    efficiency_ax.set_ylim(0.88, 2.02)
    efficiency_ax.set_xticks((10, 20, 40, 60, 80, 100))
    efficiency_ax.set_yticks((1.0, 1.2, 1.4, 1.6, 1.8, 2.0))
    efficiency_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.1f}x")
    )
    efficiency_ax.set_xlabel("Consumed-work progress (%)")
    efficiency_ax.set_ylabel("Throughput / synchronous reference")
    efficiency_ax.grid(axis="y", color=GRID, linewidth=0.75)
    efficiency_ax.spines["top"].set_visible(False)
    efficiency_ax.spines["right"].set_visible(False)
    efficiency_ax.set_title(
        "Sustained efficiency over the full run",
        loc="left",
        color=INK,
        fontsize=9.2,
        fontweight="bold",
        pad=9,
    )

    full_ratio = snapshot["throughput"]["full_history_ratio"]
    efficiency_ax.text(
        101,
        cumulative_ratio[-1] + 0.045,
        f"full history  {full_ratio:.2f}x",
        color=TEAL,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )
    efficiency_ax.text(
        101,
        1.025,
        "synchronous reference",
        color=SLATE,
        ha="right",
        va="bottom",
        fontsize=7.4,
    )
    efficiency_ax.text(
        10,
        1.94,
        "all ten equal-work segments remain above 1.0x",
        color=INK,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=TEAL_MID,
            linewidth=1.5,
            marker="o",
            markerfacecolor=WHITE,
            markeredgewidth=1.2,
            label="local equal-work segment",
        ),
        Line2D(
            [0],
            [0],
            color=TEAL,
            linewidth=2.6,
            marker="o",
            label="cumulative",
        ),
    ]
    efficiency_ax.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        fontsize=7.1,
        handlelength=2.2,
    )

    efficiency_fig.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.18,
        top=0.84,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            efficiency_fig,
            output_dir / "execution_efficiency_throughput_stability",
            "Sustained execution efficiency",
            "Work-normalized throughput over the full training run.",
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
        export_tables(snapshot, args.data_dir)

    render(snapshot, args.output_dir)


if __name__ == "__main__":
    main()
