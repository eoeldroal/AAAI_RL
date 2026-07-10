#!/usr/bin/env python3
"""Verify rollout-selection artifacts and optional extracted DataProto groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from scripts.migration.select_rollout_groups import ROLLOUT_N, _canonical_prompt_hash, _scalar
from verl import DataProto


def _validate_manifest(manifest: pd.DataFrame, expected_groups: int) -> None:
    required = {
        "run",
        "feed_step",
        "attempt_count",
        "archive_name",
        "prompt_hash",
        "reward_scores",
        "response_hashes",
        "actual_route",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if len(manifest) != expected_groups:
        raise ValueError(f"manifest groups={len(manifest)} expected={expected_groups}")
    if manifest.duplicated(["run", "feed_step"]).any():
        raise ValueError("manifest contains duplicate run/feed_step groups")
    if not manifest.attempt_count.eq(ROLLOUT_N).all():
        raise ValueError(f"manifest attempt_count must always equal {ROLLOUT_N}")
    if manifest.archive_name.eq("").any():
        raise ValueError("manifest contains an empty archive_name")
    archive_counts = manifest.groupby("run").archive_name.nunique().to_dict()
    expected_archives = {"M5": 8, "nocispo": 8, "RLonly": 8}
    if archive_counts != expected_archives:
        raise ValueError(f"archive shard counts differ: expected={expected_archives} actual={archive_counts}")


def _parse_run_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run root must be NAME=PATH, got {value!r}")
        name, path_text = value.split("=", 1)
        if name in roots:
            raise ValueError(f"duplicate run root: {name}")
        path = Path(path_text).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        roots[name] = path
    return roots


def _validate_run_roots(
    manifest: pd.DataFrame,
    run_roots: dict[str, Path],
    archives_only: bool,
) -> None:
    expected = set(str(run) for run in manifest.run.unique())
    actual = set(run_roots)
    if archives_only:
        if actual:
            raise ValueError("--archives-only cannot be combined with --run-root")
        return
    if actual != expected:
        raise ValueError(
            f"run roots must exactly match manifest runs: expected={sorted(expected)} actual={sorted(actual)}"
        )


def _stable_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    work = frame.copy()
    work["_priority"] = [
        hashlib.sha256(f"{row.run}|{row.feed_step}|{row.prompt_hash}".encode()).hexdigest()
        for row in work.itertuples(index=False)
    ]
    return work.sort_values("_priority").head(count).drop(columns=["_priority"])


def _verify_extracted_group(row: object, run_root: Path) -> None:
    group = run_root / str(row.feed_step)
    accs: list[float] = []
    lengths: list[int] = []
    min_versions: list[int] = []
    max_versions: list[int] = []
    resume_counts: list[int] = []
    reward_scores: list[float] = []
    response_hashes: list[str] = []
    prompt_hash: str | None = None
    for attempt_index in range(ROLLOUT_N):
        data_path = group / f"attempt_{attempt_index}" / "gen_batch.dp"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        data = DataProto.load_from_disk(data_path)
        current_hash = _canonical_prompt_hash(_scalar(data.non_tensor_batch.get("raw_prompt"), []))
        if prompt_hash is None:
            prompt_hash = current_hash
        elif current_hash != prompt_hash:
            raise ValueError(f"prompt hash differs across attempts in {group}")
        accs.append(float(_scalar(data.non_tensor_batch.get("acc"), 0.0)))
        reward_scores.append(float(data.batch["rm_scores"].sum().item()))
        response = data.batch["responses"]
        tokens = response[data.batch["response_mask"].bool()].detach().cpu().numpy()
        response_hashes.append(hashlib.sha256(tokens.tobytes()).hexdigest())
        lengths.append(int(data.batch["response_mask"].sum().item()))
        minimum = int(_scalar(data.non_tensor_batch.get("min_global_steps"), 0))
        min_versions.append(minimum)
        max_versions.append(int(_scalar(data.non_tensor_batch.get("max_global_steps"), minimum)))
        resume_counts.append(int(_scalar(data.non_tensor_batch.get("partial_rollout_resume_count"), 0)))

    expected = {
        "prompt_hash": str(row.prompt_hash),
        "accs": list(row.accs),
        "response_lengths": list(row.response_lengths),
        "min_versions": list(row.min_versions),
        "max_versions": list(row.max_versions),
        "resume_counts": list(row.resume_counts),
        "reward_scores": list(row.reward_scores),
    }
    actual = {
        "prompt_hash": prompt_hash,
        "accs": accs,
        "response_lengths": lengths,
        "min_versions": min_versions,
        "max_versions": max_versions,
        "resume_counts": resume_counts,
        "reward_scores": reward_scores,
    }
    if str(row.response_hashes):
        expected["response_hashes"] = str(row.response_hashes).split(";")
        actual["response_hashes"] = response_hashes
    if actual != expected:
        raise ValueError(f"DataProto sample mismatch for {row.run}/{row.feed_step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--run-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--samples-per-shard", type=int, default=2)
    parser.add_argument("--archives-only", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-archive-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_dir = args.selection_dir.resolve()
    summary = json.loads((selection_dir / "selection_summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(selection_dir / "selection_manifest.parquet")
    _validate_manifest(manifest, expected_groups=int(summary["groups"]))
    run_roots = _parse_run_roots(args.run_root)
    _validate_run_roots(manifest, run_roots, archives_only=args.archives_only)

    if not args.skip_checksums:
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=selection_dir, check=True)
    archive_names = sorted(manifest.archive_name.unique())
    for archive_name in archive_names:
        archive_path = selection_dir / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        if not args.skip_archive_test:
            subprocess.run(["zstd", "-t", str(archive_path)], check=True, capture_output=True)

    for run_name, run_root in run_roots.items():
        run_frame = manifest[manifest.run == run_name]
        for row in run_frame.itertuples(index=False):
            for attempt_index in range(ROLLOUT_N):
                data_path = run_root / str(row.feed_step) / f"attempt_{attempt_index}" / "gen_batch.dp"
                if not data_path.is_file():
                    raise FileNotFoundError(data_path)
        samples = pd.concat(
            [
                _stable_sample(shard_frame, args.samples_per_shard)
                for _, shard_frame in run_frame.groupby("archive_name", sort=True)
            ],
            ignore_index=True,
        )
        for row in samples.itertuples(index=False):
            _verify_extracted_group(row, run_root)

    print(
        f"verified groups={len(manifest)} archives={len(archive_names)} extracted_runs={len(run_roots)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
