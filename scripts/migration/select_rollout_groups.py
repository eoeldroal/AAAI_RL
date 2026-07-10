#!/usr/bin/env python3
"""Build a compact rollout census and a deterministic paper-analysis sample.

The async rollout dump stores one prompt group per numeric directory and eight
``attempt_*`` DataProto pickles below it.  The numeric directory is a feed-step,
not a policy parameter version, so sampling must inspect the DataProto metadata.

This script is intentionally migration-oriented:

1. Scan every group once and write one compact Parquet row per prompt group.
2. Select whole prompt groups (all eight attempts) using deterministic strata,
   paired-prompt windows, and rare-event supplements.
3. Write a manifest and, unless disabled, one tar.zst archive per run plus a
   SHA-256 sidecar.

Original rollout dumps are read-only and are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from verl import DataProto

ROLLOUT_N = 8
RESPONSE_CAP = 8192
CENSUS_SCHEMA_VERSION = 3
_PROMPT_METADATA_BY_HASH: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_id: str
    root: Path


@dataclass
class GroupSummary:
    run: str
    run_id: str
    feed_step: int
    source_group: str
    prompt_hash: str
    prompt_uid: str
    source_subtype: str
    tau_char_length: int
    tau_message_count: int
    prompt_char_length: int
    prompt_token_length: int
    complete: bool
    min_param_version: int
    max_param_version: int
    partial_span: int
    k: int
    k_class: str
    route_class: str
    attempt_count: int
    missing_attempts: str
    correct_attempts: int
    raw_correct_attempts: int
    truncated_attempts: int
    truncated_correct_attempts: int
    partial_attempts: int
    unique_response_count: int
    response_length_min: int
    response_length_mean: float
    response_length_max: int
    response_length_std: float
    generation_time_min: float
    generation_time_mean: float
    generation_time_max: float
    generation_time_std: float
    raw_bytes: int
    accs: list[float]
    reward_scores: list[float]
    response_hashes: str
    response_lengths: list[int]
    min_versions: list[int]
    max_versions: list[int]
    resume_counts: list[int]
    generation_times: list[float]


def _json_normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_normalize(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_normalize(value.item())
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_json_normalize(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported prompt value: {type(value).__name__}")


def _canonical_prompt_hash(raw_prompt: Any) -> str:
    payload = json.dumps(
        _json_normalize(raw_prompt),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _generation_time(data: DataProto) -> float:
    metrics = data.meta_info.get("metrics", []) if data.meta_info else []
    if not metrics:
        return 0.0
    value = metrics[0].get("generate_sequences", 0.0)
    return float(value or 0.0)


def _route_class(run_name: str, k: int) -> str:
    if k == 0:
        return "zero_variance_rl" if run_name == "RLonly" else "sft"
    if k == ROLLOUT_N:
        return "zero_variance_rl"
    return "informative_rl"


def _classify_group(
    run_name: str,
    reward_scores: list[float],
    attempt_count: int,
) -> tuple[int, str, str]:
    k = sum(score > 0.0 for score in reward_scores)
    if attempt_count != ROLLOUT_N:
        return k, "incomplete", "incomplete"
    if k == 0:
        k_class = "k0"
    elif k == ROLLOUT_N:
        k_class = "k8"
    else:
        k_class = "k1_7"
    return k, k_class, _route_class(run_name, k)


def _response_hash(data: DataProto, response_mask: Any) -> str:
    response = data.batch["responses"]
    tokens = response[response_mask.bool()].detach().cpu().numpy()
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def _scan_group(task: tuple[str, str, str]) -> dict[str, Any]:
    run_name, run_id, group_path_text = task
    group_path = Path(group_path_text)
    feed_step = int(group_path.name)
    accs: list[float] = []
    reward_scores: list[float] = []
    lengths: list[int] = []
    min_versions: list[int] = []
    max_versions: list[int] = []
    resume_counts: list[int] = []
    generation_times: list[float] = []
    prompt_hash: str | None = None
    prompt_token_length = 0
    missing_attempts: list[int] = []
    raw_bytes = 0

    for attempt_idx in range(ROLLOUT_N):
        attempt_dir = group_path / f"attempt_{attempt_idx}"
        data_path = attempt_dir / "gen_batch.dp"
        if not data_path.is_file():
            missing_attempts.append(attempt_idx)
            continue
        raw_bytes += data_path.stat().st_size
        meta_path = attempt_dir / "meta.json"
        if meta_path.is_file():
            raw_bytes += meta_path.stat().st_size

        data = DataProto.load_from_disk(data_path)
        if prompt_hash is None:
            raw_prompt = _scalar(data.non_tensor_batch.get("raw_prompt"), default=[])
            prompt_hash = _canonical_prompt_hash(raw_prompt)
            prompt_width = int(data.batch["prompts"].shape[-1])
            prompt_token_length = int(data.batch["attention_mask"][..., :prompt_width].sum().item())

        acc = float(_scalar(data.non_tensor_batch.get("acc"), default=0.0))
        if not np.isfinite(acc) or not (np.isclose(acc, 0.0) or np.isclose(acc, 1.0)):
            raise ValueError(f"acc must be binary in {data_path}, got {acc!r}")
        acc = float(round(acc))
        reward_score = float(data.batch["rm_scores"].sum().item())
        if not np.isfinite(reward_score):
            raise ValueError(f"rm_scores must be finite in {data_path}, got {reward_score!r}")
        response_mask = data.batch.get("response_mask")
        if response_mask is None:
            raise KeyError(f"response_mask missing from {data_path}")
        response_length = int(response_mask.sum().item())
        min_version = int(_scalar(data.non_tensor_batch.get("min_global_steps"), default=0))
        max_version = int(_scalar(data.non_tensor_batch.get("max_global_steps"), default=min_version))
        resume_count = int(_scalar(data.non_tensor_batch.get("partial_rollout_resume_count"), default=0))

        accs.append(acc)
        reward_scores.append(reward_score)
        lengths.append(response_length)
        min_versions.append(min_version)
        max_versions.append(max_version)
        resume_counts.append(resume_count)
        generation_times.append(_generation_time(data))

    attempt_count = len(accs)
    k, k_class, route_class = _classify_group(run_name, reward_scores, attempt_count)
    if prompt_hash is None:
        prompt_hash = ""
        prompt_metadata = {
            "prompt_uid": "",
            "source_subtype": "",
            "tau_char_length": 0,
            "tau_message_count": 0,
            "prompt_char_length": 0,
        }
    else:
        prompt_metadata = _PROMPT_METADATA_BY_HASH.get(prompt_hash)
        if prompt_metadata is None:
            raise KeyError(f"prompt hash {prompt_hash} from {group_path} is absent from the dataset")

    min_param_version = min(min_versions, default=0)
    max_param_version = max(max_versions, default=0)
    summary = GroupSummary(
        run=run_name,
        run_id=run_id,
        feed_step=feed_step,
        source_group=str(group_path),
        prompt_hash=prompt_hash,
        prompt_uid=str(prompt_metadata["prompt_uid"]),
        source_subtype=str(prompt_metadata["source_subtype"]),
        tau_char_length=int(prompt_metadata["tau_char_length"]),
        tau_message_count=int(prompt_metadata["tau_message_count"]),
        prompt_char_length=int(prompt_metadata["prompt_char_length"]),
        prompt_token_length=prompt_token_length,
        complete=attempt_count == ROLLOUT_N,
        min_param_version=min_param_version,
        max_param_version=max_param_version,
        partial_span=max_param_version - min_param_version,
        k=k,
        k_class=k_class,
        route_class=route_class,
        attempt_count=attempt_count,
        missing_attempts=",".join(str(index) for index in missing_attempts),
        correct_attempts=k,
        raw_correct_attempts=sum(acc > 0.0 for acc in accs),
        truncated_attempts=sum(length >= RESPONSE_CAP for length in lengths),
        truncated_correct_attempts=sum(
            length >= RESPONSE_CAP and acc > 0.0 and score <= 0.0
            for length, acc, score in zip(lengths, accs, reward_scores, strict=True)
        ),
        partial_attempts=sum(
            count > 0 or hi != lo for count, lo, hi in zip(resume_counts, min_versions, max_versions, strict=False)
        ),
        unique_response_count=-1,
        response_length_min=min(lengths, default=0),
        response_length_mean=float(np.mean(lengths)) if lengths else 0.0,
        response_length_max=max(lengths, default=0),
        response_length_std=float(np.std(lengths)) if lengths else 0.0,
        generation_time_min=min(generation_times, default=0.0),
        generation_time_mean=float(np.mean(generation_times)) if generation_times else 0.0,
        generation_time_max=max(generation_times, default=0.0),
        generation_time_std=float(np.std(generation_times)) if generation_times else 0.0,
        raw_bytes=raw_bytes,
        accs=accs,
        reward_scores=reward_scores,
        response_hashes="",
        response_lengths=lengths,
        min_versions=min_versions,
        max_versions=max_versions,
        resume_counts=resume_counts,
        generation_times=generation_times,
    )
    return asdict(summary)


def _scan_response_profile(task: tuple[str, int, str]) -> dict[str, Any]:
    run_name, feed_step, group_path_text = task
    group_path = Path(group_path_text)
    response_hashes: list[str] = []
    for attempt_index in range(ROLLOUT_N):
        data_path = group_path / f"attempt_{attempt_index}" / "gen_batch.dp"
        data = DataProto.load_from_disk(data_path)
        response_mask = data.batch["response_mask"]
        response_hashes.append(_response_hash(data, response_mask))
    return {
        "run": run_name,
        "feed_step": feed_step,
        "response_hashes": ";".join(response_hashes),
        "unique_response_count": len(set(response_hashes)),
    }


def _enrich_selected_responses(manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    tasks = [(str(row.run), int(row.feed_step), str(row.source_group)) for row in manifest.itertuples(index=False)]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, maxtasksperchild=1000) as pool:
        profiles = list(pool.imap_unordered(_scan_response_profile, tasks, chunksize=8))
    profile_frame = pd.DataFrame(profiles)
    base = manifest.drop(columns=["response_hashes", "unique_response_count"])
    return base.merge(profile_frame, on=["run", "feed_step"], how="left", validate="one_to_one")


def _numeric_group_dirs(root: Path) -> list[Path]:
    groups = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    groups.sort(key=lambda path: int(path.name))
    return groups


def _load_prompt_metadata_map(dataset_path: Path) -> dict[str, dict[str, Any]]:
    dataset = pd.read_parquet(
        dataset_path,
        columns=["prompt", "prompt_uid", "extra_info", "tau_messages"],
    )
    mapping: dict[str, dict[str, Any]] = {}
    for row in dataset.itertuples(index=False):
        prompt_hash = _canonical_prompt_hash(row.prompt)
        prompt_messages = _json_normalize(row.prompt)
        tau_messages = json.loads(str(row.tau_messages))
        extra_info = row.extra_info if isinstance(row.extra_info, dict) else {}
        metadata = {
            "prompt_uid": str(row.prompt_uid),
            "source_subtype": str(extra_info.get("source_data_source", "unknown")),
            "tau_char_length": len(str(row.tau_messages)),
            "tau_message_count": len(tau_messages),
            "prompt_char_length": sum(
                len(str(message.get("content", ""))) for message in prompt_messages if isinstance(message, dict)
            ),
        }
        previous = mapping.setdefault(prompt_hash, metadata)
        if previous != metadata:
            raise ValueError(f"prompt hash collision: {previous['prompt_uid']!r} and {metadata['prompt_uid']!r}")
    if len(mapping) != len(dataset):
        raise ValueError(f"dataset prompts are not unique: rows={len(dataset)} hashes={len(mapping)}")
    return mapping


def _write_census(
    spec: RunSpec,
    output_dir: Path,
    workers: int,
    chunk_size: int,
    max_groups: int | None,
) -> Path:
    suffix = f".sample_{max_groups}" if max_groups is not None else ""
    final_path = output_dir / f"census_{spec.name}{suffix}.parquet"
    if final_path.is_file():
        schema_names = set(pq.read_schema(final_path).names)
        required = {"reward_scores", "complete", "source_subtype", "response_hashes"}
        if not required <= schema_names:
            raise ValueError(f"incompatible census schema in {final_path}; missing {sorted(required - schema_names)}")
        print(f"[CENSUS_REUSE] {spec.name}: {final_path}", flush=True)
        return final_path

    partial_path = final_path.with_suffix(".partial.parquet")
    partial_path.unlink(missing_ok=True)
    groups = _numeric_group_dirs(spec.root)
    if max_groups is not None:
        groups = groups[:max_groups]
    total = len(groups)
    print(f"[CENSUS_START] {spec.name}: {total} groups, workers={workers}", flush=True)
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    started = time.time()
    tasks = ((spec.name, spec.run_id, str(path)) for path in groups)

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=workers, maxtasksperchild=2000) as pool:
            for index, row in enumerate(pool.imap_unordered(_scan_group, tasks, chunksize=8), start=1):
                buffer.append(row)
                if len(buffer) >= chunk_size or index == total:
                    table = pa.Table.from_pylist(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(partial_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    buffer.clear()
                if index % 1000 == 0 or index == total:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"[CENSUS_PROGRESS] {spec.name}: {index}/{total} ({index / elapsed:.1f} groups/s)",
                        flush=True,
                    )
    finally:
        if writer is not None:
            writer.close()

    if not partial_path.is_file():
        raise RuntimeError(f"census output was not created for {spec.name}")
    partial_path.replace(final_path)
    print(f"[CENSUS_DONE] {spec.name}: {final_path}", flush=True)
    return final_path


def _priority(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _choose_unique_prompt_rows(frame: pd.DataFrame, quota: int, namespace: str) -> pd.DataFrame:
    if frame.empty or quota <= 0:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    work["_prompt_priority"] = [
        _priority(namespace, row.run_id, row.prompt_hash) for row in work.itertuples(index=False)
    ]
    work["_row_priority"] = [
        _priority(namespace, row.run_id, row.prompt_hash, row.feed_step) for row in work.itertuples(index=False)
    ]
    work = work.sort_values(["_prompt_priority", "_row_priority"]).drop_duplicates("prompt_hash")
    return work.head(quota).drop(columns=["_prompt_priority", "_row_priority"])


def _add_reason(selected: dict[tuple[str, int], set[str]], rows: pd.DataFrame, reason: str) -> None:
    for row in rows.itertuples(index=False):
        selected.setdefault((str(row.run), int(row.feed_step)), set()).add(reason)


def _apply_target_budget(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
    target_groups: int,
) -> dict[tuple[str, int], set[str]]:
    """Fit the candidate union to a deterministic target without dropping core evidence."""
    if target_groups <= 0:
        raise ValueError("target_groups must be positive")
    if target_groups > len(frame):
        raise ValueError(f"target_groups={target_groups} exceeds census rows={len(frame)}")

    row_by_key = {(str(row.run), int(row.feed_step)): row for row in frame.itertuples(index=False)}

    def order(keys: set[tuple[str, int]], namespace: str) -> list[tuple[str, int]]:
        return sorted(
            keys,
            key=lambda key: _priority(
                namespace,
                row_by_key[key].run_id,
                row_by_key[key].prompt_hash,
                key[1],
            ),
        )

    candidate_keys = set(selected)
    mandatory = {
        key for key, reasons in selected.items() if any(not reason.startswith("stratum_") for reason in reasons)
    }
    if len(mandatory) > target_groups:
        raise ValueError(
            f"core/event candidates ({len(mandatory)}) exceed target_groups={target_groups}; "
            "raise the target or lower event quotas"
        )

    kept = set(mandatory)
    strata: dict[str, set[tuple[str, int]]] = {}
    for key, reasons in selected.items():
        for reason in reasons:
            if reason.startswith("stratum_"):
                strata.setdefault(reason, set()).add(key)
    if len(kept) + sum(not bool(keys & kept) for keys in strata.values()) > target_groups:
        raise ValueError("target is too small to retain core events and one row per stratum")
    stratum_queues = {reason: order(keys, f"coverage:{reason}") for reason, keys in sorted(strata.items())}
    for reason, keys in sorted(strata.items()):
        if not keys & kept:
            kept.add(stratum_queues[reason][0])

    while len(kept) < target_groups:
        added = False
        for reason in sorted(stratum_queues):
            queue = stratum_queues[reason]
            while queue and queue[0] in kept:
                queue.pop(0)
            if not queue:
                continue
            kept.add(queue.pop(0))
            added = True
            if len(kept) == target_groups:
                break
        if not added:
            break

    for key in order(candidate_keys - kept, "candidate_fill"):
        if len(kept) == target_groups:
            break
        kept.add(key)

    if len(kept) < target_groups:
        all_keys = set(row_by_key)
        for key in order(all_keys - kept, "census_fill"):
            if len(kept) == target_groups:
                break
            selected.setdefault(key, set()).add("target_fill")
            kept.add(key)

    return {key: set(selected[key]) for key in kept}


def _select_strata(frame: pd.DataFrame, selected: dict[tuple[str, int], set[str]], quota: int) -> None:
    work = frame.copy()
    work["pv_bin"] = (work["max_param_version"] // 5).astype(int)
    for (run, pv_bin, k_class), group in work.groupby(["run", "pv_bin", "k_class"], sort=True):
        rows = _choose_unique_prompt_rows(group, quota, f"stratum:{run}:{pv_bin}:{k_class}")
        _add_reason(
            selected,
            rows,
            f"stratum_{run}_pv{pv_bin * 5:03d}_{pv_bin * 5 + 4:03d}_{k_class}",
        )


PAIR_WINDOWS = (
    ("c2_stable", "M5", "nocispo", 45, 69),
    ("c2_pre_storm", "M5", "nocispo", 70, 77),
    ("c2_storm", "M5", "nocispo", 78, 85),
    ("c2_recovery", "M5", "nocispo", 86, 95),
)


def _select_paired(frame: pd.DataFrame, selected: dict[tuple[str, int], set[str]], quota: int) -> None:
    for name, left_run, right_run, lo, hi in PAIR_WINDOWS:
        left = frame[(frame.run == left_run) & frame.max_param_version.between(lo, hi)].copy()
        right = frame[(frame.run == right_run) & frame.max_param_version.between(lo, hi)].copy()
        common = sorted(set(left.prompt_hash) & set(right.prompt_hash), key=lambda prompt: _priority(name, prompt))
        common = common[:quota]
        center = (lo + hi) / 2.0
        for run_name, run_frame in ((left_run, left), (right_run, right)):
            cohort = run_frame[run_frame.prompt_hash.isin(common)].copy()
            cohort["_distance"] = (cohort.max_param_version - center).abs()
            cohort["_priority"] = [
                _priority(name, run_name, row.prompt_hash, row.feed_step) for row in cohort.itertuples(index=False)
            ]
            cohort = (
                cohort.sort_values(["prompt_hash", "_distance", "_priority"])
                .drop_duplicates("prompt_hash")
                .drop(columns=["_distance", "_priority"])
            )
            _add_reason(selected, cohort, f"paired_{name}")


def _select_teacher_panel(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
    prompt_quota: int,
) -> None:
    cohorts = {
        ("nocispo", "early"): frame[(frame.run == "nocispo") & frame.max_param_version.between(20, 50)],
        ("RLonly", "early"): frame[(frame.run == "RLonly") & frame.max_param_version.between(20, 50)],
        ("nocispo", "late"): frame[(frame.run == "nocispo") & frame.max_param_version.between(130, 160)],
        ("RLonly", "late"): frame[(frame.run == "RLonly") & frame.max_param_version.between(130, 160)],
    }
    common = set.intersection(*(set(cohort.prompt_hash) for cohort in cohorts.values()))
    prompts = sorted(common, key=lambda prompt: _priority("teacher_panel", prompt))[:prompt_quota]
    for (run_name, phase), cohort in cohorts.items():
        center = 35.0 if phase == "early" else 145.0
        rows = cohort[cohort.prompt_hash.isin(prompts)].copy()
        rows["_distance"] = (rows.max_param_version - center).abs()
        rows["_priority"] = [
            _priority("teacher_panel", run_name, phase, row.prompt_hash, row.feed_step)
            for row in rows.itertuples(index=False)
        ]
        rows = rows.sort_values(["prompt_hash", "_distance", "_priority"]).drop_duplicates("prompt_hash")
        _add_reason(selected, rows, f"paired_teacher_{phase}_{run_name}")


def _trajectory_class(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(["max_param_version", "feed_step"])
    classes = ordered.k_class.tolist()
    if ordered.assign(pv_bin=ordered.max_param_version // 5).pv_bin.nunique() < 2:
        return "insufficient"
    if all(value == "k0" for value in classes):
        return "persistent_k0"
    if all(value == "k1_7" for value in classes):
        return "persistent_mixed"
    if all(value != "k0" for value in classes):
        return "persistent_positive"
    positive = [value != "k0" for value in classes]
    changes = sum(left != right for left, right in zip(positive, positive[1:], strict=False))
    if changes >= 2:
        return "oscillatory"
    if not positive[0] and any(positive[1:]):
        return "improvement"
    if positive[0] and not all(positive[1:]):
        return "regression"
    return "other_transition"


def _policy_phase(run_name: str, version: int) -> str:
    if run_name == "M5":
        if version < 70:
            return "stable"
        if version <= 77:
            return "pre_storm"
        if version <= 85:
            return "storm"
        if version <= 95:
            return "recovery"
        return "post_storm"
    if version <= 50:
        return "early"
    if version >= 130:
        return "late"
    return "middle"


def _annotate_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    ordered = work.sort_values(["run", "prompt_hash", "max_param_version", "feed_step"]).copy()
    keys = ["run", "prompt_hash"]
    ordered["_pv_bin"] = (ordered.max_param_version // 5).astype(int)
    ordered["_positive"] = ordered.k_class != "k0"
    ordered["_mixed"] = ordered.k_class == "k1_7"
    previous = ordered.groupby(keys, sort=False)._positive.shift()
    ordered["_change"] = previous.notna() & ordered._positive.ne(previous)
    groups = ordered.groupby(keys, sort=False)
    profile = groups.agg(
        bins=("_pv_bin", "nunique"),
        any_positive=("_positive", "max"),
        all_positive=("_positive", "min"),
        all_mixed=("_mixed", "min"),
        first_positive=("_positive", "first"),
        changes=("_change", "sum"),
    )
    profile["trajectory_class"] = np.select(
        [
            profile.bins < 2,
            ~profile.any_positive,
            profile.all_mixed,
            profile.all_positive,
            profile.changes >= 2,
            ~profile.first_positive & profile.any_positive,
            profile.first_positive & ~profile.all_positive,
        ],
        [
            "insufficient",
            "persistent_k0",
            "persistent_mixed",
            "persistent_positive",
            "oscillatory",
            "improvement",
            "regression",
        ],
        default="other_transition",
    )
    return work.merge(
        profile[["trajectory_class"]].reset_index(),
        on=keys,
        how="left",
        validate="many_to_one",
    )


def _select_longitudinal(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
    prompts_per_class: int = 100,
) -> None:
    eligible = frame[frame.trajectory_class != "insufficient"]
    for (run_name, trajectory_class), class_frame in eligible.groupby(["run", "trajectory_class"], sort=True):
        prompt_hashes = sorted(
            class_frame.prompt_hash.unique(),
            key=lambda prompt: _priority("longitudinal", run_name, trajectory_class, prompt),
        )[:prompts_per_class]
        for prompt_hash in prompt_hashes:
            rows = class_frame[class_frame.prompt_hash == prompt_hash].sort_values(["max_param_version", "feed_step"])
            positions = sorted({0, len(rows) // 2, len(rows) - 1})
            phases = ("early", "late") if len(positions) == 2 else ("early", "mid", "late")
            for phase, position in zip(phases, positions, strict=True):
                _add_reason(
                    selected,
                    rows.iloc[[position]],
                    f"longitudinal_{run_name}_{trajectory_class}_{phase}",
                )


HOLDOUT_QUOTAS = {"M5": 197, "nocispo": 399, "RLonly": 404}


def _select_random_holdout(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
) -> dict[str, tuple[int, int]]:
    populations: dict[str, tuple[int, int]] = {}
    for run_name, quota in HOLDOUT_QUOTAS.items():
        run_frame = frame[frame.run == run_name].copy()
        run_frame["_priority"] = [
            _priority("random_holdout_v1", run_name, row.feed_step) for row in run_frame.itertuples(index=False)
        ]
        rows = run_frame.sort_values("_priority").head(quota).drop(columns=["_priority"])
        _add_reason(selected, rows, "random_holdout")
        populations[run_name] = (len(run_frame), len(rows))
    return populations


def _select_analysis_coverage(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
) -> None:
    for run_name, run_frame in frame.groupby("run", sort=True):
        for source_subtype, source_frame in run_frame.groupby("source_subtype", sort=True):
            rows = _choose_unique_prompt_rows(
                source_frame,
                64,
                f"source:{run_name}:{source_subtype}",
            )
            _add_reason(selected, rows, f"source_{run_name}_{source_subtype}")

        for column, label in (
            ("tau_char_length", "tau_length"),
            ("prompt_token_length", "prompt_length"),
        ):
            low = run_frame[column].quantile(0.01)
            high = run_frame[column].quantile(0.99)
            _add_reason(
                selected,
                _choose_unique_prompt_rows(run_frame[run_frame[column] <= low], 64, f"{label}_low:{run_name}"),
                f"{label}_low_tail",
            )
            _add_reason(
                selected,
                _choose_unique_prompt_rows(run_frame[run_frame[column] >= high], 64, f"{label}_high:{run_name}"),
                f"{label}_high_tail",
            )


def _select_matched_event_pairs(
    frame: pd.DataFrame,
    selected: dict[tuple[str, int], set[str]],
) -> None:
    work = frame.copy()
    work["match_pv_bin"] = (work.max_param_version // 5).astype(int)
    work["match_prompt_bin"] = (work.prompt_token_length // 128).astype(int)
    work["match_tau_bin"] = (work.tau_char_length // 4000).astype(int)
    work["match_response_bin"] = (work.response_length_max // 1024).astype(int)

    for run_name, run_frame in work.groupby("run", sort=True):
        nontruncated = run_frame[run_frame.truncated_attempts == 0]
        long_p99 = nontruncated.response_length_max.quantile(0.99)
        long_p90 = nontruncated.response_length_max.quantile(0.90)
        latency_p99 = run_frame.generation_time_max.quantile(0.99)
        latency_p90 = run_frame.generation_time_max.quantile(0.90)
        specs = (
            ("partial", run_frame.partial_attempts > 0, run_frame.partial_attempts == 0, 100, False),
            ("span_gt1", run_frame.partial_span > 1, run_frame.partial_span == 0, 34, False),
            (
                "truncated_correct",
                run_frame.truncated_correct_attempts > 0,
                run_frame.truncated_attempts == 0,
                100,
                False,
            ),
            (
                "truncated_wrong",
                (run_frame.truncated_attempts > 0) & (run_frame.truncated_correct_attempts == 0),
                run_frame.truncated_attempts == 0,
                100,
                False,
            ),
            (
                "gate_mismatch",
                run_frame.raw_correct_attempts != run_frame.correct_attempts,
                run_frame.raw_correct_attempts == run_frame.correct_attempts,
                67,
                False,
            ),
            (
                "long_nontruncated",
                (run_frame.truncated_attempts == 0) & (run_frame.response_length_max >= long_p99),
                (run_frame.truncated_attempts == 0) & (run_frame.response_length_max <= long_p90),
                67,
                False,
            ),
            (
                "latency_p99",
                run_frame.generation_time_max >= latency_p99,
                run_frame.generation_time_max <= latency_p90,
                67,
                True,
            ),
        )
        for event_name, case_mask, control_mask, quota, match_response in specs:
            cases = _choose_unique_prompt_rows(run_frame[case_mask], quota, f"event_case:{event_name}:{run_name}")
            controls = run_frame[control_mask].copy()
            controls["_priority"] = [
                _priority("event_control", event_name, run_name, row.prompt_hash, row.feed_step)
                for row in controls.itertuples(index=False)
            ]
            controls = controls.sort_values("_priority")
            match_columns = [
                "match_pv_bin",
                "k_class",
                "source_subtype",
                "match_prompt_bin",
                "match_tau_bin",
            ]
            if match_response:
                match_columns.append("match_response_bin")
            exact_buckets: dict[tuple[Any, ...], list[Any]] = {}
            relaxed_buckets: dict[tuple[Any, ...], list[Any]] = {}
            for control_row in controls.itertuples(index=False):
                exact_key = tuple(getattr(control_row, column) for column in match_columns)
                relaxed_key = (control_row.match_pv_bin, control_row.k_class)
                exact_buckets.setdefault(exact_key, []).append(control_row)
                relaxed_buckets.setdefault(relaxed_key, []).append(control_row)
            for bucket in [*exact_buckets.values(), *relaxed_buckets.values()]:
                bucket.reverse()

            used_controls: set[tuple[str, int]] = set()

            def pick_control(
                bucket: list[Any],
                case_prompt_hash: str,
                used: set[tuple[str, int]] = used_controls,
            ) -> Any | None:
                while bucket:
                    candidate = bucket.pop()
                    candidate_key = (str(candidate.run), int(candidate.feed_step))
                    if candidate_key not in used and candidate.prompt_hash != case_prompt_hash:
                        return candidate
                return None

            for case in cases.itertuples(index=False):
                exact_key = tuple(getattr(case, column) for column in match_columns)
                relaxed_key = (case.match_pv_bin, case.k_class)
                control_row = pick_control(exact_buckets.get(exact_key, []), str(case.prompt_hash))
                if control_row is None:
                    control_row = pick_control(
                        relaxed_buckets.get(relaxed_key, []),
                        str(case.prompt_hash),
                    )
                if control_row is None:
                    continue
                control_key = (str(control_row.run), int(control_row.feed_step))
                used_controls.add(control_key)
                pair_id = _priority(event_name, run_name, case.feed_step, control_key[1])[:16]
                _add_reason(
                    selected,
                    pd.DataFrame([case._asdict()]),
                    f"event_{event_name}_case_{pair_id}",
                )
                _add_reason(
                    selected,
                    pd.DataFrame([control_row._asdict()]),
                    f"event_{event_name}_control_{pair_id}",
                )


def _select_events(frame: pd.DataFrame, selected: dict[tuple[str, int], set[str]]) -> None:
    for run, run_frame in frame.groupby("run", sort=True):
        partial = run_frame[run_frame.partial_attempts > 0]
        _add_reason(selected, _choose_unique_prompt_rows(partial, 512, f"partial:{run}"), "partial")

        truncated = run_frame[run_frame.truncated_attempts > 0]
        _add_reason(selected, _choose_unique_prompt_rows(truncated, 512, f"truncated:{run}"), "truncated")

        nontruncated = run_frame[run_frame.truncated_attempts == 0].copy()
        if not nontruncated.empty:
            threshold = nontruncated.response_length_max.quantile(0.99)
            long_tail = _choose_unique_prompt_rows(
                nontruncated[nontruncated.response_length_max >= threshold],
                256,
                f"long_tail_p99:{run}",
            )
            _add_reason(selected, long_tail, "long_response_tail")

        latency_threshold = run_frame.generation_time_max.quantile(0.99)
        latency = _choose_unique_prompt_rows(
            run_frame[run_frame.generation_time_max >= latency_threshold],
            256,
            f"latency_tail_p99:{run}",
        )
        _add_reason(selected, latency, "latency_tail")

        spans = run_frame[run_frame.partial_span > 1]
        _add_reason(
            selected,
            _choose_unique_prompt_rows(spans, 256, f"partial_span:{run}"),
            "partial_span_gt_1",
        )

    storm = frame[(frame.run == "M5") & frame.max_param_version.between(78, 85)]
    _add_reason(selected, _choose_unique_prompt_rows(storm, 1024, "M5_storm_78_85"), "storm_M5_pv78_85")

    plateau = frame[(frame.run == "nocispo") & frame.max_param_version.between(160, 190)]
    _add_reason(
        selected,
        _choose_unique_prompt_rows(plateau, 256, "nocispo_plateau_160_190"),
        "plateau_nocispo_pv160_190",
    )


def _build_manifest(
    census_paths: list[Path],
    output_dir: Path,
    stratum_quota: int,
    pair_quota: int,
    target_groups: int,
    archive_shards: int,
    enrichment_workers: int,
    enrich_responses: bool,
) -> Path:
    frames = [pd.read_parquet(path) for path in census_paths]
    census = pd.concat(frames, ignore_index=True)
    incomplete = census[~census.complete]
    if not incomplete.empty:
        print(
            f"[INCOMPLETE_GROUPS] total={len(incomplete)} by_run={incomplete.groupby('run').size().to_dict()}",
            flush=True,
        )
    frame = census[census.complete].copy()
    frame["resumed_attempts"] = frame.resume_counts.map(lambda values: sum(int(value) > 0 for value in values))
    frame["cross_version_attempts"] = [
        sum(int(minimum) != int(maximum) for minimum, maximum in zip(mins, maxes, strict=True))
        for mins, maxes in zip(frame.min_versions, frame.max_versions, strict=True)
    ]
    frame["pv_bin"] = (frame.max_param_version // 5).astype(int)
    frame = _annotate_trajectories(frame)
    selected: dict[tuple[str, int], set[str]] = {}
    _select_strata(frame, selected, stratum_quota)
    _select_paired(frame, selected, pair_quota)
    _select_teacher_panel(frame, selected, prompt_quota=500)
    _select_longitudinal(frame, selected)
    holdout_populations = _select_random_holdout(frame, selected)
    _select_analysis_coverage(frame, selected)
    _select_matched_event_pairs(frame, selected)
    selected = _apply_target_budget(frame, selected, target_groups)

    selected_keys = pd.DataFrame(
        [(run, feed_step, ";".join(sorted(reasons))) for (run, feed_step), reasons in selected.items()],
        columns=["run", "feed_step", "selection_reason"],
    )
    manifest = frame.merge(selected_keys, on=["run", "feed_step"], how="inner", validate="one_to_one")
    manifest["raw_k"] = manifest.raw_correct_attempts
    manifest["clean_k"] = manifest.correct_attempts
    manifest["gate_k"] = np.where(manifest.run == "RLonly", ROLLOUT_N, manifest.clean_k)
    manifest["actual_route"] = np.where(
        (manifest.run != "RLonly") & (manifest.gate_k == 0),
        "sft",
        "rl",
    )
    manifest["clean_correct_attempts"] = manifest.correct_attempts
    manifest["truncated_wrong_attempts"] = manifest.truncated_attempts - manifest.truncated_correct_attempts
    manifest["source_data_source"] = manifest.source_subtype
    manifest["policy_phase"] = [
        _policy_phase(str(row.run), int(row.max_param_version)) for row in manifest.itertuples(index=False)
    ]
    manifest["matched_event_tags"] = [
        ";".join(reason for reason in str(value).split(";") if reason.startswith("event_"))
        for value in manifest.selection_reason
    ]
    manifest["is_random_holdout"] = manifest.selection_reason.str.contains(r"(?:^|;)random_holdout(?:;|$)", regex=True)
    manifest["holdout_inclusion_probability"] = [
        holdout_populations[str(row.run)][1] / holdout_populations[str(row.run)][0] if row.is_random_holdout else np.nan
        for row in manifest.itertuples(index=False)
    ]
    manifest["holdout_sample_weight"] = 1.0 / manifest.holdout_inclusion_probability
    population = (
        frame.groupby(["run", "pv_bin", "k_class"], sort=True).size().rename("stratum_population").reset_index()
    )
    manifest = manifest.merge(population, on=["run", "pv_bin", "k_class"], how="left", validate="many_to_one")
    selected_counts = (
        manifest.groupby(["run", "pv_bin", "k_class"], sort=True).size().rename("stratum_selected").reset_index()
    )
    manifest = manifest.merge(
        selected_counts,
        on=["run", "pv_bin", "k_class"],
        how="left",
        validate="many_to_one",
    )
    manifest["stratum_sample_fraction"] = manifest.stratum_selected / manifest.stratum_population
    if enrich_responses:
        print(
            f"[RESPONSE_ENRICH_START] groups={len(manifest)} workers={enrichment_workers}",
            flush=True,
        )
        manifest = _enrich_selected_responses(manifest, workers=enrichment_workers)
        print("[RESPONSE_ENRICH_DONE]", flush=True)
    manifest = manifest.sort_values(["run", "max_param_version", "feed_step"]).reset_index(drop=True)
    manifest = _assign_archive_shards(manifest, archive_shards)
    manifest_path = output_dir / "selection_manifest.parquet"
    manifest.to_parquet(manifest_path, compression="zstd", index=False)
    csv_manifest = manifest.drop(
        columns=[
            "accs",
            "reward_scores",
            "response_hashes",
            "response_lengths",
            "min_versions",
            "max_versions",
            "resume_counts",
            "generation_times",
            "missing_attempts",
        ]
    )
    csv_manifest.to_csv(output_dir / "selection_manifest.csv", index=False)
    for run_name, run_frame in manifest.groupby("run", sort=True):
        run_frame.to_parquet(output_dir / f"selection_manifest_{run_name}.parquet", compression="zstd", index=False)
        csv_manifest[csv_manifest.run == run_name].to_csv(
            output_dir / f"selection_manifest_{run_name}.csv", index=False
        )
    print("[SELECTION_COUNTS]", flush=True)
    print(manifest.groupby("run").size().to_string(), flush=True)
    print(f"[SELECTION_TOTAL] {len(manifest)} groups", flush=True)
    return manifest_path


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_archive_shards(manifest: pd.DataFrame, archive_shards: int) -> pd.DataFrame:
    if archive_shards <= 0:
        raise ValueError("archive_shards must be positive")
    assigned = manifest.copy()
    assigned["archive_shard"] = 0
    assigned["archive_name"] = ""
    assigned["archive_member"] = assigned.feed_step.astype(str)
    for run_name, run_frame in assigned.groupby("run", sort=True):
        shard_count = min(archive_shards, len(run_frame))
        heap = [(0, shard_index) for shard_index in range(1, shard_count + 1)]
        heapq.heapify(heap)
        order = sorted(
            run_frame.index,
            key=lambda index: (
                -int(assigned.at[index, "raw_bytes"]),
                _priority(
                    "archive_shard",
                    assigned.at[index, "run_id"],
                    assigned.at[index, "prompt_hash"],
                    assigned.at[index, "feed_step"],
                ),
            ),
        )
        for index in order:
            shard_bytes, shard_index = heapq.heappop(heap)
            assigned.at[index, "archive_shard"] = shard_index
            assigned.at[index, "archive_name"] = (
                f"rollout_sample_{run_name}.part-{shard_index:02d}-of-{shard_count:02d}.tar.zst"
            )
            heapq.heappush(heap, (shard_bytes + int(assigned.at[index, "raw_bytes"]), shard_index))
    return assigned


def _group_archive_members(group_path: Path, run_root: Path) -> list[str]:
    members: list[str] = []
    for attempt_index in range(ROLLOUT_N):
        attempt = group_path / f"attempt_{attempt_index}"
        data_path = attempt / "gen_batch.dp"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        members.append(str(data_path.relative_to(run_root)))
        meta_path = attempt / "meta.json"
        if meta_path.is_file():
            members.append(str(meta_path.relative_to(run_root)))
    return sorted(members)


def _archive_shard(
    spec: RunSpec,
    archive_name: str,
    shard_rows: pd.DataFrame,
    output_dir: Path,
    zstd_level: int,
    zstd_threads: int,
) -> Path:
    archive_path = output_dir / archive_name
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    members: list[str] = []
    for source in shard_rows.source_group:
        members.extend(_group_archive_members(Path(source), spec.root))
    members = sorted(members)
    member_stem = archive_path.name.removesuffix(".tar.zst")
    text_list_path = output_dir / f"selected_files_{member_stem}.txt"
    null_list_path = output_dir / f"selected_files_{member_stem}.nul"
    text_list_path.write_text("\n".join(members) + "\n", encoding="utf-8")
    null_list_path.write_bytes(("\0".join(members) + "\0").encode("utf-8"))
    partial_path.unlink(missing_ok=True)
    print(
        f"[ARCHIVE_START] {spec.name}: {len(shard_rows)} groups/{len(members)} files -> {archive_path}",
        flush=True,
    )
    tar = subprocess.Popen(
        [
            "tar",
            "--format=posix",
            "--sort=name",
            "--numeric-owner",
            "--owner=0",
            "--group=0",
            "--mtime=@0",
            "--pax-option=delete=atime,delete=ctime",
            "--no-recursion",
            "--null",
            "-C",
            str(spec.root),
            "-T",
            str(null_list_path.resolve()),
            "-cf",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert tar.stdout is not None
    zstd = subprocess.Popen(
        ["zstd", f"-{zstd_level}", f"-T{zstd_threads}", "-f", "-o", str(partial_path)],
        stdin=tar.stdout,
    )
    tar.stdout.close()
    zstd_rc = zstd.wait()
    tar_rc = tar.wait()
    if tar_rc != 0 or zstd_rc != 0:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"archive failed for {spec.name}: tar={tar_rc}, zstd={zstd_rc}")
    partial_path.replace(archive_path)
    null_list_path.unlink(missing_ok=True)
    subprocess.run(["zstd", "-t", str(archive_path)], check=True, capture_output=True)
    listing = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-tf", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if listing != members:
        raise RuntimeError(f"archive member mismatch for {archive_path}: expected={len(members)} actual={len(listing)}")
    digest = _sha256_file(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    print(
        f"[ARCHIVE_DONE] {spec.name}: {archive_path.stat().st_size} bytes sha256={digest}",
        flush=True,
    )
    return archive_path


def _archive_run(
    spec: RunSpec,
    manifest: pd.DataFrame,
    output_dir: Path,
    zstd_level: int,
    zstd_threads: int,
) -> list[Path]:
    rows = manifest[manifest.run == spec.name]
    selected_groups = sorted(str(Path(source).relative_to(spec.root)) for source in rows.source_group)
    (output_dir / f"selected_groups_{spec.name}.txt").write_text("\n".join(selected_groups) + "\n", encoding="utf-8")
    shards = [(str(name), shard.copy()) for name, shard in rows.groupby("archive_name", sort=True)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(shards))) as executor:
        futures = [
            executor.submit(
                _archive_shard,
                spec,
                archive_name,
                shard_rows,
                output_dir,
                zstd_level,
                zstd_threads,
            )
            for archive_name, shard_rows in shards
        ]
        return [future.result() for future in futures]


def _write_sha256sums(output_dir: Path, paths: list[Path]) -> Path:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(set(paths), key=lambda item: item.name):
        lines.append(f"{_sha256_file(path)}  {path.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _create_control_archive(output_dir: Path, control_paths: list[Path], zstd_level: int, zstd_threads: int) -> Path:
    archive_path = output_dir / "rollout_selection_control.tar.zst"
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    members = sorted(path.name for path in control_paths)
    null_list_path = output_dir / ".control_files.nul"
    null_list_path.write_bytes(("\0".join(members) + "\0").encode("utf-8"))
    partial_path.unlink(missing_ok=True)
    tar = subprocess.Popen(
        [
            "tar",
            "--format=posix",
            "--sort=name",
            "--numeric-owner",
            "--owner=0",
            "--group=0",
            "--mtime=@0",
            "--pax-option=delete=atime,delete=ctime",
            "--no-recursion",
            "--null",
            "-C",
            str(output_dir),
            "-T",
            str(null_list_path.resolve()),
            "-cf",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert tar.stdout is not None
    zstd = subprocess.Popen(
        ["zstd", f"-{zstd_level}", f"-T{zstd_threads}", "-f", "-o", str(partial_path)],
        stdin=tar.stdout,
    )
    tar.stdout.close()
    zstd_rc = zstd.wait()
    tar_rc = tar.wait()
    null_list_path.unlink(missing_ok=True)
    if tar_rc != 0 or zstd_rc != 0:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"control archive failed: tar={tar_rc}, zstd={zstd_rc}")
    partial_path.replace(archive_path)
    subprocess.run(["zstd", "-t", str(archive_path)], check=True, capture_output=True)
    listing = subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-tf", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if listing != members:
        raise RuntimeError(f"control archive member mismatch: expected={members} actual={listing}")
    digest = _sha256_file(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path


def _default_specs(repo_root: Path) -> list[RunSpec]:
    rollout_root = repo_root / ".cache" / "rollout_dump"
    return [
        RunSpec(
            name="M5",
            run_id="f5ugxklh",
            root=rollout_root
            / "openr1_async_hpt_qwen25_math_1_5b_M5_20260708_195507"
            / "qwen25_math_1_5b_openr1_async_hpt_M5_cleanasync_async-hpt-openr1"
            / "GBS1_N8_in1536_out8192",
        ),
        RunSpec(
            name="nocispo",
            run_id="oki4kv8u",
            root=rollout_root
            / "openr1_async_hpt_qwen25_math_1_5b_M5abl_nocispo_20260709_232320"
            / "qwen25_math_1_5b_openr1_async_hpt_M5abl_nocispo_async-hpt-openr1"
            / "GBS1_N8_in1536_out8192",
        ),
        RunSpec(
            name="RLonly",
            run_id="qzsnwc08",
            root=rollout_root
            / "openr1_async_hpt_qwen25_math_1_5b_RLonly_20260710_065549"
            / "qwen25_math_1_5b_openr1_async_RLonly_grpo_async-hpt-openr1"
            / "GBS1_N8_in1536_out8192",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/migration/rollout_selection"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--stratum-quota", type=int, default=64)
    parser.add_argument("--pair-quota", type=int, default=250)
    parser.add_argument("--target-groups", type=int, default=20_000)
    parser.add_argument("--max-groups-per-run", type=int)
    parser.add_argument(
        "--prompt-dataset",
        type=Path,
        default=Path("datas/openr1_hpt_main_v2/train.parquet"),
    )
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--zstd-threads", type=int, default=4)
    parser.add_argument("--archive-shards", type=int, default=8)
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--skip-response-enrichment", action="store_true")
    return parser.parse_args()


def main() -> int:
    global _PROMPT_METADATA_BY_HASH

    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dataset = args.prompt_dataset
    if not prompt_dataset.is_absolute():
        prompt_dataset = repo_root / prompt_dataset
    _PROMPT_METADATA_BY_HASH = _load_prompt_metadata_map(prompt_dataset)
    print(
        f"[PROMPT_METADATA_MAP] {len(_PROMPT_METADATA_BY_HASH)} prompts from {prompt_dataset}",
        flush=True,
    )
    specs = _default_specs(repo_root)
    for spec in specs:
        if not spec.root.is_dir():
            raise FileNotFoundError(spec.root)

    census_paths = [
        _write_census(
            spec,
            output_dir,
            workers=args.workers,
            chunk_size=args.chunk_size,
            max_groups=args.max_groups_per_run,
        )
        for spec in specs
    ]
    if args.census_only:
        print(f"[CENSUS_ONLY_DONE] schema_version={CENSUS_SCHEMA_VERSION}", flush=True)
        return 0
    manifest_path = _build_manifest(
        census_paths,
        output_dir,
        stratum_quota=args.stratum_quota,
        pair_quota=args.pair_quota,
        target_groups=args.target_groups,
        archive_shards=args.archive_shards,
        enrichment_workers=min(args.workers, 48),
        enrich_responses=not args.skip_response_enrichment,
    )
    manifest = pd.read_parquet(manifest_path)
    census_summary = pd.concat(
        [pd.read_parquet(path, columns=["run", "complete"]) for path in census_paths],
        ignore_index=True,
    )

    summary = {
        "groups": int(len(manifest)),
        "attempts": int(len(manifest) * ROLLOUT_N),
        "raw_bytes": int(manifest.raw_bytes.sum()),
        "by_run": {str(key): int(value) for key, value in manifest.groupby("run").size().items()},
        "by_k_class": {
            f"{run}:{k_class}": int(value)
            for (run, k_class), value in manifest.groupby(["run", "k_class"]).size().items()
        },
        "archive_shards": int(manifest.archive_name.nunique()),
        "census_groups": int(len(census_summary)),
        "census_complete_groups": int(census_summary.complete.sum()),
        "census_incomplete_groups": int((~census_summary.complete).sum()),
        "census_by_run": {str(key): int(value) for key, value in census_summary.groupby("run").size().items()},
        "random_holdout_groups": int(manifest.is_random_holdout.sum()),
    }
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    tar_version = subprocess.run(["tar", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[
        0
    ]
    zstd_version = subprocess.run(
        ["zstd", "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "created_at_unix": time.time(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "rollout_n": ROLLOUT_N,
        "response_cap": RESPONSE_CAP,
        "stratum_quota": args.stratum_quota,
        "pair_quota": args.pair_quota,
        "target_groups": args.target_groups,
        "prompt_dataset": str(prompt_dataset),
        "prompt_dataset_sha256": _sha256_file(prompt_dataset),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "git_head": git_head,
        "census_schema_version": CENSUS_SCHEMA_VERSION,
        "archive_shards_per_run": args.archive_shards,
        "holdout_quotas": HOLDOUT_QUOTAS,
        "paired_windows": [list(window) for window in PAIR_WINDOWS],
        "analysis_limitations": [
            "tau is present for every prompt, so missing-tau fallback is not empirically identified",
            "rollout dumps lack learner version/current log-probs/advantages; use preserved W&B history for "
            "staleness and clip analyses",
            "population prevalence must be estimated from is_random_holdout rows, not the enriched sample",
        ],
        "zstd_level": args.zstd_level,
        "zstd_threads": args.zstd_threads,
        "response_hashes_enriched": not args.skip_response_enrichment,
        "tar_version": tar_version,
        "zstd_version": zstd_version,
        "runs": [{**asdict(spec), "root": str(spec.root)} for spec in specs],
    }
    metadata_path = output_dir / "selection_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    restore_path = output_dir / "RESTORE.md"
    restore_path.write_text(
        "# Rollout selection restore\n\n"
        "1. Verify the control archive with its `.sha256` sidecar and extract it.\n"
        "2. Put all rollout `*.tar.zst` shards beside `SHA256SUMS`.\n"
        "3. Run `python scripts/migration/verify_rollout_selection.py --selection-dir <dir> "
        "--archives-only`.\n"
        "4. Extract each run's eight shards into separate `M5`, `nocispo`, and `RLonly` roots.\n"
        "5. Verify the extracted DataProto groups with `python "
        "scripts/migration/verify_rollout_selection.py --selection-dir <dir> "
        "--run-root M5=<M5-root> --run-root nocispo=<nocispo-root> "
        "--run-root RLonly=<RLonly-root>`.\n",
        encoding="utf-8",
    )

    archive_paths: list[Path] = []
    if not args.no_archive:
        for spec in specs:
            archive_paths.extend(
                _archive_run(
                    spec,
                    manifest,
                    output_dir,
                    zstd_level=args.zstd_level,
                    zstd_threads=args.zstd_threads,
                )
            )

    control_paths = sorted(
        [
            *census_paths,
            *output_dir.glob("selection_manifest*"),
            *output_dir.glob("selected_groups_*.txt"),
            *output_dir.glob("selected_files_*.txt"),
            metadata_path,
            summary_path,
            restore_path,
        ],
        key=lambda path: path.name,
    )
    checksum_path = _write_sha256sums(output_dir, [*control_paths, *archive_paths])
    control_archive = _create_control_archive(
        output_dir,
        [*control_paths, checksum_path],
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
    )
    print(f"[CONTROL_ARCHIVE] {control_archive}", flush=True)
    print(f"[DONE] outputs in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
