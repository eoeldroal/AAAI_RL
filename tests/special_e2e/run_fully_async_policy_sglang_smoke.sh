#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

OPENR1_HPT_DATA_DIR="${OPENR1_HPT_DATA_DIR:-${VERL_ROOT}/datas/openr1_hpt_smoke}"
TRAIN_FILES="${TRAIN_FILES:-${OPENR1_HPT_DATA_DIR}/train.parquet}"
VAL_FILES="${VAL_FILES:-${OPENR1_HPT_DATA_DIR}/test.parquet}"
OPENR1_HPT_METADATA_FILE="${OPENR1_HPT_DATA_DIR}/metadata.json"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

if ! python - "${TRAIN_FILES}" "${VAL_FILES}" "${OPENR1_HPT_METADATA_FILE}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

train_path, val_path, metadata_path = map(Path, sys.argv[1:])
if not train_path.exists() or not val_path.exists() or not metadata_path.exists():
    raise SystemExit(1)

train = pd.read_parquet(train_path)
required = {"prompt_uid", "tau_messages"}
if not required.issubset(train.columns):
    raise SystemExit(1)
if train["tau_messages"].isna().any():
    raise SystemExit(1)
if train["tau_messages"].astype(str).str.len().eq(0).any():
    raise SystemExit(1)
PY
then
    rm -f "${TRAIN_FILES}" "${VAL_FILES}" "${OPENR1_HPT_METADATA_FILE}"
    python examples/data_preprocess/openr1_hpt.py \
        --local_save_dir "${OPENR1_HPT_DATA_DIR}" \
        --normalize_data_source numina_olympiads \
        --tokenizer_path "${MODEL_PATH}" \
        --max_prompt_tokens 1024 \
        --max_response_tokens 2048
fi

export NUM_GPUS=8
export N_GPUS_ROLLOUT=4
export N_GPUS_TRAINING=4
export ACTOR_STRATEGY=fsdp2
export ROLLOUT_NAME=sglang
export MODEL_PATH
export TRAIN_FILES
export VAL_FILES

# This cluster mounts /tmp with noexec. SGLang 0.5.12's default NUMA V2 path
# creates executable wrapper scripts under /tmp for scheduler subprocesses.
# Use SGLang's in-process NUMA binding path instead.
export SGLANG_NUMA_BIND_V2=0

export N_RESP_PER_PROMPT=4
export TRAIN_PROMPT_MINI_BSZ=4
# Fully async caps rollout steps by len(train_dataloader) * trainer.total_epochs.
# The generated OpenR1 HPT smoke dataset has 12 train rows. The strict HPT smoke
# keeps tau_messages in the train parquet itself and points the tau lookup at
# the same file. With gamma=1.0 and fail_on_missing_tau=True, this exercises the
# HPT SFT route without relying on missing-tau fallback.
export TOTAL_ROLLOUT_STEPS=7680
export VAL_BEFORE_TRAIN=False
export TRAINER_TOTAL_EPOCHS=640
export TEST_FREQ=-1
export PARTIAL_ROLLOUT=True
# HPT learner-row assembly can consume more queue samples than the base
# required_samples per update. Keep the smoke's sync interval at 4, bound the
# completed queue at 256 samples, and set staleness above that queue cap so the
# completed-queue budget, not staleness pause, is the first backpressure point.
export STALENESS_THRESHOLD=19.0

exec bash tests/special_e2e/run_fully_async_policy.sh \
    async_training.partial_rollout=True \
    async_training.max_inflight_prompt_groups=32 \
    async_training.max_completed_prompt_groups=256 \
    async_hpt.enabled=True \
    async_hpt.gamma=1.0 \
    async_hpt.tau_dataset_path="${TRAIN_FILES}" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    algorithm.norm_adv_by_std_in_grpo=False \
    "$@"
