#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

DATA_DIR="${DATA_DIR:-${VERL_ROOT}/../data/gsm8k}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/train.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_DIR}/test.parquet}"

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
    python examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"
fi

export NUM_GPUS=8
export N_GPUS_ROLLOUT=4
export N_GPUS_TRAINING=4
export ACTOR_STRATEGY=fsdp2
export ROLLOUT_NAME=sglang
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export TRAIN_FILES
export VAL_FILES

# This cluster mounts /tmp with noexec. SGLang 0.5.12's default NUMA V2 path
# creates executable wrapper scripts under /tmp for scheduler subprocesses.
# Use SGLang's in-process NUMA binding path instead.
export SGLANG_NUMA_BIND_V2=0

export N_RESP_PER_PROMPT=2
export TRAIN_PROMPT_MINI_BSZ=2
# Fully async derives train progress as
# total_rollout_steps / (required_samples * trigger_parameter_sync_step).
# With this smoke's 2 required samples and the base script's 4-step sync
# interval, 80 rollout steps exercise 10 trainer progress cycles and cross
# multiple parameter-sync boundaries.
export TOTAL_ROLLOUT_STEPS=80
export VAL_BEFORE_TRAIN=False
export TRAINER_TOTAL_EPOCHS=10
export TEST_FREQ=-1

exec bash tests/special_e2e/run_fully_async_policy.sh "$@"
