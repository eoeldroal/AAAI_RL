#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

OPENR1_HPT_DATA_DIR="${OPENR1_HPT_DATA_DIR:-${VERL_ROOT}/../data/openr1_hpt_smoke}"
TRAIN_FILES="${TRAIN_FILES:-${OPENR1_HPT_DATA_DIR}/train.parquet}"
VAL_FILES="${VAL_FILES:-${OPENR1_HPT_DATA_DIR}/test.parquet}"
OPENR1_HPT_TAU_FILE="${OPENR1_HPT_TAU_FILE:-${OPENR1_HPT_DATA_DIR}/tau.parquet}"
OPENR1_HPT_METADATA_FILE="${OPENR1_HPT_DATA_DIR}/metadata.json"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ] || [ ! -f "${OPENR1_HPT_TAU_FILE}" ] || [ ! -f "${OPENR1_HPT_METADATA_FILE}" ]; then
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
export OPENR1_HPT_TAU_FILE

# This cluster mounts /tmp with noexec. SGLang 0.5.12's default NUMA V2 path
# creates executable wrapper scripts under /tmp for scheduler subprocesses.
# Use SGLang's in-process NUMA binding path instead.
export SGLANG_NUMA_BIND_V2=0

export N_RESP_PER_PROMPT=4
export TRAIN_PROMPT_MINI_BSZ=4
# Fully async derives train progress as
# total_rollout_steps / (required_samples * trigger_parameter_sync_step).
# With this smoke's 4 required samples and the base script's 4-step sync
# interval, 480 rollout steps exercise 30 trainer progress cycles and cross
# many parameter-sync / partial-rollout boundaries.
export TOTAL_ROLLOUT_STEPS=480
export VAL_BEFORE_TRAIN=False
export TRAINER_TOTAL_EPOCHS=30
export TEST_FREQ=-1
export PARTIAL_ROLLOUT=True
export STALENESS_THRESHOLD=1.0

exec bash tests/special_e2e/run_fully_async_policy.sh \
    async_training.partial_rollout=True \
    async_training.max_inflight_prompt_groups=16 \
    async_training.max_completed_prompt_groups=32 \
    async_hpt.enabled=True \
    async_hpt.gamma=1.0 \
    async_hpt.tau_dataset_path="${OPENR1_HPT_TAU_FILE}" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=False \
    async_hpt.trajectory_scheduler.enabled=True \
    algorithm.norm_adv_by_std_in_grpo=False \
    "$@"
