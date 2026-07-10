#!/usr/bin/env bash
# Phase-0 smoke: VANILLA synchronous GRPO (no HPT) to confirm the v0 RayPPOTrainer
# path runs on this box (torch2.11 / sglang0.5.12 / transformers5 / B200) BEFORE
# wiring the paper-HPT routing into fit(). ~20 steps, no val, no wandb, no ckpt.
#
# Entry: verl.trainer.main_ppo with trainer.use_v1=false -> main_ppo_v0.TaskRunner
# -> RayPPOTrainer (the class recipe/paper_hpt subclasses). Config values for
# model / data / reward / sglang / model-override are borrowed from the proven
# async launcher; async/HPT knobs are dropped.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

# RL conda env exposes the CUDA13/cuDNN/NCCL wheel stack via an activation hook;
# source it directly so SGLang subprocesses see the same loader path under nohup.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

# Keep JIT / tmp artifacts on an exec-friendly, short path (cluster /tmp may be noexec;
# Ray AF_UNIX socket path length is limited).
RUNTIME_CACHE_DIR="$(cd "${VERL_ROOT}/.." && pwd)/.rt"
mkdir -p "${RUNTIME_CACHE_DIR}"/{tmp,triton,torchinductor,xdg}
export TMPDIR="${RUNTIME_CACHE_DIR}/tmp"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE_DIR}/torchinductor"
export XDG_CACHE_HOME="${RUNTIME_CACHE_DIR}/xdg"

DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main"          # strip (paper-faithful)
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-1.5B"

SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m verl.trainer.main_ppo \
    trainer.use_v1=false \
    data.train_files="${DATA_DIR}/train.parquet" \
    "data.val_files=['${DATA_DIR}/AIME24/test.parquet','${DATA_DIR}/AIME25/test.parquet','${DATA_DIR}/AMC23/test.parquet','${DATA_DIR}/MATH-500/test.parquet','${DATA_DIR}/Minerva/test.parquet','${DATA_DIR}/Olympiad-Bench/test.parquet']" \
    data.val_batch_size=512 \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=1536 \
    data.max_response_length=8192 \
    data.train_batch_size=16 \
    data.shuffle=True \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=16384 \
    +actor_rollout_ref.model.override_config.rope_theta=40000 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.clip_grad=80.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=9728 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=9728 \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.json_model_override_args='{\"max_position_embeddings\":16384,\"rope_theta\":40000}'" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=verl/utils/reward_score/math_verify_adapter.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.timeout=30.0 \
    +reward.reward_kwargs.compute_score_in_executor=True \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=8192 \
    "trainer.logger=['console']" \
    trainer.project_name=paper-hpt-phase0 \
    trainer.experiment_name=phase0_smoke_sync_grpo_qwen25_math_1_5b \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.log_val_generations=10 \
    trainer.balance_batch=False \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=20 \
    "$@"
