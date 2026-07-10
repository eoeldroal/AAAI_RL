#!/usr/bin/env bash
# Paper-faithful SYNCHRONOUS HPT main run — Qwen2.5-Math-1.5B.
# Reproduces the original UPT/HPT CODE (arXiv:2509.04419) on modern verl / B200.
#
# ⚠️ REQUIRES Phase-1 wiring to EXECUTE the HPT path (else it crashes at the first
# actor update: paper_hpt_dual_loss requires `hpt_is_sft`, which only the routing
# sets). Phase-1 = (1) gated routing hook in RayPPOTrainer.fit() after reward /
# before advantage, (2) dataset tau passthrough (tgt_* fields). This file is the
# pinned CONFIG; run it only after Phase-1 lands. See recipe/paper_hpt/README.md.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

# RL conda env CUDA13/cuDNN/NCCL wheel stack — source directly for nohup/sglang.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

RUNTIME_CACHE_DIR="$(cd "${VERL_ROOT}/.." && pwd)/.rt"
mkdir -p "${RUNTIME_CACHE_DIR}"/{tmp,triton,torchinductor,xdg}
export TMPDIR="${RUNTIME_CACHE_DIR}/tmp"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE_DIR}/torchinductor"
export XDG_CACHE_HOME="${RUNTIME_CACHE_DIR}/xdg"

MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/models/Qwen2.5-Math-1.5B}"
DATA_DIR="${DATA_DIR:-${VERL_ROOT}/datas/openr1_hpt_main}"        # STRIP (paper-faithful), NOT v2
# Paper's entropy_math grader (loaded by upt_v6_adapter in a spawn subprocess).
ENTROPY_MATH="${ENTROPY_MATH:-${VERL_ROOT}/../Unify-Post-Training/hpt/verl/verl/mix_src/entropy_math/__init__.py}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-paper_hpt_sync_qwen25_math_1_5b_beta03}"

SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m verl.trainer.main_ppo \
    trainer.use_v1=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=false \
    +algorithm.paper_hpt.enable=true \
    +algorithm.paper_hpt.gamma=0.0 \
    +algorithm.paper_hpt.beta=0.3 \
    +algorithm.paper_hpt.success_value=1.0 \
    data.train_files="${DATA_DIR}/train.parquet" \
    "data.val_files=['${DATA_DIR}/AIME24/test.parquet','${DATA_DIR}/AIME25/test.parquet','${DATA_DIR}/AMC23/test.parquet','${DATA_DIR}/MATH-500/test.parquet','${DATA_DIR}/Minerva/test.parquet','${DATA_DIR}/Olympiad-Bench/test.parquet']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=1536 \
    data.max_response_length=8192 \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.custom_loss_fn=recipe.paper_hpt.paper_hpt_loss.paper_hpt_dual_loss \
    +actor_rollout_ref.actor.loss_scale_factor=8192 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=9728 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=9728 \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.json_model_override_args='{\"max_position_embeddings\":16384,\"rope_theta\":40000}'" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=verl/utils/reward_score/upt_v6_adapter.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.entropy_math_path="${ENTROPY_MATH}" \
    +reward.custom_reward_function.reward_kwargs.use_process_pool=True \
    +reward.custom_reward_function.reward_kwargs.process_timeout=30.0 \
    +reward.reward_kwargs.compute_score_in_executor=True \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=8192 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=async-hpt-openr1 \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.log_val_generations=10 \
    trainer.rollout_data_dir="${VERL_ROOT}/.cache/rollout_dump/${EXP_NAME}_${RUN_TIMESTAMP}" \
    trainer.validation_data_dir="${VERL_ROOT}/.cache/val_dump/${EXP_NAME}_${RUN_TIMESTAMP}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.balance_batch=False \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    "$@"
