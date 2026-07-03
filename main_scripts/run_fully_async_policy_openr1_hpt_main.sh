#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/ lives directly under the repo root (one level up).
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# The RL environment installs CUDA 13 runtime libraries through pip wheels.
# Source the checked activation hook directly so nohup/non-interactive launches
# expose the same loader path to SGLang subprocesses without relying on an
# interactive conda activation round-trip.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main"
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-7B"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Triton/TorchInductor build shared objects at runtime. The cluster /tmp is
# mounted noexec, so keep JIT artifacts on an executable project filesystem.
# Keep this path short because Ray also creates AF_UNIX sockets under TMPDIR.
RUNTIME_CACHE_DIR="$(cd "${VERL_ROOT}/.." && pwd)/.rt"
mkdir -p \
    "${RUNTIME_CACHE_DIR}/tmp" \
    "${RUNTIME_CACHE_DIR}/triton" \
    "${RUNTIME_CACHE_DIR}/torchinductor" \
    "${RUNTIME_CACHE_DIR}/xdg"
export TMPDIR="${RUNTIME_CACHE_DIR}/tmp"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE_DIR}/torchinductor"
export XDG_CACHE_HOME="${RUNTIME_CACHE_DIR}/xdg"

# Async rollout dump uses the skip path in write-only mode. Keep the dump
# directory unique per run so analysis artifacts never mix across runs.
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_main_${RUN_TIMESTAMP}"

# require_batches=2 * ppo_mini_batch_size(64) * rollout.n(8) = 1024 rows/fit_step,
# matching the synchronous baseline's train_batch_size(128)*rollout.n(8) scale
# and its train_batch_size/ppo_mini_batch_size=2 mini-batch steps per update.
# max_completed_prompt_groups scales down with it (2048/8=256) to keep the
# completed-queue cap, not staleness, as the first backpressure point. Do not
# reintroduce require_batches=16: that was a units bug (queue samples are
# prompt groups, not rows -- see docs/AsyncBudget_RL.md's "Known pitfall") that
# silently ran fit_steps 8x larger than intended. See docs/AsyncBudget_RL.md
# for the full derivation and docs/Codemap_RL.md for the rollouter call graph.
REQUIRE_BATCHES=2
MAX_COMPLETED_PROMPT_GROUPS=256

# Keep only engine-runtime environment at the command boundary. All training,
# rollout, validation, and HPT settings are explicit Hydra overrides below.
SGLANG_NUMA_BIND_V2=0 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${DATA_DIR}/train.parquet" \
    "data.val_files=['${DATA_DIR}/AIME24/test.parquet','${DATA_DIR}/AMC23/test.parquet','${DATA_DIR}/MATH-500/test.parquet']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.val_batch_size=512 \
    data.shuffle=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=16384 \
    +actor_rollout_ref.model.override_config.rope_theta=40000 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=80.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=16384 \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.json_model_override_args='{\"max_position_embeddings\":16384,\"rope_theta\":40000}'" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    critic.strategy=fsdp2 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=8192 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=async-hpt-openr1 \
    trainer.experiment_name=qwen25_math_7b_openr1_async_hpt \
    trainer.val_before_train=False \
    trainer.save_freq=50 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=500 \
    trainer.test_freq=10 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=6 \
    rollout.n=8 \
    rollout.total_rollout_steps=128000 \
    async_training.staleness_threshold=2.0 \
    async_training.require_batches=${REQUIRE_BATCHES} \
    async_training.partial_rollout=True \
    async_training.trigger_parameter_sync_step=4 \
    async_training.use_trainer_do_validate=False \
    async_training.max_inflight_prompt_groups=24 \
    async_training.max_completed_prompt_groups=${MAX_COMPLETED_PROMPT_GROUPS} \
    skip.async_rollout.enable=True \
    skip.async_rollout.dump_dir="${ROLLOUT_DUMP_DIR}" \
    skip.async_rollout.steps=[] \
    skip.async_rollout.all_steps=True \
    skip.async_rollout.action=dump \
    async_hpt.enabled=True \
    async_hpt.gamma=0.0 \
    async_hpt.tau_dataset_path="${DATA_DIR}/train.parquet" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    "$@"
