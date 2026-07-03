#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/는 repo root 바로 아래에 있으므로 한 단계 위가 VERL_ROOT다.
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# [각주 1] 이 스크립트는 Qwen2.5-Math-1.5B용 async RL + HPT main 비교 run이다.
# 기존 7B main launcher를 덮어쓰지 않고 별도 파일로 둔다. 이유는 7B run과
# 1.5B run이 같은 데이터/목표를 공유하더라도 HPT 강도(beta)와 모델 경로가
# 다르므로, 실행 기록과 rollout dump를 파일명 수준에서 분리해야 하기 때문이다.

# RL conda env는 CUDA 13 runtime/cuDNN/NCCL wheel stack을 activation hook으로
# 노출한다. nohup/non-interactive 실행에서도 SGLang subprocess가 같은 loader
# path를 보도록 hook을 직접 source한다.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main"
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-1.5B"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# [각주 2] Qwen2.5-Math 계열은 LUFFY/UPT 설정과 맞추기 위해 긴 context를 쓴다.
# 아래 Hydra override에서 max_position_embeddings=16384, rope_theta=40000을
# actor model과 SGLang rollout engine 양쪽에 모두 전달한다. 한쪽만 바꾸면
# trainer/rollout이 서로 다른 context 해석을 할 수 있으므로 두 경로를 함께 둔다.

# Triton/TorchInductor는 runtime에 shared object를 만든다. cluster /tmp는
# noexec일 수 있으므로 JIT artifact는 실행 가능한 repo 인접 파일시스템에 둔다.
# Ray AF_UNIX socket path 제한 때문에 경로는 짧게 유지한다.
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

# Async rollout dump는 replay가 아니라 write-only 분석 기록이다. main run에서는
# post-update rollout까지 품질 분석 대상이므로 all_steps=True로 전체 feed step을
# 저장한다. 디스크 비용이 크면 smoke/debug 전용 스크립트에서만 제한한다.
#
# 이 덤프의 prepare_data는 전체 generation DataProto를 직렬화+디스크 기록하는 무거운
# 작업이다. 동기로 실행하면 rollout의 단일 asyncio event loop를 점유해 SGLang 요청
# 제출을 굶기고 gen throughput을 런이 길어질수록 단조 붕괴시킨다(py-spy로 loop 시간의
# 대부분이 덤프 직렬화임을 확인). 그래서 SkipManager.annotate(async 경로)가 prepare_data를
# executor로 offload해 loop를 blocking하지 않게 하고 덤프는 그대로 유지한다. 실비용은
# 디스크(~GB/hour, all_steps=True)와 background 직렬화 CPU다. 장기 런에서 디스크가
# 빠듯하면 all_steps=False + steps=[...]로 일부 step만 저장해 제한한다.
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_qwen25_math_1_5b_${RUN_TIMESTAMP}"

# [각주 3] batch scale은 UPT train_batch_size=128 prompt groups와 맞춘다.
# queue sample은 row가 아니라 prompt group이다. rollout.n=8이므로
# ppo_mini_batch_size(32) * require_batches(4) = 128 prompt groups,
# 즉 fit_step마다 128 * 8 = 1024 generated rows를 본다.
# 64 * 2도 prompt-batch parity는 맞지만, 32 * 4가 HPT learner-row divisibility와
# async trainer scheduling에 더 덜 brittle해서 1.5B 첫 main run 기본값으로 둔다.
REQUIRE_BATCHES=4
MAX_COMPLETED_PROMPT_GROUPS=2048

# [각주 4] UPT와의 대응 관계:
#   UPT SWITCH_GATE=0
#     -> 8개 rollout 중 성공 0개인 prompt만 offline/SFT target을 넣는다.
#   ours async_hpt.gamma=0.0
#     -> success_count / rollout.n <= 0.0, 즉 0/8 전멸 prompt만 SFT route.
# 따라서 gamma=0.0은 routing/gate 관점에서 UPT SWITCH_GATE=0에 대응한다.
#
#   UPT SFT_LOSS_COEF=0.3 for Qwen2.5-Math-1.5B
#     -> 별도 SFT CE loss에 0.3을 곱한다.
#   ours async_hpt.beta=0.3
#     -> tau* SFT row의 terminal pseudo reward를 0.3으로 둔다.
# 따라서 beta=0.3은 SFT intervention strength 관점에서 UPT의 1.5B 계수에 대응한다.
#
# [각주 5] sft_beta_mode=constant는 classic HPT를 완전히 복제하려는 control이
# 아니라 우리 DR-001 main method다. classic token-mean SFT에 더 가깝게 맞추는
# control은 length_inverse가 더 적절하지만, 이 파일은 "우리 방법 main"을 위한
# launcher이므로 constant를 사용한다.
#
# [각주 6] DR-001~003에서 고정된 objective 계약:
#   - B_eff/prompt_equal 분모를 쓰지 않고 branch_blind reduction을 쓴다.
#   - actor loss aggregation은 seq-mean-token-sum-norm, L_max=8192로 둔다.
#   - SFT row는 self-detach CE 의미를 갖고, entropy/KL auxiliary에서 제외한다.
# 이 값들은 단순 튜닝값이 아니라 우리 방법의 정체성이므로 비교 run에서도 유지한다.

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
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=8192 \
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
    trainer.experiment_name=qwen25_math_1_5b_openr1_async_hpt_beta03_constant \
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
    async_training.max_inflight_prompt_groups=96 \
    async_training.max_completed_prompt_groups=${MAX_COMPLETED_PROMPT_GROUPS} \
    skip.async_rollout.enable=True \
    skip.async_rollout.dump_dir="${ROLLOUT_DUMP_DIR}" \
    skip.async_rollout.steps=[] \
    skip.async_rollout.all_steps=True \
    skip.async_rollout.action=dump \
    async_hpt.enabled=True \
    async_hpt.gamma=0.0 \
    async_hpt.beta=0.3 \
    async_hpt.loss_aggregation=branch_blind \
    async_hpt.sft_beta_mode=constant \
    async_hpt.sft_entropy_enabled=False \
    async_hpt.sft_kl_enabled=False \
    async_hpt.tau_dataset_path="${DATA_DIR}/train.parquet" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    "$@"
