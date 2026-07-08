#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/는 repo root 바로 아래에 있으므로 한 단계 위가 VERL_ROOT다.
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# [M6 = signal-density + parity] M5(clean-async: fp32-head + queue384)가 val6 mean@8 최고
# 38.47@50까지 상승했으나 train score가 ~step25부터 정체(신호-기아: 성공률 ~0.6에서 k=8 포화
# 그룹이 배치 gradient 밀도를 희석). M6는 M5 레시피를 동결하고 "검증 계층이 높은" 델타 3개
# (B0/B1'/B3)만 켠다. 탐색 축(B2=KL-Cov)은 코드로 준비돼 있으나 config로 꺼둔다(단일 축 귀인 유지,
# §5.9). M5의 최고 val 체크포인트에서 resume한다(바닥부터 재등반 안 함 — §5.12).
#
#   [B0] grad clip parity: optim.clip_grad 80.0 → 1.0
#     - 무엇: 전역 gradient-norm 상한을 UPT parity 값(1.0)으로. 성분/토큰 삭제가 아니라 노름>1.0
#       일 때만 방향 보존 스케일다운. 실측: M4 95스텝(치명폭풍 포함) grad_norm 최대 1.198(1회),
#       M5 작동점 0.011-0.028 → 평시 발동 0회. 순수 parity 위생 + 프릭 배치 보험(Adam EMA가
#       못 잡는 단발 스파이크). 성능 효과 ≈0 (개선 아니라 "정정"). 출처 UPT ppo_trainer.yaml grad_clip=1.0.
#
#   [B1'] semantic-aware queue eviction: +async_hpt.queue_evict_zero_variance=True
#     - 무엇: 완료 큐가 overflow(M5 실측 ~7940 drop)할 때 축출 우선순위를 "나이순"에서
#       "k==n 전정답(zero-variance) RL 그룹 우선"으로. 전정답 그룹은 GRPO advantage가 항등 0
#       (배치 dead weight)이라 가장 안전한 victim. 힌트 없으면 기존 popleft로 fail-open.
#     - 왜: 생산 ~293그룹/스텝 vs 소비 128 → 165그룹/스텝이 이미 버려진다. "버릴지"는 강제이나
#       "무엇을 버릴지"는 자유 → 정보 0인 것을 버리고 유익(다소 stale)한 것을 살린다. 정체의
#       확인된 성분(신호밀도)을 직접 겨냥. 학습 수학 불변(전송층 스케줄링만; 힌트=int, 큐는
#       의미론 무지). 배치는 여전히 trim+carryover로 128그룹 고정.
#     - 게이지: count/mq_evicted_zero_variance ↑ (B1' 활성 신호), count/mq_dropped_samples 대비.
#     - 정직 공시: 소비 배치의 hpt/offline_data_ratio가 ~0.21→0.23-0.25로 기계적 상승(조성 변화,
#       교사 라우팅 증가 아님). k=8 비중(성공률 0.6에서 15-20%)이 상한.
#
#   [B3] staleness 추가 축소: max_completed_prompt_groups 384 → 256
#     - 무엇: 완료 큐 상한을 384→256. 유효 staleness를 ScaleRL k≤12 경계 쪽으로 더 당긴다
#       (현재 유효 k~13-20). B1'의 축출 압력 상승과 순방향 결합(소비 배치가 더 젊어짐).
#     - 감시: idle_ratio↑(트레이너 굶음)면 320으로 완화. 배치는 trim+carryover로 큐 깊이와 분리.
#
#   [B2 = OFF] KL-Cov 탐색 축 (loss_mode=cispo_klcov). 지금은 끈다.
#     - 코드는 준비 완료(core_algos.compute_policy_loss_cispo_klcov + losses.py RL-마스크 스레딩 +
#       hpt_config 화이트리스트). 활성화는 아래 한 줄만 바꾸면 된다:
#         actor_rollout_ref.actor.policy_loss.loss_mode=cispo   →   =cispo_klcov
#       (선택) actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.0002 (기본), ppo_kl_coef=0.1 (기본).
#     - 투입 조건(사전등록 §5.12): M6-core가 resume 후 3 eval(15스텝) 내 3-val MA 미갱신 →
#       그 시점 최고 체크포인트에서 loss_mode=cispo_klcov 단일 델타로 resume. 실패 시 fresh+B2.
#     - 왜 지금 끄나: KL-Cov는 1.5B/CISPO/붕괴-후-복원 조건에서 미검증 조합(부품은 검증, 조합은
#       가설). 검증 계층이 낮아 core 번들에 섞으면 귀인/신뢰도가 무너진다(§5.12 스켑틱 판정).
#
# 동결(M5에서 그대로): fp32 lm_head(L1), advstd=True, β0.3 constant, 관대 보상, CISPO(clip_low=10/
#   high=0.28), loss_agg=seq-mean-token-sum-norm/8192, gamma=0.0, entropy_coeff=0.001,
#   C1 decoupling(entry/token-IS/C_w=2.0), temp 1.0/val 0.6. 델타는 B0/B1'/B3 셋뿐.
#
# ── 이어달리기(resume) ─────────────────────────────────────────────────────────
# M5의 "검증된 최고 val" 체크포인트에서 이어간다. resume_mode=resume_path(auto 금지 — auto는
# latest_checkpointed_iteration.txt의 최신 스텝을 잡아 정점이 아닌 곳에서 재개될 수 있다).
# ★ 실행 전 RESUME_FROM_STEP을 M5가 수확한 최고 val 스텝으로 설정하라 ★ (현재 최고 38.47@50,
# M5가 계속 상승 중이므로 완주/중단 시점의 실제 best-val 스텝으로 갱신). 복원: actor 가중치 +
# optimizer + lr_scheduler + rng + StatefulDataLoader 위치. 미복원: 큐 내 in-flight 수백 프롬프트
# (46k 중 무시가능). 첫 2-3 fit-step은 구정책 롤아웃 소화 구간이므로 rollout_corr/kl 감시.
RESUME_FROM_STEP="${RESUME_FROM_STEP:-50}"
RESUME_FROM_PATH="${VERL_ROOT}/checkpoints/async-hpt-openr1/qwen25_math_1_5b_openr1_async_hpt_M5_cleanasync/global_step_${RESUME_FROM_STEP}"

# RL conda env는 CUDA 13 runtime/cuDNN/NCCL wheel stack을 activation hook으로
# 노출한다. nohup/non-interactive 실행에서도 SGLang subprocess가 같은 loader
# path를 보도록 hook을 직접 source한다.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

# openr1_hpt_main_v2: 전처리에서 OpenR1 시스템 프롬프트를 train에 유지하고 val eval셋에
# 동일 시스템 프롬프트를 주입한 데이터셋(train↔val 프롬프트 정합). M5와 동일.
DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main_v2"
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-1.5B"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# [reward] 채점기는 HF Math-Verify(verl math_verify_adapter)로 통일. M5와 동일 근거.

# Triton/TorchInductor JIT artifact는 실행 가능한 repo 인접 파일시스템에 둔다.
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

# Async rollout dump는 write-only 분석 기록. ※ 디스크 여유가 빠듯하면 all_steps=False로 제한.
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_qwen25_math_1_5b_M6_${RUN_TIMESTAMP}"
TRAIN_DUMP_DIR="${VERL_ROOT}/.cache/train_dump/openr1_async_hpt_qwen25_math_1_5b_M6_${RUN_TIMESTAMP}"
VAL_DUMP_DIR="${VERL_ROOT}/.cache/val_dump/openr1_async_hpt_qwen25_math_1_5b_M6_${RUN_TIMESTAMP}"

# batch scale은 UPT train_batch_size=128 prompt groups와 맞춘다(M5와 동일).
REQUIRE_BATCHES=4
# [B3] 384→256: staleness 다이얼을 ScaleRL k<=12 경계 쪽으로 추가 축소.
MAX_COMPLETED_PROMPT_GROUPS=256

# Keep only engine-runtime environment at the command boundary. All training,
# rollout, validation, and HPT settings are explicit Hydra overrides below.
SGLANG_NUMA_BIND_V2=0 \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${DATA_DIR}/train.parquet" \
    "data.val_files=['${DATA_DIR}/AIME24/test.parquet','${DATA_DIR}/AIME25/test.parquet','${DATA_DIR}/AMC23/test.parquet','${DATA_DIR}/MATH-500/test.parquet','${DATA_DIR}/Minerva/test.parquet','${DATA_DIR}/Olympiad-Bench/test.parquet']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=1536 \
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
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152 \
    actor_rollout_ref.actor.clip_ratio_low=10.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.0002 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=0.1 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
    actor_rollout_ref.rollout.max_model_len=9728 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=9728 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_fp32_lm_head=True \
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
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=False \
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
    "trainer.logger=['console','wandb']" \
    trainer.project_name=async-hpt-openr1 \
    trainer.experiment_name=qwen25_math_1_5b_openr1_async_hpt_M6_signaldensity \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=500 \
    trainer.test_freq=5 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
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
    training_dump.enable=True \
    training_dump.dir="${TRAIN_DUMP_DIR}" \
    training_dump.sample_every_n_steps=20 \
    training_dump.max_rows=256 \
    training_dump.dtype=bf16 \
    training_dump.offload=True \
    async_hpt.enabled=True \
    async_hpt.gamma=0.0 \
    async_hpt.beta=0.3 \
    async_hpt.loss_aggregation=branch_blind \
    async_hpt.sft_beta_mode=constant \
    async_hpt.sft_entropy_enabled=False \
    async_hpt.sft_kl_enabled=False \
    async_hpt.rl_old_logprob_source=entry \
    +async_hpt.entry_proximal=recent \
    async_hpt.k_max=null \
    async_hpt.queue_evict_zero_variance=True \
    async_hpt.tau_dataset_path="${DATA_DIR}/train.parquet" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    "$@"
