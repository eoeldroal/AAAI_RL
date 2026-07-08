#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/는 repo root 바로 아래에 있으므로 한 단계 위가 VERL_ROOT다.
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# [M7 = full-stack fresh] M5(clean-async: fp32-head + queue384)를 복제해 4델타를 국소 적용한
# "바닥부터" 런이다. M6(resume + core B0/B1'/B3, B2 off)와 달리, M7은 B2(cispo_klcov)를 켜므로
# 반드시 fresh여야 한다: KL-Cov는 "엔트로피 붕괴를 처음부터 막는(maintain)" 것으로만 검증됐고
# (Cui et al. 2505.22617), 이미 플로어(0.02–0.15)에 앉은 체크포인트에서 켜는 "복원(restore)"은
# 미검증이라 resume하면 "cispo_klcov 효과"와 "복원 여부"가 뒤섞여 판독 불능이다(Improvement_RL §5.12.7).
# 그래서 M5-recipe를 그대로 두고 아래 4델타만 얹어 step0부터 재출발한다.
#
#   [B0] grad clip parity: optim.clip_grad 80.0 → 1.0
#     - UPT parity(ppo_trainer.yaml grad_clip=1.0). 성분/토큰 삭제가 아니라 전역 노름>1.0일 때만
#       방향 보존 스케일다운. M4/M5 실측 grad_norm 최대 1.198(1회)이라 사실상 무해 = "정정".
#
#   [B1'] semantic-aware queue eviction: async_hpt.queue_evict_zero_variance=True
#     - 완료 큐 overflow(M5 실측 ~7940 drop) 시 축출 우선순위를 "나이순"→"k==n 전정답(zero-variance)
#       RL 그룹 우선". 전정답 그룹은 GRPO advantage 항등 0(dead weight)이라 가장 안전한 victim.
#       힌트=int(hpt_gate.zero_variance_evict_hint), 큐는 의미론 무지, 힌트 없으면 popleft fail-open.
#     - fresh 초반(성공률 낮음)엔 k=8이 드물어 거의 무발동, 성공률 ~0.4+로 오르는 mid-run(≈step40+)부터
#       작동. 학습 수학 불변(전송 스케줄링만). 게이지 count/mq_evicted_zero_variance ↑.
#     - 정직 공시: k=8 제거로 소비 배치 hpt/offline_data_ratio 소폭 상승(조성 변화, 교사 증가 아님).
#
#   [B3] staleness 추가 축소: max_completed_prompt_groups 384 → 256
#     - 유효 staleness를 ScaleRL k≤12 경계 쪽으로. B1' 축출 압력과 순방향(소비 배치가 더 젊어짐).
#     - 감시: idle_ratio↑(트레이너 굶음)면 320으로 완화.
#
#   [B2] KL-Cov 탐색 축: policy_loss.loss_mode cispo → cispo_klcov (★M7의 주역★)
#     - CISPO 본체 + KL-Cov 오버레이: 매 업데이트마다 RL 토큰을 Cov(logp, A)로 랭크,
#       상위 kl_cov_ratio(0.0002 = 논문 검증값)에만 KL(π_old‖π) 페널티(ppo_kl_coef=0.1)를 더해
#       엔트로피 붕괴를 주도하는 소수 pivotal 토큰의 sharpening만 감쇠. 나머지 토큰은 CISPO 그대로.
#       SFT(교사) 토큰은 Cov 선택에서 제외(hpt_sft_token_mask) — 모방 타깃이라 감쇠 대상 아님.
#     - 기대: M5의 step20–25 엔트로피 붕괴(1.1→0.14)를 늦추거나 막음 → dead-zone(AIME/AMC) stumble
#       여지 유지. 1.5B/CISPO/async 미검증 조합(부품 검증, 조합 가설)이므로 이 fresh 런이 그 검증.
#     - 다이얼: 3h에 엔트로피/val 무변화면 kl_cov_ratio 0.0002→0.0004, ppo_kl_coef 0.1→0.3 상향 여지.
#
# 동결(M5 그대로): fp32 lm_head(L1), advstd=True, β0.3 constant, 관대 보상, CISPO upper-only
#   (clip_low=10/high=0.28/c=10), loss_agg=seq-mean-token-sum-norm/8192, gamma=0.0,
#   entropy_coeff=0.001, C1 decoupling(entry/token-IS/C_w=2.0), temp 1.0/val 0.6.
#
# ── 3시간 go/no-go (사전등록, matched-step vs M5) ─────────────────────────────────
# M7은 fresh라 M5의 알려진 궤적과 스텝별 직접 비교된다. 대조 앵커(M5 lenient6 mean@8):
#   step10 33.66 · 15 34.62 · 20 34.41 · 25 32.45 · 30 31.04 · (M5 entropy_mean step20–25 0.14–0.31).
# 판정: (a) M7 entropy_mean이 같은 구간에서 M5보다 높게 유지(>0.3–0.5) AND matched-step val ≥ M5
#   → cispo_klcov 유효, M7이 방향(완주 = 메인 런). (b) 엔트로피 여전히 붕괴 OR val ≤ M5
#   → M6 방향(resume+core, B2 선반). 핵심 조기 tell = M5의 step25–30 dip(32.5→31.0)을 M7이 피하는가.
# 게이지: actor/entropy_mean, actor/klcov_selected_tokens(오버레이 활성 확인), rollout_corr/kl,
#   count/mq_evicted_zero_variance, val-core/*/acc/mean@8.
# 안전(상시): rollout_corr/kl>1 2스텝 연속·truncation>10%·ESS<0.25 → 중단.
#
# [디스크] fresh 3h ≈ 5.9GB/step × ~65 ≈ ~400GB. M5 정지 후 여유 ~378G면 step~60에서 벽 →
#   이전 런(M2/M3/M4/M) rollout_dump 정리로 여유 확보(사장님 영역). 벽 근처면 all_steps=False로.

# RL conda env는 CUDA 13 runtime/cuDNN/NCCL wheel stack을 activation hook으로
# 노출한다. nohup/non-interactive 실행에서도 SGLang subprocess가 같은 loader
# path를 보도록 hook을 직접 source한다.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

# openr1_hpt_main_v2: train↔val 프롬프트 정합 데이터셋(M5와 동일).
DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main_v2"
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-1.5B"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# [reward] 채점기는 HF Math-Verify(verl math_verify_adapter)로 통일(M5와 동일 근거).

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
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_qwen25_math_1_5b_M7_${RUN_TIMESTAMP}"
TRAIN_DUMP_DIR="${VERL_ROOT}/.cache/train_dump/openr1_async_hpt_qwen25_math_1_5b_M7_${RUN_TIMESTAMP}"
VAL_DUMP_DIR="${VERL_ROOT}/.cache/val_dump/openr1_async_hpt_qwen25_math_1_5b_M7_${RUN_TIMESTAMP}"

# Fresh launch: no resume path (B2=cispo_klcov requires maintain-from-start; see header).

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
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo_klcov \
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
    trainer.experiment_name=qwen25_math_1_5b_openr1_async_hpt_M7_fullstack_klcov \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.resume_mode=disable \
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
