#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/는 repo root 바로 아래에 있으므로 한 단계 위가 VERL_ROOT다.
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# ============================================================================
# [M5abl_nocispo = M5 − C2(CISPO)] Ablation_RL.md §13.2 신 격자 1순위 arm (2026-07-09)
# 델타 (M5 대비 정확히 2 + 스텝캡):
#   actor.policy_loss.loss_mode: cispo -> vanilla   (g-슬롯을 Clip-Higher로 되돌림)
#   actor.clip_ratio_low: 10.0 -> 0.2               (vanilla 하한 복원; high=0.28 유지 = M과 밴드 정렬)
#   trainer.total_training_steps: 500 -> 200        (§13.3: 비교 창 50-90; 벽~100 이후는 폭풍-해부 보너스)
#   trainer.save_freq: 5 -> 10                       (체크포인트 수 절감; ablation은 수확 아닌 궤적/서명 분석)
# C1(decoupling)은 유지: rl_old_logprob_source=entry + entry_proximal=recent + rollout_is=token(C_w=2.0).
# 판독(§3 M−cispo, §12.3): low-prob death 밀도가 애초 >0였나(≈0이면 M5≈M5−cispo 자체가 결과),
#   clipfrac_top20entropy 라우터, 50-90 창 평균 val vs M5(38.47 peak@50, 창 평균 ~37.3).
# 규율: 폭풍 창 val 판정 제외 · 단일 val 차 ±0.7 노이즈 · 서명 1차(§8).
# ============================================================================

# [M5 = clean-async] M4(관대-harvest + adv-std, 관대=정직 val6 최고 36.27@step70)를 기저로,
# HPT(sync, 41.9) 대비 우리만의 병리 — async 폭발-붕괴 한계순환과 그로 인한 길이 붕괴 —
# 를 겨냥한 "clean-async" arm이다. M4 레시피 전체를 동결하고 델타 2개만 넣는다
# (단일 축 귀인 유지 — §5.9의 "한 번에 여러 노브 = 귀인 불가" 교훈).
#
#   [L1] rollout logprob FP32화: +rollout.engine_kwargs.sglang.enable_fp32_lm_head=True
#     - 무엇: SGLang의 lm_head matmul만 fp32로 계산(logits_processor.py) → rollout-side
#       logprob이 bf16 양자화 없이 산출된다. 추론 엔진 "전체" fp32화가 아니라 마지막 head뿐.
#     - 왜: C1(decoupled IS)·C2(CISPO)의 보정은 생성 시점 logprob을 앵커로 쓴다(Draft A.5).
#       그 앵커가 bf16이면 phantom off-policy 신호가 상시 섞여 CISPO가 과잉 이동한다.
#       ScaleRL은 이 fp32-head 단일 수정으로 천장 A +0.09(그들이 잰 최대 단일 이득)를 보고.
#     - 기대(정직): 우리 실측 RL-only mismatch는 KL 0.003nat로 온건 → A 이득 크기는 불확실,
#       메모리·사이클 강건성 이득은 확실. training/rollout_probs_diff의 명목 0.05-0.16은
#       SFT행 placeholder(rollout_log_probs=0, hpt_assembler.py:240) 오염 수치이므로 이 지표
#       원본으로 판정 금지 — RL-only 게이지가 필요하다(로깅 수정은 사용자 확인 후 별도 추가).
#     - 배선 검증: RolloutConfig.engine_kwargs가 그대로 ServerArgs로 전달(async_sglang_server.py
#       288/341/420), 로컬 SGLang 0.5.12에 enable_fp32_lm_head 필드/CLI 실재, replica.py:365가
#       이 async 어댑터를 로드. SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1로 context_length
#       =9728 충돌 처리(기존 기조 유지). 출처: SGLang issue #10490.
#     - 되돌림: 이 델타는 학습 수학·보상·advantage를 안 건드리므로 회귀 리스크 극소. 문제 시 라인 제거.
#
#   [L2] staleness 다이얼 축소: max_completed_prompt_groups 768→384
#     - 무엇: 완료 큐 상한을 절반으로 → 소비 시점 유효 staleness ~2-3 param-version →
#       ~1-1.5로. (§3.5: 명목 lag는 설계상 유계지만 큐 백로그 버퍼링으로 소비가 낡아지는 경로.)
#     - 왜: 한계순환은 지연×이득 진동자다(rollout_KL ≈ lag × 정책 이동속도). advstd 이득은
#       엔진이라 보존하되, 지연을 절반으로 낮춰 런어웨이 발화 문턱을 높이고 성장을 늦춘다.
#       HPT는 sync(지연 0)라 SFT-long-form 견인이 폭발 없이 4-6.5k로 성장·유지된다(UPT Fig.7);
#       우리 길이 붕괴(→1.2k)는 async 사이클의 산물이므로, 지연 축소가 그 근본을 부분 완화한다(§5.11).
#     - 기대(정직): 부분해다. 지배 동력(advstd × 절단-관대 연료)은 남으므로 사이클을 약화할 뿐
#       완치가 아니다. 완치가 필요하면 P2(KL-Cov 이동감쇠)로 escalation.
#     - 무해성: 배치는 trim+carryover로 128 그룹 고정이라 큐 깊이와 분리(§5.8.6) → 학습 수학 불변.
#       드롭률은 생산-소비 차로 결정되어 큐 크기와 무관. 감시: idle_ratio↑(트레이너 굶음) 시 512로 완화.
#
# 조기 게이지(clean-async 첫 20-30 fit-step, §5.11):
#   - L1: (로깅 수정 후) RL-only rollout_probs_diff 하강 여부.
#   - 사이클: rollout_corr/kl 피크 < 2-3 AND ESS 바닥 > 0.7 → 사이클 진정(P2 불필요);
#            여전히 KL 8+ / ESS 0.2 스파이크면 이득 다리 지배 → P2(KL-Cov) 투입 트리거.
#   - 길이(근본): 훈련 응답 길이가 ~1.2k에서 HPT식 3-5k로 성장하면 길이-붕괴 근본 해결 입증.
#
# 동결(M4에서 그대로): advstd=True, β0.3 constant, 관대 보상(P0 없음), CISPO(clip_low=10/high=0.28),
#   loss_agg=seq-mean-token-sum-norm, ppo_max_token_len=49152(⑦ 상속: OOM 폴백 49152→32768),
#   gamma=0.0, entropy_coeff=0.001, C1 decoupling(entry/token-IS/C_w=2.0). 델타는 위 2개뿐.
# 대조군: M4(체크포인트 harvest/M4_global_step_30_val35.16 · _70_val36.27 보존)가 without-clean-async 곡선.
#
# [각주 1] 이 스크립트는 Qwen2.5-Math-1.5B용 async RL + HPT의 **M5(clean-async) run**이다.
# M4 런처를 그대로 상속하되 위 L1/L2 두 델타만 켠다. ablation 두 축(docs/Ablation_RL.md §3):
#   - C1 = decoupling: async_hpt.rl_old_logprob_source=entry + entry_proximal=recent(옵션 B),
#     algorithm.rollout_correction.rollout_is=token(TIS-w, C_w=2.0), rollout_rs=null(rejection OFF).
#   - C2 = CISPO g-slot: actor.policy_loss.loss_mode=cispo + upper-only 클립
#     (clip_ratio_low=10.0으로 하한 비활성, 상단 cap=1+clip_ratio_high=1.28). 이 1.28은
#     이동-partition 스케일이자 M−cispo(vanilla)의 상단 밴드와 정렬 — C2 축을 "메커니즘
#     (gradient 死 vs 유지)"만으로 격리한다(DR-005 §6/정정4).
# 그 외 학습-문제 인자·공통 기반(DR-001~003)은 M4와 동일. val_before_train=True로 논문 baseline
# 대조의 step-0 앵커를 남긴다.

# RL conda env는 CUDA 13 runtime/cuDNN/NCCL wheel stack을 activation hook으로
# 노출한다. nohup/non-interactive 실행에서도 SGLang subprocess가 같은 loader
# path를 보도록 hook을 직접 source한다.
export CONDA_PREFIX="${HOME}/miniconda3/envs/RL"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh"

# openr1_hpt_main_v2: 전처리에서 OpenR1 시스템 프롬프트를 train에 유지하고 val eval셋에
# 동일 시스템 프롬프트를 주입한 데이터셋(train↔val 프롬프트 정합). 구 openr1_hpt_main은
# 시스템 프롬프트가 strip되어 학습/평가가 서로 다른 조건에 놓였다(백업으로 보존).
DATA_DIR="${VERL_ROOT}/datas/openr1_hpt_main_v2"
MODEL_PATH="${VERL_ROOT}/models/Qwen2.5-Math-1.5B"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# [reward] 채점기는 HF Math-Verify(verl math_verify_adapter)로 통일한다. 근거:
# (1) 실측상 후보 채점기(entropy_math/prime_math/math_reward/math_dapo) 중 recall이
#     가장 높아 저평가가 가장 적고(박스 없이도 표현식 추출 + 기호적 등가 판정),
#     그 상위집합이라 과대인정도 없다. (2) 커스텀 fn 하나로 train·val에 동일 적용해
#     default_compute_score의 data_source 라우팅이 train↔val 채점을 가르는 것을 막는다.
# (3) Math-Verify는 timeout/parse 실패 시 예외 대신 0으로 degrade해 async reward loop를
#     죽이지 않는다(이전 upt_v6의 raise 크래시 계열을 원천 차단).

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

# Async rollout dump는 replay가 아니라 write-only 분석 기록이다. all_steps=True로 전체
# feed step을 저장한다(길이/사이클/absorbing-loop 포렌식 소스). prepare_data는 executor로
# offload되어 rollout event loop를 blocking하지 않는다. 실비용은 디스크(~GB/hour)와 background
# 직렬화 CPU다. ※ 디스크 여유가 빠듯하면(현재 볼륨 ~98%) all_steps=False + steps=[...]로 제한.
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_qwen25_math_1_5b_M5abl_nocispo_${RUN_TIMESTAMP}"

# 학습-side per-token dump (docs/Ablation_RL.md §10). loss 경계 텐서
# (log_probs/old_log_probs/rollout_log_probs/advantages/mask)를 저장한다.
# read-only·sampled·offload라 학습을 막지 않는다. run별 유니크 dir.
TRAIN_DUMP_DIR="${VERL_ROOT}/.cache/train_dump/openr1_async_hpt_qwen25_math_1_5b_M5abl_nocispo_${RUN_TIMESTAMP}"

# Val 생성물 JSONL 덤프(관측성 전용). rollouter actor에서 wandb table이 침묵 드롭되는
# 버그(tracking.py `wandb.run is None` 가드)와 무관하게 val 출력을 디스크에 남긴다.
VAL_DUMP_DIR="${VERL_ROOT}/.cache/val_dump/openr1_async_hpt_qwen25_math_1_5b_M5abl_nocispo_${RUN_TIMESTAMP}"

# Scratch launch only: no resume path.

# [각주 3] batch scale은 UPT train_batch_size=128 prompt groups와 맞춘다.
# queue sample은 row가 아니라 prompt group이다. rollout.n=8이므로
# ppo_mini_batch_size(32) * require_batches(4) = 128 prompt groups,
# 즉 fit_step마다 128 * 8 = 1024 generated rows를 본다.
REQUIRE_BATCHES=4
# [L2] 768→384: staleness 다이얼을 절반으로. 유효 lag ~2-3 → ~1-1.5 param-version →
# 한계순환의 지연 항 축소(런어웨이 발화 문턱↑). 배치는 trim+carryover로 128 그룹 고정이라
# 큐 깊이와 분리(§5.8.6) → 학습 수학 불변. 드롭률은 생산-소비 차라 큐 크기 무관.
# 감시: idle_ratio↑(트레이너 굶음)면 512로 완화 (Improvement_RL §5.11).
MAX_COMPLETED_PROMPT_GROUPS=384

# [각주 4] UPT와의 대응 관계:
#   UPT SWITCH_GATE=0 -> 8개 rollout 중 성공 0개인 prompt만 offline/SFT target.
#   ours async_hpt.gamma=0.0 -> success_count / rollout.n <= 0.0, 즉 0/8 전멸 prompt만 SFT route.
#   UPT SFT_LOSS_COEF=0.3 (Qwen2.5-Math-1.5B) -> ours async_hpt.beta=0.3 (constant).
# beta=0.3 constant는 UPT 1.5B 계수 정합이자 관대-지표 34.0을 달성한 D0·M의 공통값이다.
#
# [각주 6] DR-001~003에서 고정된 objective 계약(M5 동결):
#   - branch_blind reduction(B_eff/prompt_equal 분모 미사용).
#   - actor loss aggregation은 seq-mean-token-sum-norm, L_max=8192.
#   - SFT row는 self-detach CE 의미를 갖고 entropy/KL auxiliary에서 제외.
#
# [각주 7] rollout sampling parity: temperature=1.0만 명시(top_p/top_k는 verl 기본값
# top_p=1.0/top_k=-1 = UPT와 동일). val은 temperature=0.6, top_p=0.95 명시.
#
# [각주 8] clip은 UPT와 의도적으로 다르다. UPT는 loss_remove_clip=True(동기라 unclipped 안전).
# 우리는 fully-async off-policy라 trust region 필수(DR-005 §5): staleness는 w-슬롯(TIS,
# rollout_is=token, C_w=2.0), movement는 g-슬롯(CISPO, 상단 cap 1.28)이 흡수. clip_ratio_low
# =10.0(1-10<0 → 하한 무효 = CISPO upper-only) + clip_ratio_high=0.28 + clip_ratio_c=10.0.
#
# [각주 9] grain은 UPT와 다르다. UPT ppo_mini_batch_size=64(fit당 SGD 2회), 우리 32(fit당 4회).
# fit당 128 prompt-group(=1024 row) 스케일은 동일, optimizer grain만 더 잘다.
#
# [각주 10] D0 정체성 명시(동작 불변, 앵커 고정): actor.ppo_epochs=1,
# async_hpt.rl_old_logprob_source=entry(C1 ON), async_hpt.k_max=null(학습-시점 staleness 드롭
# OFF — 낡음은 async_training.staleness_threshold=2.0 예산 레벨 + 절단 IS C_w로만 제어).

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
    actor_rollout_ref.actor.optim.clip_grad=80.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
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
    trainer.experiment_name=qwen25_math_1_5b_openr1_async_hpt_M5abl_nocispo \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=200 \
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
    async_hpt.tau_dataset_path="${DATA_DIR}/train.parquet" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    "$@"
