#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# main_scripts/는 repo root 바로 아래에 있으므로 한 단계 위가 VERL_ROOT다.
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VERL_ROOT}"

# [M4] 목표 재정의(무조건 관대-val6 극대화, Improvement_RL.md §5.10) 하의 수확(harvest) arm이다.
# 34.0을 달성한 두 run(D0·M/resume40b)의 공통 인자로 회귀하고, 그 위에 신규 재료 1개
# (advantage std 정규화)와 staleness 다이얼을 얹는다. M3 대비 델타:
#   ① 채점 일치: P0 절단 게이트 2개 제거 → 훈련 보상 = 평가 채점(관대, verl 표준).
#      (관대+CISPO 조합은 죽은 칸이 아님 — M 33.76, resume40b 34.00 실측. M2의 26 사망은
#       β1.0 leninv/라우팅 축이 용의자였고 그 축을 아래 ④로 제거한다. §5.9-§5.10)
#   ④ SFT 강도: beta 1.0→0.3, length_inverse→constant — 両34-달성 run의 공통값.
#   ⑤ 신규 재료: norm_adv_by_std=True — 희소 성공(1/8) 신호 ~3× 증폭(하드벤치 직격),
#      참조 구현 parity. SFT singleton 행은 mean=0/std=1 특례로 β_r 보존(계약 테스트 고정).
#      되돌림 기준(사전 등록): entropy_mean>3.5 또는 관대 val 2연속 하락 시 이 축만 False 회귀.
#   ⑥ staleness 다이얼: max_completed_prompt_groups 384→768 — trim+carryover가 배치 크기와
#      큐 깊이를 분리했으므로(§5.8.6) 큐 상한은 이제 순수 staleness(~1.5 param-version) 손잡이다.
#      드롭률은 생산-소비 차로 결정되어 큐 크기와 무관(≈38% 유지); 이 드롭은 낭비가 아니라
#      C1(decoupled+TIS)이 유의미하게 일하는 레짐을 유지하는 가격이다(논문 분석용).
#   ⑦ 순수 속도: ppo_max_token_len_per_gpu 32768→65536(+log_prob 동일) — 마이크로배치 상각,
#      학습 수학 불변. 실측 allocated 53.6GB@32k의 선형 외삽 ~95GB@64k(용량 183GB의 52%).
#      OOM 폴백 사다리(사전 등록): 65536 → 49152 → 32768.
# 유지: C1 decoupling + C2 CISPO(기여 축 — ⑥이 이들의 작동 레짐을 복원), G4 인프라
# (trim+carryover), val JSONL 덤프(관대/정직 이중 곡선의 offline 산출 = 부패 조기경보).
# 실행 규율: §5.8.3 수확 원칙 — 관대 val 2연속 하락 시 즉시 종료하고 최고점 checkpoint 수확.
# (M3의 "완주 서약"은 정직-baseline용이며 M3 완주 후에 본 스크립트를 실행한다. §5.10)
#
# [각주 1] 이 스크립트는 Qwen2.5-Math-1.5B용 async RL + HPT의 **M4(lenient-harvest + adv-std) run**이다.
# D0 런처(run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_main.sh)를 그대로 상속하되
# ablation 두 축(docs/Ablation_RL.md §3)을 명시적으로 켠다:
#   - C1 = decoupling: async_hpt.rl_old_logprob_source=entry + entry_proximal=recent(옵션 B),
#     algorithm.rollout_correction.rollout_is=token(TIS-w, C_w=2.0), rollout_rs=null(rejection OFF).
#   - C2 = CISPO g-slot: actor.policy_loss.loss_mode=cispo + upper-only 클립
#     (clip_ratio_low=10.0으로 하한 비활성, 상단 cap=1+clip_ratio_high=1.28). 이 1.28은
#     이동-partition 스케일이자 M−cispo(vanilla)의 상단 밴드와 정렬 — C2 축을 "메커니즘
#     (gradient 死 vs 유지)"만으로 격리한다(DR-005 §6/정정4). CISPO cap은 upstream verl
#     단일 채널(clip_ratio_low/high)로 지정하며, 별도 cispo_epsilon_high knob은 폐기됨.
# 그 외 학습-문제 인자·공통 기반(DR-001~003)은 D0와 동일. val_before_train=True로 켜 논문 baseline
# 대조의 step-0 앵커를 남긴다. M−cispo(C1만)·M−dec(C2만)는 이 파일에서 해당 축만 꺼서 얻는다.
# 기존 7B main launcher를 덮어쓰지 않고 별도 파일로 둔다. 이유는 7B run과
# 1.5B run이 같은 데이터/목표를 공유하더라도 HPT 강도(beta)와 모델 경로가
# 다르므로, 실행 기록과 rollout dump를 파일명 수준에서 분리해야 하기 때문이다.

# RL conda env는 CUDA 13 runtime/cuDNN/NCCL wheel stack을 activation hook으로
# 노출한다. nohup/non-interactive 실행에서도 SGLang subprocess가 같은 loader
# path를 보도록 hook을 직접 source한다.
# [각주 ⑦-보강] 64k 토큰 마이크로배치는 가변 길이 패킹의 단편화에 민감하다.
# expandable_segments로 allocator 단편화를 완화한다(reserved 172GB@32k 실측의 주범).
# Ray worker는 driver 환경을 상속하므로 여기서 export하면 트레이너까지 전파된다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
#     죽이지 않는다(이전 upt_v6의 raise 크래시 계열을 원천 차단). 따라서 UPT entropy_math
#     scorer 경로 의존성은 제거한다.

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
ROLLOUT_DUMP_DIR="${VERL_ROOT}/.cache/rollout_dump/openr1_async_hpt_qwen25_math_1_5b_M4_${RUN_TIMESTAMP}"

# 학습-side per-token dump (docs/Ablation_RL.md §10). 위 생성 dump가 담을 수 없는
# loss 경계 텐서(log_probs/old_log_probs/rollout_log_probs/advantages/mask)를 저장한다.
# A1(정답×길이)·A6c(death 밀도) offline 분석의 유일한 소스라 D0 baseline부터 켠다.
# read-only·sampled·offload라 학습을 막지 않는다(§10 무게 검토). run별 유니크 dir.
TRAIN_DUMP_DIR="${VERL_ROOT}/.cache/train_dump/openr1_async_hpt_qwen25_math_1_5b_M4_${RUN_TIMESTAMP}"

# Val 생성물 JSONL 덤프(관측성 전용). rollouter actor에서 wandb table이 침묵 드롭되는
# 버그(tracking.py `wandb.run is None` 가드)와 무관하게 val 출력을 디스크에 남긴다.
VAL_DUMP_DIR="${VERL_ROOT}/.cache/val_dump/openr1_async_hpt_qwen25_math_1_5b_M4_${RUN_TIMESTAMP}"

# Scratch launch only: no resume path.

# [각주 3] batch scale은 UPT train_batch_size=128 prompt groups와 맞춘다.
# queue sample은 row가 아니라 prompt group이다. rollout.n=8이므로
# ppo_mini_batch_size(32) * require_batches(4) = 128 prompt groups,
# 즉 fit_step마다 128 * 8 = 1024 generated rows를 본다.
# 64 * 2도 prompt-batch parity는 맞지만, 32 * 4가 HPT learner-row divisibility와
# async trainer scheduling에 더 덜 brittle해서 1.5B 첫 main run 기본값으로 둔다.
REQUIRE_BATCHES=4
# 2048→384: 배치 유계화(원본 healthy M의 ~365그룹 재현, 최소 128의 3배). 큐 상한=배치 상한
# → 스텝 30분→~8분·OOM 제거·데이터 신선. 감시: idle_ratio↑면 512로 완화 (Improvement_RL §5.8.4)
MAX_COMPLETED_PROMPT_GROUPS=768

# [각주 4] UPT와의 대응 관계:
#   UPT SWITCH_GATE=0
#     -> 8개 rollout 중 성공 0개인 prompt만 offline/SFT target을 넣는다.
#   ours async_hpt.gamma=0.0
#     -> success_count / rollout.n <= 0.0, 즉 0/8 전멸 prompt만 SFT route.
# 따라서 gamma=0.0은 routing/gate 관점에서 UPT SWITCH_GATE=0에 대응한다.
#
#   UPT SFT_LOSS_COEF=0.3 for Qwen2.5-Math-1.5B
#     -> 별도 SFT CE loss에 0.3을 곱한다.
#   ours async_hpt.beta=0.3 (constant)
#     -> tau* SFT row의 terminal pseudo reward를 0.3으로 둔다.
# 따라서 beta=0.3 constant는 UPT 1.5B 계수 정합이자, 관대-지표 34.0을 달성한
# D0·M 두 run의 공통값이다(§5.10 — β1.0 leninv는 M2 사망의 제1용의자로 제거).
#
# [각주 5] M4의 SFT는 "약하게, RL이 지표를 직접 밀게" 두는 배분이다: SFT 질량 ~5-10%
# (train_dump 실측 기준)로, 그래디언트의 90%+가 관대-지표를 직접 최적화하는 RL에 간다.
#
# [각주 6] DR-001~003에서 고정된 objective 계약:
#   - B_eff/prompt_equal 분모를 쓰지 않고 branch_blind reduction을 쓴다.
#   - actor loss aggregation은 seq-mean-token-sum-norm, L_max=8192로 둔다.
#   - SFT row는 self-detach CE 의미를 갖고, entropy/KL auxiliary에서 제외한다.
# 이 값들은 단순 튜닝값이 아니라 우리 방법의 정체성이므로 비교 run에서도 유지한다.
#
# [각주 7] rollout sampling parity: UPT exp_scripts/train.sh는 학습 rollout에
# temperature=1.0만 명시하고 top_p/top_k는 두지 않아 verl config 기본값
# (top_p=1.0, top_k=-1)을 쓴다. 우리 rollout config 기본값도 동일하므로
# (rollout.yaml/rollout.py), 여기서도 temperature=1.0만 명시하고 top_p/top_k는
# 기본값에 맡겨 UPT와 런처 형태·유효 샘플링을 일치시킨다. (val은 양쪽 모두
# temperature=0.6, top_p=0.95를 명시하므로 val_kwargs는 그대로 둔다.)
#
# [각주 8] clip은 UPT와 의도적으로 다르다(선언). UPT exp_scripts/train.sh는
# loss_remove_clip=True로 PPO ratio clip을 끈다 — UPT가 동기(near-on-policy)라
# ratio가 1 근처에 머물러 unclipped가 안전하기 때문이다. 우리는 fully-async
# off-policy라 trust region이 필수다(DR-005 §5: raw/no-clip 금지). 단 M은 그 trust
# region을 두 partition으로 분리한다: staleness는 w-슬롯(TIS, rollout_is=token, C_w=2.0)이
# down-weight로 흡수하고, movement는 g-슬롯(CISPO)이 상단 cap 1.28로 유계화한다. 따라서
# 여기서는 clip_ratio_low=10.0(1-10<0 → 하한 무효 = CISPO upper-only, 하한은 원논문이 비활성)
# + clip_ratio_high=0.28(상단 cap=1.28, 이동-partition 스케일) + clip_ratio_c=10.0(cispo에선
# 미사용, upstream vanilla와 공유하는 상수라 유지)를 둔다. D0/M−cispo의 vanilla Clip-Higher와
# 상단 값(0.28)을 같게 맞춰, C2 축(vanilla↔CISPO)이 "같은 지점에서 gradient 死 vs 유지"만으로
# 갈리게 한다. all-SFT 국면에선 SFT self-detach(ρ≡1)라 clip이 no-op이다.
#
# [각주 9] grain은 UPT와 다르다(선언). UPT는 ppo_mini_batch_size=64(128/64 =
# fit당 SGD 2회), 우리는 32(fit당 4회)다. 둘 다 fit당 128 prompt-group(=1024 row)
# 스케일은 동일하고(각주 3) optimizer grain만 더 잘다 — async trainer scheduling과
# HPT learner-row divisibility에 덜 brittle하려는 선택이다.
#
# [각주 10] D0 정체성 명시 (docs/Ablation_RL.md). 아래 세 값은 verl 기본값과
# 동일해 동작은 바뀌지 않지만, ablation 비교의 앵커(D0)를 코드에 못박기 위해
# 명시한다. 누락이 아니라 의도된 고정이다.
#   - actor.ppo_epochs=1
#       A6c(g-슬롯 CISPO) 정보성 논증이 이 값에 의존한다(DR-005 §4.1: anchor B +
#       ppo_epochs=1이라 r 표류가 작아 death 표면이 작다). 기본 1이지만 명시 고정.
#   - async_hpt.rl_old_logprob_source=rollout
#       coupled anchor 고정. entry(decoupling, DR-004 §9)는 A5 arm에서만 켠다.
#       현재 Literal["rollout"]이라 강제되지만, entry 추가 후에도 D0가 명시로 남게 한다.
#   - async_hpt.k_max=null
#       RL row의 학습-시점 staleness 드롭을 의도적으로 끈다. 낡음은 예산 레벨
#       (async_training.staleness_threshold=2.0)로만 제어하고, A5에서 절단 IS(C_w)로
#       보정한다(DR-005 §7-1의 C_w×k_max 공동 조율). A5 arm에서도 null로 고정해
#       델타가 anchor 한 축만 벌어지게 한다.

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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.actor.clip_ratio_low=10.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo \
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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.max_model_len=9728 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=9728 \
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
    trainer.experiment_name=qwen25_math_1_5b_openr1_async_hpt_M4_lenient_advstd \
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
    async_hpt.tau_dataset_path="${DATA_DIR}/train.parquet" \
    async_hpt.tau_messages_key=tau_messages \
    async_hpt.fail_on_missing_tau=True \
    async_hpt.trajectory_scheduler.enabled=True \
    "$@"
