# Onboarding: AAAI_RL (async-HPT `verl` fork)

> **REFERENCE ONLY — tracked codebase onboarding.** 이 문서는 fork의 코드와 실험 역사를 이해하기
> 위한 저장소 내 원본이며, 현재 StreamWeave 원고의 주장·용어·증거 위계를 소유하지 않는다. 논문
> 작업은 `papers_RL/README.md`에서 시작하고, 이 문서는 구현 provenance가 필요할 때만 참조한다.
> 코드 분석 기준은 `644bdd60` (2026-07-10)이며, 이후 paper-only commit은 아래 구현 경계를 바꾸지
> 않는다. 문서 내용은 2026-07-17에 마지막으로 검증했다.

이 문서는 커밋 기록을 근거로 **verl 업스트림 부분**과 **이 리포지토리 고유 작업(async-HPT)**을 분리해
정리한 온보딩 자료다.

> **분석 기준 이후의 핵심 변화.**
> 저장소는 이제 **신 B200 서버로의 핸드오프(마무리) 단계**에 있다. 지난 세대 대비 두 가지가
> 크게 바뀌었으니 먼저 알고 시작한다:
> 1. **연구 결론이 뒤집혔다(§6).** 옛 앵커 M(=CISPO)이 아니라 **CISPO를 뺀 nocispo가 현행 main**이다.
>    문서 곳곳의 "M을 먼저"류 지시문은 이행 완료된 역사다 — 현행 진실은 `Ablation_RL.md` **§14**.
> 2. **HPT 구현이 둘로 늘었다(§5).** 비동기(연구 코어)와 동기(논문 재현 baseline)가 공존한다.

## 1. 리포지토리 정체

`verl`(ByteDance/volcengine의 RL-for-LLM 프레임워크)을 클론해 그 위에 하나의 연구 라인을 얹은
포크다: **HPT (Hybrid Post-Training)** — 프롬프트 그룹 단위로 on-policy RL과 expert trajectory(τ)
지도학습을 라우팅하는 목적함수를, **fully-asynchronous RL 런타임**(rollout 생성과 학습이 겹쳐 도는
구조) 위에서 구현한 것. (구명칭 "Hybrid Policy Training"은 폐기, UPT/arXiv:2509.04419 계열.)

총 2,807커밋 중 마지막 **101개**(2026-07-02 ~ 2026-07-10, 9일)가 이 프로젝트 고유 작업이고 나머지
전부가 verl 업스트림이다.

## 2. verl 업스트림 vs 프로젝트 작업 — 정확한 경계

```
git log --oneline 93d94c60^..HEAD   # 프로젝트 고유 101커밋
```

| | |
|---|---|
| **Fork 시작점 커밋** | `93d94c60` "Document clean verl export baseline" (goonco, 2026-07-02) |
| **직전 업스트림 마지막** | `91666d99` "[rollout] fix: support SGLang FP8 …" (GEM, 2026-07-02) |
| **verl 업스트림 커밋 수** | 2,706 |
| **프로젝트 고유 커밋 수** | 101 (148 files, +28,072/−345) |
| **프로젝트 작성자** | `goonco` (19, 07-02~03 초기 이식/설정) → `eoeldroal` (82, 07-03~10 본작업) |
| **HEAD** | `644bdd60` "Record final migration footprint" (eoeldroal, 2026-07-10) |

즉 `git log 93d94c60^..HEAD`가 곧 "이 프로젝트가 실제로 한 일"의 전체다. verl 자체의 아키텍처를
알고 싶다면 별도로 upstream 문서(`docs/extend_guide.rst` 등)를 봐야 하고, 이 온보딩은 프로젝트
고유 부분만 다룬다. AGENTS.md의 원칙("업스트림 계약으로 확장, 병렬 스택을 만들지 않는다")대로,
건드린 표면은 좁게 유지된다.

## 3. 프로젝트가 건드린 표면

- **`verl/experimental/fully_async_policy/`** — 비동기 arm의 홈. 신규 7파일(`hpt_assembler.py`,
  `hpt_config.py`, `hpt_gate.py`, `hpt_payload.py`, `hpt_rollout_accumulator.py`, `hpt_training.py`,
  `training_dump.py`) + 기존 driver/rollouter/trainer 수정.
- **`recipe/paper_hpt/`** — **신규 동기 arm**(§5). UPT 논문 코드 재현 baseline. 공유 트리는
  default-off 훅 2곳(`custom_loss_fn` + `algorithm.paper_hpt.enable`)으로만 건드린다.
- **`verl/workers/utils/losses.py::ppo_loss`** — branch-blind 정책 손실(연구 코어). 3모드.
- **`verl/trainer/ppo/`** — `core_algos.py`(cispo/cispo_klcov 추가), `metric_utils.py`,
  `rollout_corr_helper.py`(off-policy correction, RL row 전용).
- **`scripts/migration/` + `docs/MIGRATION.md`** — 핸드오프 bundle 워크플로(§7).
- **`tests/special_RL/`** — CPU-only 계약 테스트 13개. **`main_scripts/`** — launcher 19개.
- **`docs/`** — 설계/운영 문서 13개 + `papers_RL/`(초안 `Draft.tex`/`Efficiency.tex`, 슬라이드 생성기).

## 4. 문서 지도 (읽는 순서)

| 문서 | 역할 |
|---|---|
| `AGENTS.md` | 이 리포에서 일할 때의 durable한 규칙(기여 정책, 엔지니어링 원칙, 테스트 원칙) |
| `docs/Overview_RL.md` | 이 fork가 **무엇이고 왜** 이런 모양인가 (문제의식 + 설계 기여 + 보증 G1~G5) |
| `docs/Codemap_RL.md` | 코드가 **어디**에 있고 런이 **어디서** 깨지는지 (심볼 기준 + "Where Did It Break?") |
| `docs/Readme_RL.md` | 환경 셋업, 실행, 로그 점검 |
| `docs/AsyncBudget_RL.md` | 큐/staleness/HPT 배치 사이징. **trim+carryover 패러다임** + Operating Principles P1~P6 |
| `docs/Debug_RL.md` | lint/profiling/성능 진단 + landmine(fsdp2 등) |
| `docs/DR-001~005*.md` | 손실함수 설계 결정 기록 (이론적 근거) |
| **`docs/Ablation_RL.md`** | ablation 설계·분석. **§14(2차 재앵커링)가 현행 단일 진실** — §0~§13은 역사 |
| `docs/Improvement_RL.md` | 실제 run 병리 분석 + M-계열 개선 캠페인 시간축(§5.7~§5.13) |
| `docs/MIGRATION.md` | 핸드오프 bundle 범위·검증 절차 |

> **문서 읽는 요령:** Ablation·Improvement는 결론에 도달한 *과정*을 층층이 보존한다.
> "M을 최우선으로"·"M=CISPO 앵커" 같은 문장을 만나면 **역사 기록**임을 기억하고,
> 현행 판정은 항상 `Ablation_RL.md §14`로 확인한다.

## 5. 시스템 아키텍처 + 두 개의 HPT 구현

기존 verl의 fully-async 실행 모델(rollout과 training이 겹쳐 도는 구조) 위에 프롬프트 단위 HPT
라우팅을 얹었다. 3개의 Ray 액터:

```
[FullyAsyncRollouter] --put_sample--> [MessageQueue] --get_sample--> [FullyAsyncTrainer]
   rollout 생성 + reward           (cloudpickle, deque)              학습(actor/critic/ref WG) + 업데이트
```

- **라우팅**(`hpt_gate.py`): 프롬프트 그룹 성공률 ≤ `gamma`면 SFT(τ 지도학습), 아니면 RL.
- **비대칭**: SFT는 그룹→학습 row 1개로 축약, RL은 `rollout.n`개로 확장 → "큐 샘플 수 ≠ 학습 row 수".
- **핵심 보증 G1~G5**: 부분 그룹 미방출 · RL/SFT 단일 `DataProto` 계약 공유 · RL row는 **기본**
  rollout-anchored old-logprob(entry-anchor decoupled는 flag-off 옵션) · partial 복구 시 정렬 보존 ·
  SFT row는 rollout correction 면제.
- **branch-blind policy loss**(`losses.py::ppo_loss`): SFT row는 `old=log_prob.detach()`로 ratio≡1
  (advantage-weighted NLL), RL row는 base 손실. 손실함수는 분기하지 않고 **데이터가 분기**한다.
- **배치 수집 = trim+carryover**(`fully_async_trainer.py`): 정렬 최대 배치로 trim, 잔여는 다음 스텝
  이월. 구 grow-to-align 크래시(row-alignment)를 대체 → "128그룹/fit-step = 논문 x축 parity".

**두 구현 — 같은 objective, 다른 arm** (온보딩 시 반드시 구분):

| | **async HPT** (`fully_async_policy/`) | **sync HPT** (`recipe/paper_hpt/`) |
|---|---|---|
| 역할 | 이 포크 고유의 **research/system** arm | UPT 논문 코드 재현 **baseline** arm |
| 런타임 | 3-actor 완전 비동기 | 표준 sync colocated (`RayPPOTrainer.fit`) |
| loss | branch-blind self-detach reward-injection | 명시적 dual-loss(no-clip RL + β·masked_mean SFT), dp_size 곱 없음 |
| routing | assembler 직접 조립 | template cloning + DP-divisor loss-neutral padding |
| 통합 | 포크의 홈 | default-off 훅 2곳 (켜지 않으면 byte-identical) |
| 데이터/grader | v2 / math_verify | strip / entropy_math |

> 함정: `recipe/paper_hpt/paper_hpt_trainer.py`(`PaperHptTrainer`)는 **dead code**다 —
> live 경로는 fit-hook + template cloning. sync baseline 런 `v96fvd0p`은 8×-scale 버그로 돌아
> grad/loss가 논문의 8배 → 공정화 전까지 "sync 압도" 비교 금지(§6).

## 6. 연구적 핵심 — C2 대반전과 현재 상태

이 프로젝트의 연구 서사는 M-계열 개선 캠페인(`Improvement_RL.md §5.7~§5.13)이 도달한
**2차 재앵커링**(`Ablation_RL.md §14`)으로 요약된다.

**발단.** 원조 M(decoupled+CISPO)은 "val은 순항하는데 학습 rollout이 붕괴"하는 병리를 보였다
(길이 1,184→7,500+, entropy 1.0→7.3). 근본원인은 3단 사슬: all-SFT 국면의 τ 길이-과채택(exposure
bias) → **정답 보상의 41%가 비종료 응답**이라 비종료를 양으로 강화 → entropy↑의 자기증폭. 이를
고치려다 **잠복 라우팅 버그**(`hpt_gate`가 보상을 `rm_scores[-1]` 우패딩에서 읽어 clean 응답을
항상 0으로 오독)를 발견해 `sum(-1)`로 수정했다.

**캠페인(M2→M7).** "정직(P0)한 arm은 안정적이나 천장이 낮다" vs "성능-최우선 arm은 신기루
래칫에 빠진다"의 긴장 속에서 진화. M4에서 **HPT 자가치유**(위기 시 SFT 바닥재가 정책을 회복)를
발견하고, M5(clean-async)가 앵커로 승격됐다.

**대반전(§14).** C2를 제거하는 arm **M5abl_nocispo**(`oki4kv8u`, 델타 = `loss_mode: cispo→vanilla`
+ `clip_ratio_low: 10→0.2`뿐)가 앵커를 이겼다:

| | C2=vanilla | C2=CISPO |
|---|---|---|
| **C1=decoupled** | **main = nocispo** (`oki4kv8u`) ✅ 정점 40.17/52.06@170, 190스텝 무폭풍 | M5 (`f5ugxklh`) = main+cispo |
| **C1=coupled** | D0′ — 취소 | M5abl_nodec — 후순위 |

- **C2(CISPO) = 실증 반증·기각.** outcome 열위(정점 −1.7) + ~step100 **폭풍 벽의 진범**. 기전:
  CISPO의 `sg(clip(r))·A·logπ`는 min-clip 브레이크가 없어 과확신 토큰을 계속 밀어 entropy 붕괴→KL
  폭풍. vanilla는 clip이 브레이크로 실작동. M4/M5/M7/M5R 폭풍 4런의 공통분모는 스택이 아니라
  전부 CISPO 계열이었다("구조적 벽" 가설 반증).
- **C1(decoupling) = 무런 폐쇄.** main 실측 `P(w>C_w=2)` 중앙값 0.10%, w̄=0.954 → 이 레짐은 낡음이
  낮아 coupled≈decoupled(DR-004 §6). D0′ 신규 런 취소(런처는 보존).
- **신규 H축(교사 채널) = 논문의 실제 핵심 결과.** `success_threshold: 0.0→-1.0` 센티널로 라우팅만
  봉인한 RLonly(`qzsnwc08`)와 대비 → 초반은 동등, **후반(130-160) +3.4·정점 +2.4**. 교사 채널의
  값어치는 후반 지속-상승·안정성에 있고, 논문 핵심 가설이 실증됐다.

**함의.** 방법론적으로 이건 실패가 아니라 "도출 → ablation 확인 → 반증 시 수정"의 정당한 완결이다.
논문 무게중심이 CISPO/decoupling(둘 다 이 레짐에서 조건 미충족)에서 **async-HPT 아키텍처
(transport↔semantics 분리) + 교사 채널**로 이동했다.

**코드 레벨 변경(직접 검증):** 정책손실 3모드(`vanilla`(main)/`cispo`(기각)/`cispo_klcov`(미증명)) ·
trim+carryover 수집 · truncation 처리(`zero_reward_if_truncated`/`zero_truncated_rl_advantage`,
`hpt_is_truncated_rl`) · 큐 zero-variance eviction · base model 7B→**Qwen2.5-Math-1.5B**.

## 7. 실행·디버깅 진입점

```bash
# CPU 전용 계약 테스트 (GPU/Ray 불필요, 가장 먼저)
conda activate RL && pytest tests/special_RL/ -v

# 현행 research main (nocispo) / resume
bash main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_M5abl_nocispo.sh
bash main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_M5abl_nocispo_cont.sh   # 190→300

# smoke run (8 GPU)
bash tests/special_e2e/run_fully_async_policy_sglang_smoke.sh
```

런이 깨졌을 때는 `Codemap_RL.md`의 "Where Did It Break?" 표(증상 → 먼저 볼 곳)를 먼저 확인.
핸드오프 검증은 `MIGRATION.md`의 절차(clone 일치 → dry-run → checksum → 추출 검증).

## 8. 다음 단계 (Ablation §14.5 판정 대기 원장)

1. **protocol-fair fixed-checkpoint 재평가**: {nocispo@170, @190, LUFFY}를 동일 grader·decoding·
   budget에서 문항당 32 stochastic generations의 mean@32와 문항단위 paired hierarchical
   bootstrap 10,000회 95% CI로 평가한다. 공통 evaluation-seed set은 재현성 장치이며 독립 RL
   training seed가 아니다. "LUFFY 상회/동급"의 지면 확정은 이것으로만 하고, 논문 41.9와 직접
   비교하지 않는다(grader 관대·k 불일치).
2. **nocispo_cont(190→300)**: 150-190 고원이 완만 상승 중(+0.015/step)이라 잔여 상승 확인.
3. **paper-HPT sync(`v96fvd0p`) 공정화**: 8×-scale 유효성 판별 + 동일 채점기 재채점. 둘 다 끝나기 전
   "sync 압도" 주장 금지.
4. 후순위 arm: M5abl_nodec(대각선/상호작용), M5−advstd(파리티 이탈 방어), A1(β_r 산출 선행).

run registry 전체·라벨 해독은 `../../memory/glossary.md`, 프로젝트 상태 요약은
`../../memory/projects/aaai-rl.md`.
