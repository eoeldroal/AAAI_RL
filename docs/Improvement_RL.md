# Improvement_RL — M run 병리 분석과 개선 결정 (사례 기록 · Best Practice)

Status: 분석 확정(2026-07-06) · 개선 결정 P0/P1 합의 · P0+관측성 구현(§5.6) · **★M′ 런 순수-SFT 붕괴 사후분석 완료(§5.7): 근본원인=라우팅 `rm_scores[-1]` 위치버그(P0-1이 노출), 수정 적용·테스트 통과★** · 라우팅 수정 후 재시작 완료 · **★2026-07-07: P0 런(Mprime2/3) 실측 = 정직·안정하나 val ~30% 천장 · 병리 M은 신기루로 33.76+ 지속 상승 → 성능-최우선 M 궤적 회귀(resume40) · 신규 async 배치 폭발 진단·유계화 처방 · §11 진단 DP-집계 버그 수정(§5.8)★**
범위: M run(`M_decoupled_cispo`)에서 관측된 응답길이·entropy 폭발 병리의 (1) 실험 기록, (2) 분석 방법과 결과, (3) 진단, (4) 개선 결정, (5) 재사용 가능한 분석 방법론(Best Practice). 설계 결정의 근거·이론은 DR-001~005 소관이며 여기서 재서술하지 않는다.
관련 문서: `Ablation_RL.md`(격자 설계 — 이 문서의 개선이 공통 기반을 바꾸므로 §5.4 참조) · `DR-002`(entropy 결정 — 본 문서가 개정 사유 제공) · `DR-003`(§5 drift-pacing — 본 문서 §4.2가 사전 관문 측정치에 해당)
관련 코드(symbol 기준): `losses.ppo_loss` · `core_algos.compute_policy_loss_cispo` · `core_algos.compute_grpo_outcome_advantage`(singleton 그룹 mean=0 처리) · `hpt_gate.count_successful_rollouts` · `hpt_assembler.materialize_sft_payload` / `_sft_terminal_reward` · reward_manager `dapo`(overlong_buffer/max_resp_len) · `ray_trainer`의 `calculate_entropy or entropy_coeff != 0` 게이트

---

## 0. 한 문단 요약

M run(decoupled+CISPO, v2 proper-prompting)은 val이 순항(6벤치 전반 상승)하는 동안 학습 rollout이 붕괴했다 — 응답길이 1,184→7,500+(clip 89%), RL-토큰 entropy 1.0→7.3, rollout↔train KL 12 nats. 정량(rollout dump 2계층) + 정성(Sonnet 서브에이전트 전수 정독) + wandb 전 메트릭 스윕(291키)으로 해부한 결과, 병리는 **3단계 인과 사슬**이었다: ① all-SFT 국면에서 tau의 "길이"만 채택되고 "종료"는 전이 실패(exposure bias) → ② 보상 체계가 비종료를 직접 강화(**정답 보상의 41%가 truncated 비종료 응답**) + truncated 실패의 음(−) advantage 홍수가 분포 평탄화 → ③ RL 질량 증가와 함께 자기증폭(ppo_kl 2.4 스파이크, gibberish attractor). 반면 용의선상의 lr/clip_grad/CISPO cap/버전-lag staleness/온도는 **전부 무죄**로 판정됐다(§4.3). 개선은 "비종료=실패" 원칙을 reward 한 지점에 심어 라우팅·baseline·보상을 동시에 고치고(P0), truncated row를 loss에서 제거하며, entropy bonus 제거 + beta 강화(되돌림 기준 사전 등록)로 구성된다(§5). 이 변경은 Ablation 격자의 공통 기반을 바꾸므로 **기반 v2·M′ 재앵커**를 선언한다(§5.4).

---

## 1. 실험 대장

| 항목 | D0 run | M run (본 문서의 분석 대상) |
|---|---|---|
| experiment_name | `qwen25_math_1_5b_openr1_async_hpt_beta03_constant` | `qwen25_math_1_5b_openr1_async_hpt_M_decoupled_cispo` |
| launcher | `main_scripts/..._main.sh` | `main_scripts/..._M.sh` |
| 구성 | 공통 기반만 (C1/C2 off) | 공통 기반 + C1(entry+TIS-w) + C2(CISPO 1.28) |
| 데이터 | openr1_hpt_main (v1, system prompt strip) | **openr1_hpt_main_v2** (proper prompting, eval에도 LUFFY system 주입) |
| 채점기 | (v1 계열) | `math_verify_adapter`(HF Math-Verify 통일) |
| val | 3벤치, n=8 | **6벤치**(AIME24/25·AMC23·MATH-500·Minerva·Olympiad), n=8, val_before_train |
| wandb | run-20260704 계열 | **run-20260705_195604-uvbi7wq3** (37 history rows, global_step 0–143, param_version 0–35, 6.25h) |
| rollout dump | — | `.cache/rollout_dump/...M_20260705_195531/` (48,057 그룹) |
| 요약 결과 | step10 MATH-500 13.3(포맷 붕괴)→step30 회복→step70 [11.7/46.9/71.5] | 아래 §2 |

**M run val 궤적 (mean@8, val step 0/39/79/119):**

| 벤치 | 0 | 39 | 79 | 119 |
|---|---|---|---|---|
| MATH-500 | 36.9 | 43.0 | 52.9 | **66.0** |
| AMC23 | 27.5 | 26.6 | 34.4 | 40.0 |
| Olympiad-Bench | 17.6 | 21.3 | 24.6 | 31.2 |
| Minerva | 12.4 | 14.4 | 18.2 | 23.0 |
| AIME25 | 2.5 | 5.4 | 4.2 | 8.3 |
| AIME24 | 5.4 | **4.2 (역행)** | 6.3 | 8.3 |

step0(base)이 논문 base와 정합(v1 대비 — LUFFY-strip 가설 확증)하고, 4개 저노이즈 벤치는 매끈히 단조 상승. **단 AIME24가 길이 폭발 개시 창(step 19–35)에서 유일하게 역행** — 최난도 벤치가 병리의 카나리아다(§7-BP5).

---

## 2. 관측된 병리 — 3단계 해부 (wandb 궤적)

| 지표 | step 3 | 19 | 35 | 43 | 67 | 83 | 99 | 143 |
|---|---|---|---|---|---|---|---|---|
| response_length/mean | 1,184 | 1,211 | 5,289 | 6,901 | 7,467 | 6,567 | 7,171 | ~7,500 |
| response_length/clip_ratio | 0.0005 | 0.005 | 0.315 | 0.65 | 0.79 | 0.58 | 0.70 | **0.891** |
| actor/entropy_mean (RL 토큰) | 1.01 | — | 1.45 | 1.50 | 2.07 | **7.32** | 5.77 | ~5 |
| actor/ppo_kl | 0 | 0 | 0 | 1e-4 | 7e-4 | 0.35 | 0.07 | (step91 **2.4**) |
| actor/pg_clipfrac | 0 | 0 | 0 | 2e-5 | 4e-4 | 0.040 | 0.107 | 0.070 |
| hpt/onpolicy_success_rate | 0.0002 | **0** | 0.008 | 0.020 | 0.035 | 0.054 | 0.056 | 0.29 |
| actor/hpt/sft_nll | 1.06 | 0.82 | 0.68 | 0.63 | 0.43 | 0.36 | 0.32 | ~0.35 |
| rollout_corr/kl (rollout↔train) | 0.007 | — | — | — | — | — | **12.0**(@95) | 1.2–6.8 |
| rollout_is_eff_sample_size | 0.99 | — | — | — | — | — | **0.42** | 0.42–0.77 |

- **1단계 (step 0–35, 사실상 all-SFT):** success_rate 0(step 7–19는 정확히 0, 4행 연속) → 전 그룹 SFT 라우팅. 그런데 **이 국면에서 길이가 1,211→5,289로 폭발 시작**(clip 0.5%→31%). sft_nll은 매끈히 하강 — "SFT 자체는 잘 학습되는데 생성 행동이 붕괴"하는 전형적 off-policy distillation 실패.
- **2단계 (step 35–67):** 길이 6,900–7,500, clip 65–80%. 이때도 pg_clipfrac≈0, ppo_kl≤7e-4 — **clip/이동이 아니라 gradient의 '방향'이 문제였던 구간.**
- **3단계 (step 67+):** RL 질량 증가와 함께 자기증폭 — ppo_kl 2.4 스파이크(@91), entropy_mean 7.3(top20은 10.9 ≈ 균등분포 11.9 근처), 유효 staleness 폭발(KL 12 nats, ESS 0.42).
- **역설:** 이 와중에 val(temp 0.6)은 순항. temp 0.6이 고엔트로피 attractor를 회피 + tau SFT가 성능을 견인. 즉 "val이 오른다"는 병리 부재의 증거가 못 된다(§7-BP5).

---

## 3. 분석 방법과 결과

### 3.1 정량 1 — rollout dump 중반 창 (2,504 rollouts)

방법: DataProto 직독(`response_mask` 합=길이, `acc`, UPT boxed 추출기 재사용).

| 측정 | 값 | 의미 |
|---|---|---|
| clip(≥8191) | 1,316/2,504 = 52.6% | 중반 창 기준 |
| **acc=1 중 truncated** | **108/264 = 41%** | ★ 양(+)의 advantage의 41%가 비종료 응답에 흘러 **"답 내고 끝내지 마라"를 보상이 직접 학습시킴** |
| clipped 중 boxed 없음 | 1,076 (81.8%) | reward 0 → 혼합 그룹에서 8,192토큰 음(−) advantage 홍수 |
| 프로그램적 repeatness 플래그 | **0/1,316** | 휴리스틱 재현율 0 — §7-BP1의 근거 |

### 3.2 정성 1 — clipped 250개 전수 정독 (Sonnet 서브에이전트 ×5, 샤딩)

GIBBERISH 80.4% · BOXED_BUT_CONTINUED 12.8% · REPETITION 5.6% · WANDERING 1.2% · **GENUINE_PROGRESS 0% · "12288로 늘리면 구제" 0/250**. → response_length 확대는 무익·유해로 판정(길이가 아니라 종료의 문제).

### 3.3 정량 2 — 최신 창 그룹 단위 실측 (1,500 그룹 / 11,883 rollouts, Sonnet)

개선안(truncation 처리)의 **효과를 사전 실측**하기 위한 분석. 최신 체크포인트 기준:

| 측정 | 값 |
|---|---|
| 전체 clip | **88.7%** (정답의 89.8%가 clipped — 중반 창보다 악화) |
| 현행 라우팅: SFT/RL | 28.4% / 71.6% |
| RL 그룹 중 "정답 전부 clipped" | 741/1,074 = 69.0% → **전체의 49.4%가 loss-mask만으론 죽은 그룹** |
| 새 규칙(성공=정답∧종료): SFT/RL | 77.8% / 22.2%, RL 그룹당 생존 양성 row 평균 1.32 |
| unclipped-정답 길이 | p50 **1,323**, p90 2,773 (tau 중앙값 4,172의 1/3) |

### 3.4 정성 2 — unclipped 30+30 정독 + clipped-정답 box 위치 (Sonnet)

- unclipped 정답 30/30: **일관 풀이 → boxed → 즉시 EOS** (요행 조기종료 0건) — truncation 필터 후 남는 양성 신호의 품질이 검증됨.
- unclipped 오답 30/30: 일관된-오답(건강한 음의 신호). 필터를 새는 짧은 gibberish는 3/11,883 = **0.025%** (무시 가능).
- clipped-정답: 첫 boxed가 응답의 **평균 8.2%** 지점, 이후는 추론이 아니라 반복 루프/다국어 샐러드(mode collapse).
- **`</think>` 규율: ~53%는 아예 안 닫고, 닫는 경우 100% boxed 이후** — tau의 구조(생각→`</think>`→Solution)가 자기 생성으로 전이되지 않았다는 exposure-bias의 직접 증거.

### 3.5 wandb 전 메트릭 스윕 (291 히스토리 키, Sonnet)

- **유효 staleness**: 명목 lag는 설계대로 유계(`max_partial_span` ≤ 1 항상, abort 0)인데 rollout↔train **KL 0.007→12 nats**, chi2 1.25e5, ppl_ratio ~1e8–1e10. → **구속 조건은 sync 주기가 아니라 "sync당 정책 이동 속도"**(폭발이 원인, staleness는 결과).
- **TIS 비대칭 붕괴**: cap 2.0에 걸리는 토큰 ≤7.5%뿐, 반대로 **weight<0.5로 뭉개지는 토큰 최대 59.5%**, ESS 최저 0.42, min weight 2e-9 바닥 고정. C_w가 모자란 게 아니라 하방 붕괴.
- **큐 백로그**: mq_len 0→말미 지속 ~1,400–1,550 (wandb에 없어 output.log에서 복원) — 트레이너가 8,192토큰 업데이트로 느려진 volume 효과(per-token 비용은 +13%로 정상). 소비 시점 데이터가 ~2–3버전 낡아지는 경로.
- **조성 3계층 괴리**: step143에 그룹 기준 71%가 여전히 0/8(SFT행)인데 row 기준 SFT는 5.8%, 토큰 기준 ~0.4% — 단일 지표로 조성을 읽으면 오독(§7-BP3).

---

## 4. 진단

### 4.1 인과 사슬 (확정)

```
[1] all-SFT 국면: tau(중앙값 4,172tok) CE → "길이"는 즉시 채택, "종료"는 전이 실패
    (EOS는 tau 4,172토큰 중 1토큰; 자기 생성 prefix가 tau에서 벗어나면 EOS 문맥 미도달
     — §3.4의 </think> 규율 붕괴가 직접 증거)
[2] 보상 결함: truncated-정답에 reward 1 (정답 보상의 41%) → 비종료가 양(+)으로 강화
    + truncated-실패(reward 0)의 음(−) advantage가 토큰 질량 지배 → 분포 평탄화(entropy↑)
    (Cov(logπ, Â) 부호: 정책이 좋아하는 토큰에 음의 advantage → entropy 상승)
[3] 자기증폭: entropy↑ → temp1.0 샘플링 더 무작위 → gibberish↑ → 실패↑ → 음의 질량↑
    → 정책 폭주(ppo_kl 2.4) → 유효 staleness 폭발(KL 12) → TIS가 gradient 59% 말살
```

### 4.2 DR과의 접속

이 관측은 `DR-003` §5(drift-pacing 가설)의 **사전 관문 측정치가 처음으로 충족된 사례**다 — 단 기제는 optimizer-이동이 아니라 "생성 분포의 길이 채택 vs 종료 전이의 비대칭"으로 더 구체화됐다. `DR-001` §4.3 관점에서는 `sft_beta_mode=constant`가 긴 tau에 예산을 편중(`β·|o|/L_max`, p90 tau=중앙값의 1.8배)시켜 [1]을 데이터 배분 차원에서 가중한다(→ §5 P2 카드).

### 4.3 무죄 판정 (혐의 해제 — 재수사 방지)

| 용의자 | 판정 근거 |
|---|---|
| lr=5e-6, clip_grad=80, entropy 0.001 | **UPT `exp_scripts/train.sh` 원본 그대로**(parity). grad_norm 실측 0.03–0.41로 80은 한 번도 안 걸림 |
| CISPO cap 1.28 (C2) | 폭발 개시 구간(step 35–67)에 pg_clipfrac≈0 — cap이 아예 안 걸렸으므로 범인 불가 |
| 버전-lag staleness / sync 주기 | max_partial_span≤1 항상, abort 0. 유효 staleness는 폭발의 **결과** |
| TIS C_w=2.0 | 상방 포화 ≤7.5%. 문제는 하방(정책 폭주의 결과) |
| temp 1.0 / top_p | UPT parity. val(0.6)이 오르는 건 회피이지 해결 아님 |
| tau 초과길이(>8192) 오염 | v2 metadata `token_limit_action: "filter"` — 빌드에서 이미 제거됨 |

---

## 5. 개선 결정

원칙: **"비종료 = 실패"를 시스템의 한 지점(reward)에 심으면 라우팅·baseline·보상이 한 번에 정합**해진다. 각 결정은 [근거 수치 → 기제 → 구현 지점 → 검증/되돌림]으로 기록한다.

### 5.1 P0 — 재시작 전 필수

| # | 결정 | 근거 → 기제 | 구현 지점 |
|---|---|---|---|
| 1 | **truncated ⇒ reward 0** | 41% 비종료-보상 누수 차단(§3.1) + 라우팅이 score를 읽으므로 성공=정답∧종료가 **자동 성립**(죽은 그룹 49.4% 해소, §3.3) + GRPO baseline 정합(깨끗한 정답 1+truncated-정답 7 그룹에서 mean=1→advantage 0이 되는 병리 차단) | reward_manager `dapo`: `valid_response_length ≥ max_resp_len`이면 score 0 |
| 2 | **truncated RL row의 advantage를 0으로**(advantage 계산 후) | clipped의 정성 판정: 양성 0%·구제 0/250(§3.2) → 순수 독. **구현 recon 정정**: `seq-mean-token-sum-norm` 분모 `global_batch_size`가 고정 상수(`ppo_mini_batch_size×n`, [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) `_update_actor`)라 **advantage-zeroing은 dead-row 희석이 0이고 물리적 제거와 gradient 등가** — 배치 형상/divisibility 리스크 없이. (문서 초안의 "zero-mask→5× 희석"은 live-denominator 일반론이라 우리 config엔 해당 없음.) | `FullyAsyncTrainer._fit_filter_truncated_rl_advantage`(advantage 산출 후, loss 이전) |
| 3 | **entropy_coeff 0.001→0 + `actor.calculate_entropy=True`** | entropy는 bonus(`policy_loss -= c·H`)인데 폭발 국면에 위로 미는 항 유지 중. **동반 플래그 필수**: `calculate_entropy or entropy_coeff != 0` 게이트라 coeff=0만 두면 `entropy_mean`·Ablation §11 진단 3지표가 전부 꺼짐 | 런처 2줄. `DR-002` 개정 필요 |
| 4 | **beta 0.3→1.0** (되돌림 기준 동반) | tau는 100% 깔끔 종료 + unclipped 정답 100% 깨끗(§3.4) → tau 앵커 강화가 종료 학습 지원. **단 반대 증거 존재**: [1]단계가 beta=0.3에서 이미 발화했고 UPT가 1.5B에 의도적으로 0.3 채택 → 도박이 아닌 통제 실험으로: **재시작 후 step 40까지 clip_ratio>0.3 지속 or entropy_mean>2.5면 0.3 회귀** | 런처 1줄 |
| 5 | **step-0 재시작** | 현 체크포인트는 attractor 내부(§3.2–3.4) — 재사용 금지 | — |

> P0-1/P0-2(truncation 처리)가 표준 RL을 벗어나는 지점의 정확성·안전성 상세 검토는 **§5.5**(논문 인용용).

### 5.2 P1 — 가드레일 · 관측성

| # | 결정 | 근거 |
|---|---|---|
| 6 | **트립와이어 사전 등록**: clip_ratio>0.3(5 fit 지속) / entropy_mean>2.5 / KL(rollout↔train)>1 nat 지속 / entropy_mean<0.2(반대편: 탐색 붕괴) / **AIME24·25 카나리아** | D0·M 모두 clip 30–50% 도달(step 31–35)에 개입 없이 지나침. AIME24는 step39에 유일 역행(§1) |
| 7 | **ESS 회로차단기**: `rollout_is_eff_sample_size < 0.6`이면 해당 fit 스킵(우선 수동 기준, 자동화는 선택) | 관측 최저 0.42(§3.5) — 원인 불문 유효-staleness 폭주 시 자동 제동 |
| 8 | **큐 상한 축소**: `MAX_COMPLETED_PROMPT_GROUPS` 2048→512 | 말미 백로그 ~1,400–1,550 = 소비 시점 ~2–3버전 노화(§3.5) |
| 9 | 관측성: mq_len wandb 스칼라 추가 · centered advantage mean 별도 로깅(현행 `critic/advantages/mean`은 score와 alias — §7-BP3) | §3.5 |

### 5.3 P2 — 조건부 카드 (트리거 명시) · 기각 원장

| 카드 | 트리거 | 근거 |
|---|---|---|
| `sft_beta_mode=length_inverse` | P0 후에도 clip 재상승 | constant의 긴-tau 예산 편중 중화(§4.2). DR-001 축 2로 이미 구현 |
| tau 길이 선호/필터(≤4096) | length_inverse로도 부족 | 성공 응답 p50 1,323 vs tau p50 4,172의 3× 미스매치(§3.3). parity 이탈이라 증거 축적 후 |
| rollout min_p | 비상용만 | 샘플링 parity 훼손 — 정책 gradient 편향 |

기각 원장 (본 병리 관련):
- **response_length 12288 확대**: 구제 0/250, 비용 +42% KV — 기각.
- **overlong soft penalty**(`overlong_buffer.enable=True`): truncated-실패의 음의 advantage를 **더 키워** 평탄화 루프 강화 — 마스킹이 맞고 페널티는 방향이 반대.
- **sync 주기 단축(trigger 4→2)**: 유효 staleness는 결과이지 원인(§4.3) — 표적 아님.
- **KL 앵커 도입**: UPT parity 이탈 + `ref.use_ref=False` 인프라 — 최후 수단으로만 보류.

### 5.4 프로그램 차원 — 기반 v2 선언

P0 1–4는 `Ablation_RL.md` §2의 **공통 기반**(전 격자점 고정: entropy 0.001, beta 0.3, 보상 정의)을 바꾼다. 따라서 기완료 D0와의 델타는 더 이상 단일-요인이 아니다. 결정: **기반 v2를 선언하고 M′(=기반 v2 + C1 + C2)을 새 앵커로** — strongest-first 원칙 그대로 M′ 먼저, D0′ 재실행은 M′이 유의미할 때만. `DR-002` 개정 + truncation 처리의 신규 DR 노트 1장을 남긴다.

---

## 5.5 truncated-row 제거의 정확성·안전성 분석 (P0-1/P0-2 심화 · 논문 인용용)

P0-1(truncated⇒reward 0)과 P0-2(advantage 계산 후 truncated row 제거)는 표준 GRPO를 벗어나는 유일한 지점이다. 이 절은 그 이탈이 **정확히 어디서·얼마나 일어나며 왜 안전한지**를 추정량 수준에서 규정한다. 논문에서 방법의 정당화로 인용할 수 있도록 자족적으로 서술한다.

### 5.5.1 이탈의 정확한 규정 — 추정량 차분

우리 정책-gradient는 다음으로 쓸 수 있다:

```
g_ours = g_GRPO(새 reward 정의) − Σ_{truncated row} (그 row의 항)
```

두 가지가 핵심이다. **(1) baseline은 건드리지 않는다** — advantage는 truncated row를 실패로 센 채(제거 *전*) 계산되므로, 살아남는 모든 row의 advantage는 **새 reward 정의 하의 표준 GRPO 값과 완전히 동일**하다. **(2) 이탈의 전부는 "truncated row의 억제 항 생략" 하나**다. reward를 0으로 만드는 것(P0-1)은 라벨/baseline/라우팅을 고치는 것이지 이탈이 아니다(올바른 라벨로의 교정); 이탈은 그 0-reward row를 gradient에서 빼는 것(P0-2) 하나뿐이다.

### 5.5.2 순서 불변식 — advantage 먼저, 제거 나중 (load-bearing)

제거를 advantage 계산 *전*에 하면 baseline이 왜곡된다. 예: **2 clean-정답 + 6 truncated-실패**.
- **올바른 순서**(채점→advantage→제거): reward `[1,1,0,0,0,0,0,0]`, baseline 0.25 → 정답 각 **+0.75**. 제거 후 이 2개로 학습. baseline이 "6/8 실패한 어려운 문제"임을 반영.
- **틀린 순서**(제거 먼저): 6개를 먼저 빼면 `[1,1]`, baseline 1.0 → advantage **0**. baseline이 실패를 잊어 생존자가 우연히 다 정답이면 "변별 없음"으로 오인 → gradient 0.

즉 truncated row는 advantage 시점엔 그룹에 **남아 baseline을 만들고**, loss 시점엔 **빠져야** 한다. singleton 그룹은 `compute_grpo_outcome_advantage`가 mean=0으로 처리하므로 순서 위반 시 이 fallback에 걸린다.

### 5.5.3 조성 전수 열거 (k_c=clean정답, k_w=종료오답, k_t=truncated; k_c+k_w+k_t=8)

| 조성 | 라우팅 | loss에 남는 것 | 판정 |
|---|---|---|---|
| k_c=0 (k_w·k_t 무관) | SFT | tau (advantage=β) | 기존 gamma=0 HPT와 동일 동작. 종료-오답 음의 신호도 버려지나 이는 새 조치 아닌 HPT 설계 |
| k_c≥1, k_t=0 | RL | **표준 GRPO 완전 보존** (아무것도 제거 안 됨) | 이탈 0 |
| k_c≥1, k_t≥1 혼합 | RL | 정답(+)·종료오답(−) 유지, trunc만 제거 | 양·음 균형 유지, 아티팩트만 제외 |
| k_c≥1, 나머지 전부 trunc (케이스 B) | RL | 양의 row만 | §5.5.5(b) 우려 대상 |
| k_c=8 | RL | advantage 전원 0 → gradient 없음 | 표준 GRPO 고유 성질(낭비이지 위험 아님; DAPO는 dynamic sampling으로 제거) |

**중요**: "음의 신호를 다 없애는 것"이 아니다. 종료-오답(k_w)의 음의 advantage는 어느 조성에서도 살아 있다. 제거되는 것은 오직 "잘림" 아티팩트 row뿐이며, 정성 실측에서 이 경계가 "gibberish vs 의미 있는 오답" 경계와 **99.97% 일치**(clipped 250/250 degenerate, unclipped-오답 30/30 coherent-wrong, 새는 짧은 gibberish 0.025%)했다. 필터는 **틀림 여부가 아니라 잘림 여부**로 건다 — 이것이 DAPO Overlong Filtering의 원리다.

### 5.5.4 억제 항 생략의 정당화 (3)

1. **생략하는 항의 효과는 이미 나쁜 쪽으로 실측됐다.** truncated-오답 row는 M run 내내 reward 0으로 loss에 남아(혼합 그룹에서 음의 advantage) 바로 그 억제를 수행했고, 결과가 entropy 0→7.3 폭발이다. "남기는 팔"은 미지의 대안이 아니라 **이미 실패가 관측된 팔**이다. 기제: 좋은 응답이 이미 끝난 늦은 위치(2000~8192; 정답 p50=1,323)에서 한쪽만 누르는 억제는 확률 질량을 특정 대안으로 못 보내고 vocabulary 전체로 퍼뜨린다(평탄화 → entropy↑ → 자기증폭).
2. **이탈 크기가 병리에 비례(자기조절)**. 재시작 시점 base는 clip 0.05%(step3 clip_ratio 0.0005)라 제거 row ≈ 0, 추정량 ≈ 표준 GRPO. 필터는 병리가 자랄 때만 그에 비례해 문다.
3. **선례**: DAPO(arXiv:2503.14476) Overlong Filtering이 같은 처방(잘린 샘플 loss 마스킹)으로 entropy 안정화 보고.

### 5.5.5 실재하는 잔여 우려 (3)와 방어

정직하게, 다음 셋은 이론적으로 실재한다. 각각의 방어와 담당 감시 장치를 명시한다.

- **(a) 도피처 허점.** 종료-오답은 −로 억제되나 truncated는 무벌점 → 램블의 상대 확률이 *수동적으로* 오를 수 있음. 방어: ① gradient는 무벌점 지대를 능동적으로 찾아가지 않음(끄는 힘도 0). ② 램블 증가 프롬프트는 k_c=0이 되어 **SFT 재라우팅 → tau가 종료를 직접 교정**(도피처의 출구가 교정 장치로 연결). ③ **`clip_ratio>0.3` 트립와이어가 잡음** — 이 허점 때문에 트립와이어는 장식이 아니라 안전 논증의 일부다.
- **(b) 케이스 B류 양-편향 → 붕괴 방향.** 붕괴 국면 RL 생존 row가 양의 것 위주(그룹당 +1.32개 vs 종료오답 ~0.6개)라 배치가 "강화만" 쪽으로 기울 수 있음(현 폭발의 반대 실패 모드=mode collapse). 방어: **반대편 트립와이어 `entropy_mean<0.2`**. 또한 재시작 초기는 base 성공률≈0(step3 0.0002)이라 all-SFT 부트스트랩 국면이라 무대가 작음.
- **(c) 케이스 A의 좋은 음의 신호 소실.** 종료-오답의 음의 신호가 SFT 재라우팅으로 버려짐. 이는 gamma=0 HPT 기존 설계이지 새 조치 부작용이 아님 — 아깝다면 gamma 설계 논의(별건).

### 5.5.6 결론 — 조건부 안전

- **정확성**: 살아남는 모든 row의 advantage = 새 reward 정의 하 표준 GRPO 값. baseline 왜곡 없음. ✅
- **안전성**: 이탈은 "아티팩트 row 억제 항 생략" 하나. 그 항을 남기는 대안은 이번 run에서 실패 실측. 이탈 크기는 건강할수록 0 수렴. ✅
- **조건부**: (a) 도피처와 (b) 양-편향은 실재하므로 **`clip_ratio>0.3`·`entropy_mean<0.2` 두 트립와이어가 이 설계의 필수 구성요소**다. 트립와이어 없이 필터만 배포하는 것은 권장하지 않는다.

---

## 5.6 구현 기록 (2026-07-06)

Step-0 recon으로 확정한 단일 출처·정규화 사실(§5.1 P0-2 근거)에 따라 P0 + 관측성(문서 #9)을 구현했다. P1 트립와이어는 코드가 아니라 이미 로깅되는 지표의 수동 감시 규율이라 별도 구현이 없다(§5.2).

| 조치 | 구현 위치 (symbol) | 게이트 플래그 (기본 off) |
|---|---|---|
| P0-1 truncated⇒reward 0 | `reward_loop/reward_manager/dapo.py::DAPORewardManager.run_single` (async 경로의 유일 reward 출처; routing `reward_score`와 advantage `rm_scores` 동시 공급) | `reward.reward_kwargs.zero_reward_if_truncated` |
| P0-2 truncated RL advantage 0 | `fully_async_policy/fully_async_trainer.py::FullyAsyncTrainer._fit_filter_truncated_rl_advantage` (fit 루프 `_fit_compute_advantage` 직후) | `reward.reward_kwargs.zero_truncated_rl_advantage` |
| P0-3 entropy bonus 제거+진단 유지 | 런처: `actor.entropy_coeff=0.0` + `actor.calculate_entropy=True` | — |
| P0-4 SFT 강화 | 런처: `async_hpt.beta=1.0` | — |
| 관측성 mq_len | `_fit_...`에서 `batch.meta_info["fully_async/mq_len"]`(자동 승격) | — |
| 관측성 centered advantage | `_collect_metric_aggregation_weights`: `critic/advantages/centered_mean`·`/centered_absmean`(RL 토큰, SFT 제외) | — |

**런처**: `main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_Mprime.sh` (M.sh 보존, M′=기반 v2 새 앵커; experiment_name `..._Mprime_v2`, dump dir `_Mprime_`). 위 4 플래그 on.

**설계 선택 (recon 결과)**: P0-2는 물리적 row 제거가 아니라 **advantage-zeroing** — 분모 `global_batch_size`가 고정 상수라 희석 0·제거와 등가이면서 배치 형상/divisibility 리스크가 없고, row가 남으므로 `clip_ratio`(attention_mask 기반) 트립와이어가 자동으로 온전(§5.1 P0-2, Q2/Q3 소멸). `workers/reward_manager/dapo.py::__call__`의 overlong 로직은 async에서 미사용(dead code)이라 건드리지 않음.

**테스트**: `tests/special_RL/test_hpt_truncation_handling_on_cpu.py`(CPU 28개, 컴포넌트→통합 규합 모듈) — 전체 스위트 **110개 통과, 회귀 0**.
- P0-1(8): truncated-정답 0점+acc 보존 · truncated-오답 0 유지+flag · 종료응답 유지 · **경계=cap(텐서폭 아님)** · max_resp_len=None 폴백 · scalar score 처리 · **overlong 페널티 뒤 truncation이 0으로 덮음** · opt-in.
- P0-2(9): truncated RL만 0(+RL-분모 frac) · all-SFT 불침해 · hpt키 없는 base RL · truncated 無 no-op · 경계 길이 · opt-in · reward_kwargs 부재 안전 · advantages 부재 가드 · **§5.5.2 순서 불변식**.
- 등가성(2): **advantage-zeroing ≡ 물리적 제거**를 실제 `compute_policy_loss_cispo` gradient로 검증(고정 분모) + zeroed row gradient=0; **live-denominator였다면 3:1 희석 발생**을 대조로 고정(향후 seq-mean 정규화가 live로 바뀌면 이 테스트가 실패로 경보).
- 스모크 계약이 Mprime.sh를 자동으로 유효 config로 검증.

**미적용(합의된 보류, §5.2/§5.3)**: ESS 자동 차단기·큐 상한 축소·P2 카드 — P0가 원인을 제거하므로 "재발 시" 카드. 트립와이어는 수동 감시.

---

## 5.7 M′ 런 붕괴 사후분석 (2026-07-06): 라우팅 위치 버그 — P0가 노출한 잠복 결함

M′ 런(uz72mzb9, P0 전부 켬)을 291스텝 돌린 결과 **`critic/score/mean`·`hpt/onpolicy_success_rate`가 전 스텝 정확히 0**, val이 M보다 낮고(step159 MATH-500 55.8 vs M 71.1) 이후 하락. 처음엔 "P0-1 데드락"으로 의심했으나 M vs M′ 공정 비교 + 반사실 검정으로 **다른 근본 원인**을 확정했다.

### 증상과 잘못된 첫 가설
- 학습 dump: **100% SFT, RL 행 0** (전 스텝). token_level_scores=beta=1.0(SFT), advantages=beta×토큰수(정상 SFT NLL). → 순수 SFT.
- 첫 가설("정답이 다 truncated라 P0-1이 정당하게 0")은 **반증됨**: 롤아웃 dump상 M′은 clean-correct를 **25% 그룹에서 실제 생성**(raw per-rollout acc 0.20 > clean 비율 0.15 → dump acc는 raw 확정). clean-correct의 `rm_scores` terminal=**1.0**(P0-1이 안 건드림, is_truncated=False). 그런데 라우팅 success=0. → 생성단계 reward=1.0인데 트레이너는 0 = 파이프라인 disconnect.

### 근본 원인 (정량 확정)
`hpt_gate.extract_score_values`가 `reward_score`를 `rm_scores[-1]`(**마지막 텐서 인덱스**)에서 읽는다. 그러나 보상은 **terminal 토큰(valid_response_length−1)**에 놓이고 그 뒤는 우패딩(0)이다. 따라서:
- **truncated 응답**(budget 소진, valid=마지막): 보상이 마지막 인덱스에 있어 `[-1]`가 우연히 봄.
- **clean(조기종료) 응답**: 보상이 valid−1<마지막에 있어 `[-1]`는 항상 **0(패딩)**을 읽음.

dump로 정량 확증(500그룹):

| routing | M′ per-rollout | M′ GROUP≥1 | M per-rollout | M GROUP≥1 |
|---|---|---|---|---|
| 현행 `rm_scores[-1]` | **0.000** | **0.000** | 0.353 | 0.696 |
| 수정 `rm_scores.sum(-1)` | 0.053 | **0.236** | 0.427 | 0.746 |

M′ 현행=0.000이 wandb onpolicy=0과 **완벽 일치**. 즉 **M는 truncated-correct(보상@마지막)를 `[-1]`가 잡아 성공을 부트스트랩했고**(=우리가 없애려던 "비종료 보상" 경로), **P0-1이 그 truncated 보상을 0으로 만들자 clean 보상은 위치버그로 원래 안 보였으므로 양쪽 다 0 → 전 그룹 SFT.** **우리 P0가 버그를 만든 게 아니라, M가 몰래 의존하던 잠복 라우팅 버그를 노출**시킨 것.

부수 확인: advantage 경로는 무관(정상 `token_level_rewards.sum(-1)` 사용) — 버그는 **라우팅 성공 카운트에만** 있었다. 그래서 M는 truncated-correct 그룹으로 RL이 돌았고 clean-correct 그룹만 잘못 SFT로 샜다(치명적 아님). M′는 P0-1로 truncated가 0이 되며 치명화.

### 수정 (적용 완료)
`extract_score_values`의 rm_scores 분기: `[-1]` → `sum(-1)` ([hpt_gate.py](../verl/experimental/fully_async_policy/hpt_gate.py)). verl의 sequence-reward 리덕션(`core_algos` GRPO·`metric_utils`)과 정합. 위치 무관하게 terminal 보상을 봄. 회귀·통합 테스트는 `tests/special_RL/test_hpt_truncation_handling_on_cpu.py`(28개, 5섹션: P0-1 게이트/라우팅 위치/P0-2/등가성/**end-to-end 파이프라인**)로 규합 — 통합 테스트가 gate→route→advantage→P0-2→CISPO loss 전 단계를 합성 그룹으로 흘려 상호작용을 검증(clean-correct→RL·양의 gradient, truncated→0 gradient, 구 `[-1]`이면 SFT로 샜음을 대조). 전체 스위트 110 통과·회귀 0.

효과(반사실): 이 수정만으로 M′ 라우팅이 **clean-correct 그룹 23.6%를 RL로 복구** → 순수 SFT 데드락 해소. RL 그룹 내에서 P0-1/P0-2가 의도대로 작동(clean-correct 양의 advantage, truncated 0) → **종료를 향한 gradient** 제공.

### 과학적 함의 (중요)
M의 성공(onpolicy 0.31)은 **상당 부분 truncated-correct**였다(라우팅 `[-1]`가 그것만 봤으므로). 즉 **M의 "학습"은 대체로 비종료-정답을 강화**하던 것 — P0의 문제의식이 실증된 셈. 라우팅 위치버그 수정 후에야 P0(비종료 보상 제거 + 종료 gradient)가 제대로 평가된다.

### 다음 (재시작 전)
1. 라우팅 위치버그 수정(적용됨) — **필수**(없으면 P0-1 런은 순수 SFT).
2. 재시작 시 P0-1/P0-2/entropy=0 유지. beta=1.0은 §6 되돌림 기준으로 감시(위치버그 수정으로 24% RL이 생기므로 길이 압력 재평가 필요).
3. 이 수정은 M(비-P0) 런에도 라우팅 정확도 개선(+5%p, clean-correct 복구)이라 공통 기반에 반영.

---

## 5.8 라우팅 수정 후 실측과 전략 전환 (2026-07-07)

§5.7의 라우팅 수정(`rm_scores[-1]`→`sum(-1)`) 후, 기반 v2(P0)를 beta·entropy 스윕으로 재실측했다. 결론: **정직한 개선은 안정적이나 성능 천장이 낮다.** 현 국면의 목적이 성능 최우선(val6 mean@8 극대화)임에 따라 판정 기준을 바꿔 **병리 M 궤적으로 회귀**했고, 그 회귀가 신규 운영 병리(async 배치 폭발)를 노출했다. 이 절은 실측·결정·신규 병리·관측성 후속을 기록한다.

### 5.8.1 M-계열 대장 확장

| run | Δ(vs M) | 라우팅 | 운명 |
|---|---|---|---|
| M (원본) | — | `[-1]`(구버그) | 병리 상승(신기루), §2 분석 대상 |
| M′/Mprime | +P0 전부, β1.0, ent0 | `[-1]` | 순수 SFT 붕괴(§5.7) |
| Mprime2 | P0, β0.6, ent0 | `sum(-1)` | 안정, val 정체 |
| Mprime3 | P0, β0.6, ent.001 | `sum(-1)` | 가장 건강, val ~30 정체 |
| **resume40** | 원본 M @step40 이어감(no P0) | `sum(-1)` | **성능-우선 회귀(현재 실행)** |

### 5.8.2 val6 mean@8 궤적 — 교차점이 결정을 규정

| step | M(병리) | Mprime2 | Mprime3 |
|---|---|---|---|
| 0 | 17.05 | 16.67 | 16.58 |
| 10 | 19.14 | **25.60** | 19.29 |
| 20 | 23.41 | **28.01** | **27.75** |
| 30 | 29.49 | — | 28.79 |
| 40 | **33.76** | — | 29.79 |

- **교차점**: step≤20에서 P0(Mprime2/3)가 M을 **앞선다**(+4~6pt). step 30+에서 M이 역전(40: 33.76 vs 29.79). **M의 우위는 오직 병리(길이폭발·신기루) 국면에서만** 생긴다 — §5.7의 "M의 성공은 대체로 truncated-correct"를 **outcome으로 확증**.
- **정직 경로의 급감속**: Mprime3 step20→40 **+2.0pt** vs M **+10.3pt**. P0가 비종료 보상(정답의 41%)을 제거하자 남은 clean-correct 신호가 base 모델(temp1.0·고난도)에서 너무 희소해 val을 못 민다. = **base 능력의 정직한 천장 ~30%.**

### 5.8.3 결정 — 성능 최우선 회귀 (2026-07-07)

목적이 "val 극대화"이면 판정 기준에서 "정직/신기루"는 무의미하고 **도달 val + 실행 생존**만 남는다. M(33.76+, 상승 중) ≫ 정직 천장(~30). 따라서:
- **resume40 = 원본 M을 최신 checkpoint(step 40)에서 이어감**(현 코드 = 라우팅수정 + no P0 = 신기루 보존). 라우팅 변경은 **SFT→RL 단방향**(clean-correct 복구; resume 시점 param40 근방 +0~3%p, 균등추정 +12.9%p)이라 M 로직을 크게 안 바꾸고 오히려 신호↑ — 이어달리기 허용(bit-identical 아님).
- **best-val 수확 원칙**: 궤적이 꼬리에서 붕괴(entropy→균등)해도 무방 — `test_freq`/`save_freq`를 조여 **최고점 checkpoint만 취한다.**
- **정직 경로(P0/기반 v2, §5.4)는 틀린 게 아니라 천장이 낮은 것** — "정당한 ~30 초과"가 필요할 때의 카드로 **파킹(기각 아님)**. 미봉 갭 = 능동적 종료 유도(§5.3 P2 `length_inverse`/brevity; 인과 [1] 미봉, BP7).

### 5.8.4 신규 운영 병리 — async 배치 폭발 (P1-8이 예고한 백로그의 급성화)

resume40가 노출한 것은 reward/length 병리의 **운영 하류 효과**다(신규 인과, 학습 목표와 무관):
- 길이폭발(resp_len~7,600, clip 0.90)로 학습기 forward/backward가 비싸짐 → 스텝 20–30분 → 학습기가 rollout(6GPU)에 뒤처짐 → **완료 큐 누적**(collected 365→1,962 그룹, `stale_trajectory_processed` 168→21,768) → 다음 스텝 배치 폭증(토큰 22M→90M) → 자기증폭.
- **실효 처리율은 일정(~50.5k tok/s)** — 스텝 시간은 순전히 배치 토큰수에 비례(느려진 게 아니라 일이 3–4배). GPU 8장 91–99%·경합 0 → OOM/thrash 아님.
- 대가: 3시간에 step 40→47(≈7스텝), 신규 val 0개, 메모리 160/183GB(88%, OOM 위험). **성능-우선인데 배치 폭발이 val 상승을 기어가게 만듦.**

**처방(= §5.2 P1-8 실현)**: `max_completed_prompt_groups` 2048→**384**(원본 healthy M의 ~365그룹 배치 재현; 학습 최소 128그룹의 3배 여유; 큐 상한 = 배치 상한이자 신선도 자동 개선). `staleness_threshold`는 불건드림(큐 상한이 노화를 자동 완화). `test_freq`/`save_freq` 10→5(수확). 감시: `fully_async/trainer/idle_ratio`≈0 유지(상승 시 512로 완화). 예상: 스텝 ~8분(**3–4× val 처리량**), OOM 제거, 데이터 신선. — 이어달리기 절차: step 50 저장 대기 → val6@50 확인 → kill → 위 config로 step_50 재개.

### 5.8.5 관측성 후속 — §11 진단 DP-집계 버그 수정

P0-3(§5.1)이 상시 요구한 `entropy_mean`·`entropy_top20_mean`·`pg_clipfrac_top20entropy`(Ablation §11)가 멀티-rank 학습에서 crash했다(`Metric.aggregate_dp: [3,5]`). 원인: `compute_entropy_clip_diagnostics`가 all-SFT 마이크로배치에서 `{}` 반환 → §11 키가 rank별로 다른 개수 → DP 정렬 불변식 위반. 수정: **token-weighted sum/count 컴포넌트로 재구성**(항상 5키 방출, `finalize_entropy_clip_diagnostics`가 환원 후 비율화 — 공유 집계 인자가 몫에서 상쇄돼 정확한 token-weighted mean). 테스트 20개(진단 13 + M-anchor 7; crash 재현 + 실 `reduce_metrics` 다중-iter 정합 + ppo_loss 방출 통합). 이로써 §11 지표가 resume40에서 정상 방출(entropy_mean 6.3→7.6 관측 = 병리 진행의 정직한 계측).

---

## 6. 판정의 갱신 이력 — beta 사례

초기 판정(2026-07-05): "tau가 잘 종료되므로 beta↑가 종료를 가르친다 — 타당". 이후 wandb 단계 해부(§2)가 "길이 폭발이 beta=0.3의 all-SFT 국면에서 이미 발화"를 밝혀 판정 신뢰도를 하향. 최종: **채택하되 되돌림 기준을 사전 등록**(P0-4). — 증거가 판정을 갱신하면 결정을 뒤집는 대신 **실패 기준을 명시해 통제 실험으로 전환**하는 것이 방법론이다.

---

## 7. Best Practice — 로그 분석 방법론 (재사용 가능한 것)

**BP1. 휴리스틱을 믿지 말고 내용을 읽어라(정성 분석은 서브에이전트로).** UPT `repeatness`는 명백한 반복 루프에조차 재현율 0(§3.1). 250개 전수 정독(Sonnet ×5 샤딩)이 GIBBERISH 80.4%를 밝혔고, 이것이 "길이 확대 무익" 판정의 유일한 근거였다. 분류가 아니라 **판정이 걸린 질문**(구제 가능한가?)을 정독으로 물어라.

**BP2. wandb는 콘솔이 아니라 바이너리를 읽어라.** output.log는 라인당 4,096자 truncate(벤치 누락 오인 유발). `.wandb` datastore를 `nested_key`(슬래시 경로)+`value_json`으로 파싱(신형 포맷은 `item.key`가 빈 문자열). 단 mq_len처럼 **wandb에 아예 없는 신호**는 output.log에만 있다 — 양쪽을 상호 보완으로.

**BP3. alias·조성-산술 지표를 식별하고 나서 읽어라.** 이 run에서만: `actor/entropy`≡`entropy_loss`(합계라 RL 토큰 수에 오염 — per-token은 `entropy_mean`) · `critic/advantages/mean`≡`score/mean`(centering 안 보임) · `num_turns/mean` 3.0→2.06은 "멀티턴 붕괴"가 아니라 SFT row(메시지 3)·RL row(2)의 조성 산술(5.8% SFT → 2.058, 관측치와 정확 일치). **서브에이전트의 발견은 채택 전 산술로 교차검증하라** — 이번에 2건 기각됐다.

**BP4. 조성을 층화하지 않으면 모든 추세를 오독한다.** HPT는 그룹/row/토큰 3계층의 SFT 비중이 100%/100%/47% → 71%/5.8%/0.4%로 괴리한다(§3.5). "entropy가 오른다"도 "무엇의 entropy인가"(RL 토큰 한정, `p_success` 조건화)를 먼저 물어라 — `Ablation_RL.md` §12.1과 동일 원칙, 본 사례가 실증.

**BP5. val 상승은 학습 건강의 증거가 아니다.** val(temp 0.6)은 attractor를 회피하고 tau SFT가 견인해 병리 내내 상승했다. 학습 건강은 **train-side 신호**(clip_ratio, entropy_mean, rollout↔train KL, ESS)로 읽어라. 카나리아는 최난도 벤치(AIME24가 유일하게 역행)다.

**BP6. 보상의 의미론을 실패 모드에 대조하라.** "마지막 boxed만 채점"은 정상 응답엔 무해하지만, 비종료 병리 하에서는 **비종료에 보상을 주는 결함**이 된다(41%). 채점기·보상은 고정 사양이 아니라 현재 정책의 실패 모드와의 상호작용으로 재감사 대상이다.

**BP7. 개선안의 충분성은 인과 사슬에 사상해 판정하라.** 각 조치를 사슬의 고리([1][2][3])에 대응시키면 "무엇이 안 덮이는가"가 기계적으로 드러난다 — 본 사례에서 [1](SFT발 길이 채택)은 어떤 P0로도 직접 안 덮이며, 그래서 되돌림 기준(P0-4)과 P2 카드가 존재한다. 커버리지 주장 없이 조치 목록만 나열하지 마라.

**BP8. 개입 전 효과를 dump로 사전 실측하라.** "라우팅 성공 정의 변경"은 구현 전에 최신 1,500 그룹에 시뮬레이션(§3.3)해 죽은 그룹 49.4% 해소와 재배분(77.8/22.2)을 수치로 확정했다. 설계 논증만으로 배포하지 않는다.

**BP9. 무죄 판정을 기록하라.** 혐의 해제된 용의자 표(§4.3)가 없으면 다음 세션이 같은 손잡이(lr, sync, cap)를 재수사한다. parity 확인(원본 스크립트 대조)은 가장 싼 무죄 증명이다.

**BP10. 사고 후 개입 기준을 사전 등록하라.** D0·M 모두 clip 30–50% 시점(step 31–35)에 아무도 개입하지 않았고 100+ step의 컴퓨트가 낭비됐다. 트립와이어(P1-6)는 수치·지속조건·행동까지 박아야 작동한다.

**BP11. 비표준 조치는 표준 추정량으로부터의 명시적 차분으로 규정하라.** "표준 RL을 벗어난다"는 우려는 이탈 지점을 정확히 국한할 때만 반증 가능해진다 — `g_ours = g_GRPO − Σ(제외 항)`처럼 쓰면(§5.5.1), 이탈이 baseline이 아니라 특정 항 하나임이 드러나고, 그 차분의 크기가 병리에 비례(건강하면 0으로 수렴)함을 보여 "정상 학습을 처음부터 왜곡한다"는 반론을 기각할 수 있다. 조치의 안전성은 "괜찮아 보인다"가 아니라 **차분의 규정 + 그 차분을 남기는 대안의 실측 실패 + 잔여 허점의 감시 장치 대응**(§5.5.4–5.5.6)으로 논증하라.

---

## 8. 유지보수

- P0/P1 구현 시 본 문서 §5의 각 행에 구현 커밋/파일을 채워 넣고 Status를 갱신한다.
- M′ run 개시 후: §1 대장에 M′ 행 추가, 트립와이어 판정 결과(특히 beta 되돌림 여부)를 §6에 이어 기록한다.
- **2026-07-07 상태(§5.8 집약)**: M′/Mprime2/3 실측 완료 · 성능-최우선 전환 · async 배치 폭발 진단·유계화 처방 · §11 진단 DP-집계 버그 수정. **성능-우선 국면에선 판정 기준이 바뀐다** — train-side 건강(BP5)보다 **val6 + 실행 생존**이 우선. 단 정직 경로(P0/기반 v2)는 파킹일 뿐 기각 아니며, "정당한 ~30 초과"가 목표가 되는 순간 §5.3 P2(종료 유도)로 복귀한다.
- resume40 이어달리기 결과(step 50+ val6, 배치 유계화 후 스텝시간·`idle_ratio`)는 §5.8.4에 이어 기록한다.
- 본 문서의 분석 스크립트는 세션 scratchpad(휘발)에 있었다 — 재사용할 것(wandb datastore 파서, dump 그룹 스캐너)은 `tools/` 또는 `tests/special_RL/` 인근으로 승격을 검토한다(`Ablation_RL.md` §12.5의 "분석 스크립트 고정" 원칙).
- 결정의 근거·이론이 필요한 독자는 DR을 인용한다 — 이 문서는 사례와 절차의 기록이지 이론의 출처가 아니다.
