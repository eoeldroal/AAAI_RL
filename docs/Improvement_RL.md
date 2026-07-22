# Improvement_RL — M run 병리 분석과 개선 결정 (사례 기록 · Best Practice)

_Last updated: 2026-07-10_

> **Paper-use status (2026-07-22).** This is a historical pathology/debugging record, not
> the current paper narrative or evidence hierarchy. Use it only for Appendix-level cost,
> limitation, and implementation context after applying the claim boundary in
> `papers_RL/Full_Paper_Draft_ko.md`.

Status: 분석 확정(2026-07-06) · 개선 결정 P0/P1 합의 · P0+관측성 구현(§5.6) · **★M′ 런 순수-SFT 붕괴 사후분석 완료(§5.7): 근본원인=라우팅 `rm_scores[-1]` 위치버그(P0-1이 노출), 수정 적용·테스트 통과★** · 라우팅 수정 후 재시작 완료 · **★2026-07-07: P0 런(Mprime2/3) 실측 = 정직·안정하나 val ~30% 천장 · 병리 M은 신기루로 33.76+ 지속 상승 → 성능-최우선 M 궤적 회귀(resume40) · 신규 async 배치 폭발 진단·유계화 처방 · §11 진단 DP-집계 버그 수정 · **유계 leg 수집-정렬 크래시 근본수정(§5.8.6: 성장루프→trim+carryover, 낭비0·crash제거, 68 tests green)**(§5.8)★** · **★2026-07-08: 전 run dump 전수 감사(§5.9) — mirage=단일근원 확정(M2 래칫 85% 실측·D0 "길이안정"은 로깅 세대왜곡으로 철회·C1/C2 증폭기 가설 기각) · Mprime2/3=leninv였음(§5.8.1 정정) · 정직 arm 절단 45–55% 고원("하강" 관측은 윈도우 요행) · 로깅 왜곡 카탈로그+코드 4세대 지도 → **M3(=M2+P0 단일델타, β1.0 leninv, val JSONL 덤프, 500스텝 완주 서약) 결정·실행★** · **★목표 재정의(관대-val6 무조건 극대화, §5.10): 지표 이중성 발견(val6=관대 acc; M3 관대31.2/정직28.0) → 관대+CISPO 죽은칸 판정 정정(M 33.76·resume40b 34.00) → **M4 설계·구현 완료**(P0 제거·β0.3const·adv-std ON(게이트 완화+계약테스트 42passed)·큐768 staleness다이얼·packing 65536) — M3 완주 후 사용자 직접 발사★**
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
| M (원본, uvbi7wq3) | — | `[-1]`(구버그) | 병리 상승(신기루), §2 분석 대상. 33.76@40 |
| M′/Mprime (uz72mzb9) | +P0 전부, **β1.0 const**, ent0 | `[-1]` | 순수 SFT 붕괴(§5.7) |
| M′v2 (olh2hynl) | P0, **β1.0 const**, ent0 | `sum(-1)` | 건강(val 29.7@30, 정직계 최속)했으나 P0-4 기준(clip>0.3) 발화로 step34 조기 종료 — 기준 자체가 과보정이었음(§5.9.4) |
| Mprime2 (t661hw58) | P0, **β0.6 length_inverse**, ent0 | `sum(-1)` | step21 종료, val 28.0@20 |
| Mprime3 (7nvpern9) | P0, **β0.6 length_inverse**, ent.001 | `sum(-1)` | step41 종료, val 29.8@40. 절단 45–55% 고원(§5.9.1) |
| resume40/40b | 원본 M @step40 이어감(no P0) | `sum(-1)` | 무제한 leg 폐기(§5.8.4) → 유계 leg 수확 34.0@50 후 종료 |
| M2 (7aesrah1) | scratch·mirage 유지, **β1.0 length_inverse**, G4 인프라 | `sum(-1)` | 신기루 래칫 재발(dump 85%), val 26.1 정점 후 하락, step65 종료(§5.9.1) |
| **M3** | **M2 + P0-1/2 (단일델타)** | `sum(-1)` | **발사 대기 — 현행 anchor(§5.9.4)** |

> 정정(2026-07-08): 이 표의 구판은 Mprime2/3를 "β0.6"으로만 적어 **length_inverse 모드를 누락**했다(런처 307–309행 + train_dump 텐서 β_r=β·8192/len 이중 확인, §5.9.2). §5.3의 P2 leninv 카드는 Mprime2/3 시점에 이미 플레이된 것이다.

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

**원본 run 전 구간 staleness 실측(2026-07-07, wandb uvbi7wq3 전수)** — "step_40이 불안정한 산 위인가"의 판정 데이터: step 1–19만 신선(rollout↔현재 KL 0.007–0.05, ESS 0.95–0.99)했고 **그 신선 구간에서 이미 entropy 0.007→2.2·길이 1,184→7,178·절단 0→0.74가 완성** = §4.1 만성 엔진은 staleness 무관(§4.3 재확증). step 20부터 큐 폭발(KL 0.56→최대 12, ESS 0.42–0.83)로 혼돈 레짐이었고 **val@30·val@40 자체가 혼돈 안의 점수**다. 같은 혼돈에서 원본 30→40은 +4.3pp, resume 40→50은 −2.9pp — **이 레짐의 val은 부호 예측 불가한 고분산 랜덤워크**. 유계화 leg의 현실적 기대: step_40 정책의 자체 드리프트(스텝당 ppo_kl ±0.4–0.5)가 1-버전 신선도에서도 **KL ~1 nat 바닥**을 만든다(실측: resume 첫 스텝, 큐 419그룹에서 rKL 1.12·ESS 0.87) — 유계 leg ≈ "step 41–42 조건의 동결"이지 step 1–19 회귀가 아님.

**수확 leg 트립와이어(사전 등록, §5.2 P1-6의 재보정 — 기존 entropy_mean>2.5 기준은 정직-재시작용이라 이 leg에 부적용)**: ① ESS<0.6 중단 검토(P1-7) ② rKL>3 nats 3스텝 지속 → 중단 ③ entropy_mean(§11)>8.0 또는 +0.15/step×3 → leg 종료, leg 2(entropy_coeff=0 + `calculate_entropy=True` 동반 필수) 검토 ④ 절단율<0.6 ∧ score 하락 동반(48–49 시그니처) → 즉시 중단 ⑤ val 규칙: val@45≥33.5 계속 / 33.0–33.5 val@50까지만 / <33.0 종료(수확=argmax val) / 어느 시점이든 <32.5 즉시 종료 ⑥ AIME24 카나리아 병행. 기각 재확인: overlong penalty(§5.3 원장 — 방향 반대)·sync 4→2(표적 아님)·entropy_coeff 즉시 0(단일변수 A/B 훼손, 41–49 실패는 staleness로 설명돼 0.001 유죄 증거 없음).

**결과(2026-07-07 실측) — stale-배치 학습은 운영 문제를 넘어 실제 능력 훼손이었다**: resume40의 step 41–49 전 구간이 폭발 배치 레짐(rollout↔현재 KL **1.1–4.5 nats**, 시퀀스 16–23% IS 하한 폐기, ESS 0.72–0.87, 스텝당 ppo_kl ±0.4–0.5)에서 학습됐고, step 48–49에서 분포 붕괴(resp_len 7,696→4,825 · 절단율 0.90→0.37 · train score 0.50→0.39 · onpolicy 성공 0.40→0.27 · SFT 주입 2배 78k→160k tok; grad_norm은 0.05–0.24로 내내 안정 = 수치 폭발 아님, **레짐 붕괴**). **val6@50 = 30.38 (vs @40 33.29, mean@8 −2.91pp · best@8 −2.93pp, 6벤치 중 4–5개 동방향 하락)**. step_50 정책은 더 짧게 생성(절단↓)하므로 능력이 유지됐다면 val이 올랐어야 하는데 하락 → 신기루 회계 붕괴만이 아니라 **실 능력 훼손의 실증**. 판정(사전 기준 val6@50<32.5): **step_50 폐기, 두 번 검증된 최고점 step_40(33.76/33.29 = 평가 노이즈 바닥 ±0.5 실측)에서 유계배치(384)로 41–50 재주행**. 부수 소득: `actor/entropy` 6.4→3.8 "폭락"은 seq-mean-token-sum-norm의 길이 비례 착시(길이비 0.63 ≈ 엔트로피비 0.60)였고 token-weighted `entropy_mean`(§11)은 7.65→6.58 완만 하락 — §11 지표의 첫 실전 기여.

### 5.8.5 관측성 후속 — §11 진단 DP-집계 버그 수정

P0-3(§5.1)이 상시 요구한 `entropy_mean`·`entropy_top20_mean`·`pg_clipfrac_top20entropy`(Ablation §11)가 멀티-rank 학습에서 crash했다(`Metric.aggregate_dp: [3,5]`). 원인: `compute_entropy_clip_diagnostics`가 all-SFT 마이크로배치에서 `{}` 반환 → §11 키가 rank별로 다른 개수 → DP 정렬 불변식 위반. 수정: **token-weighted sum/count 컴포넌트로 재구성**(항상 5키 방출, `finalize_entropy_clip_diagnostics`가 환원 후 비율화 — 공유 집계 인자가 몫에서 상쇄돼 정확한 token-weighted mean). 테스트 20개(진단 13 + M-anchor 7; crash 재현 + 실 `reduce_metrics` 다중-iter 정합 + ppo_loss 방출 통합). 이로써 §11 지표가 resume40에서 정상 방출(entropy_mean 6.3→7.6 관측 = 병리 진행의 정직한 계측).

### 5.8.6 유계배치 leg의 수집-정렬 크래시와 근본수정 (trim + carryover)

**증상**: resume40b(유계 384) 재시작이 step_40 val 재현 후 ~57분·7 fit-step 만에 `ValueError: HPT learner-row-aware collection could not form a trainable batch ... learner_rows=3095 required_multiple=256 queue_samples=512 max_queue_samples=512`로 사망(`fully_async_trainer._get_samples_from_queue`). OOM·NCCL·신기루 붕괴 아님 = **순수 수집 로직 크래시**(뒤 수천 줄 `Dropped ... TaskCancelled`은 드라이버 종료의 뒤처리 폭포).

**근본원인 (공유 학습기 계약)**: 학습기는 배치 행수가 `mini_batch×n`(=32×8=**256**)의 배수여야 한다 — `_update_actor`([ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) `ppo_mini_batch_size *= rollout.n`)가 "모든 그룹=n행" 표준RL 가정을 박고, `make_iterator`가 `assert rows % mini == 0`으로 **하드 강제**. HPT의 SFT 그룹(1행)이 이 가정을 깬다: `learner_rows = 8·RL + 1·SFT`, 256=8×32이라 정렬엔 `#SFT ≡ 0 (mod 8)` 필요. 기존 수집은 **정렬될 때까지 큐를 성장**(도착순, RL/SFT 선택 불가)시켰는데 → (a) 정상 시 **2.81× 과수집**(실측 avg 2,158행 vs 의도 768) + 정렬대기 avg 40.8s/스텝, (b) 잔차 mod 8을 못 맞추면 **발산→raise**(8번째 수집에서 512그룹/3,095행까지 당겨도 미정렬).

**비효율 = 크래시와 동일 뿌리**: "정렬까지 성장"이 조금 자라면 느리고(과수집), 무한히 자라면 터진다. 유계화(384)는 `max_queue_samples`를 512로 좁혀 정렬창을 조여 발산을 **노출**시켰을 뿐(원본 2048창은 "대개 운좋게" 정렬돼 안 터졌지만 배치 폭발).

**근본수정 층위 판정**: 정렬은 공유 학습기의 load-bearing 계약이라 제거하려면 (L1) 공유 `make_iterator` 관용화 = 전 학습경로 폭발반경, 또는 (L2) `num_mini_batch`로 mini-batch 크기 적응 = **학습 동역학 변경**(AGENTS "do not change the learning problem to fix utilization" 위반). 채택 = **(L3) HPT 수집에서 재조정** — HPT가 불변식을 깨는 그 지점, 학습 계약(256행 mini-batch·그래디언트) 완전 보존 = 순수 plumbing.

**수정 = 성장루프 삭제 → trim + carryover**(`_plan_row_alignment_deferral` 정확 subset-sum): required_samples 그룹 수집 → 1 mini-batch 미만이면만 성장 → 잔차 행을 **하위 배수로 trim**하되, 떼어낸 그룹을 **폐기 않고 다음 스텝으로 이월(carryover)**. 이월은 다음 스텝 우선소비(`protected_prefix`)로 1스텝 내 학습, 항상 `<256`행 유계. 효과: **crash 제거**(trim은 항상 정렬 도달), **과수집 제거**(768행 고정 → ~2.8× 빠름·정렬대기 0), **낭비 0**(discard 아님). 성장루프+crash raise는 삭제 = 순 복잡도 감소.

**로깅(AGENTS "모든 단위 명시·drop 가시화·loss 분모 불변")**: 이월행은 이번 스텝 배치에 부재 → 손실·어드밴티지·분모에 0 기여(다음 스텝 전량). 단일 trimmed 배치가 하류 전체(값·가중치·버전메타·stale)를 관통 → 값↔가중치 자동 정합. 명시 메트릭 **add 4개**(`hpt_carryover_in_groups`·`_out_groups`·`_row_alignment_deferred_rows`·`_fresh_pulled_groups`) + 기존 `hpt_collected_queue_samples`(의미 불변, 값만 유계 반영). 항등식 `fresh_pulled = retained + out − in` 테스트 고정. 정직 부수효과: 이월 샘플은 학습 시 실제 1버전 staler → staleness 지표 소폭 상승(버그 아님).

**테스트(RL conda env)**: `test_hpt_trainer_queue_contract.py`에 헬퍼 subset-sum 5(정렬됨/정확잔차/pure-RL/carryover보호/불능→None) + 통합 3(trim-유계·**크래시조성 재현→trim성공**·carryover 왕복·항등식). 기존 성장테스트(`...uses_completed_budget...`)는 오버슈트를 정답으로 박제했으므로 trim 동작으로 갱신. 전체 68 passed(회귀 0).

---

## 5.9 전 run 전수 감사 — mirage 단일근원 확정과 M3 결정 (2026-07-08)

M2(scratch·mirage·leninv β1.0·G4 인프라)가 val 26.1@50 정점 후 하락 전환한 것을 계기로, **6개 run의 rollout dump 양·질 전수 해부 + train_dump 텐서 검증 + wandb 바이너리 재파싱 + run↔코드 세대 지도**를 서브에이전트 fan-out으로 수행했다. 교차검증이 기존 판정 4건을 뒤집었고(아래), 마지막 두 라운드가 같은 결론을 강화하며 수렴했다.

### 5.9.1 dump 실측 — mirage가 유일 근원, objective는 속도 조절자일 뿐

| run (보상) | mirage% 궤적 (pv 정렬) | 절단 50% 돌파 | clean-pos 궤적 | 질적 시그니처 |
|---|---|---|---|---|
| M (mirage, CISPO+dec) | 0→39(pv17)→**82(pv30)** | pv9 | 12.5→**3–8% 퇴화** | 정답 후 다국어 토큰샐러드로 8192까지 |
| D0 (mirage, vanilla clip) | 0→19(pv9)→42(pv30)→**66(pv69)** | pv11 | 7–13% 정체 | **동일 샐러드, clean-양성까지 침식** |
| M2 (mirage, leninv1.0, G4) | 0→54.5(pv35)→**84.6(pv60)** | ~pv13 | 신기루에 매몰 | 동일. r<0.72 토큰 1.6→50% 폭증(그래디언트 말살) |
| 정직 3-arm (M′v2/Mp2/Mp3) | **정확히 0% (전 윈도우)** | — (45–55% 고원) | 11→**~30% 성장** | 토큰루핑 멸종, 실패 97%가 boxed 단 진짜 시도 |

- **D0 "길이 안정" 판정 철회**: 로그 clip 0.45는 세대 왜곡(§5.9.3), dump p50은 pv30부터 8192 고정. vanilla clip은 래칫을 못 막고 **기울기만 절반** → C1/C2 증폭기 가설 기각(§4.3 무죄 유지·확장). D0의 34.0 포화는 "깨끗한 천장"이 아니라 mirage 42→66% 부패 속의 표류 도달점.
- **M2의 말기 "회복" 지표는 위장**: pos_rate 26%·dead 44% 중 양성의 85%가 절단 — 보상뿐 아니라 **HPT 라우터도 오염**(가짜 P>0가 dead group을 SFT에서 RL로 빼돌림 = 리프트 원천 차단).
- **정직 arm의 한계도 실측**: 절단은 45–55% 고원에서 **하강하지 않음**(초기 "51→28% 하강" 관측은 50-그룹 윈도우 요행 — pv해상도 재측정에서 pos 0.20↔0.43, trunc 0.40↔0.56 진동, 로그 평균 0.28과 정합). "boxed-early-then-ramble"은 정직 arm에도 잔존 = **종료는 여전히 미학습**, 롤아웃 ~절반이 zeroed로 낭비.
- 축 판정: **ent 0 vs 0.001 무차별**(Mprime2 vs 3: val·다양성·엔진 분리 불가, 5-gram Jaccard 0.01대로 붕괴 없음), **β 0.6 vs 1.0 무차별**(step20 27.0/28.0/27.75).

### 5.9.2 train_dump 텐서 검증 — 전제 정정 2건

- **Mprime2/3는 length_inverse β0.6이었다**(§5.8.1 표 정정 참조). β_r=β·8192/len이 전 SFT행에서 0.2% 이내로 성립 — leninv는 설계대로 동작.
- **SFT 질량 실측**: leninv arm 19–69%(배치 |A|·len 질량 기준; 참조구현 ~23–30%급 이상) — "SFT가 약해서 리프트 부재" 서사는 **const-0.3 세대(M/D0, ~5–10%)에만 부분 성립**. leninv arm의 리프트 부재는 SFT 강도가 아니라 mirage(라우터 오염) 또는 시간 부족.

### 5.9.3 로깅 왜곡 카탈로그 + run↔코드 세대 지도 (cross-run 비교 시 필독)

| 왜곡 | 경계 커밋 | 피해 사례 |
|---|---|---|
| response_length/*가 tau행 포함 계산 | `f75f9976`(07-04 13:46) 이전 | D0 로그 clip 0.45 vs dump 62–65% — "길이 안정" 착시의 근원 |
| onpolicy_success_rate debias | `58488791`(07-04 20:44) 이전 | D0 세대 성공률 비교 불가 |
| §11 entropy에 zeroed행 포함 | `40674d28`(07-07) 이전 | Mprime3 엔트로피 의미가 현행과 다름 |
| val 생성물 침묵 드롭 | 미수정(전 run) | rollouter actor에서 `wandb.run=None` 가드 — val 출력이 어디에도 없음 → M3부터 `trainer.validation_data_dir`로 우회 |
| ckpt 지표줄 = 4–5 fit-step 가중평균 | 구조적 | dump 포인트 샘플과 직비교 금지. 역으로 dump 50-그룹 단일윈도우도 ±15pp 요동 — **cross-run 주장은 dump ≥2윈도우 + 로그 상호대조로만** |

세대 지도(wandb base-commit 실측): **G1** `14d2a78` D0 / **G2** `38019bf` M·M′ / **G3** `f34a930` Mprime2/3·resume40 (grow수집·zeroed포함 entropy) / **G4** `5ebd614`+ resume40b·M2·M3 (trim+carryover=**정확히 128그룹/fit-step=논문 x축 parity**·유계·전체 지표). **세대 다른 run의 로그 지표 직비교는 무효.** Mprime3 resume 안이 기각된 이유이기도 하다(G3 ckpt를 G4 코드로 잇기 = 연속 아님).

### 5.9.4 결정 — M3: 정직 × 최강 SFT × G4, 단일델타 + 완주 서약

- **M3 = M2 + P0-1/2** (그 외 전부 동일: β1.0 leninv·ent0.001·CISPO+decoupled·384 유계) → M3−M2 = 보상 정직성 단일축, M3−Mprime3 = (G4 인프라, β1.0, 신선배치) 델타.
- `trainer.validation_data_dir` 추가(관측성 전용) — 하드벤치 절단-vs-능력 판정을 최초로 가능하게 함.
- **P0-4 되돌림 기준(clip>0.3) 폐기**: 건강한 정직 arm 전부가 위반하면서 건강했음(M′v2가 이 기준의 희생) — 과보정 실증. 대체 게이트: ① 데이터축 매칭(60k/120k/226k rows에서 Mprime3 21.8/28.1/29.8 대비 −1.5pp 이상 열세면 재검토) ② `truncated_rl_frac` 고원(≥0.45, pv~30) 재확인 시 tau-선호 카드 개봉 ③ entropy_mean<0.2(SFT 과구동 붕괴)·ESS<0.6 차단기 유지 ④ AIME 카나리아.
- **실행 서약: 500 fit-step 완주, 게이트 외 조기 종료 금지** — 실패자산 "가속/건강 중 조기 종료"(정직 3-arm 전부 step 21–41 사망) 반복 방지. 논문 41.9는 500스텝 산물; 우리는 그 구간 데이터가 없다.
- 대기 카드(순서): ① **tau 짧은-궤적 선호 가중**(P2 확장 — tau p50 4,172라 단순 ≤4096 필터는 교사궤적 절반 상실+어려운 문제일수록 tau 길어 부적합, 선호-가중 설계로) ② **A1(token-mean)** — 둘 다 절단 고원 G4 재확인 시. 기각 원장(overlong penalty·12288·sync단축) 준수 유지.
- 기대치(냉정): M3 단독 착지 추정 30–33. 41.9는 [절단 고원 붕괴 + 완주 + 하드벤치 회복]이 모두 성립해야 사정권 — 첫 관문이 M3다.

### 5.9.5 방법론 소득 (BP 추가 후보)

- 서브에이전트 대량 해부는 유효하되 **맹신 금지**: 단일-윈도우 결론(플라이휠 가속)이 pv-해상도 재측정으로 기각, 에이전트 전제 오류(const 가정)가 텐서 검증으로 정정. 모든 dump 주장은 복수 윈도우 + 로그 상호대조.
- 로그와 dump가 다르면 **어느 쪽도 기본 신뢰하지 말고 계산 코드를 읽어라** — 두 착시(D0 clip, Mprime3 43%)의 해소가 모두 코드/커밋 추적에서 나왔다.
- 사용자 직관 가설("배치 크기 아닌가", "체계가 다르다")이 두 번 결정적이었다 — 분석은 반론 검증을 우선하라.

---

## 5.10 목표 재정의(관대-지표 극대화)와 M4 설계 (2026-07-08)

§5.9의 지표 이중성 발견 직후, 운영 목표가 재정의됐다: **"정직/신기루 구분 없이 관대-val6(=논문 Table 3와 같은 채점)를 무조건 극대화한다."** 이 절은 그 재정의 하에서의 재분석과 M4 설계·구현을 기록한다.

### 5.10.1 관대 훈련보상은 표준이다 — P0의 지위 재평가

- **verl 표준·UPT 실코드 검증**: 참조 구현의 훈련 보상엔 절단 게이트가 없다(관대). 라우팅 P도 관대 점수로 계산. 즉 41.9를 만든 훈련이 관대 보상 그 자체다. **P0-1/2는 우리의 비표준 개입**이며, 측정된 비용 = 명목 데이터 31% zero化 + 관대-양성 행 ~8% 폐기 + 훈련-평가 불일치.
- 단 P0가 병리의 원인은 아니다 — 관대 훈련이 우리 파이프라인에서 자기파괴(M 붕괴·M2 26 사망)하거나 34에서 부패-포화(D0)하는 것에 대한 응급처치였다(§5.1 당시 근거 유효). 재정의된 목표 하에서는 "부패를 막는" 대신 "**정점을 수확**"(§5.8.3)하는 전략으로 전환한다.

### 5.10.2 셀 판정 정정 — "관대+CISPO"는 죽은 칸이 아니다

§5.9까지의 서사("M3−P0=M2=사망")는 과잉 일반화였다. 관대-지표 원장:

| run | 엔진 | β | 라우팅 | 관대 val6 |
|---|---|---|---|---|
| D0 | vanilla+coupled | 0.3 const | 버그 | **34.04**@191k행, 246k까지 유지 |
| M | CISPO+dec | 0.3 const | 버그 | **33.76**@155k; 유계-resume(resume40b) **34.00@50, 자발 중단** |
| M2 | CISPO+dec | **1.0 leninv** | 수정 | 26.1 사망 |
| M3(정직) | CISPO+dec | 1.0 leninv | 수정 | 31.2 상승 중(완주 예정) |

→ 관대+CISPO+β0.3은 34에 도달한다. **M2 사망의 용의자는 CISPO가 아니라 {β1.0 leninv의 SFT 질량 20–70% 전용, 라우팅 수정(버그의 우연한 clean-SFT 커리큘럼 제거), G4 신선도(신기루 피드백 조임)} — 3자 미분리.** CISPO/decoupled의 관대-지표 비용은 155k행까지 미미(33.7 동률)하나 래칫 인구 가속 2×는 실측(§5.9.1) — 수확 규율로 관리한다.

### 5.10.3 큐·속도 재검토 (기각/채택 원장)

- **큐 확대는 드롭을 못 줄인다**: 정상상태 드롭률 = (생산−소비)/생산 ≈ 38%, 큐 크기와 무관(대기줄 수학). 큐 상한의 실제 의미는 — trim+carryover(§5.8.6)가 배치 크기를 128그룹으로 분리 고정했으므로 — **순수 staleness 다이얼**이다(384=0.75, 768=1.5, 2048=4 param-version).
- **staleness↔처리량 배타성**: 드롭 제거의 유일한 레버는 GPU 재배분(5/3이 +33%로 최적, scratch라 world_size 제약 없음)이나, 생산<소비가 되는 순간 큐가 비어 staleness가 소멸 → C1(decoupled+TIS)이 무의미해짐. **드롭 38%는 낭비가 아니라 C1 분석 레짐의 유지 비용.** 5/3은 "분석 없이 속도만 필요할 때"의 카드로 등록.
- **미니배치 병합(32→64) 기각**: use_dynamic_bsz 하에서 GPU 포화는 마이크로배치(토큰 packing)가 결정, 미니배치는 옵티마이저 단위일 뿐. 이득 ~1–5%에 학습 동역학 변경 + 전 run(mini=32)과의 비교성 파괴 — AGENTS 원칙 위반 거래.
- **토큰 packing 32768→65536 채택(⑦)**: 학습 수학 불변의 순수 속도(+5–10%). 메모리: allocated 53.6GB@32k 실측의 선형 외삽(토큰당 ~1.27MB, 어휘 152k logits가 지배) → ~95GB@64k = 용량 52%. 수확체감상 2×가 스윗스팟(3×는 +1–2%에 위험만 증가). ~~`expandable_segments` 동반~~, OOM 폴백 사다리(65536→49152→32768) 사전 등록. **[개정 2026-07-08]** 실행 사고(§5.10.4)로 정정: `expandable_segments`가 SGLang TorchMemorySaver와 비호환이라 **영구 제거**, packing은 **49152(1.5×) 확정**(65536은 미검증 추측이라 되돌림). 최종값은 §5.10.4 ⑦.

### 5.10.4 M4 설계 (= 34-달성 공통 인자 + 신규 1 + 다이얼 1 + 속도 1)

| # | 델타 (vs M3) | 근거 |
|---|---|---|
| ① | P0 2줄 제거 | 채점 일치(§5.10.1) |
| ④ | β 1.0→0.3, leninv→const | 両34-달성 공통값; M2 제1용의자 제거 |
| ⑤ | norm_adv_by_std=True | 희소 성공 ~3× 증폭(하드벤치 직격) + 참조 parity. **SFT 안전성 검증**: singleton 그룹은 mean=0/std=1 특례(core_algos)라 β_r 보존 — 계약 테스트로 고정. 되돌림 기준: entropy_mean>3.5 또는 관대 val 2연속 하락 시 이 축만 False |
| ⑥ | 큐 384→768 | staleness ~1.5버전 = C1 작동 레짐 복원(논문 분석). 차단기: ESS<0.6 / rKL>3 지속 / 초반 3-val ESS<0.85 지속 시 384 재시작 |
| ⑦ | 토큰 packing 32768→**49152**(1.5×), expandable_segments 없이 | §5.10.3 — 아래 실행 사고 참조 |
| 유지 | **CISPO+decoupled(기여 축)**, G4 trim+carryover, val JSONL, ent 0.001, mini 32, sum-norm 집계 | §5.10.2; token-mean(A1)은 기제 근거 약화(Adam 스케일 흡수)로 보류 |

**실행 규율**: M3는 §5.9.4 완주 서약대로 끝까지(관대 31.2 상승 중 — 그 자체가 34 도전 1차 시도). 종료 후 M4를 **사용자가 직접 발사**. M4는 §5.8.3 수확 원칙(관대 val 2연속 하락 = 즉시 정점 수확)으로 운영하며, val JSONL로 관대/정직 이중 곡선을 병산해 정직 곡선을 부패 조기경보로 쓴다.

**구현 기록(2026-07-08)**: hpt_config의 norm_adv 하드게이트를 "명시적 bool 요구"로 완화(+why 주석), 신규 계약 테스트 `test_hpt_norm_adv_std_contract_on_cpu.py`(양 모드 허용/미설정 거부/singleton β_r 보존/std 증폭률) 추가, `run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_M4.sh` 작성(전 델타 헤더 문서화). 계약 테스트 42 passed.

**실행 사고와 ⑦ 철회(2026-07-08, 1차 발사)**: M4 1차 발사가 롤아웃 init에서 즉사 — `RuntimeError: TorchMemorySaver is disabled ... expandable_segments is not supported yet`. 원인은 ⑦의 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 **SGLang의 TorchMemorySaver(async 가중치 동기화 checkpoint_engine 경로)와 비호환**이라는 것. 학습 로직(P0 제거·β0.3·adv-std·큐768·게이트 완화)은 전부 무죄 — SGLang 스케줄러 예외가 rollouter 액터를 죽여 전체 run 종료. **조치: expandable_segments만 영구 제거하고 packing은 1.5×(49152)로 되살림.** 크래시 원인은 packing 값이 아니라 expandable_segments 한 줄이었으므로 65536을 32768로 되돌린 건 과잉조치였다(65536은 실행 기회조차 없었음 = 미검증 추측). reserved 172GB는 캐시 고수위일 뿐 allocated(53.6GB@32k)만 안 넘으면 OOM이 아니며, 49152의 allocated ~74GB(40%)는 defrag 없이도 여유가 크다(65536=52%은 다음 카드). OOM 폴백(사전 등록): 49152→32768, 발생 시 롤아웃 init 직후라 발견이 쌈. 교훈: **allocator env는 트레이너·롤아웃 워커가 Ray 상속으로 공유하므로 SGLang 제약(TorchMemorySaver)을 함께 만족해야 한다.** M4 델타(①④⑤⑥⑦)는 이 형태로 재발사 준비 완료.

---

## 5.11 M4 실측 종합과 M5(clean-async) 설계 (2026-07-08)

M4를 재발사해 완주 관측했다. 결과·논문 대조·사용자 반박 3건·관측성 버그·M5 도출을 기록한다.

### 5.11.1 M4 실측 — 정직 34 돌파, 그리고 자가치유

- **val6(관대=정직, gap 0 전 구간)**: 17.56(base) → 27.73(5) → 33.34(15) → **정점 36.27(step70)** → 이후 34–36 진동(5차 사이클). D0/M의 34.04/33.76은 **mirage**였고, M4의 36.27은 **정직(절단 없음)** — 프로젝트 최초의 정직 34+이자 최고치. 자산 2점 보존: `checkpoints/harvest/M4_global_step_{30_val35.16,70_val36.27}`.
- **자가치유(신규 발견)**: 길이 폭발→cap-hit→mirage 붕괴 후 스스로 간결-정답으로 회복. 기제 = **HPT SFT 바닥재** — 위기 시 게이트가 offline_data_ratio를 ~82%로 올려 능력을 살려두고, 능력이 오르니 RL이 실신호를 되찾아 길이 런어웨이가 자멸. HPT 구조의 자가안정성.
- **parity 확정**: UPT 순정 클론 전수 대조 결과 M4는 학습-핵심 노브(advstd=grpo_use_std=True, β0.3 const, gamma0=SWITCH_GATE0, cap8192, lr5e-6, entropy0.001, CISPO)에서 논문과 일치. **남은 델타는 async(staleness 큐)·CISPO(vs PPO-clip)·250스텝(vs 500)뿐.** 4세대의 "승리 조치"는 사실상 우리 비표준 개입을 걷어내고 논문으로 복귀한 것이었다.

### 5.11.2 천장 정체(~35)의 3층 진단 — 전부 async 특유

Sonnet 덤프분석(val 1546문항×9시점, train rollout .dp) + 곡선적합으로 확정:

1. **k=0 벽(주 병목, 75–80%)**: val k=0(8샘플 전패) 비율 **36%(AIME 77%)가 step15 이후 미동**. GRPO는 전패 그룹서 advantage 항등 0 → gradient 0. 오류 질적분류: 접근오류/지식결손 59%·근접실수 26%·자기모순 give-up 15%(k=0 전용). **비종결 0%·채점손실 0%**(생산 채점기로 오답 500행 재채점 → **0건 반전**, grader-false-negative는 eval서 무시가능).
2. **길이 붕괴(HPT 대비)**: HPT는 SFT-long-form 견인으로 4–6.5k로 성장·유지(UPT Fig.7 "does not regress"), M4는 폭발→붕괴→**1.2k**. **레시피가 동일하므로 원인은 async 사이클.** 단 "너무 짧다"는 오해 — val 정답 길이는 난이도별로 조절됨(AIME정답 2685ch > MATH 1494ch)이고 오답이 정답보다 김(rambling-실패). 문제는 양이 아니라 **생산적 긴-추론 모드의 부재**(§3.4 exposure-bias: `</think>` 53% 미종결과 동일근원).
3. **한계순환(불안정)**: rollout_KL 0.04↔16, ESS 0.98↔0.15, entropy 0.2↔8.6, 주기 9–11. 지연(staleness)×이득(advstd)×연료(절단-관대 보상)의 지연-피드백 진동자. §3.5 재확인 — **구속조건은 sync주기가 아니라 sync당 이동속도**.
- **곡선적합**: post-transient(step≥15) 3함수형 A≈34.8–35.1. 단 **val@70=36.27이 적합을 이미 초과** → 슬로우 그라인드 실재, A추정은 하한으로만. 41.9는 같은 레시피 스텝 연장으로 도달 불가(ScaleRL 어법: A 문제이지 B 문제 아님) — 새 A-레버 필요.

### 5.11.3 논문 대조 레버 원장 (ScaleRL/Entropy/LUFFY/async, 로컬 PDF 정독)

- **UPT Table3(1.5B) per-bench**: AIME24 16.6·AIME25 17.8·AMC 51.0·MATH500 81.0·Minerva 37.5·Olympiad 47.3 = **41.9**. eval: AIME류 avg@32/temp0.6, 그 외 avg@1.
- **ScaleRL A-레버(천장, ±0.02 마진 밖만)**: **fp32 lm_head A 0.52→0.61(+0.09, 최대 단일)**; CISPO 0.595(**보유**); batch512→2048 0.605→0.645(예산 보류); length 14k→32k 0.61→0.645(원장 기각); zero-var drop 0.52→0.54; staleness k=8 최적·k=12 열화. **B-레버(마진 내, 천장 무관)**: loss-aggregation(prompt/token-avg 0.535 — 우리 현 seq-sum-norm이 이미 token-avg 계열)·advantage-norm(batch 0.530). Clip-Cov는 gradient 영점화라 **거부권 위반 기각**; KL-Cov(7B +2.0/32B +6.4)는 P2.
- **LUFFY**: 1.5B서 34.7(<HPT 41.9)이라 SFT 전면교체는 숫자상 부지지. 지지되는 것은 **게이트-행 한정 하이브리드**(τ 그룹합류로 전패 그룹에 음의 advantage + shaped ratio f=x/(x+γ), γ=0.1) — dead-zone·긴-추론 전이 실패의 문헌적 정답.

### 5.11.4 사용자 반박 3건 — 전부 코드로 재검증

1. **L1(fp32) 정정**: `use_fused_kernels=True` 단독은 backend 기본 None이라 **step1 크래시**(engine은 fused 분기, monkey-patch는 스킵). 게다가 현 커널은 bf16 matmul **후** fp32 업캐스트라 정밀도 이득 0. → **진짜 해법은 rollout-side `engine_kwargs.sglang.enable_fp32_lm_head=True`**(logits_processor.py:898 fp32 matmul, replica.py:365가 이 async 어댑터 로드, 로컬 SGLang 0.5.12 필드 실재, smoke 통과). 이게 IS 앵커(rollout logprob)를 직접 정확화 = Draft A.5 유효성 계층의 토대.
2. **L5(zero-var) whiten 우려 기각**: 인용된 masked_whiten(core_algos:466)은 `grpo_vectorized` 경로. 우리 런처는 `adv_estimator=grpo`(compute_grpo_outcome_advantage) → **whiten 없음**. 실증: `critic/advantages/max=2.4748` 매 스텝 상수 = 1/8그룹 해석해. 단 L5는 config 아닌 취약 trim+carryover 수집기 코드 → 이번 묶음서 제외.
3. **L2(staleness) 정정**: 768은 max_completed_prompt_groups(큐 depth)이지 staleness_threshold(=2.0)가 아님. "검증 밖 2–3자릿수"는 그룹수↔staleness 혼동 오류. 재계산 유효 lag ~2–3 param-version(k≈13–20, ScaleRL 열화경계 k=12 근처). 768→384는 lag를 ~절반으로 낮추는 **부분해**(사이클 주범 advstd 이동속도는 엔진이라 불변).

### 5.11.5 관측성 버그 — `rollout_probs_diff` placeholder 오염과 RL-only 수정

- **버그**: `calculate_debug_metrics`(debug/metrics.py)가 response 토큰만 마스킹, SFT행 제외 안 함. HPT SFT행의 rollout_log_probs는 placeholder 0(hpt_assembler.py:240) → `|actor_prob − exp(0)=1|`을 평균에 포함. M4 명목 probs_diff 0.05–0.16은 **이 오염 수치**(pearson 0.84도 이 탓). FAQ의 0.005/0.01 기준으로 이 원본을 읽으면 오독.
- **진짜 엔진 불일치**: SFT-제외 지표(rollout_corr/*) step1(동일가중치) KL **0.003nat**, ESS 0.996 — 실재하되 온건. bf16 수준.
- **수정(구현 완료)**: `calculate_debug_metrics(data, exclude_rows=)` 추가 — 기존 전-행 키는 불변(back-compat), `exclude_rows`(=hpt_is_sft) 주면 `_rl` 키 5종 + `sft_rows_excluded` 추가 방출. 호출부(separation/ray_trainer.py:552, entry-mode라 이 super()가 실행됨) 수정. 계약테스트 `test_rollout_probs_diff_rl_only_on_cpu.py` 4개(전-행 오염 재현·RL-only 격리·all-SFT valid=0·비HPT 무변경), special_RL 139 passed.

### 5.11.6 M5 = clean-async (L1+L2), 그리고 정직한 무게중심

- **M5-P0(단일 축 귀인)**: M4 전 레시피 동결 + 델타 2개 — **L1** `enable_fp32_lm_head=True`(config), **L2** 큐 768→384(config). 런처 `run_..._M5.sh` 작성(diff 확인: 델타 정확히 2줄 + 이름). 로깅 축(5.11.5)은 L1의 유일 게이지라 M5 발사 전 필수 — 완료.
- **조기 게이지(첫 20–30 fit-step)**: L1 = RL-only rollout_probs_diff↓; 사이클 = rollout_corr/kl 피크<2–3 & ESS바닥>0.7(아니면 P2); 길이(근본) = 훈련 길이가 1.2k→HPT식 3–5k 성장 시 길이-붕괴 근본해결 입증.
- **정직한 한계**: clean-async는 **위생 + 사이클/길이붕괴 부분해**이지 41.9 돌파 카드가 아니다. ScaleRL 어법상 L1만 A축(단 우리 mismatch 온건 → 이득 불확실), L2는 부분해. ~~41.9의 무게중심은 P1(LUFFY-hybrid)~~ **[정정 §5.11.9: P1은 scope 밖 — 무게중심은 완주 + sync-HPT parity 도달]**. 사이클 완치가 필요하면 **P2(KL-Cov 이동감쇠)** — rollout_KL/ESS 게이지로 판정.
- **기각/보류 원장(반박·검증 반영)**: L3(β조정 — 우리 SFT가 이미 논문 per-token 2×라 근거소멸) 기각 · L4(batch-norm — B축·advstd 엔진도 끔) P2강등 · L6(prompt-agg — 현 모드가 이미 token-avg 최적 + SFT 8× 증폭 부작용) 철회 · L5(zero-var — 취약코드·마진A) 다음 arm 대기 · 마스킹/trim/overlong penalty 거부권 유지.
- **운영 리스크**: 디스크 볼륨 99% — M5 발사 전 구런 rollout_dump·사멸런 ckpt 정리 필요(all_steps=True 유지 시). ~~M4는 M5 준비까지 control로 유지 후 회수.~~ **[갱신 2026-07-08]** 디스크 정리 완료(587G 확보), M4 회수(GPU 8장 해제)·M5 발사됨 — 실측은 §5.11.8.

### 5.11.7 SFT 정규화 함수형 — 논문 intensive vs 우리 extensive (L3/L6 정밀, 특이점)

dead-zone(k=0) 정체의 원인이 "SFT가 약해서"인지 코드로 검증한 결과 — **가설 기각, 단 구조적 차이는 실재하며 발현 시점만 다르다.**

- **손실 수식 대조**(양쪽 코드 직독): 논문(mix_actor) = `RL: Σ(-A·ρ)/8192`(token-sum, **extensive** — train.sh가 `loss_remove_token_mean=True`로 yaml 덮어씀) `+ 0.3·mean(SFT NLL)`(token-**mean**, **intensive**). 우리(seq-mean-token-sum-norm) = RL·SFT 모두 extensive/8192, SFT는 A=0.3.
- **per-token 실효 가중(SFT:평균RL)**: 우리 = `0.3/|Ā|` = **구성 무관 상수 ≈0.40**(|Ā|≈0.75). 논문 = `0.075/(s_tok·|Ā|)` = **s_tok(마이크로 내 SFT 토큰 비중)에 반비례**. 현재 s_tok≈0.54 → 논문 0.185. **즉 지금은 우리 SFT가 논문보다 per-token ~2.2× 강하다 → β 상향 근거 소멸(L3 기각).**
- **엔드게임 크로스오버**: s_tok 0.54(현재) 논문0.185<우리0.40 / 0.25 ≈동등 / 0.10 논문0.75>우리0.40(1.9×) / 0.05 논문1.50(3.8×). 논문 mean-norm은 **내장 커리큘럼** — 미해결 문제가 줄수록 각 교사 시범에 gradient를 자동 집중; 우리 extensive는 반대로 희석. **차이는 지금이 아니라 dead-zone 축소 국면(후반)에 발현.**
- **P2 카드 "intensive-SFT 모드"**: β_r ∝ 1/s_tok(작은 코드)로 논문식 고정예산 복원 — 장기 런 엔드게임 대비. 단일변수 원칙상 즉시 투입 안 함. Draft.tex A.4 Reduction 절 보강 재료.
- **L6(loss_agg) 동반 판정**: `seq-mean-token-sum-norm`은 ScaleRL 분류상 이미 token-avg 최적(A 0.535). prompt-avg로 바꾸면 "프롬프트당 1표"가 RL 8행 vs SFT singleton 1행의 비대칭을 깨 **SFT 가중 ~8× 증폭**(HPT 고유 부작용) → 철회. **loss_agg는 건드리지 않는다.**

### 5.11.8 M5(clean-async) 라이브 실측 (2026-07-08, 진행 중)

- **발사 정상**: L1 `enable_fp32_lm_head: True` 크래시 없이 SGLang init 반영(fork 배선검증 라이브 확인), GPU 8장 가동.
- **관측성 버그 라이브 확증**: `rollout_probs_diff_mean`(전-행) **0.136** vs `_rl_mean`(RL-only) **0.005** = **~25× 차** — M4의 "정밀도 병리 0.05–0.16"이 ~96% **SFT placeholder 오염**이었음이 실증됨(pearson_rl 0.998, sft_rows_excluded 48–56). 로깅 수정이 의도대로 작동.
- **M5가 M4를 앞섬(val, gap 0)**: 16.46(0)·28.29(5)·**33.66(10)** vs M4 17.56·27.73·**27.68**. M5 val@10이 M4 val@15(33.34) 수준 = **5스텝 앞섬**. M4는 5–10 평탄 후 점프, M5는 매끄러운 상승.
- **시그니처: 덜 움직이며 더 학습**(step10 M5 vs M4): ppo_kl **0.0005 vs 0.0010**·rollout_KL **0.010 vs 0.016**(둘 다 M5 낮음)인데 reward **0.425 vs 0.392**·onS 0.285 vs 0.253(M5 높음). L2(신선샘플)+L1(깨끗한 앵커/샘플)의 "적은 이동 대비 큰 학습" 시그니처.
- **fp32 이중 경로(뉘앙스)**: 진짜 RL mismatch는 0.004–0.011로 온건(§5.11 예측대로 → A 이득 불확실) — 그러나 fp32는 IS 앵커뿐 아니라 **샘플링 분포 자체를 충실화**하므로, 빠른 val엔 이 rollout-품질 경로가 기여했을 수 있다(이전 과소평가 지점).
- **결정적 시험 통과 — 사이클 진정 확증(step 11–15)**: M4가 폭발한 바로 그 구간을 M5는 **폭발 없이** 통과. 위험구간 peak 대조 — rollout_KL M4 **9.21** vs M5 **0.124**(step14 미세 blip, 즉시 감쇠) = **~74× 낮음**; ESS M4 0.20 붕괴 vs M5 **0.95–0.99 유지**; entropy M4 8.15 vs M5 **max 1.14**; 길이 M4 6197 폭주 vs M5 **1.2–2.1k 유지**(step11–13엔 오히려 하강). reward M4 step15 **0.199 붕괴** vs M5 **0.529 상승**. val@15 M5 **34.62**(gap 0) vs M4 33.34. → **L2(+L1)가 지연-피드백 진동을 "부분해" 예측을 넘어 거의 제거.** 단 step14 blip(rollKL 0.124·ppo_kl 0.028·entropy 1.14)이 잔존 진동성 = advstd 이득 존재(예측대로), 그러나 M4 대비 ~74× 작고 자가감쇠.
- **귀인 유보 + 다음 관문**: L1+L2 2델타 동시 → 개별분리 불가(사이클 진정은 L2 유력, val 가속은 L1의 샘플품질 경로 기여 가능); n=1 seed. **다음 결정 지점 = dead-zone(AIME) 전환율** — clean-async는 사이클/길이붕괴를 잡았으나 k=0 벽엔 무효 예상. AIME 정체 시 대응은 §5.11.9(P1 scope-out·엔트로피·intensive-norm) 참조.

### 5.11.9 LUFFY 대조 정정·교사 강도·β 원장 철회·scope 재정의 (2026-07-08, 세션 정련)

M5 진행 중 사용자 반박으로 이전 판정 3건을 정정하고 목표·scope를 재확정한다.

- **LUFFY 기준점 정정 (산수)**: 붙여넣은 비교표의 LUFFY-Zero "Avg 42.1"은 **6개 벤치 단순평균이 아니다**(단순=37.2, 크기가중=50.6, 둘 다 아님). Base·Instruct는 표기AVG=6개평균 일치하나 학습 3행(SFT/RL/LUFFY)만 +4.4~4.9 초과 → **표가 두 출처 병합**(per-bench 열 ≠ Avg 열). **유일한 공정 기준 = 로컬 재현 mean@8(우리와 동일 harness) = 39.58.** (붙인 표 per-bench도 mean@8 아님 — AMC 46.8인데 실제 56.56.) M5 val@20 34.41 vs 39.58 = **−5.17, 6개 전부 뒤짐**(이전 "AMC/Minerva 앞섬"은 잘못된 표 탓 → 철회). 단 M5는 step20 이른 값·M4를 매 스텝 앞섬(val@10 +6·@20 +1.8) → 정점 ~37–38 예상, 잔여는 dead-zone(AIME/AMC/Olympiad 격차 최대). **목표 위계: LUFFY mean@8 39.58 = 프로토콜 일치 실전목표; 논문 41.9 = avg@32/avg@1이라 우리 mean@8과 직접비교 불가 → HPT 로컬 mean@8 재현으로 진짜 목표 확정 권고.**
- **교사 trajectory 강도: LUFFY vs 우리 (코드 확정)**: LUFFY `n_prefix=1`(train_luffy.sh) — **전 프롬프트** 8개 중 1개를 교사로, 같은 advantage 그룹에서 **RL-대조**(교사 +adv / 실패롤아웃 −adv) + shaped ratio로 off-policy 보정. 우리 게이트(hpt_gate.py) — **k=0만** `on_remove=8/off_add=1`(실패 8 버리고 교사 1을 SFT β0.3). **LUFFY가 넓고(전프롬프트 vs k=0만) 강함(RL-대조 vs SFT-모방).** 그러나 **논문 HPT도 우리와 동일**(mix_trainer `on_remove=8/off_add=1`, gamma0, β0.3 SFT)**인데 LUFFY를 이김**(41.9 > LUFFY) → **선택적·약한 교사가 넓은·강한 교사를 이긴다 = 교사 강도는 우리 격차의 원인이 아니다.** 격차는 async-HPT가 sync-HPT 천장 미도달일 뿐. → 교사 강화(LUFFY화)는 불필요 + scope 밖.
- **β 원장 철회 (커밋 정밀)**: M2/M3(`033e96a2`)와 M5(HEAD `ca560fcd`)는 **런타임 학습 코드 동일**(`5ebd614b` 이후 `df720458`=gate 검증만·로깅 수정=관측만). 그러나 config가 다축: **advstd False→True**(결정타)·β 1.0leninv→0.3const·큐. 따라서 "β1.0 실패(M2 26사망·M3 31정체)"는 **{advstd-off × leninv × β1.0} 합작이라 β를 격리 못 함 → §5.11.7의 실패원장 논거 철회.** advstd-on 레짐에서 β>0.3은 **미검증**(테스트된 적 없음). β 상수 인상 반대의 **유일 생존 논거는 parity**(우리가 이미 논문 per-token 2.2×, §5.11.7) — "올리면 실패한다"는 단정은 과했으므로 함께 철회.
- **P1 scope-out 확정**: Draft.tex의 주장 = async-HPT **아키텍처**(transport/semantics 분리). P1(LUFFY shaped-ratio off-policy loss, ~400줄 신규 estimator)은 **다른 방법**("HPT+LUFFY")이라 기여를 흐리고 취약코드 대규모 → **철회.** LUFFY는 이겨야 할 **기준점**이지 채택할 방법 아님. scope 안 레버 = ① 완주(500스텝) ② clean-async(완료) ③ **P2 intensive-norm**(신규 방법 아닌 **논문 HPT 집계(mean/intensive)와의 parity 수정**, §5.11.7).
- **엔트로피 급락 등록 (신규 관측)**: M5 entropy_mean 정점 1.10(step5) → **step20–25 0.14–0.31 급락.** dead-zone 연결 — 탐색 붕괴 → k=0 성공 stumble 불가 → 벽 자기강화. 값 0.001은 parity(논문 HPT도 0.001로 41.9)이나 **우리 async/CISPO/advstd가 sync/PPO보다 sharpening이 강해 같은 0.001에 더 급락.** Entropy-Mechanism 논문: 엔트로피 보너스는 무딘 도구(KL-Cov 선호). **판정: M5 진행 중 불변(단일축 오염 금지). M5 완주 후 별도 단일-델타 arm(entropy_coeff 0.001→0.003) 후보로 등록 — 기대 중간(무딘 도구), 근본은 생성 다양성 회복.**

---

## 5.12 M6 설계 — 신호밀도(B0/B1'/B3) + 탐색축 장전(B2) (2026-07-08)

M5가 예상(정점 ~37–38)을 넘어 계속 상승했고(§5.12.1), 이를 근거로 **M6 = M5 최고 체크포인트에서 resume + 검증계층 높은 3델타(B0/B1'/B3)**를 설계한다. 탐색축 B2(KL-Cov)는 **코드로 완성하되 config로 꺼서** 준비만 한다(단일축 귀인). 7축 증거스윕 + 3축 구현감사 + 4장 스켑틱 패널 + SAPO/cispo_klcov 냉정검토를 근거로 한다.

### 5.12.1 M5 완주 궤적 실측과 전망 (정체→반등, 점근 ~40)

- **val6 mean@8 궤적**: 16.46(0)·28.29(5)·33.66(10)·34.62(15)·34.41(20)·**32.45(25)·31.04(30)**[3연속 하락, 10–12×SEM 실질]·**35.99(35)**[반등]·37.05(40)·37.81(45)·**38.47(50)**[신기록·4연속 상승]. M4 동스텝 대비 계속 앞섬. **진동 구조 확정**: val은 ±2.5 진폭·주기 10–25스텝으로 느린 상승 평균 둘레를 진동 — **단일 판독 금지, 3-val MA로 판정.** advstd-off(M2/M3)도 진동(M3 val@15 −4.1) → 진동은 추정기 고유, advstd는 진폭 ~2배 증폭·수준 +7pt.
- **갭 해부 (vs LUFFY-local 39.58, val@45)**: AIME24 −2.92·AIME25 **0(닫힘)**·AMC −3.44·MATH-500 −1.88·Minerva −0.19(정점은 추월)·Olympiad −2.19. **닫힌 것/추월한 것(MATH·Minerva·AIME25) 외 잔여는 어려운 문제 질량(AIME24·AMC·Olympiad).** AIME24가 5스텝 만에 −6.25→−2.92로 전환 시작 = dead-zone이 늦게(1.5B 특성, UPT Fig.6) 열리는 신호.
- **전망(정직)**: 감쇠비 ~0.72 외삽 점근 ≈ **39.8**(대역 38.9–42). **39.58은 점근 한복판 = 자력 도달 가능권**(중심 스텝 ~80). **41.9-등가(우리 mean@8)는 점근 상단 꼬리 = 시간 내 어렵다**(사용자 판단 동의). 단 41.9는 avg@32/@1 프로토콜이라 애초 직접 바 아님 — 공정 바 = 동일 harness LUFFY-local 39.58.

### 5.12.2 정체 진단 — 신호-기아(signal starvation), 고착 아님

- **train score 정체(확인된 성분)**: 5-스텝 평균 0.216→0.344→0.481→0.518→0.544 후 **~step25부터 0.53–0.62 평탄**(가속 소멸). 성공률 ~0.6에서 **k=8 전정답 그룹이 늘며 배치 gradient 밀도 희석**(k=8은 GRPO advantage 0 = dead mass).
- **고착 아님 판정**: 저점이 얕아짐(36.59 vs 이전 31.04), offline_ratio 신저점(0.176 — dead-zone 여전히 느리게 전환), KL/길이/엔트로피 붕괴 없음. 정책이 굳은 게 아니라 신호 밀도 문제 → **B1'의 표적.** 진짜 "고착"이면 B2(탐색)가 답이고, 이는 게이트로 분기(§5.12.8).

### 5.12.3 M6 번들 — 검증계층으로 분리

| 레버 | 내용 | 상태 | 구현 지점 | 근거 |
|---|---|---|---|---|
| B0 | clip_grad 80→1.0 | core | 런처 1줄 | UPT parity; 실측 무해(§5.12.5) |
| B1' | 큐 의미론-인지 축출(k=8 우선) | core | message_queue/hpt_gate/rollouter/hpt_config | ScaleRL 천장 A+0.02; 폐기 롤아웃 재활용(§5.12.4) |
| B3 | max_completed_prompt_groups 384→256 | core | 런처 1줄 | ScaleRL k≤12 경계 |
| B2 | cispo_klcov(KL-Cov 오버레이) | **OFF(장전)** | core_algos/losses/actor config | Cui +2.0@7B, AIME 최대이득(§5.12.7) |

**번들이 단일-델타 규율을 안 깨는 이유**: (i) 네 레버 각자 독립 flag+전용 게이지로 개별 롤백/귀인 가능, (ii) 메커니즘 직교(최적화/전송·배치구성/staleness/토큰손실), (iii) B0/B1'/B3는 **가중치-의미론 불변축**(전송·최적화 안전장치)이라 resume 전후 비교가 준-통제. B2만 학습 수학을 바꾸므로 분리 대기.

### 5.12.4 B1' — 사용자 제안의 재해석: "staleness로 버리는 샘플을 쓰자"

사용자 통찰: 비동기에서 버리는 것은 강제(생산 ~293 vs 소비 128 → 165그룹/스텝 폐기, M5 실측 ~7940 drop)이나 **"무엇을 버릴지"는 자유**. 원안(트레이너-측 k=8 필터+리필, ~180 LOC)을 **큐-네이티브 축출**로 재설계:

- **의미론-인지 축출**: overflow 시 우선순위 [① k=8 zero-variance RL 그룹 → ② 나이순]. k=8은 GRPO advantage 항등 0 = 가장 안전한 victim. 정보 0인 것을 버리고 유익(다소 stale)한 것을 살림 = 사용자 의도 정확 구현. 소비 배치가 **오히려 더 젊어짐**(B3와 순방향).
- **★구현감사 정정 (cloudpickle-bytes)**: 큐 엔트리는 RolloutSample 객체가 아니라 `ray.cloudpickle.dumps(...)` **불투명 bytes**(rollouter:1043). 따라서 힌트를 객체 필드에 못 넣음 → **side-channel int로 분리**(put_sample(sample, evict_hint)). 이게 전송/의미론 분리를 **더 깨끗하게** 만듦: 큐는 int만 읽고 HPT 의미론 무지. deque는 `(hint, payload)` 튜플 저장, get_sample이 hint를 벗겨 payload만 반환 → 소비자 계약 완전 보존.
- **fail-open 내장**: 힌트 없으면 `first_hinted_index→None`→ popleft(기존 동작 비트단위 동일) → 2026-07-07류 fail-closed 크래시 경로 원천 부재. 비-HPT/flag-off 런은 `_saw_evict_hint=False`로 스캔조차 건너뜀(O(1)).
- **기각한 확장**: "SFT를 축출 보호"·"stale RL을 IS로 더 소비"는 소비 혼합비 왜곡(offline 57%, UPT Table6 교사-과잉 악화 레짐) 및 ScaleRL k≤12 위반이라 **기각.** 축출 대상은 오직 정보 0(k=8), SFT는 절대 미대상(RL/SFT 혼합비 보존).
- **정직 공시**: k=8 제거로 소비 배치 `offline_data_ratio` ~0.21→0.23–0.25 기계적 상승(조성 변화지 교사 라우팅 증가 아님). 게이지 `count/mq_evicted_zero_variance`로 활성 확인.

### 5.12.5 B0 해설 — norm clipping은 "삭제"가 아니다

사용자 우려("너무 많이 클리핑해 없애나?") 해소: `clip_grad`는 per-token/성분 삭제(PPO/CISPO clip)와 **다른 층위** — 1.5B 전체 gradient의 **단일 L2 노름 하나**를 보고, 노름>max_norm일 때만 **방향 보존 스케일다운**. 실측: M5 작동점 grad_norm 0.011–0.028(상한 1.0의 35–90배 아래), M4 95스텝(치명폭풍 포함) 1.0 초과 **단 1회(1.198@20, ×0.83)**. 즉 평시 발동 0. **넣는 이유 = (a) parity 위생**(80은 최초 async 런처에서 복사된 유래불명 보일러플레이트; UPT=1.0), **(b) Adam EMA가 못 잡는 단발 프릭 배치 보험**(advstd 2.83× × 길이 플레어 겹칠 때). **성능 효과 ≈0이라 개선 아니라 "정정"으로 분류.** 극단적 최소 델타를 원하면 빼도 무방(단 "왜 80?"이 재현성 검토에 남음).

### 5.12.6 SAPO 대체 검토 — 기각(백업 유지)

DR-005가 이미 심사한 축. CISPO와 SAPO는 **같은 g-슬롯 대체재**(결합 아님). 오늘 3근거로 기존 "backup" 판정 **강화 유지**: (1) **clipfrac 실측 ~0** — M5 pg_clipfrac 평균 0.032%·최대 0.34% = CISPO 캡이 잠들어 있어 gate 모양차의 효과 상한이 ~0(SAPO가 고치는 ratio-폭주 레짐 부재; ESS 0.95+·KL 0.002); (2) **CISPO와 head-to-head 부재** — SAPO 논문은 GSPO·GRPO-R2만 이김, Qwen3-30B **MoE** 검증(1.5B dense 근거 0); (3) **집계 하드코딩 함정** — `compute_policy_loss_sapo`가 `loss_agg_mode="seq-mean-token-mean"` 강제 → 우리 계약(sum-norm/8192) 조용히 깸 = 길이 병리 재초대 위험. **발동 조건 사전등록**: pg_clipfrac 유의 상승(캡 빈발) 또는 adv-양수 ratio 폭주 시그니처 → 계약-호환 포크+τ 스윕한 g-슬롯 단독 arm.

### 5.12.7 cispo_klcov — "믿을 만한 알고리즘"이 아니라 근거 있는 가설 (정직)

**기성 알고리즘 아님 — 이번 세션 설계 조합.** 신뢰도 계층 분해:
- **계층1 CISPO 본체: 강함**(ScaleRL 실증 A=0.595 + M4/M5 라이브).
- **계층2 KL-Cov 단독: 중간·생각보다 약함**. 이론(공분산 dH≈−Cov)은 독립 재확인(2511.05993/2509.26114), 코드는 저자가 verl 본가 병합(PR #1830). **그러나** 방법 이득의 강한 독립 재현 부재 + 후속 논문들이 KL-Cov 대신 각자 다른 해법 제시 = **미합의 경합지대**; 이득이 스케일 순방향(7B+2.0→32B+6.4)이라 **1.5B에서 축소·소멸 가능**(실증 0).
- **계층3 cispo_klcov 조합: 가설**. CISPO 위 KL-Cov 실증 어디에도 없음. DR-005 슬롯 틀로 구조 건전성 논증 가능(추가 페널티항=슬롯 비점유, g(1)=1, SFT 마스킹)이나 **구조 합법 ≠ 경험 유효**.
- **우리 데이터의 결정적 힌트**: 엔트로피 플로어는 상승을 **안 막았다**(M4 좋은구간·M5 4연속상승 모두 플로어 0.10–0.15). 즉 KL-Cov 표적은 "확인된 현재 병리"가 아니라 "미래 정체 가설".
- **maintain vs restore 비대칭 (핵심)**: 논문 이득은 전부 **붕괴 전 시작** 조건. 우리는 이미 플로어에 앉은 체크포인트에서 켬. (a) KL-Cov는 올리는 힘이 아니라 내리는 힘 제거 → 최선도 "유지"; (b) 붕괴는 부분 비가역(대안 continuation logit이 덮어써짐 = 사용자의 "고착"); (c) 복원된 다양성이 유용하단 보장 없음. **단 복원 경로 실존 증거**: M5 플레어(엔트로피 0.02→1.0 스파이크)는 자연 엔트로피 유입원(SFT 주입·음수-adv gradient·긴 응답) 존재를 보임 — KL-Cov의 현실적 복원 = 이 유입을 1–3스텝 재붕괴로부터 지켜 누적. **미검증이라 gated arm, 실패 시 fresh+B2(논문 검증 조건).**

### 5.12.8 시작점 — resume 우세, fresh는 게이트 뒤로

- **resume 근거**: (1) 재등반 ~2h+ 면제(감사: 코드 0줄, 가중치+옵티마이저+스케줄러+rng+데이터로더 위치 복원, in-flight 수백 프롬프트만 유실=무시); (2) B1'가 작동하는 고성공 레짐 직행; (3) 초반 분기(M4 step12 폭발) 재주사위 회피; (4) 가중치-의미론 불변 델타라 귀인 깨끗; (5) 하방 0(체크포인트 계보 보존). **`resume_mode=resume_path` + 명시 경로**(auto 금지 — 최신 스텝 잡음). 실행 전 `RESUME_FROM_STEP`을 실제 best-val 스텝으로.
- **fresh 기각**: M6-core에 탐색(엔트로피) 축이 없어 fresh는 같은 플로어 종착지로 ~2.3h 늦게 재수렴할 뿐. 고착 해독제(B2)는 resume에서도 작동.
- **의사결정 나무(사전등록)**: M6-core resume → **게이트1**(3 eval=15스텝 내 3-val MA 갱신?): 예→계속 / 아니오→**게이트2**(그 시점 최고 ckpt에서 loss_mode=cispo_klcov 단일델타 resume): 갱신→계속 / 실패→**게이트3**(fresh+B2, maintain 조건 = "고착" 최종 검증). 안전게이트(KL>1 2연속·truncation>10%·ESS<0.25) 상시.
- **디스크**: 소모 ~5.9GB/스텝(ckpt 3.8 + rollout dump 2.1) → 여유 378G에서 **~스텝110 소진.** 39.58 중심창(스텝80)은 안쪽, +3h창(스텝115)은 바깥 → **이전 런 rollout_dump 정리로 활주로 연장**(스텝100 전 필요). ckpt 정점 보존 필수(현재 step_50=38.47).

### 5.12.9 구현·테스트 요약 (2026-07-08 구현 완료)

- **B1'**: `message_queue.py`(deque `(hint,payload)` 튜플·`first_hinted_index` 순수함수·`_select_evict_index`·`evicted_hinted` 통계·client `put_sample(evict_hint)`), `hpt_gate.py`(`zero_variance_evict_hint` 순수 헬퍼), `fully_async_rollouter.py`(힌트 계산+전달+monitor 게이지), `hpt_config.py`(`queue_evict_zero_variance` 필드).
- **B2**: `core_algos.py`(`compute_policy_loss_cispo_klcov` 등록 — CISPO 본체 + RL-토큰만 Cov topk KL 오버레이), `losses.py`(화이트리스트+`hpt_sft_token_mask` 조건부 스레딩), `hpt_config.py`(화이트리스트+`CISPO_FAMILY` 클립검사), `actor.py`(docstring). **off = loss_mode=cispo**(kl_cov_ratio/ppo_kl_coef는 inert 대기); **on = loss_mode=cispo_klcov 한 줄.**
- **테스트**: `test_queue_semantic_eviction_on_cpu.py`(7: 힌트정책+선택+end-to-end), `test_cispo_klcov_on_cpu.py`(3: ratio0→CISPO동일·오버레이+grad유한·SFT제외). special_RL 전체 **150 통과**(스텁 `_CollectingQueue`에 evict_hint 미러링).
- **런처**: `run_..._M6.sh` = M5 상속 + B0(clip_grad1.0)·B1'(queue_evict_zero_variance=True)·B3(256)·resume·B2 kл params 장전(loss_mode=cispo). bash -n·override diff 검증.

---

## 5.13 C2 대반전 — 폭풍 벽의 진범은 CISPO, vanilla가 main을 탈환 (2026-07-09~10)

M6/M7(§5.12) 이후 이틀의 아크. 결론부터: **M5abl_nocispo(= M5 − CISPO, decoupled+vanilla)가 신 main이다** — 정점 lenient6 naive 40.17 / weighted 52.06 @step170으로 M5(38.47/50.97)를 완전 돌파하고 LUFFY-local(39.58/51.21)을 초과했으며, 그 과정에서 M-시리즈를 괴롭힌 "폭풍 벽"의 원인이 스택이 아니라 **CISPO 자체**임이 확정됐다. 현행 격자·레지스트리는 `Ablation_RL.md` §14가 단일 진실.

### 5.13.1 M7 만성진동 → 중단 (kill-gate 발화)

M7(4델타 동시)은 step 90부터 KL 점화 → 100-130에서 **만성 진동**(재점화 3회, ESS 최저 0.079, 길이폭발 6.2k↔단문붕괴 0.9k 왕복, score 기준선 미복귀 28스텝). 사전등록 kill-gate("나디르 후 ~10스텝 내 ESS>0.8 복귀 실패 또는 재점화") 발화로 중단. 인프라 포렌식(55만 줄 로그에 에러 0·디스크 사건과 무상관·KL 점화가 디스크 고갈보다 22분 선행)으로 **순수 알고리즘 현상** 판정. cispo_klcov(B2)는 이점 미증명으로 종결: 정점 38.33<38.47, matched-step 우위는 노이즈 이내, 핵심 기전(엔트로피 플로어 방어)도 실패(M7 0.10-0.13 < M5 0.14-0.33 — 선택 토큰이 배치당 수 개인 동종요법 수준).

### 5.13.2 M5R 대조실험 — "구조적 벽" 가설 (반증될 운명의 중간 결론)

M7 델타가 원인인지 가리려 순정 M5 구성을 건강한 global_step_95에서 재개(M5R) → **step 104에서 동일 붕괴**(엔트로피 5.99 폭주, val@105=0.01, SGLang 크래시로 사망). 이로써 "M7의 4델타"는 무죄가 됐고, 당시 결론은 "엔트로피 플로어×고활용×~100스텝의 **스택 구조적 벽**"이었다. M4(80-88)·M5(81-84)·M7·M5R 4런 정합. — 이 결론은 §5.13.3에서 반증된다. 방법론 교훈: M5R이 배제한 것은 "델타"까지였고, 4런의 진짜 공통분모(전부 CISPO 계열)는 한 변수 더 아래에 있었다.

### 5.13.3 nocispo가 가설을 반증 — 벽 = CISPO 귀인 확정

ablation 1순위 arm M5abl_nocispo(`oki4kv8u`, 델타 = `loss_mode: cispo→vanilla` + `clip_ratio_low: 10→0.2`뿐)가 같은 스택으로 **벽 영역을 무폭풍 통과**: step 95-190에서 KL≤2.21·ESS≥0.86·엔트로피 0.2 안정·score 0.60→0.65 상승. 전 구간 폭풍 0회(초기 step20 KL 7.0 미니스파이크 1회 자가소화뿐).

**기전(실측 뒷받침)**: CISPO의 `−sg(clip(r))·A·logπ`는 모든 토큰에 gradient를 유지하는 것이 설계 명분이지만, 뒤집으면 **min-clip의 비관성(브레이크)이 없다** — 과확신 토큰이 계속 밀려 엔트로피 붕괴→길이폭발→KL 폭풍으로 이어진다. vanilla Clip-Higher는 `pg_clipfrac` 0.5~9%로 브레이크가 실작동(M5의 clipfrac≈0과 대조). 이 레짐(entry anchor라 r≈1)에서 CISPO가 살릴 "죽은 피벗"은 애초에 거의 없었고(DR-005 §4.1의 조건부 예측과 정합), 남는 것은 브레이크 상실의 비용뿐이었다.

### 5.13.4 성능 — main 승격 (사전등록 규칙 3종 충족 후)

| lenient6 | naive / weighted | 비고 |
|---|---|---|
| **nocispo 정점 @170** | **40.17 / 52.06** | 대형벤치 주도(MATH 78.5·Olympiad 43.6·Minerva 31.9) — 요행 아님 |
| nocispo 150-190 창평균 | 39.22 / 51.34 | 사상 최고 고원, 완만 상승 중 절단(190, 재개 장전) |
| nocispo val@190 (절단점) | 39.70 / 51.30 | naive로 LUFFY 2번째 초과 지점 |
| M5 정점 @50 | 38.47 / 50.97 | 구 main |
| LUFFY-local | 39.58 / 51.21 | protocol-fair 목표 |
| paper-HPT sync `v96fvd0p` | 29.49 / 39.97 @50 | **비교 무효 2중**: 8×-scale as-run(유효 clip 10) + **채점기 불일치**(boxing-필수 entropy_math, 하방편향 — 우리 lenient 척도 아님). "압도" 주장 금지, 동일 채점기 재채점 선행 |

판정 절차: step 20-30의 +3.6~+5.8 "우위"는 M5 골짜기 신기루로 기각(당시 보류) → 40-50에서 역전당함 → 55+ 재역전 → **폭풍 벽 통과 + 150-190 고원 확립 후에야** 사전등록 3규칙(창평균/정점/안정성) 전부 충족으로 승격. 논문 41.9와의 직접 비교는 프로토콜 불일치(관대 grader +1~4·k=8 vs 32)로 금지 — "상회/동급"의 지면 확정은 동일 grader·decoding·budget의 fixed-checkpoint mean@32와 문항단위 paired hierarchical bootstrap 재평가로만.

### 5.13.5 격자 재편 + H축 신설 (요약 — 상세는 Ablation §14)

- C2 축 완료(main vs M5), C1 축 무런 폐쇄(main 실측 `P(w>C_w=2)` 중앙값 0.10% → 디커플링 준-불활성 판정, 구 D0는 세대각주 참고), D0′ 취소.
- **신규 H축 = RLonly**(`qzsnwc08`): 교사 채널의 순기여 격리 — 논문 핵심 가설의 직접 검증. `async_hpt.success_threshold=-1.0` 센티널로 **HPT를 켜둔 채 라우팅만 봉인**(전 그룹 "성공" 판정 → SFT 불발 → k=0 그룹은 adv 0 RL 행 = 순정 GRPO 의미론). 앵커·IS·큐가 main과 100% 동일한 단일축. 기동 검증 통과(step0 17.03, num_sft≡0). **결과(조기절단@162)**: 정점 37.73@30 후 플래토·후반 불안정(30.89@155 딥) — 초반(20-50)은 main과 동등하나 **후반(130-160 창) 격차 +3.4, 정점 +2.4** → 교사 채널의 값어치는 후반 지속-상승 동력과 안정성. **논문 핵심 가설 실증.**
- **RLonly v1 크래시 원장**: `async_hpt.enabled=False`는 비-HPT old-logprob 경로(version-1 save/restore, `actor_save_model_to_cpu`)로 진입 → **fsdp2 비호환 잠복 버그**("No DTensor-type parameters") 즉사. HPT 런들이 전부 우회해 와서 미노출. 회피 = 센티널 방식(v2). 상세: `Debug_RL.md`.

### 5.13.6 방법론 수확 (재사용 가능한 것)

1. **공통분모 회귀 오류**: N개 실패 런의 공통 원인을 찾을 때 "공유 구성요소 전부"가 용의자다 — M5R은 델타를 배제했지만 CISPO는 4런 모두의 기저였다. 대조군은 "의심 변수 하나를 뺀" 런이어야 완결된다.
2. **신기루 규율의 가치**: 초기 +5.8 우위에서 승격했다면 틀린 이유(골짜기 비교)로 결론에 도달했을 것 — 사전등록 판정 규칙이 "맞는 이유로 맞는 결론"을 강제했다.
3. **ablation이 main을 이기면 그것이 새 main** — 격자는 라벨만 뒤집으면 수용된다(레시피 동결 + 단일축 원칙 덕). 서사도 "도출(DR-005) → ablation 반증 → 수정"으로 오히려 강해진다(§8 정신).
4. **콘솔 로그 4096자 truncate 함정 재확인**: val 부재 오판 방지 — `.wandb` 바이너리에서 history record의 `item.nested_key`("/".join) + `value_json`으로 추출이 정답.

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
