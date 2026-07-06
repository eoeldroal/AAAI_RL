# Ablation_RL — Async-HPT Ablation Study 설계

Status: 설계 확정(M-앵커 2×2로 재배향, 2026-07-04) · D0 런처 반영·run 완료 · C1 decoupling / C2 CISPO config+loss+routing 구현 완료(런처/run은 별도) · A1은 config-only · 라벨 대응: 구 A5≡M−cispo, 구 A6c≡M(§3)
범위: DR-001~005가 내린 **결정 요소를 D0 기준으로 하나씩 뒤집어** 각 결정의 기여를 격리하는 ablation. 결정의 근거·이론은 각 DR 소관이며, 이 문서는 그 결정들을 **실행 가능한 실험**으로 배치한다.
관련 문서: `DR-001-loss-normalization_1.md`(집계) · `DR-002-auxiliary-terms_1.md`(aux) · `DR-003-offpolicy-supervised-branch_1.md`(SFT branch) · `DR-004-offpolicy-rl-branch_1.md`(decoupling) · `DR-005-rl-objective-composition_1.md`(결합 문법) · `Codemap_RL.md`(코드 위치)
관련 코드: `verl/workers/utils/losses.py::ppo_loss` · `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla` · `verl/trainer/ppo/rollout_corr_helper.py` · `verl/experimental/fully_async_policy/hpt_{config,training}.py`
전제: fully-async + HPT, GRPO(Dr.GRPO), `ppo_epochs=1` fit_step당 SGD≈4, partial rollout.

---

## 0. 한 문단 요약

이 ablation은 **M(예상 최강 구성)을 앵커로 두고 요소를 하나씩 덜어내는(leave-one-out) 2×2 요인 설계**다. 축은 정확히 둘 — **C1 = decoupling**(entry anchor B + TIS-w, DR-004 §8/§9)과 **C2 = CISPO**(M의 upper-only g-슬롯, DR-005 §6) — 이고, 그 외 확정 결정 전부(DR-001 집계 / DR-002 aux 마스크 / DR-003 self-detach + Clip-Higher·Dr.GRPO 등)는 **공통 기반**으로 전 격자점에 고정된다. 격자 네 점: **M**(= 기반+C1+C2, main run, 신규 런처) · **M−cispo**(C2 제거 = decoupled+TIS-w+vanilla/Clip-Higher; 구 A5) · **M−dec**(C1 제거 = 원조 CISPO 형태, 캡 충돌 관측 arm; 신규) · **D0**(둘 다 제거 = 기존 main run, **이미 완료**). 실행은 measure-first가 아니라 **strongest-first**다: M을 먼저 돌려 기완료 D0와 비교(풀스택 총 이득)하고, 귀속이 필요할 때만 나머지 두 코너를 돌린다. §11 pivotal 지표는 always-on이라 사전 게이트가 아니라 **사후 귀속**을 담당한다. **A1**(집계 mode flip, DR-001 검증)은 C1/C2와 직교하는 별도 축으로 유지한다. rejection(B1)·SAPO·belt(A4)·GSPO·a3po·mis는 확정 제외(§6, 원장). 각 arm의 판정은 최종 성능이 아니라 **DR가 예측한 관측 서명**으로 한다.

---

## 1. 설계 원리

두 규율을 계승하되, 실행 방향은 재배향한다(2026-07-04 결정).

- **strongest-first / 귀속은 사후에.** 옛 판(D0 앵커)은 measure-first 게이트(사전점검 통과 후 착수)였다. 재배향 후: **M을 무조건 먼저 돌린다.** DR-004 §11 라우터·death-밀도 전제는 "무엇을 만들지의 사전 게이트"에서 "**각 요소가 실제로 값을 했는지의 사후 판독**"으로 지위가 바뀐다 — §11 지표가 M run 안에서 always-on이라 판독 비용이 0이기 때문이다. 빌드는 무조건, 판독은 결과로.
- **단일-요인 격리 — 기준점은 M.** 모든 arm은 M에서 **정확히 한 요소**만 덜어낸다. 공통 기반(DR-001~003 + §2 전제)은 전 격자점 고정, 딸려 흔들리는 축은 통제 손잡이로 고정한다(A1의 β_r, 전 arm의 k_max·γ, M·M−dec의 `C_g=1.28` 동일값 고정). 2×2가 완전하므로 단독 효과(각 코너 vs D0)와 한계 효과(M vs M−X)가 모두 나온다.

DR-001~003(구현·확정)은 공통 기반으로 항상 켜져 있고, DR-004/005(신규 구현)가 두 축을 이룬다. A1은 DR-001 결정의 도출-확인용 직교 flip으로 격자 밖에 둔다. DR-005 추정량 `∇Ĵ = E[w̄·g(r)·A·∇logπ]` 기준으로 C1=w-슬롯, C2=g-슬롯, A1=집계층이다.

---

## 2. 공통 기반 — 인자별 구성 (전 격자점 고정 · D0 = all-off 코너)

D0 = "확정된 DR 결정 전부(공통 기반) + 두 축(C1/C2)은 off". 아래 (a)~(g)는 격자의 **모든 점**에서 동일하게 켜져 있고, (e)의 anchor와 (f)의 g-슬롯만 C1/C2 축으로 움직인다. 층별 값과 근거:

### (a) 집계층 — DR-001
| 인자 | D0 값 | 근거 |
|---|---|---|
| `actor.loss_agg_mode` | `seq-mean-token-sum-norm` | RL 인센티브(Dr.GRPO): 자기-길이 divisor 금지 → 상수 divisor |
| `actor.loss_scale_factor` | `8192` (=L_max=max_response_length) | sum-norm의 상수 분모 |
| `async_hpt.sft_beta_mode` / β | `constant` / `beta=0.3` | SFT 배분 = "토큰당 정액". 0.3은 UPT SFT_LOSS_COEF(1.5B)에 대응 |
| `async_hpt.loss_aggregation` | `branch_blind` | B_eff/prompt-equal 분모 제거 |

### (b) advantage — 전제
| 인자 | D0 값 | 근거 |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | HPT enable 강제 |
| `algorithm.norm_adv_by_std_in_grpo` | `False` (Dr.GRPO) | std 정규화 없음, UPT 파리티 |

### (c) SFT branch — DR-003
| 인자 | D0 값 | 근거 |
|---|---|---|
| SFT old_log_prob | `log_prob.detach()` → ρ≡1 (NLL) | provenance 없음 → 보정·통계적 trust region 둘 다 불요 |
| belt (entry-snapshot+clip on SFT) | **OFF (미구현)** | 통계적 불필요; drift-pacing은 미검증 가설 |
| SFT의 correction/rejection/staleness | 면제 (G5) | rollout에서 안 나온 데이터 |

### (d) auxiliary — DR-002
| 인자 | D0 값 | 근거 |
|---|---|---|
| `actor.entropy_coeff` (RL) | `0.001` | RL 탐색 정칙화, UPT 파리티 |
| `async_hpt.sft_entropy_enabled` | `False` → SFT 마스킹 | entropy는 분포를 평평하게 → "확신 있게 모방"과 상충 |
| KL 전부 (`use_kl_loss`, `kl_*`, `sft_kl_enabled`) | 전부 off / False | reference model 없음 → anchor-KL은 non-issue |

### (e) w-슬롯 / staleness — DR-004 (coupled 기본)
| 인자 | D0 값 | 근거 |
|---|---|---|
| `async_hpt.rl_old_logprob_source` | **`rollout`** (융합) | D0 기본값. `entry`는 C1 arm(M·M−cispo)에서 구현된 opt-in 경로 |
| `rollout_correction.rollout_is` | `null` (+ old==rollout이라 어차피 IS≡1, inert) | correction dormant |
| `rollout_correction.rollout_rs` | `null` (OFF) | support 절단은 별도 결정 |
| `rollout_is_threshold` (C_w) | `2.0` (entry 모드에서만 실효) | 절단 IS 임계 |
| `async_hpt.k_max` | **`null`** (RL row 학습-시점 드롭 없음) | 낡음은 예산 레벨로만 제어(§2 pin 참조) |
| `rollout.calculate_log_probs` | `True` | anchor에 필수 |

### (f) g-슬롯 / trust-region — DR-004 §1-8 / DR-005 §4.0
| 인자 | D0 값 | 근거 |
|---|---|---|
| clip 형태 | Clip-Higher (비대칭) | 현행 런처 실태. async staleness가 clip 자체는 강제(no-clip 금지, DR-005 §5) |
| `actor.clip_ratio_low` / `high` | `0.2` / `0.28` | 상방 여유(entropy 붕괴 완화, DAPO) |
| `actor.clip_ratio_c` (dual-clip) | `10.0` | A<0·r 폭주 유계화 |

### (g) 라우팅 / 스케일 — 문맥 (이번 ablation에서 고정)
`async_hpt.gamma=0.0`(0/8 전멸 prompt만 SFT, UPT SWITCH_GATE=0 대응) · `rollout.n=8` · `ppo_epochs=1` · `ppo_mini_batch_size=32`×`require_batches=4`=128 prompt group · `partial_rollout=True` · `trigger_parameter_sync_step=4` · `staleness_threshold=2.0`.

### D0 명시 고정 (3-pin) — 암묵 기본값을 앵커로 못박음

아래 셋은 verl 기본값과 동일해 **동작을 바꾸지 않지만**, ablation 델타가 D0에서 한 축만 벌어지도록 코드에 명시했다(런처 [각주 10]). **누락이 아니라 의도된 고정이다.**

| pin | 이유 |
|---|---|
| `actor.ppo_epochs=1` | C2(CISPO) 기여 판독 논증이 이 값에 의존(DR-005 §4.1: anchor B + ppo_epochs=1이라 r 표류↓ → death 표면↓) |
| `async_hpt.rl_old_logprob_source=rollout` | coupled anchor = D0/M−dec 코너. entry는 C1 arm(M·M−cispo)에서만 켠다. entry 추가 후에도 D0가 명시로 남게 함 |
| `async_hpt.k_max=null` | RL 학습-시점 staleness 드롭을 **의도적으로 끔**. 낡음은 예산 레벨(`staleness_threshold`)로만 제어하고 C1 arm에서 절단 IS(C_w)로 보정. 전 격자점 null 고정 |

> `k_max=null`의 함의: 전 격자점이 낡은 RL row를 학습 시점에 버리지 않으므로, C1 arm의 w 분포 꼬리가 k_max 컷 없이 넓게 나올 수 있다. 이는 결함이 아니라 "낡음을 드롭이 아니라 절단 IS로 다룬다"는 decoupling의 정합적 형태다(DR-005 §7-1의 C_w×k_max 공동 조율을 "C_w 단독"으로 단순화).

---

### M — 앵커 명세 (신규 런처 = D0 인자 + 아래 knob)

| knob | 값 | 축 |
|---|---|---|
| `async_hpt.rl_old_logprob_source` | `entry` | C1 |
| `async_hpt.entry_proximal` | `recent` (옵션 B: 매 fit_step 현재 가중치 forward, DR-004 §8-2) | C1 |
| `algorithm.rollout_correction.rollout_is` / `rollout_is_threshold` | `token` / `2.0` (C_w) | C1 |
| `algorithm.rollout_correction.rollout_rs` | `null` (rejection OFF 확정) | C1 |
| `actor.policy_loss.loss_mode` | `cispo` | C2 |
| `actor.clip_ratio_low` | `10.0` (CISPO upper-only: `1−10<0` → 하한 무효; 원논문이 lower bound 비활성) | C2 |
| `actor.clip_ratio_high` | `0.28` (상단 cap `C_g = 1+0.28 = 1.28`, 이동-partition 스케일; M−cispo 밴드 정렬) | C2 |
| 그 외 전부 | §2 공통 기반과 동일 | — |

---

## 3. Ablation arms — 2×2 격자

| | C2=CISPO | C2=vanilla(Clip-Higher) |
|---|---|---|
| **C1=decoupled** | **M** (앵커 = `uvbi7wq3`, pre-fix·step175, 부록 레지스트리) | **M−cispo** (구 A5) |
| **C1=coupled** | **M−dec** (신규) | **D0** (완료 ✅ = `gvqi3cgq`, 부록 픽스) |

라벨 대응(상호참조 보존): **A5 ≡ M−cispo · A6c ≡ M** · DR-004 §10의 B0 ≡ M−cispo, B1 ≡ 제외(§6 원장). (B0′=`entry_proximal=mis`는 코드가 `Literal["recent"]`로만 열어 **미구현·제외**; §6 원장.)

각 arm: **바꾸는 인자 → 격리하는 기여 → 관측해야 할 차이 → 통제 → 해석**.

### A1 · 집계 mode `sum-norm → seq-mean-token-mean` (DR-001, config-only)

**바꾸는 인자.** `actor.loss_agg_mode=seq-mean-token-mean` 한 줄.
**격리.** RL 인센티브 편향(Dr.GRPO 위반) = 응답 길이 조작 유인.
**필수 통제.** mode를 바꾸면 SFT 실효 배분이 딸려 흔들린다. **β_r를 조정해 SFT 실효예산을 D0와 동일하게 고정**(DR-001 §4.3 lemma를 통제 도구로). β_r를 ablate하는 게 아니라 통제 손잡이로 쓴다.

**관측해야 할 차이 — "정답/오답 길이의 갈라짐"이 핵심 서명.**
| 지표 | D0 (sum-norm) | A1 (token-mean) 기대 | 메커니즘 |
|---|---|---|---|
| 오답(A<0) 응답 길이 | 평탄 | **증가 ↑** | 자기-길이 divisor → 길게 틀려도 토큰당 벌점 희석("공짜") |
| 정답(A>0) 응답 길이 | 평탄 | **정체/감소 ↓** | 짧을수록 토큰당 강화가 세짐 |
| 두 곡선 간격 | 거의 불변 | **시간 따라 벌어짐** | 위 둘의 합 = 인센티브 편향 |
| SFT token 점유율 | (기준) | **D0와 동일해야** | 다르면 통제 실패 → 재실행 |

**증폭 예상.** `gamma=0.0`이라 A<0 다수 레짐 → 일반 Dr.GRPO 재현보다 오답 길이 폭주가 더 크게 보여야 한다.
**해석.** 갈라짐이 보이면 sum-norm 선택의 실증. 안 보이면 이 레짐에서 길이 게임 미발동(sum-norm은 동기 관례·Dr.GRPO로 독립 정당 → 설계 불변). 최종 성능 단독은 약한 증거, **길이 서명이 1차**.

### M−cispo (구 A5) · M에서 C2 제거 — decoupled(entry, TIS-w) + vanilla Clip-Higher

**바꾸는 인자 (한 세트).** `rl_old_logprob_source: rollout→entry` · anchor=**옵션 B**(매 fit_step 현재 가중치로 old_log_prob forward, 가중치 스왑 없음) · `rollout_is` ON(C_w=2.0 절단) · `rollout_rs` OFF.
**격리.** staleness 보정을 clip에서 분리. clip은 `r=π_current/π_entry`(이번 이동분)에만, 낡음 `w=π_entry/π_rollout`은 절단 IS로 곱셈.
**사후 판독 (실행 조건 아님 — strongest-first).** D0(기완료) 로그의 `pg_clipfrac`×staleness 상관과 이 arm의 `P(w>C_w)`가 decoupling 실효를 판독한다. `P(w>C_w)≈0`이면 이 레짐의 낡음이 낮아 M−cispo≈D0가 되는데, **그 자체가 결과**다(DR-004 §6 "이점은 조건부"의 실증). 재배향 후 이 판독은 착수 게이트가 아니라 귀속 근거다(§1).
**필수 통제.** k_max(null)·cliprange·γ를 D0와 동일. **주의: D0는 "무보정"이 아니라 "융합 clip + 장치② dormant(IS≡1)"** — `(M−cispo)−D0`는 "보정 추가"가 아니라 "융합을 분리".

**관측해야 할 차이 — "clip-frac 분리 + stale 토큰 부활".**
| 지표 | D0 (융합) | M−cispo (분리) 기대 | 의미 |
|---|---|---|---|
| clip-frac | 높음(낡음+이동 뒤섞임) | **r-clip-frac ≪ D0** | staleness 성분이 clip→w로 이동 |
| w 포화율 `P(w>C_w)` | (없음) | **0보다 유의미** | 절단되는 낡음 질량. ≈0이면 M−cispo≈D0 |
| 동결됐던 stale 토큰 | 얼어붙음 | **gradient 흐름 재개** | "낡았지만 이번엔 안 과격한" 토큰 복귀 |
| RL:SFT 유효비 | (기준) | **D0와 동일해야** | rejection OFF라 RL row 안 버림 → 조성 불변 |
| step-time | (기준) | **+≈actor forward 1/3** | entry forward 비용(이득 0이어도 내는 고정비) |

**해석.** r-clip-frac↓ + stale 토큰 부활 + 성능↑ → 이 레짐에서 decoupling 이득. `P(w>C_w)≈0`이면 "이 run의 낡음이 낮음"(사후 판독으로 그렇게 읽힘 = M−cispo≈D0). 성능↓이면 절단(편향) IS < 얼리기의 정직한 음수(GRPO라 stale-advantage 위협은 완화, DR-004 §6).

### M (앵커, 구 A6c) · 풀스택 — decoupled + TIS-w + CISPO-sg (main run, 신규 런처)

**구성.** C1(w-슬롯=TIS-w) + C2(g-슬롯 `min-clip → g(r)=sg(min(r, C_g))`, `C_g=1.28=1+ε_h`) 전부 켠 예상 최강점.
**왜 C1+C2를 함께 갖나?** coupled에서 CISPO를 켜면 staleness 캡(`C_w`)과 이동 캡(`C_g`)이 하나의 ρ 위에서 충돌(DR-005 §4.2c). decoupled 위라야 두 캡의 의미가 분리된다 — 그래서 M은 둘을 함께 갖고, C2 단독의 캡-충돌 형태는 M−dec가 실측한다.
**필수 통제.** `C_g=1.28`(=`1+clip_ratio_high`)은 g-슬롯의 **이동 partition** 캡이다 — decoupling이 staleness를 w-슬롯(`C_w=2.0`)으로 분리했으므로 이동 스케일에 맞춘 값이며(DR-005 §6/정정4), M−cispo의 상단 밴드(0.28)와 정렬해 C2 축을 "같은 지점 gradient 死 vs 유지"만으로 격리한다. 구현은 upstream 단일 채널(`clip_ratio_low=10`으로 하한 비활성 + `clip_ratio_high=0.28`). ScaleRL/ms-swift의 `5.0`은 coupled·async ratio용이라 이동 캡에 쓰지 않는다. min()의 비관성(악화 방향에서만 gradient) 상실을 감시.

**관측해야 할 차이 — CISPO 한계 기여의 판정은 `M − (M−cispo)`.**
| 지표 | M−cispo (vanilla clip) | M (CISPO) 기대 | 의미 |
|---|---|---|---|
| death 밀도(clip된 low-prob 피벗 비율, A 부호별) | 일부 피벗 gradient=0 | **≈0** | 죽던 피벗 토큰이 캡 값으로 복귀 |
| **M−cispo의 death 밀도가 애초에 >0였나** | — | **사후 판독** | ≈0이면 살릴 게 없어 M≈M−cispo — 그 자체가 결과(DR-005 §4.1 검증) |
| entropy 추이 | (기준) | 탐색 특성 변해 이동 가능 | 지표로만 추적, 선제 조정 금지 |
| grad-norm / 발산 | 안정 | **A>0·r 폭주 구간 spike 감시** | 비관성 상실 → PPO면 멈출 곳을 계속 밀어붙임 |
| `ppo_kl`류 모니터링 | 기존 의미 | **의미 바뀜** | 추정량 클래스가 REINFORCE-with-coefficient로 변경 |

**해석.** M−cispo에서 피벗-death가 실재했고 CISPO가 되살려 성능↑ → g-슬롯 교체 정당. **M−cispo death 밀도≈0이면 M≈M−cispo** = "CISPO가 나쁜 게 아니라 이 레짐이 gradient-死를 안 만듦"(DR-005 §4.1 예측의 검증). grad-norm spike/발산 → vanilla clip의 비관성이 안전장치로 일하고 있었던 것.

### M−dec (신규) · M에서 C1 제거 — coupled(rollout anchor) + CISPO-sg

**바꾸는 인자.** M에서 `rl_old_logprob_source=rollout`으로 되돌리고 `rollout_correction` 블록을 내린다(coupled에선 w≡1이라 inert — validation이 명시 설정을 거부). g-슬롯 CISPO는 유지.
**격리.** CISPO 단독 효과 + **캡 충돌의 실측**(DR-005 §4.2c): coupled에서는 staleness와 이동이 하나의 ρ에 섞여 `C_g` 캡 하나가 두 역할을 겸한다 — 원조 CISPO(MiniMax-M1)가 정확히 이 형태다.
**관측.** `M − (M−dec)` = decoupling의 한계 기여(CISPO 위). `M−dec − D0` = CISPO 단독. staleness가 낮은 레짐이면 M−dec≈M 예상 — 그 자체가 "decoupling 이점은 조건부"(DR-004 §6) 판정.
**통제.** `C_g=1.28`은 M과 **동일 값 고정**(캡 의미는 이동→ρ로 달라져도 값 고정이 단일-요인 조건). 이 이동용 캡이 coupled ρ(staleness 포함)에 걸리면 상단 포화가 커져 캡 충돌이 그대로 관측된다. validation은 coupled+cispo를 RAISE가 아니라 **WARN**으로 허용한다(이 arm의 존재가 이유).

---

## 4. arm 구성과 실행 순서 (2×2 격자)

```
              C2=CISPO            C2=vanilla(Clip-Higher)
C1=decoupled  M (앵커, 1순위) ───  M−cispo (구 A5)
               │                     │
C1=coupled    M−dec (신규)   ───  D0 (완료 ✅, 재사용)
```

델타 해석: `M−D0` = 풀스택 총이득(M만 돌리면 즉시 가용) · `M−(M−cispo)` = CISPO 한계 기여 · `M−(M−dec)` = decoupling 한계 기여 · 각 코너`−D0` = 단독 효과.

실행 순서: **① M(신규 런처, 최우선)** → ② M이 D0와 유의미하게 다를 때만 M−cispo·M−dec(귀속) → ③ A1(직교 검증, config-only). 신규 run 최대 3(+A1), D0는 재사용. 구현 비용: C1(entry-forward anchor B) + C2(CISPO loss) **두 knob이 격자 전체를 생성** — arm 간 차이는 전부 런처 인자.

---

## 5. 공통 통제 & 지표

**arm 간 고정 (통제 인자):** `k_max`(null)·cliprange(축 아닐 때)·`ppo_epochs`(1)·RL:SFT 목표비(γ=0.0)·lr(5e-6)·grad-clip(80)·모델·데이터.

**모든 run에서 항상 보고:** RL:SFT 유효 row/token 비(post-mask) — 없으면 "안정성 개선"이 진짜인지 배치가 조용히 SFT-지배가 된 건지 구분 불가(DR-004 §10). w 분포·포화율, r-clip-frac vs 융합 clip-frac(같은 이름의 다른 지표라 대시보드 구분 표기, DR-005 §8), `max_partial_span`, step-time.

**교란 통제 체크리스트:**
1. D0는 "no correction"이 아니라 "융합 + IS dormant" — `(M−cispo)−D0`를 "보정 유무"로 오독 금지.
2. A1은 β_r로 SFT 예산 고정 안 하면 단일-요인 아님.
3. 하이퍼 이식 금지: gate 인자가 ρ→r로 바뀌면 ε 의미가 전부 바뀐다.
4. `C_w × k_max` 공동 조율(D0는 k_max=null이라 C_w 단독).

---

## 6. 제외 원장 — A4(SFT belt) 및 확정 제외 목록

DR-003의 belt(SFT self-detach→entry-snapshot+clip)는 격자에서 뺀다. C1(decoupling)과 겉만 닮았지(둘 다 "π_entry+clip") **다른 branch·다른 provenance**다: A4는 SFT row(생성 정책 없음 → IS 공집합, 순수 최적화 pacing), C1은 RL row(rollout provenance 있음 → 진짜 off-policy 보정). 더 결정적으로, **SFT가 일으킨 drift는 rollout provenance를 가진 RL row 쪽(=C1의 w)에서 이미 보정**되므로 SFT에 또 벨트를 거는 것은 RL 처리와 중복(DR-003 §4). belt는 사전 계측에서 SFT-induced drift가 확인될 때만 여는 별도 ablation으로 남긴다(DR-003 §7). 같은 원장에서 rejection(B1)·SAPO·GSPO·a3po·mis도 확정 제외다 — 각각 DR-005 §5(조성왜곡 기각 / 같은 g-슬롯 대체재 / Lemma 3 위반 / ⏸보류)와 DR-004 §9(mis: 이론·비용 열등, probe 전용)가 근거다.

---

## 7. 우선순위와 컴퓨트

**M이 최우선** — main run이자 헤드라인 델타 `M−D0`를 단독으로 준다(D0 기완료). 귀속 run(M−cispo·M−dec)은 M이 D0와 유의미하게 다를 때만 — 차이가 없으면 격자는 조기 종료되고 그것도 답이다(풀스택 무효 = 두 축 모두 이 레짐에서 조건 미충족). **A1**은 config-only라 컴퓨트 남을 때 언제든. 컴퓨트 절단 순서: A1 → M−dec → M−cispo (**M은 절단 불가**).

---

## 8. 서술 원칙

ablation은 **선택을 낳은 것이 아니라 도출을 확인하는 위치**에 둔다(DR-001 §8, DR-003 §7). "여러 mode 시도해 최선을 골랐다"(경험적 튜닝) ❌ → "테제에서 도출 → ablation이 각 고리를 검증"(도출 후 확인) ⭕. C1/C2는 "놓친 안전장치"가 아니라 "가설의 처치군"으로 프레이밍 → 어느 결과가 나와도 방법이 안 무너진다(M≈D0조차 "두 축 모두 이 레짐에서 조건 미충족"이라는 결과다).

---

## 9. DR 정합 매트릭스

| arm | 검증하는 DR 결정 | 대응 추정량 층(DR-005) |
|---|---|---|
| M (앵커) | DR-004+005 풀스택: decoupling × CISPO 결합(§5 판정표 ✅) | w-슬롯 + g-슬롯 |
| M−cispo (구 A5) | DR-004: staleness/clip 분리(decoupling)의 조건부 이점 | w-슬롯 |
| M−dec (신규) | DR-005 §4.2c: coupled-CISPO 캡 충돌의 실증 + CISPO 단독 효과 | g-슬롯 |
| A1 | DR-001: sum-norm이 RL 인센티브(Dr.GRPO)를 지킨다 | 집계층 |
| (제외) A4 | DR-003: SFT는 벨트 불요 — decoupling의 w와 중복이라 별도 계측 후에만 | — |

DR-002(aux RL-only 마스크)는 D0에 이미 고정, 별도 arm 없이 SFT 마스킹으로 상시 유지.

---

## 10. 계측 (로깅) — 기존으로 충분한 것 vs 유일 갭

**원칙: wandb 추가는 pivotal-token 3지표(§11)뿐, 그 외 0 + 학습-side dump 1개.** live triage 지표는 대부분 이미 충분하고, "치밀한 사후 분석"에 구조적으로 없는 건 (a) loss 경계 텐서 하나(→ 아래 dump)와 (b) **per-token entropy**(→ §11 live 3지표)뿐이다.

**기존 wandb/롤아웃 dump로 충분히 obtainable → 신규 없음:**
- w 분포(`P(w>C_w)` 포함)·off-policy 거리(KL/PPL): `rollout_corr/rollout_is_*`, `compute_offpolicy_metrics` **자동 전파**. D0에선 old==rollout이라 trivial(≡1), C1 arm(M·M−cispo)에서 그대로 w-분포가 됨.
- r/fused clip-frac: `actor/pg_clipfrac(+lower)` (arm이 별도 run이라 run 정체성으로 구분).
- staleness: `stale_traj_count`(계산됨) + `trajectory_param_versions`(meta) + `fully_async/partial/max_partial_span`.
- RL:SFT 조성: `hpt/{num_rl_routed,num_sft,offline_data_ratio}` + `rollout_rs_masked_fraction`.
- 응답 길이(집계): `response_length/*`. 정답/오답 길이는 아래 학습-dump가 reward째 담아 offline 산출.

**유일 갭 = loss 경계 텐서** (생성 dump는 `generate_sequences_single` 출력만 = reward·학습 이전). death 밀도(C2 판독)는 `current log_prob + advantage + per-token clip 결정`이 필요한데 어디에도 없음.

**추가한 것 — 학습-side per-token dump** (`training_dump.py`, `training_dump.*` config, 기본 off):

| 항목 | 내용 |
|---|---|
| 통합 지점 | `FullyAsyncTrainer._fit_dump_data`(fit_step의 기존 훅, `_fit_update_actor` 직후 = 텐서 수렴 지점) |
| 담는 것 | per-token `response_mask,log_probs,old_log_probs,rollout_log_probs,advantages,hpt_is_sft,token_level_scores` + per-row `uid,prompt_uid,min/max_global_steps` + meta(step/param_version/rows) |
| offline 산출 | A1 정답×길이 · C1 w/r 분해·위치별 w · C2 death 밀도 |
| config | `enable`(off) · `dir` · `sample_every_n_steps`(20) · `max_rows`(256) · `dtype`(bf16) · `offload`(true) |

**무게 (걸림돌 아님):** read-only(라이브 batch를 이동/캐스팅/변경 안 함 — per-tensor 독립 CPU clone) · **sampled**(1/N)+**max_rows**로 볼륨 유계 · **offload**(background thread, 이전 write 진행 중이면 그 step skip → 트레이너 절대 non-block). 권장 기본값이면 이미 감내 중인 생성 dump(`all_steps=True`)보다 가볍다. 반대로 매-step 동기면 검증된 event-loop starvation 실패 모드로 직행(Readme_RL).

**검증:** `tests/special_RL/test_training_dump_on_cpu.py`(CPU, 15개) — read-only 불변성, round-trip fidelity, row cap, dtype cast, offload flush, busy-skip, config 계약. base-RL(HPT 필드 없음) 경로도 커버.

## 11. 분석 지표 — entropy·clip 해상도 (pivotal-token 질문)

C1(decoupling)·C2(CISPO)의 판정은 **집계 clip-frac이 아니라 pivotal-token 해상도**로 읽어야 한다. 이 절은 그 판정을 가능케 하는 최소 live 지표를 고정한다. 근거는 DR-004 §11(측정 라우터)·DR-005 §8(계측)과 문헌이며, 여기서 재서술하지 않고 배치만 한다.

### 11.1 질문과 문헌 근거

낮은 집계 clip-frac은 무해하지 않다 — 학습을 지배하는 **고엔트로피 소수(≈20%) forking 토큰**에 clip이 집중되면 집계값이 낮아도 피벗 신호가 죽는다.
- Wang et al., **NeurIPS 2025**(arXiv:2506.01939): 상위 20% 고엔트로피 토큰이 RLVR 성능 향상의 거의 전부를 만든다.
- **MiniMax-M1 / CISPO**(arXiv:2506.13585): rare·low-prob reflective 토큰(However/Wait/Recheck)이 높은 IS ratio로 clip되어 gradient가 소실. 원인은 **movement(다중 inner update)**이지 staleness가 아니다.
- **DAPO**(arXiv:2503.14476): 대칭 clip 상방이 저확률 토큰 성장을 억제 → entropy 붕괴(Clip-Higher가 완화, D0에 이미 반영).

### 11.2 기존 지표로 왜 안 되나

| 지표 | 한계 |
|---|---|
| `actor/entropy_loss`(=`actor/entropy`) | `seq-mean-token-sum-norm` **합계**라 RL 토큰 수에 오염. HPT 커리큘럼에서 RL 조성이 0→다수로 커지면 per-token entropy가 평탄/하락이어도 이 값은 상승(실측 확인) → **per-token 아님** |
| `actor/pg_clipfrac` | 집계 + SFT 토큰(ratio≡1) 희석. **어떤** 토큰이 clip되는지 층화 없음 |
| §10 per-token dump | w/r·death-density·길이의 오프라인 substrate이나 **entropy를 담지 않음**(`log_probs,old_log_probs,rollout_log_probs,advantages,token_level_*`만) → entropy 해상도 분석 불가 |

### 11.3 추가한 live 지표 3개 (RL-only · detached · forward 0)

`core_algos.compute_entropy_clip_diagnostics` → `losses.ppo_loss`에서 방출. 이미 계산된 텐서(entropy, ratio, clip predicate)만 재활용하므로 forward/backward·vocab 축소·통신 추가 없음.

| 지표 | 계산 | 읽는 법 |
|---|---|---|
| `actor/entropy_mean` | RL 토큰 per-token entropy 평균 | 오염된 entropy_loss 대체 → **entropy 실제 붕괴 여부** |
| `actor/entropy_top20_mean` | 상위 20% 엔트로피 토큰 평균 | pivotal 토큰의 붕괴(중요한 곳의 entropy) |
| `actor/pg_clipfrac_top20entropy` | 상위 20% 엔트로피 토큰의 clip-active 비율 | ★핵심★ `÷ actor/pg_clipfrac(전체)` ≫ 1 이면 **clip이 pivotal에 집중** |

- 빈 RL 마이크로배치(초기 all-SFT)엔 아무것도 방출 안 함(graceful, 학습 무영향). SFT 토큰은 `sft_entropy_enabled`와 무관하게 **항상 제외**.
- clip predicate는 `core_algos.pg_clip_active_mask`(공유 소스)라 `pg_clipfrac`과 정의 정합 → 비율 비교가 apples-to-apples.

### 11.4 라우팅으로 읽기 (DR-004 §11에 pivotal 해상도 추가)

| 관측 | 판정 |
|---|---|
| `clipfrac_top20entropy / pg_clipfrac ≈ 1` | clip이 pivotal에 안 쏠림. 집계가 낮으면 **고칠 것 없음**(출구 ①) |
| `≫ 1` + staleness **비례** | staleness 원인 → **C1(decoupling)이 값을 함** |
| `≫ 1` + staleness **무관** | movement/과클립 → **C2(CISPO)가 값을 함** (CISPO 문헌과 정합: 원인은 movement) |
| `entropy_mean`/`top20` 하락 추세 | 붕괴 서명 → C2 기여 가능성↑. 단 커리큘럼 조성 이동과 교란되므로 `hpt/p_success*`로 조건화해 읽는다 |

### 11.5 §10 dump와의 역할 분담 — 두 notion의 "pivotal"

| 층 | 담당 | 시점 |
|---|---|---|
| §10 per-token dump | w/r 분해·위치별 w·**low-prob** death 밀도(C2)·정답×길이(A1) | 오프라인, 임의 층화 |
| §11 live 3지표 | **high-entropy** 해상도 clip 집중·붕괴 | 실시간 라우팅 |

"pivotal"에는 두 정의가 공존한다: **low-prob**(CISPO 진단 — dump의 `log_probs`로 오프라인 산출) vs **high-entropy**(80/20 forking — entropy 필요 → live). entropy가 dump에 없어 생긴 갭을 live 3지표가 메운다 = **"wandb 추가 0"의 유일한 정당 예외**(3개, forward 0).

### 11.6 검증

`tests/special_RL/test_entropy_clip_diag_on_cpu.py`(CPU, 6개): clip predicate 상/하한, 값 정확성, **저엔트로피 clip이 pivotal 지표를 오염시키지 않음**, 빈 RL 마이크로배치 graceful, SFT 배제, detached float. `pg_clipfrac`(loss)과 predicate 정합은 `pg_clip_active_mask` 공유로 구성상 보장.

---

## 12. 분석 절차 — 로그·dump에서 DR 판정으로

§3이 arm별 **예측 서명**을, §10·§11이 **어디서 무엇이 나오는지**를 고정했다. 이 절은 run 종료 후 그 로그·dump를 **어떻게 계산해 DR 판정에 이르는지**의 절차다. 원칙: 판정은 최종 성능 단독이 아니라 **서명(signature) 일치**로 내린다(§8).

### 12.0 입력 아티팩트
| 소스 | 내용 | 용도 |
|---|---|---|
| wandb 스칼라(run별) | `actor/pg_clipfrac(+lower)`·`ppo_kl`·`entropy_mean`·`entropy_top20_mean`·`pg_clipfrac_top20entropy`(§11)·`hpt/*`(조성)·`rollout_corr/*`(C1 arm: M·M−cispo)·`response_length/*`·`fully_async/*`(staleness·partial) | 라이브 triage·추세·라우팅 |
| per-token dump(§10) | `log_probs,old_log_probs,rollout_log_probs,advantages,response_mask,hpt_is_sft,token_level_scores/rewards` + `uid,prompt_uid,min/max_global_steps` + meta(step,param_version) | 오프라인 정밀 분해 |
| run 정체성 | 격자 4점(M·M−cispo·M−dec·D0) + A1 각각 별도 run | arm 간 델타 |

### 12.1 공통 전처리 (de-confounding — 먼저 안 하면 오독)
1. **entropy_loss ≠ entropy**: 붕괴 판정에 `entropy_loss`(합계, RL 토큰 수에 오염) 금지 → `entropy_mean`/`entropy_top20_mean`(§11).
2. **SFT 희석 제거**: 집계 지표는 SFT 토큰(ratio≡1)이 분모에 섞임 → dump 파생은 `hpt_is_sft==False` 필터, 라이브는 이미 RL-only.
3. **커리큘럼 조성 통제**: RL 조성이 학습 중 0→다수로 이동 → 모든 추세를 `hpt/offline_data_ratio`·`hpt/p_success*`로 **층화/조건화**해 읽는다.
4. **arm 델타는 run 간**: D0(fused)와 C1 arm(decoupled)의 `pg_clipfrac`은 이름만 같은 **다른 양** → 대시보드 구분(§8, DR-005 §8).

### 12.2 dump 파생 계산 (정확한 수식, `hpt_is_sft==False` 행만)
```
w_t = exp(old_log_probs − rollout_log_probs)   # C1 on(M·M−cispo): π_entry/π_roll ; C1 off(D0·M−dec): ≡1
r_t = exp(log_probs − old_log_probs)           # C1 on: π_θ/π_entry ; C1 off: π_θ/π_roll (fused)
clip_active_t = pg_clip_active_mask(r_t, advantages, ε_low, ε_high)   # §11 공유 predicate
staleness_row = param_version(meta) − max_global_steps(row)
resp_len_row  = response_mask.sum(-1) ;  correct_row = advantages>0 (또는 token_level_scores)
```
파생: `P(w>C_w)`(포화율)·위치별 `w_t`(partial 꼬리=앞토큰이 가장 낡음)·r-clip-frac·**low-prob death 밀도** `= mean( clip_active & (exp(log_probs)<τ_lowprob), A 부호별 )`.

### 12.3 arm별 레시피 (계산 → 판정; 서명 상세는 §3)
| arm | 계산 | 판정 규칙 |
|---|---|---|
| **A1** | dump에서 정답/오답(A 부호)별 `resp_len` 추세를 D0 vs A1. SFT 점유율 동일 확인(통제). | 오답↑·정답↓·간격 벌어짐 → sum-norm 실증. 간격 불변 → 이 레짐 길이게임 미발동(설계 불변). 성능 단독은 약증. |
| **staleness 판독** | D0 wandb `pg_clipfrac` vs `staleness_row` 상관(binned scatter). | 무상관 → 낡음 낮음 → M−cispo≈D0·M−dec≈M 예상. **사후 귀속 참고이지 실행 게이트 아님**(§1). |
| **M−cispo** | 해당 dump로 w/r 분해: `P(w>C_w)`, r-clip-frac vs D0 fused, 동결→흐름 전환된 stale 토큰. RL:SFT 유효비 불변, step-time 델타. | r-clip-frac≪D0 + `P(w>C_w)`>0 + 성능↑ → decoupling 이득. `P(w>C_w)`≈0 → 이 run 낡음 낮음(그 자체가 결과). 성능↓ → 절단 편향 < 얼리기. |
| **M의 CISPO 기여 판독** | M−cispo dump low-prob death 밀도 + 라이브 `clipfrac_top20entropy`>0 여부. | ≈0 → 살릴 것 없음 → M≈M−cispo(DR-005 §4.1 검증). >0 → CISPO 기여 성립. |
| **M** | M dump death 밀도 재계산(≈0 기대) + grad-norm spike·발산 감시 + `ppo_kl` 의미 변경 유의. | death↓ + 성능↑ → g-슬롯 교체 정당. spike/발산 → vanilla 비관성이 안전장치였음. |

### 12.4 통합 라우터 (pivotal 해상도 = §11)
`clipfrac_top20entropy / pg_clipfrac` × staleness 상관:
- 비율 ≈1 → clip이 pivotal에 안 쏠림 → 집계 낮으면 종료(출구 ①).
- 비율 ≫1 & staleness **비례** → **C1(decoupling)이 값을 함** — M의 이득이 w-슬롯에서 옴.
- 비율 ≫1 & staleness **무관** → **C2(CISPO)가 값을 함**(movement 원인, CISPO 문헌 정합).
- `entropy_mean`/`top20` 하락(p_success 조건화 후) → 붕괴 서명, C2 기여 보강.

### 12.5 재현성·한계
- 파생 계산은 ad hoc 셀 금지 → dump+wandb export를 읽는 **분석 스크립트/노트북 1개**로 고정(입력=run dir, 출력=arm별 서명 표). arm 재실행 시 동일 스크립트로 재산출.
- **한계**: dump에 `entropy`가 없어 (a) 코호트 고정 entropy 붕괴 곡선, (b) high-entropy death 밀도는 **오프라인 불가**. (a)는 라이브 `entropy_top20_mean`(조성 조건화), (b)는 라이브 `clipfrac_top20entropy`로 대체. 정밀 오프라인이 필요하면 dump에 `entropy` 필드 추가가 유일 경로(§10 확장, entropy는 이미 계산되므로 forward 0).

---

## 부록: 런처 D0 검증 기록

`main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_main.sh`를 §2 명세와 대조한 결과 **D0를 정확히 가리킴**(잘못 걸린 값 없음). 명시값(loss_agg_mode, clip 0.2/0.28/10, entropy 0.001, grpo+Dr.GRPO, gamma/beta 등)과 검증된 암묵 기본값(`rl_old_logprob_source="rollout"` Literal 강제, `ppo_epochs=1`, `rollout_is/rs=null`, `alpha=1.0`)이 모두 D0와 일치. §2의 3-pin(ppo_epochs·rl_old_logprob_source·k_max)을 명시로 반영해 앵커를 코드에 못박았다([각주 10]).

### ★D0 실행 픽스 (2026-07-06): `gvqi3cgq`★

D0 코너의 실행은 **`eoeldroal-sogang-university/async-hpt-openr1/gvqi3cgq`** 로 고정한다(wandb `run-20260704_205442-gvqi3cgq`, display_name **`qwen25_math_1_5b_openr1_async_hpt_beta03_constant`**). 검증 완료:
- 정체 = D0(beta03_constant, C1 off·C2 off — cispo/entry/decoupled 흔적 0), beta=0.3 constant.
- 데이터 = **v2 proper-prompting**(롤아웃 raw_prompt에 LUFFY system prompt 존재; step0 MATH-500 40.3 = v2-consistent, v1-strip ~13 아님).
- val = 6-벤치(AIME24/25·AMC23·MATH-500·Minerva·Olympiad), step ~271까지 완주(성숙).
- 최종 val: MATH-500 72.3 / AMC23 50.0 / Olympiad 38.5 / Minerva 26.7 / AIME24 9.2 / AIME25 7.5. clip_ratio 0.57·length 5774(M보다 길이폭발 덜함).

**★캐비엇 — pre-fix 라우팅★**: gvqi3cgq는 hpt_gate 라우팅 위치버그(`rm_scores[-1]`) 수정 **이전** 코드로 돌았다(diff patch에 sum-fix·P0 없음). 비-P0 런에서 이 버그 영향은 **~5%**(측정: M에서 legacy 0.696 vs fixed 0.746 — clean-only-success 그룹을 SFT로 과소라우팅). 따라서 D0는 "약간 비관적"이나 근사 baseline으로 유효. **D0·M(uvbi7wq3)은 둘 다 pre-fix라 그 사이 델타(C1/C2)는 라우팅 세대가 일치**해 내부 정합적이지만, **fixed-routing M′과의 델타는 라우팅 수정이 섞임**(비-P0 ~5%). 상세: `Improvement_RL.md §5.7`.

### 실행 레지스트리 (run pin, project `async-hpt-openr1`, 2026-07-06)

| 코너 | wandb run | display_name | 코드 세대 | 비고 |
|---|---|---|---|---|
| **D0** | `gvqi3cgq` (run-20260704_205442) | `..._beta03_constant` | pre-fix | 위 §D0 픽스 참조(완주 step~271) |
| **M** | `uvbi7wq3` (run-20260705_195604) | `..._M_decoupled_cispo` | pre-fix | **step 175에서 중단(미성숙)** → `M−D0`(C1/C2)는 공통 스텝 ≤159에서 비교. D0와 라우팅 세대 일치 |
| **M′** | `olh2hynl` (run-20260706_173147) | `..._Mprime_v2` | **post-fix (라우팅 sum(-1) + P0-1/2 + entropy0 + beta1.0)** | 재실행(진행 중). `M′` = M + P0 이므로 **2×2 격자 밖의 상위 앵커**(P0 축 추가). `M′−D0`=풀스택, `M′−M`=P0+라우팅수정 |
| ~~M′(폐기)~~ | ~~`uz72mzb9` (run-20260706_051453)~~ | `..._Mprime_v2` | pre-fix 라우팅 | **버그 라우팅으로 순수-SFT 붕괴(success 전 스텝 0). 비교 금지·폐기**(Improvement_RL.md §5.7) |

#### 공정 비교 방법 (cross-generation, 2026-07-06 방향 고정 — 실행은 나중에)

라우팅 위치버그(`rm_scores[-1]`)는 **메트릭만이 아니라 라우팅 결정=학습 레짐 자체**를 바꿨다(gamma=0.0에서 clean-correct 그룹을 SFT로 오배정 → pre-fix D0·M은 초기 과-SFT + 길이보상 레짐에서 훈련됨). 덤프 검증(2026-07-06): pre-fix M의 학습-시점 "success 우상향"은 대부분 **절단율 0→95%와 함께 truncated-correct가 뒤늦게 카운트된 착시**였고(초기 window: raw avg@8=0.13인데 버그-카운트=0.00), post-fix M′은 step1부터 진짜 ~0.10에서 출발. ⇒ **학습-시점 wandb 지표(critic/score·hpt/onpolicy_success_rate·batch 통계)는 라우팅 세대(pre↔post)가 다르면 스텝정렬 비교 무효.**

- **채택 방향**: 재실행 대신 **라우팅-무관 신호로만** 교차비교 — (a) **롤아웃 덤프 raw `acc`의 param_version 정렬 avg@8 궤적**(그룹균등=per-rollout mean, 채점기 직접 산출이라 `rm_scores[-1]`·row-weighting·P0-1 게이팅 전부 무관), (b) `val-core/*/mean@8`(홀드아웃). 두 축이 실제 "생성 능력" 궤적을 준다. 스크립트는 §12.5 substrate 재사용(입력=run별 rollout_dump dir, 버킷=gen_batch `global_steps`=param_version, 지표=mean acc + 절단율).
- **캐비엇**: 이 비교조차 D0·M은 버그 레짐에서 *훈련된* 체크포인트라 baseline이 오염(측정은 공정하나 학습 과정은 아님). 정밀 factorial이 필요하면 D0(+격자) post-fix 재실행이 유일 해법. **M′−M은 5-인자 묶음**(P0-1·P0-2·entropy0·beta1.0·+라우팅수정)이라 단일축 귀속 불가 = "개선 총합 앵커"로만 유효; P0 순효과 분리 시 `M+P0만`(beta·entropy 유지) arm 별도.
- 진단 상세: `Improvement_RL.md §5.7`(라우팅버그) + 본 세션 덤프분석.

## 유지보수

이 문서는 실험 *설계·분석*의 단일 진실 출처다. arm이 추가/변경되면 §3·§4·§9와 함께 **분석 절차 §11·§12**도 갱신한다(지표·수식·판정 규칙이 arm과 어긋나면 안 됨). 수치 판정선(예: "오답 길이 +X%")이 정해지면 §3 각 arm에, `clipfrac_top20entropy/pg_clipfrac`의 "≫1" 임계 등 라우터 판정선이 정해지면 §12.4에 박는다. §12.5의 분석 스크립트가 이 절차의 실행 대응물이다. 결정의 *근거*는 여기서 재서술하지 말고 해당 DR을 인용한다(line 번호 회피, symbol 기준 — `Codemap_RL.md` 관례).
