# Ablation_RL — Async-HPT Ablation Study 설계

_Last updated: 2026-07-10_

> **Paper-use status (2026-07-22).** This file is the run and ablation ledger. Section 14
> remains the canonical main-run record, but public claims, terminology, benchmark values,
> and evidence status are controlled by `papers_RL/Full_Paper_Draft_ko.md`. Earlier
> M/CISPO-first instructions are historical experiment design, not the current paper plan.

Status: 설계 확정(M-앵커 2×2로 재배향, 2026-07-04) · D0 런처 반영·run 완료 · C1 decoupling / C2 CISPO config+loss+routing 구현 완료(런처/run은 별도) · A1은 config-only · 라벨 대응: 구 A5≡M−cispo, 구 A6c≡M(§3) · 2026-07-09 1차 재앵커링(§13): 격자 기준점을 M5로 이설 · **2026-07-10 2차 재앵커링(§14): C2 ablation(M5abl_nocispo)이 main을 탈환 — 신 main = decoupled+vanilla(`oki4kv8u`, 정점 40.17>LUFFY-local 39.58), CISPO는 "outcome 열위+폭풍 유발" 판정으로 격하, "구조적 폭풍 벽" 가설 반증(벽=CISPO 귀인), C1은 w-통계로 무런 폐쇄, 신규 축 H(교사 채널)=RLonly **완료**(조기절단@162 — 교사 채널 기여 후반 +3.4 실증, §14.4). 현행 격자·규율은 §14가 최우선.**
범위: DR-001~005가 내린 **결정 요소를 D0 기준으로 하나씩 뒤집어** 각 결정의 기여를 격리하는 ablation. 결정의 근거·이론은 각 DR 소관이며, 이 문서는 그 결정들을 **실행 가능한 실험**으로 배치한다.
관련 문서: `DR-001-loss-normalization_1.md`(집계) · `DR-002-auxiliary-terms_1.md`(aux) · `DR-003-offpolicy-supervised-branch_1.md`(SFT branch) · `DR-004-offpolicy-rl-branch_1.md`(decoupling) · `DR-005-rl-objective-composition_1.md`(결합 문법) · `Codemap_RL.md`(코드 위치)
관련 코드: `verl/workers/utils/losses.py::ppo_loss` · `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla` · `verl/trainer/ppo/rollout_corr_helper.py` · `verl/experimental/fully_async_policy/hpt_{config,training}.py`
전제: fully-async + HPT, GRPO(Dr.GRPO), `ppo_epochs=1` fit_step당 SGD≈4, partial rollout.

---

## 0. 한 문단 요약

> **★현행 상태(2026-07-10)★**: 이 절(§0~§12)은 설계 당시(M=CISPO 앵커) 기준의 원 설계 기록이다. 이후 두 차례 재앵커링을 거쳐 **현행 격자·main·규율은 §14**(main = nocispo = decoupled+**vanilla**, CISPO는 ablation에서 기각)가 단일 진실이다. 아래 본문은 방법론(leave-one-out·서명 판정·통제 원칙)의 출처로 유효하되, "M을 먼저"류의 실행 지시문은 이행 완료된 역사다.

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

> **격자 밖 개선 레버 (Improvement_RL.md §5.12, M6):** g-슬롯에 `cispo_klcov`(CISPO 본체 + KL-Cov 오버레이 — 고공분산 RL 토큰에만 KL 감쇠)가 구현·등록됨. 이 2×2 격자(C1/C2 귀인)의 arm이 아니라 M-앵커 위 탐색축 개선이며, 현재 **off(loss_mode=cispo)**로 장전만. 전송층 개선 B1'(큐 zero-variance 축출)·B3(`max_completed_prompt_groups` 384→256)도 격자 밖. 상세·게이트·근거는 `Improvement_RL.md` §5.12. **실측 결과(2026-07-09, M7·M5R — §13.4)**: cispo_klcov 이점 미증명(정점 38.33<M5 38.47·엔트로피 플로어 방어 실패), M5R 대조실험이 불안정 원인을 델타가 아닌 스택 구조로 판정 → 이 레버들은 신 격자에서도 계속 제외.

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

> **2026-07-09 재배향 → 2026-07-10 재갱신**: 이 절의 순서·재사용 판단은 구 격자(pre-fix 세대) 기준이다. 1차 재앵커링(M5 기반)은 §13, **현행 격자·규율·레지스트리는 §14**(main=nocispo)가 최우선이다.

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

~~**M이 최우선** — main run이자 헤드라인 델타 `M−D0`를 단독으로 준다(D0 기완료). 귀속 run(M−cispo·M−dec)은 M이 D0와 유의미하게 다를 때만 — 차이가 없으면 격자는 조기 종료되고 그것도 답이다(풀스택 무효 = 두 축 모두 이 레짐에서 조건 미충족). **A1**은 config-only라 컴퓨트 남을 때 언제든. 컴퓨트 절단 순서: A1 → M−dec → M−cispo (**M은 절단 불가**).~~ **← 이행 완료·승계(2026-07-10)**: M(→M5 계보)은 실행됐고, "M−cispo"에 해당하는 arm이 오히려 main을 이겨(§14.1) 우선순위 구조 자체가 갱신됐다. 현행 컴퓨트 배분은 §14.5 판정 대기 원장을 따른다.

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

## 13. 1차 재앵커링 (2026-07-09) — 격자를 M5 기반으로 이설

> **★승계 고지(2026-07-10)★**: 이 절의 격자(앵커=M5)와 실행 규율은 1순위 arm(M5−cispo)의 실측이 **main 자체를 뒤집으면서**(§14) 부분적으로 승계되었다. 특히 §13.3의 "90~100스텝 캡"과 §13.4의 "구조적 벽" 판정은 **반증**되었다(벽의 진범 = CISPO, §14.1). 역사 기록으로 보존하며, 현행 격자·규율·레지스트리는 **§14**를 본다.

### 13.1 왜 이설하는가

구 격자 기준점(M=`uvbi7wq3`, D0=`gvqi3cgq`)은 pre-fix 라우팅 세대다. 이후 개선 캠페인(M′→M2/M3/M4/M5, `Improvement_RL.md` §5.7~5.12)이 main을 **M5(cleanasync, peak lenient6 38.47@50)**로 옮겼고, M5의 동결 레시피는 §2 공통 기반에서 다음 4곳이 이탈했다:

| 축 | §2 구 기반 | M5 (신 기반) |
|---|---|---|
| 라우팅 | pre-fix (`rm_scores[-1]` 버그, 비-P0 ~5%) | post-fix (sum-fix) |
| `norm_adv_by_std_in_grpo` | `False` (Dr.GRPO/UPT 파리티) | **`True`** (M4 도입 성능 레버 — 파리티 이탈) |
| rollout logprob | bf16 | **fp32 head** (L1) |
| `max_completed_prompt_groups` | 768 | **384** (L2) |

따라서 `M5 − 구D0`는 C1/C2에 위 4개가 섞인 **5+인자 묶음**이라 귀속 불가(부록 cross-generation 방법론이 이미 경고한 함정의 재현). **신 격자 = "M5 레시피 동결 + 정확히 한 축 제거"**로 재정의하고, 1차 산출물은 한계 기여 `M5−(M5−X)`다.

### 13.2 신 격자 코너와 기존 run 재활용 판정

| 코너 | 구성 델타 (M5 런처 기준) | 상태 · 우선순위 |
|---|---|---|
| **M5** (§13 당시 앵커 — §14에서 C2 arm으로 격하) | — | **완료 ✅** `f5ugxklh` (step 96 SIGTERM 종료·건강, 50-90 창 확보) |
| **M5−cispo** | `loss_mode=vanilla` + `clip_ratio_low=0.2` | 신규 · **1순위** (C2 한계 기여) |
| **M5−dec** | `rl_old_logprob_source=rollout` + `entry_proximal` 제거 + `rollout_is=null` | 신규 · **2순위** (C1 한계 기여 + 캡 충돌 실측) |
| **D0′** | 위 둘 동시 제거 | 신규 · 옵션 (상호작용항 필요 시만) |
| M5−advstd | `norm_adv_by_std_in_grpo=False` | 후보 (§13.5 미커버 1의 방어 arm, config 1줄) |
| A1 | `loss_agg_mode=seq-mean-token-mean` + β_r 통제값 | 보류 — **DR-001 §4.3 lemma로 β_r 산출이 선행** |

**기존 run 재활용 판정 (2026-07-09 로컬 wandb output.log 실측 검증):**
- **`uvbi7wq3` ≠ D0.** experiment_name = `M_decoupled_cispo` = **C1·C2 양 축 모두 ON인 구 M 앵커**다(git 38019bf1, pre-fix, step 175 미성숙). D0(양 축 off)로 오인 금지 — 신 격자 어느 코너에도 부적합, cross-generation 참고 전용.
- `gvqi3cgq` = `beta03_constant` = 구 D0(양 축 off)가 맞다(git 14d2a78b). 그러나 **격자 코너(D0′) 대용은 불가**, 근거 셋(2026-07-09 재검증):
  (i) **6-인자 오염** — M5와의 차이가 C1/C2에 라우팅세대·advstd·L1·L2를 겹쳐 담음(§13.1).
  (ii) **스텝 의미론 비호환(로그 실측)** — gvqi3cgq의 fit-step은 ~1,994샘플(RL 1178+SFT 816, `required_training_multiple=256`)·~890초/스텝. 부록의 "완주 step~271"은 `training/global_step`(param version) 기준이며 **fit-step으로는 69**다. M5는 128그룹·~175초/스텝 → matched-step 비교 축 자체가 성립하지 않는다. 공정 축은 부록 방법론대로 **param_version 정렬 롤아웃 acc 궤적**뿐.
  (iii) 구세대 val 로깅 포맷 상이 — 콘솔 로그에서 6벤치 lenient 궤적 재추출이 깨짐(신뢰 채널 = wandb UI history 또는 dump 재채점).
  **유효한 역할 = "캠페인 총이득 앵커"**: `M5−D0`를 "C1+C2+라우팅수정+advstd+L1/L2의 **번들** 총이득"으로 정직 표기하고 param_version 축에서 비교하는 용도(부록 cross-generation 방법론이 승인)로는 쓸 수 있다. 단독효과·시너지 분해용 코너로는 ❌.
- 결론: **신 격자에서 재사용 가능한 완료 run은 M5 하나뿐**이고, 나머지 코너는 전부 신규다. 단독효과/상호작용항이 필요하면 D0′(M5 기반 both-off, ~90스텝 ≈ 4.5h 신규 run)가 유일한 경로다.

### 13.3 실행 규율 (M4·M5·M7·M5R 실측 반영)

- ~~**90~100스텝 캡.** 이 스택은 엔트로피 플로어(~0.15)×고활용(score 0.65+)×~100스텝 부근에 **구조적 폭풍 벽**을 가진다(4런 실증: M4 80-88 · M5 81-84 · M7 100-130 · M5R 104-115, §13.4). 비교 유효 창 = **matched-step 50~90**. arm당 ~4.5h로 충분하며 250스텝은 낭비이자 붕괴 위험.~~ **← 정정(2026-07-10, §14.1): "구조적" 벽이 아니었다. 벽은 CISPO(g-슬롯) 귀인 — vanilla arm(nocispo)이 같은 스택으로 190스텝 무폭풍 통과. vanilla 계열 arm은 캡 불필요(200스텝 표준), CISPO 계열만 ~100 벽 위험.**
- **판정 노이즈 규율.** naive-6 mean@8의 단일 재실행 노이즈 ≈ ±0.7pt → 단일-스텝 val 차는 무증거. **50-90 창 평균**(≈±0.25)으로 비교하고, 폭풍 창의 val은 제외(M5 val@80=0.36 전례). 서명이 1차, 성능은 보조(§8 유지).
- **사전 등록.** arm 착수 전 M5 로그·train dump에서 ① `P(w>C_w)`(w 포화율) ② `pg_clipfrac_top20entropy/pg_clipfrac`×staleness 상관 ③ low-prob death 밀도를 산출하고, §12.4 라우터로 각 arm의 **기대 결과를 본 문서에 먼저 박는다**(도출-확인 원칙 §8의 실행형).

### 13.4 격자 밖 arm 결과 원장 (2026-07-09 확정)

- **M7** (`4wl3f5do`, M5+B0/B1'/B2/B3 4델타 동시): `cispo_klcov` **이점 미증명** — 정점 38.33 < M5 38.47, matched-step 우위(+0.34@90)는 노이즈 이내, 핵심 기전(엔트로피 플로어 방어) 실패(M7 0.10-0.13 < M5 0.14-0.33, 선택 토큰 배치당 수 개). 폭풍 4회 후 만성 진동(재점화 3회, ESS 최저 0.079)으로 중단.
- **M5R** (`2hz6tp01`, M5 무변경 resume@95 — **사실상의 대조실험**): 완전 복원 확인(초기 val 37.54=앵커 일치) 후 **step 104에서 동일 붕괴** → 불안정의 원인은 M7의 델타가 아니라 ~~스택+영역의 구조적 벽~~으로 판정. **← 정정(2026-07-10, §14.1): 이 판정은 반증됨.** M5R이 배제한 것은 "M7의 4델타"까지였고, M4·M5·M5R·M7의 진짜 공통분모는 스택이 아니라 **CISPO(전 런이 cispo 계열)**였다. C2 제거 arm이 벽을 통과하며 귀인이 확정됐다.
- (비교축) `paper_hpt_sync_*` (`v96fvd0p` 등): 동기 HPT 재현 베이스라인 — 이 격자와 무관한 논문 비교축이며 arm이 아니다. **실측 보강(2026-07-10, wandb 바이너리 추출)**: 정점 naive 29.49 / weighted 39.97 @global50, step 90-100에서 붕괴(29.5→21.9), 학습 score 정점 0.43에 불과. **비교 무효 사유 둘**: (i) as-run 8×손실 스케일(유효 clip 10 — 논문 레시피 clip 1.0보다 훨씬 느슨한 보호, 후반 붕괴와 정합 가능), (ii) **채점기 불일치** — 이 런의 val은 boxing-필수 `entropy_math`(하방편향)라 우리 lenient math_verify 척도가 아니다(`paper-hpt-sync-baseline-run` 원장). → **어느 방향으로든 "압도" 주장의 baseline 사용 금지**(신뢰 baseline은 LUFFY-local 39.58/51.21). 공정 비교는 동일 채점기 재채점(math_verify, ≤8-proc — 32-proc은 OOM 실증) 후에만.

### 13.5 충분성 경계 — 이 격자로 주장할 수 있는 것과 없는 것

**커버**: C1/C2 각각의 한계 기여(신규 2 arm, §9 매트릭스의 DR-004/005 검증) · A1 집계층(DR-001, β_r 선행 후) · DR-002/003은 상시 기반/제외 원장 그대로.

**미커버와 방어선**:
1. **advstd**: 신 기반의 `norm_adv_by_std=True`는 §2(b) 파리티(False)에서의 이탈인데, M4에서 다른 변경과 묶여 도입되어 단독 격리 실측이 없다. 리뷰 방어가 필요하면 **M5−advstd**(config 1줄)를 추가한다 — D0′보다 우선순위 높음.
2. **L1/L2 (시스템 델타)**: 격자 헌장(DR 결정 검증) 밖. M4→M5 2-델타 묶음 기록(`Improvement_RL.md`)으로만 서술하고 단독 기여를 주장하지 않는다.
3. **상호작용항**: D0′ 없이는 C1×C2 시너지를 주장할 수 없다 — "각 축의 한계 기여"까지만.
4. **통계력**: arm당 단일 run이므로 50-90 창 평균으로도 0.5pt 미만의 성능 델타는 판독 불가 — 그 경우 판정은 전적으로 §3 서명·§12.4 라우터로 내린다.

---

## 14. 2차 재앵커링 (2026-07-10) — C2 ablation이 main을 탈환: 신 main = decoupled + vanilla

### 14.1 사건 — 1순위 arm(M5−cispo)이 앵커를 이겼고, 폭풍 벽의 진범을 밝혔다

§13의 1순위 arm **M5abl_nocispo**(`oki4kv8u`, M5에서 `loss_mode=vanilla`+`clip_ratio_low=0.2`만 변경) 190스텝 실측:

| 지표 (lenient6) | nocispo | M5 (구 anchor) | LUFFY-local |
|---|---|---|---|
| 정점 naive / weighted | **40.17 / 52.06** @170 | 38.47 / 50.97 @50 | 39.58 / 51.21 |
| 150-190 창평균 | **39.22 / 51.34** | (M5는 96에서 종료) | — |
| val@190 (절단점) | 39.70 / 51.30 | — | — |
| 폭풍 | **0회** (KL 최대 7.0@20 1회 자가소화, step95+ KL≤2.2·ESS≥0.86) | 1회(81-84, KL 23.1) | — |

- **성능**: M5를 창평균 기준으로도 완전 돌파(+0.7), LUFFY-local을 naive 2개 지점(170·190)에서 초과 + weighted 창평균 +0.13. 정점 170의 벤치 구성은 대형벤치 주도(MATH 78.5·Olympiad 43.6·Minerva 31.9)라 소형벤치 요행 아님.
- **벽 귀인 확정**: M4(80-88)·M5(81-84)·M7(100-130)·M5R(104-115) 폭풍 4런의 공통분모는 스택이 아니라 **전부 CISPO 계열**이었다. C2만 제거한 이 arm이 같은 스택으로 벽 영역을 무폭풍 통과 → §13.4의 "구조적 벽" 판정 반증. **기전**: CISPO의 `sg(clip(r))·A·logπ`는 min-clip **비관성(브레이크)이 없어** 과확신 토큰이 계속 밀림 → 엔트로피 붕괴 → 길이폭발 → KL 폭풍. vanilla는 `pg_clipfrac` 0.5~9%로 브레이크가 실작동(M5 clipfrac≈0 실측과 대조).
- **판정 절차 기록(§8 규율의 실행 사례)**: main 승격은 사전등록 3규칙(① 창평균 우위 ② 정점 우위 ③ 안정성 비열위) 전부 충족 후에만 내렸다. 초기(step 20-30)의 +3.6~+5.8 우위는 M5 골짜기가 만든 신기루로 **기각**했고(당시 판정 "보류"), 폭풍 벽 통과 + 150-190 고원 확립 후에야 승격했다.

### 14.2 최종 격자 (main = nocispo) — 현행 단일 진실

| | C2=vanilla | C2=CISPO |
|---|---|---|
| **C1=decoupled** | **main = nocispo** (`oki4kv8u`) ✅ | **M5** (`f5ugxklh`) ✅ = main+cispo |
| **C1=coupled** | D0′ — **취소**(아래) | M5abl_nodec — 대각선(상호작용), 런처 장전·후순위 |

- **C2 축 (CISPO) = 완료.** `main − M5` 실측: CISPO는 outcome 열위(정점 −1.7) + 폭풍 유발. DR-005의 CISPO 도출은 이 레짐에서 **실증 반증** — "도출 → ablation 확인 → 수정"(§8)의 정당한 종결이며 DR-005에 결과 노트를 남긴다.
- **C1 축 (decoupling) = 무런 폐쇄.** main 실측 w-포화율 `P(w>C_w=2)` 중앙값 **0.10%**·평온기 0.085%·최대 1.65%, w̄=0.954 → §3 사전등록 규칙("`P(w>C_w)`≈0이면 이 레짐의 낡음이 낮아 coupled≈decoupled — 그 자체가 결과", DR-004 §6)에 따라 **준-불활성 판정**. 보조 증거로 구 D0(`gvqi3cgq`)를 세대 각주와 함께 참고(§13.2). **D0′ 신규 런 취소** — 런처(`run_..._D0prime.sh`)는 장전 상태로 보존.
- **신규 축 H (교사 채널) = RLonly** (`qzsnwc08`, **완료** — 조기절단@162, 판정 §14.4) — 이 문서 최초의 DR-격자 밖 **논문-핵심-가설 축**. 결과: 교사 채널 기여 실증(후반 창 +3.4).
- **총이득 앵커** = `gvqi3cgq` (§13.2 승인 그대로: 번들 표기 + param_version 축).
- 후순위 잔여: M5abl_nodec(대각선), M5−advstd(파리티 이탈 방어), A1(β_r 산출 선행).

### 14.3 실행 규율 개정 (§13.3 승계)

- **스텝 캡 개정**: ~~90-100 캡~~ → **vanilla 계열 arm은 200스텝 표준**(무폭풍 실증). CISPO 계열 arm만 ~100 벽 위험(그 자체가 관측 대상일 때만 연장).
- **비교 창 확장**: main이 190까지 깨끗하므로 matched-step 50~190 전체가 유효 창. 노이즈 규율(단일 val ±0.7 무증거, 창평균 ±0.25, 폭풍 창 제외)은 §13.3 그대로.
- **정점 아티팩트 원장**: nocispo `global_step_170`(40.17/52.06)·`global_step_190`(39.70, 절단점) — **삭제 금지**(M5 step-50 소실 사고의 재발 방지).
- **재개 장전**: `run_..._M5abl_nocispo_cont.sh` (resume@`RESUME_FROM_STEP`, 기본 180 — **190 사용 시 명시 필요**, steps 300).

### 14.4 H축 — RLonly (교사 채널 격리 = 논문 핵심 가설의 직접 검증)

- **목적**: `main − RLonly` = 하이브리드 교사 채널(미해결 프롬프트 → LUFFY 시연 SFT)의 순기여. 논문의 sync GRPO 수치는 프로토콜(관대 grader·k 불일치)+동기성 교란이 겹쳐 **참고 행만 가능** — 하중은 이 arm이 받는다.
- **메커니즘 (v2 센티널)**: `async_hpt.success_threshold: 0.0 → -1.0` 한 줄. 게이트의 성공 판정이 `score > threshold`라 모든 스코어(0/1)가 성공 → `p_success≡1 > γ=0` → SFT 라우팅 절대 불발, k=0 그룹은 advantage 0의 RL 행으로 잔류(순정 GRPO 의미론 = 처치 그 자체). **HPT를 켜둔 채 라우팅만 봉인**하므로 앵커(entry)·token-IS·큐·스케줄러가 main과 100% 동일 = 진짜 단일축.
- **v1 실패 원장 (재발 방지)**: `async_hpt.enabled=False`로 끄면 비-HPT 경로가 old-logprob 재계산 시 version-1 save/restore(`actor_save_model_to_cpu`)로 진입하는데 이 장치가 **fsdp2 비호환**("No DTensor-type parameters" 즉사) — HPT 런들은 entry/rollout 앵커로 전부 우회해 와서 미노출이던 잠복 버그. 상세와 회피법: `Debug_RL.md`.
- **기동 검증(2026-07-10 통과)**: step0 val **17.03**(base 정합) · `hpt/num_sft≡0`·`offline_data_ratio≡0`·`num_rl_groups=384`(봉인 작동) · step19 32.57 / step39 34.63(정상 상승). ★판독 주의: `hpt/onpolicy_success_rate≡1.0`은 센티널의 집계 왜곡(무해) — 진짜 학습 성공률은 `critic/score/mean`으로 읽는다.
- **결과 (2026-07-10, 조기절단@162 — 하락 추세 확인 후 사용자 절단, as-is 사용 결정)**: 정점 **37.73@30** 후 상승이 끊기고 34.7~37.6 진동 플래토, 후반 불안정 심화(33.69@130·30.89@155 딥). **H축 판정**: 초반(20-50 창)은 main과 동등(37.0 vs 36.6 — 교사 채널 없이도 초기 학습은 됨)이나, **후반(130-160 창) 35.17 vs main 38.60 = 교사 채널 기여 +3.4**, 정점 기준 +2.4(40.17 vs 37.73). 즉 하이브리드 교사 채널의 값어치는 **후반 지속-상승 동력과 안정성**에 있다 — main의 150-190 고원·계단식 상승이 RLonly에는 없다. 논문 핵심 가설(H가 순수 async GRPO 대비 실질 기여) **실증**.

### 14.5 판정 대기 원장

1. ~~RLonly 완주(200) → H축 판정~~ **완료(2026-07-10)**: 조기절단@162로 판정 성립 — 교사 채널 기여 +3.4(후반 창)·+2.4(정점), §14.4 결과 참조. "async-HPT 핵심 기여 실증" 분기 확정.
2. **protocol-fair fixed-checkpoint 재평가**: {nocispo@170, @190, LUFFY}를 동일 grader·decoding·budget 아래 **문항당 32 stochastic generations의 mean@32 + 문항단위 paired hierarchical bootstrap 10,000회 95% CI**로 평가한다. 전 모델에 같은 재현 가능 evaluation-seed set을 쓰되, 이는 독립 training seed가 아니다. `mean@32`는 average pass@1 추정이며 pass@32가 아니다. "LUFFY 상회/동급"의 지면 확정은 이것으로만 한다. 논문 41.9와의 직접 비교는 금지(grader 관대 +1~4·k 불일치, `upt-comparison-validity` 원장).
3. **nocispo_cont(190→300)**: 150-190 고원이 완만 상승 중(+0.015/step)이라 잔여 상승 확인.
4. paper-HPT sync(`v96fvd0p`) 공정화 — ① 8×-scale 유효성 판별, ② **동일 채점기 재채점**(그 런의 val은 boxing-필수 entropy_math로 하방편향, §13.4). 둘 다 끝나기 전 "sync 압도(+10)" 주장 금지 — 현재 +10 격차에는 채점기 차이가 섞여 있다.

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

**신세대 (2026-07-09 추가 — §13 재앵커링 이후 기준):**

| 역할 | wandb run | display_name | 코드 세대 | 비고 |
|---|---|---|---|---|
| **★신 MAIN (§14)★** | `oki4kv8u` (run-20260709_232348) | `..._M5abl_nocispo` | M5 − C2 (decoupled+vanilla) | **정점 40.17/52.06@170 > LUFFY-local, 190스텝 무폭풍, step190 절단(재개 장전)**. 아티팩트: global_step_170·190 삭제 금지 |
| M5 = 구 앵커 → C2 arm | `f5ugxklh` (run-20260708_195536) | `..._M5_cleanasync` | post-fix + advstd + L1/L2 | §13 앵커였으나 §14에서 격하: **main+cispo** = C2 축의 처치군. peak 38.47@50, 폭풍 1회(81-84) |
| **H축 arm (완료)** | `qzsnwc08` (run-20260710_065617) | `..._RLonly_grpo` | main + success_threshold=-1.0 | §14.4 — 교사 채널 봉인(순수 async GRPO). **정점 37.73@30, 조기절단@162(하락 추세). 판정: 교사 채널 기여 +3.4(후반 창 130-160)·+2.4(정점)** |
| ~~RLonly v1(폐기)~~ | ~~`m3hdp3jm` (run-20260710_064153)~~ | ~~`..._RLonly_grpo`~~ | async_hpt.enabled=False | **기동 즉사** — fsdp2×save_model_to_cpu 잠복 버그(§14.4 v1 원장, Debug_RL.md). 비교 금지 |
| M7 (격자 밖) | `4wl3f5do` (run-20260709_000511) | `..._M7_fullstack_klcov` | M5 + B0/B1'/B2/B3 | §13.4 — cispo_klcov 미증명·만성진동 중단(step ~134) |
| M5R (격자 밖, 대조) | `2hz6tp01` (run-20260709_055522) | `..._M5R_resume95` | M5 무변경 resume@95 | §13.4 — step 104 동일 붕괴. ~~구조적 벽 입증~~ → **§14.1 정정: 벽=CISPO 귀인의 대조군** |
| (비교축) paper-HPT sync | `v96fvd0p` (run-20260709_204429) | `paper_hpt_sync_..._beta03` | 동기 HPT 재현 | 격자 무관. **실측: 정점 29.49/39.97@50, step90-100 붕괴 — 8×-scale + 채점기 불일치(entropy_math 하방편향) 2중으로 비교 무효, "압도" baseline 사용 금지**(선행 시도 `b1m6nke3`/`z7pmabz0`/`9jeawsty`) |

#### 공정 비교 방법 (cross-generation, 2026-07-06 방향 고정 — 실행은 나중에)

라우팅 위치버그(`rm_scores[-1]`)는 **메트릭만이 아니라 라우팅 결정=학습 레짐 자체**를 바꿨다(gamma=0.0에서 clean-correct 그룹을 SFT로 오배정 → pre-fix D0·M은 초기 과-SFT + 길이보상 레짐에서 훈련됨). 덤프 검증(2026-07-06): pre-fix M의 학습-시점 "success 우상향"은 대부분 **절단율 0→95%와 함께 truncated-correct가 뒤늦게 카운트된 착시**였고(초기 window: raw avg@8=0.13인데 버그-카운트=0.00), post-fix M′은 step1부터 진짜 ~0.10에서 출발. ⇒ **학습-시점 wandb 지표(critic/score·hpt/onpolicy_success_rate·batch 통계)는 라우팅 세대(pre↔post)가 다르면 스텝정렬 비교 무효.**

- **채택 방향**: 재실행 대신 **라우팅-무관 신호로만** 교차비교 — (a) **롤아웃 덤프 raw `acc`의 param_version 정렬 avg@8 궤적**(그룹균등=per-rollout mean, 채점기 직접 산출이라 `rm_scores[-1]`·row-weighting·P0-1 게이팅 전부 무관), (b) `val-core/*/mean@8`(홀드아웃). 두 축이 실제 "생성 능력" 궤적을 준다. 스크립트는 §12.5 substrate 재사용(입력=run별 rollout_dump dir, 버킷=gen_batch `global_steps`=param_version, 지표=mean acc + 절단율).
- **캐비엇**: 이 비교조차 D0·M은 버그 레짐에서 *훈련된* 체크포인트라 baseline이 오염(측정은 공정하나 학습 과정은 아님). 정밀 factorial이 필요하면 D0(+격자) post-fix 재실행이 유일 해법. **M′−M은 5-인자 묶음**(P0-1·P0-2·entropy0·beta1.0·+라우팅수정)이라 단일축 귀속 불가 = "개선 총합 앵커"로만 유효; P0 순효과 분리 시 `M+P0만`(beta·entropy 유지) arm 별도.
- 진단 상세: `Improvement_RL.md §5.7`(라우팅버그) + 본 세션 덤프분석.

#### Val 채점 정밀화 계획 (2026-07-06 방향 고정 — **하네스 실행은 학습 종료 후**)

**검증된 사실(2026-07-06):** (1) 보고 지표 `val-core/*/acc/*`는 **ungated 원시 채점**이다 — P0-1은 `reward`만 게이트하고 `reward_extra_info["acc"]`는 불변(`dapo.py::run_single`), val 집계는 이 `acc`를 씀(`ray_trainer._validate`); 산수로도 확증(M′ step10 MATH-500 acc 55.6 > 1−절단 52.4 → truncated-correct가 카운트됨 = 게이팅 안 됨). 따라서 **P0-1이 eval을 누른다는 우려는 사실무근.** (2) 채점기(`math_verify_adapter`→`math_verify`)는 bare-expr 추출(boxed 불요)·후보중 최고 recall·과대채점 없음.

**유일한 eval 채점 결함 = 타임아웃(FN).** `future.result(timeout=30)`이 큐대기+실행 합산이라 4-worker pool 포화 시 정답도 0 처리(`math_verify.py`). 이는 parity 무관·FP無인 **순수 correctness 버그**. no-box는 대부분 절단(정당한 0), 다중답·객관식(FN)은 eval 벤치에 드묾. 로직 완화(③④ 무조건 크레딧)는 FP 유발→parity 깸이라 **금지**; 개별 검증된 정답만 인정.

**계획(학습 종료 후 실행):** 최적/최종 체크포인트에 대해 **val-only 하드닝 재평가**.
- 체크포인트는 FSDP-sharded → `verl.model_merger`로 HF merge 필요.
- **손수 vLLM 스크립트 금지, verl 자체 val 경로 재사용**(config parity 자동 보장: v2 system-prompt 주입·temp/top_p·8192·mean@8). `validation_data_dir` 켜서 전체 생성물 덤프 + 채점 timeout↑(예 120s)·부하 격리.
- 덤프 위에서 **FN/FP 양방향 감사**로 "무오류 채점" 증명 → 논문 표는 이 숫자.
- **신뢰 규율**: 최종 적용 전 기존 `Mprime_v2/global_step_10` 체크포인트로 하네스를 돌려 wandb val-core(step10) 재현 확인(±노이즈) → config parity 검증 후에만 신뢰.
- 병행: 향후 런처에 `validation_data_dir`+하드닝 timeout 기본 배선(그러면 동일-출력 재채점으로 grader FN 순수 분리 측정도 가능).

지금은 **미실행**(현재 M′ 진행 중, 급하지 않음). 도구 존재 확인: `verl/model_merger`, `verl/trainer/main_eval.py`.

## 유지보수

이 문서는 실험 *설계·분석*의 단일 진실 출처다. arm이 추가/변경되면 §3·§4·§9와 함께 **분석 절차 §11·§12**도 갱신한다(지표·수식·판정 규칙이 arm과 어긋나면 안 됨). 수치 판정선(예: "오답 길이 +X%")이 정해지면 §3 각 arm에, `clipfrac_top20entropy/pg_clipfrac`의 "≫1" 임계 등 라우터 판정선이 정해지면 §12.4에 박는다. §12.5의 분석 스크립트가 이 절차의 실행 대응물이다. 결정의 *근거*는 여기서 재서술하지 말고 해당 DR을 인용한다(line 번호 회피, symbol 기준 — `Codemap_RL.md` 관례).
