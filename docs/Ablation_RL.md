# Ablation_RL — Async-HPT Ablation Study 설계

Status: 설계 확정 · D0(앵커) 런처 반영됨(`main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_main.sh`) · A1은 config-only, A5/A6c는 미구현
범위: DR-001~005가 내린 **결정 요소를 D0 기준으로 하나씩 뒤집어** 각 결정의 기여를 격리하는 ablation. 결정의 근거·이론은 각 DR 소관이며, 이 문서는 그 결정들을 **실행 가능한 실험**으로 배치한다.
관련 문서: `DR-001-loss-normalization_1.md`(집계) · `DR-002-auxiliary-terms_1.md`(aux) · `DR-003-offpolicy-supervised-branch_1.md`(SFT branch) · `DR-004-offpolicy-rl-branch_1.md`(decoupling) · `DR-005-rl-objective-composition_1.md`(결합 문법) · `Codemap_RL.md`(코드 위치)
관련 코드: `verl/workers/utils/losses.py::ppo_loss` · `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla` · `verl/trainer/ppo/rollout_corr_helper.py` · `verl/experimental/fully_async_policy/hpt_{config,training}.py`
전제: fully-async + HPT, GRPO(Dr.GRPO), `ppo_epochs=1` fit_step당 SGD≈4, partial rollout.

---

## 0. 한 문단 요약

이 ablation은 평평한 격자(full grid)가 아니라 **D0(확정된 DR 결정 전부를 적용한 한 점)를 앵커로 두고 DR 요소를 하나씩 뒤집는 단일-요인(leave-one-out) 설계**다. 확정·구현된 DR(001/002/003)은 결정이 이미 켜져 있으니 끄는 방향, 미구현 DR(004/005)은 기본이 꺼져 있으니 켜는 방향 — 둘 다 같은 D0 기준의 한 축 flip이다. 최소 셋만 돌린다: **A1**(집계 mode: sum-norm→token-mean, DR-001), **A5**(staleness: coupled→decoupled, DR-004), **A6c**(g-슬롯: A5 위에서 vanilla clip→CISPO, DR-005). 이 셋은 우연이 아니라 DR-005 통일 추정량의 세 층(집계 / w-슬롯 / g-슬롯)에 정확히 대응한다. A6c는 CISPO의 캡 충돌(DR-005 §4.2c) 때문에 D0가 아니라 **A5 위에** 얹으므로 실행은 `D0→A5→A6c` 사슬 + `D0→A1` 독립이 된다. DR-003의 SFT 벨트(A4)는 A5와 중복(DR-003 §4)이라 제외한다. 각 arm의 판정은 최종 성능이 아니라 **DR가 예측한 관측 서명**(A1=정답/오답 길이 갈라짐, A5=clip-frac 분리+stale 토큰 부활, A6c=death 밀도 소멸)으로 한다. 어느 결과가 나와도 도출을 확인하는 위치에 둔다.

---

## 1. 설계 원리

DR들이 공유하는 두 규율을 그대로 계승한다.

- **measure-first / 입증 책임은 기계를 더하는 쪽에** (DR-003 §5, DR-004 §11). 그래서 A5는 착수 전 사전점검, A6c는 A5의 관측을 전제로 한다.
- **단일-요인 격리.** 모든 arm은 D0에서 **정확히 한 인자**만 벌어져야 한다. 딸려 흔들리는 축은 통제 손잡이로 고정한다(A1의 β_r, A5의 k_max·cliprange·γ).

ablation 성격은 둘로 갈린다.
- **도출-확인형** (DR-001/002/003, 구현됨): branch-blind 덕에 config 한 줄. 게이트 없이 유효. 역할은 "도출의 각 고리를 검증".
- **게이트된 탐색형** (DR-004/005, 미구현): 구현 + 사전 관측이 선행 조건.

이 문서의 최소 셋은 각 성격에서 하나씩 뽑되(A1=확인형, A5=탐색형, A6c=A5의 후속), DR-005 추정량 `∇Ĵ = E[w̄·g(r)·A·∇logπ]`의 세 층을 한 번씩 건드리도록 구성했다.

---

## 2. D0 — 인자별 기본 구성 (앵커)

D0 = "확정된 DR 결정 전부 + 미구현 DR은 기본(off)". 층별 값과 근거:

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
| `async_hpt.rl_old_logprob_source` | **`rollout`** (융합) | 미구현. `ratio=π_current/π_rollout` 하나가 보정+clip 겸함 |
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
| `actor.ppo_epochs=1` | A6c 정보성 논증이 이 값에 의존(DR-005 §4.1: anchor B + ppo_epochs=1이라 r 표류↓ → death 표면↓) |
| `async_hpt.rl_old_logprob_source=rollout` | coupled anchor 고정. entry는 A5에서만 켠다. entry 추가 후에도 D0가 명시로 남게 함 |
| `async_hpt.k_max=null` | RL 학습-시점 staleness 드롭을 **의도적으로 끔**. 낡음은 예산 레벨(`staleness_threshold`)로만 제어하고 A5에서 절단 IS(C_w)로 보정. A5 arm에서도 null 고정 |

> `k_max=null`의 함의: D0/A5는 낡은 RL row를 학습 시점에 버리지 않으므로, A5의 w 분포 꼬리가 k_max 컷 없이 넓게 나올 수 있다. 이는 결함이 아니라 "낡음을 드롭이 아니라 절단 IS로 다룬다"는 decoupling의 정합적 형태다(DR-005 §7-1의 C_w×k_max 공동 조율을 "C_w 단독"으로 단순화).

---

## 3. Ablation arms

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

### A5 · staleness `coupled(rollout) → decoupled(entry, TIS w)` (DR-004, 구현 필요)

**바꾸는 인자 (한 세트).** `rl_old_logprob_source: rollout→entry` · anchor=**옵션 B**(매 fit_step 현재 가중치로 old_log_prob forward, 가중치 스왑 없음) · `rollout_is` ON(C_w=2.0 절단) · `rollout_rs` OFF.
**격리.** staleness 보정을 clip에서 분리. clip은 `r=π_current/π_entry`(이번 이동분)에만, 낡음 `w=π_entry/π_rollout`은 절단 IS로 곱셈.
**착수 전 사전점검 (거의 공짜).** D0 로그에서 `pg_clipfrac`이 staleness(`param_version − max_generation_step`)와 상관 있는지 확인(DR-004 §11). 상관 없으면 배치에 낡은 토큰이 없다는 뜻 → A5는 구성상 D0와 같아져(`w≈1`) run이 무의미. 이건 게이트-스켈레톤이 아니라 **arm 정보성 사전 확인**.
**필수 통제.** k_max(null)·cliprange·γ를 D0와 동일. **주의: D0는 "무보정"이 아니라 "융합 clip + 장치② dormant(IS≡1)"** — `A5−D0`는 "보정 추가"가 아니라 "융합을 분리".

**관측해야 할 차이 — "clip-frac 분리 + stale 토큰 부활".**
| 지표 | D0 (융합) | A5 (분리) 기대 | 의미 |
|---|---|---|---|
| clip-frac | 높음(낡음+이동 뒤섞임) | **r-clip-frac ≪ D0** | staleness 성분이 clip→w로 이동 |
| w 포화율 `P(w>C_w)` | (없음) | **0보다 유의미** | 절단되는 낡음 질량. ≈0이면 A5≈D0 |
| 동결됐던 stale 토큰 | 얼어붙음 | **gradient 흐름 재개** | "낡았지만 이번엔 안 과격한" 토큰 복귀 |
| RL:SFT 유효비 | (기준) | **D0와 동일해야** | rejection OFF라 RL row 안 버림 → 조성 불변 |
| step-time | (기준) | **+≈actor forward 1/3** | entry forward 비용(이득 0이어도 내는 고정비) |

**해석.** r-clip-frac↓ + stale 토큰 부활 + 성능↑ → 이 레짐에서 decoupling 이득. `P(w>C_w)≈0`이면 "이 run의 낡음이 낮음"(사전점검이 미리 거름). 성능↓이면 절단(편향) IS < 얼리기의 정직한 음수(GRPO라 stale-advantage 위협은 완화, DR-004 §6).

### A6c · g-슬롯 `A5 + vanilla clip → CISPO-sg` (DR-005, A5의 후속)

**바꾸는 인자.** A5는 유지(w-슬롯=TIS w), g-슬롯만 `min-clip → g(r)=sg(min(r, 1+ε_h^IS))`.
**왜 A5 위에?** coupled(D0)에서 CISPO를 켜면 staleness 캡(C_w)과 이동 캡(ε_h^IS)이 하나의 ρ 위에서 충돌(DR-005 §4.2c). decoupled 위라야 두 캡이 분리된다. → A6c는 독립 flip이 아니라 A5의 phase-2.
**필수 통제.** ε_h^IS는 clip의 0.28을 **이식 금지**, 신규 탐색. min()의 비관성(악화 방향에서만 gradient) 상실을 감시.

**관측해야 할 차이 — 판정은 `A6c − A5`.**
| 지표 | A5 (vanilla clip) | A6c (CISPO) 기대 | 의미 |
|---|---|---|---|
| death 밀도(clip된 low-prob 피벗 비율, A 부호별) | 일부 피벗 gradient=0 | **≈0** | 죽던 피벗 토큰이 캡 값으로 복귀 |
| **A5의 death 밀도가 애초에 >0였나** | — | **이게 전제** | 0이면 살릴 게 없어 A6c≈A5 |
| entropy 추이 | (기준) | 탐색 특성 변해 이동 가능 | 지표로만 추적, 선제 조정 금지 |
| grad-norm / 발산 | 안정 | **A>0·r 폭주 구간 spike 감시** | 비관성 상실 → PPO면 멈출 곳을 계속 밀어붙임 |
| `ppo_kl`류 모니터링 | 기존 의미 | **의미 바뀜** | 추정량 클래스가 REINFORCE-with-coefficient로 변경 |

**해석.** A5에서 피벗-death가 실재했고 CISPO가 되살려 성능↑ → g-슬롯 교체 정당. **A5 death 밀도≈0이면 A6c≈A5** = "CISPO가 나쁜 게 아니라 이 레짐이 gradient-死를 안 만듦"(DR-005 §4.1 예측의 검증). grad-norm spike/발산 → vanilla clip의 비관성이 안전장치로 일하고 있었던 것.

---

## 4. arm 구성과 실행 사슬

```
D0 (앵커, 현행 1.5B 런처)
 ├─ A1  = D0 + loss_agg_mode=seq-mean-token-mean  (+β_r로 SFT예산 통제)
 │        → (A1−D0) = Dr.GRPO 인센티브 편향                      [config-only]
 └─ A5  = D0 + decoupled(anchor B, TIS w C_w=2.0, rej OFF, k_max=null)
          → (A5−D0) = 순수 decoupling 효과                        [구현: DR-004 §9]
      └─ A6c = A5 + CISPO-sg g-slot
               → (A6c−A5) = gradient-死 회복(g-슬롯 교체)          [A5 위 소규모 추가]
```

총 **4 run** (D0, A1, A5, A6c). 실질 구현 비용은 A5(entry-forward anchor B) 하나, A6c는 그 위 CISPO 계수, A1은 config 한 줄.

---

## 5. 공통 통제 & 지표

**arm 간 고정 (통제 인자):** `k_max`(null)·cliprange(축 아닐 때)·`ppo_epochs`(1)·RL:SFT 목표비(γ=0.0)·lr(5e-6)·grad-clip(80)·모델·데이터.

**모든 run에서 항상 보고:** RL:SFT 유효 row/token 비(post-mask) — 없으면 "안정성 개선"이 진짜인지 배치가 조용히 SFT-지배가 된 건지 구분 불가(DR-004 §10). w 분포·포화율, r-clip-frac vs 융합 clip-frac(같은 이름의 다른 지표라 대시보드 구분 표기, DR-005 §8), `max_partial_span`, step-time.

**교란 통제 체크리스트:**
1. arm A(=D0)는 "no correction"이 아니라 "융합 + IS dormant" — A5−D0를 "보정 유무"로 오독 금지.
2. A1은 β_r로 SFT 예산 고정 안 하면 단일-요인 아님.
3. 하이퍼 이식 금지: gate 인자가 ρ→r로 바뀌면 ε 의미가 전부 바뀐다.
4. `C_w × k_max` 공동 조율(D0는 k_max=null이라 C_w 단독).

---

## 6. A4 (SFT belt) 제외 근거

DR-003의 belt(SFT self-detach→entry-snapshot+clip)는 이 최소 셋에서 뺀다. A5와 겉만 닮았지(둘 다 "π_entry+clip") **다른 branch·다른 provenance**다: A4는 SFT row(생성 정책 없음 → IS 공집합, 순수 최적화 pacing), A5는 RL row(rollout provenance 있음 → 진짜 off-policy 보정). 더 결정적으로, **SFT가 일으킨 drift는 rollout provenance를 가진 RL row 쪽(=A5의 w)에서 이미 보정**되므로 SFT에 또 벨트를 거는 것은 RL 처리와 중복(DR-003 §4). belt는 사전 계측에서 SFT-induced drift가 확인될 때만 여는 별도 ablation으로 남긴다(DR-003 §7).

---

## 7. 우선순위와 컴퓨트

**A1(재현 성격, 가장 쌈)** 은 config-only라 언제든. **A5**가 이 논문 고유 질문의 본체 — 착수 전 D0 사전점검(§3 A5) 필수. **A6c**는 A5에서 death 밀도가 실측될 때만 정보성이 살아나므로 A5 결과에 조건부. 컴퓨트 유한 시 절단 순서는 A6c → (A1 or A5)이며, A6c는 A5 관측 없이는 돌리지 않는다.

---

## 8. 서술 원칙

ablation은 **선택을 낳은 것이 아니라 도출을 확인하는 위치**에 둔다(DR-001 §8, DR-003 §7). "여러 mode 시도해 최선을 골랐다"(경험적 튜닝) ❌ → "테제에서 도출 → ablation이 각 고리를 검증"(도출 후 확인) ⭕. A5/A6c는 "놓친 안전장치"가 아니라 "가설의 처치군"으로 프레이밍 → 어느 결과가 나와도 방법이 안 무너진다.

---

## 9. DR 정합 매트릭스

| arm | 검증하는 DR 결정 | 대응 추정량 층(DR-005) |
|---|---|---|
| A1 | DR-001: sum-norm이 RL 인센티브(Dr.GRPO)를 지킨다 | 집계층 |
| A5 | DR-004: staleness/clip 분리(decoupling)의 조건부 이점 | w-슬롯 |
| A6c | DR-005: g-슬롯 교체(CISPO)가 gradient-死를 회복 | g-슬롯 |
| (제외) A4 | DR-003: SFT는 벨트 불요 — A5와 중복이라 별도 계측 후에만 | — |

DR-002(aux RL-only 마스크)는 D0에 이미 고정, 별도 arm 없이 SFT 마스킹으로 상시 유지.

---

## 10. 계측 (로깅) — 기존으로 충분한 것 vs 유일 갭

**원칙: wandb 추가는 pivotal-token 3지표(§11)뿐, 그 외 0 + 학습-side dump 1개.** live triage 지표는 대부분 이미 충분하고, "치밀한 사후 분석"에 구조적으로 없는 건 (a) loss 경계 텐서 하나(→ 아래 dump)와 (b) **per-token entropy**(→ §11 live 3지표)뿐이다.

**기존 wandb/롤아웃 dump로 충분히 obtainable → 신규 없음:**
- w 분포(`P(w>C_w)` 포함)·off-policy 거리(KL/PPL): `rollout_corr/rollout_is_*`, `compute_offpolicy_metrics` **자동 전파**. D0에선 old==rollout이라 trivial(≡1), A5에서 그대로 w-분포가 됨.
- r/fused clip-frac: `actor/pg_clipfrac(+lower)` (arm이 별도 run이라 run 정체성으로 구분).
- staleness: `stale_traj_count`(계산됨) + `trajectory_param_versions`(meta) + `fully_async/partial/max_partial_span`.
- RL:SFT 조성: `hpt/{num_rl_routed,num_sft,offline_data_ratio}` + `rollout_rs_masked_fraction`.
- 응답 길이(집계): `response_length/*`. 정답/오답 길이는 아래 학습-dump가 reward째 담아 offline 산출.

**유일 갭 = loss 경계 텐서** (생성 dump는 `generate_sequences_single` 출력만 = reward·학습 이전). death 밀도(A6c)는 `current log_prob + advantage + per-token clip 결정`이 필요한데 어디에도 없음.

**추가한 것 — 학습-side per-token dump** (`training_dump.py`, `training_dump.*` config, 기본 off):

| 항목 | 내용 |
|---|---|
| 통합 지점 | `FullyAsyncTrainer._fit_dump_data`(fit_step의 기존 훅, `_fit_update_actor` 직후 = 텐서 수렴 지점) |
| 담는 것 | per-token `response_mask,log_probs,old_log_probs,rollout_log_probs,advantages,hpt_is_sft,token_level_scores` + per-row `uid,prompt_uid,min/max_global_steps` + meta(step/param_version/rows) |
| offline 산출 | A1 정답×길이 · A5 w/r 분해·위치별 w · A6c death 밀도 |
| config | `enable`(off) · `dir` · `sample_every_n_steps`(20) · `max_rows`(256) · `dtype`(bf16) · `offload`(true) |

**무게 (걸림돌 아님):** read-only(라이브 batch를 이동/캐스팅/변경 안 함 — per-tensor 독립 CPU clone) · **sampled**(1/N)+**max_rows**로 볼륨 유계 · **offload**(background thread, 이전 write 진행 중이면 그 step skip → 트레이너 절대 non-block). 권장 기본값이면 이미 감내 중인 생성 dump(`all_steps=True`)보다 가볍다. 반대로 매-step 동기면 검증된 event-loop starvation 실패 모드로 직행(Readme_RL).

**검증:** `tests/special_RL/test_training_dump_on_cpu.py`(CPU, 15개) — read-only 불변성, round-trip fidelity, row cap, dtype cast, offload flush, busy-skip, config 계약. base-RL(HPT 필드 없음) 경로도 커버.

## 11. 분석 지표 — entropy·clip 해상도 (pivotal-token 질문)

A5(decoupling)·A6c(CISPO)의 판정은 **집계 clip-frac이 아니라 pivotal-token 해상도**로 읽어야 한다. 이 절은 그 판정을 가능케 하는 최소 live 지표를 고정한다. 근거는 DR-004 §11(측정 라우터)·DR-005 §8(계측)과 문헌이며, 여기서 재서술하지 않고 배치만 한다.

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
| `≫ 1` + A5 사전점검 staleness **비례** | staleness 원인 → **decoupling(A5)** |
| `≫ 1` + staleness **무관** | movement/과클립 → **g-슬롯/CISPO(A6c)** (CISPO 문헌과 정합: 원인은 movement) |
| `entropy_mean`/`top20` 하락 추세 | 붕괴 서명 → A6c 정보성↑. 단 커리큘럼 조성 이동과 교란되므로 `hpt/p_success*`로 조건화해 읽는다 |

### 11.5 §10 dump와의 역할 분담 — 두 notion의 "pivotal"

| 층 | 담당 | 시점 |
|---|---|---|
| §10 per-token dump | w/r 분해·위치별 w·**low-prob** death 밀도(A6c)·정답×길이(A1) | 오프라인, 임의 층화 |
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
| wandb 스칼라(run별) | `actor/pg_clipfrac(+lower)`·`ppo_kl`·`entropy_mean`·`entropy_top20_mean`·`pg_clipfrac_top20entropy`(§11)·`hpt/*`(조성)·`rollout_corr/*`(A5)·`response_length/*`·`fully_async/*`(staleness·partial) | 라이브 triage·추세·라우팅 |
| per-token dump(§10) | `log_probs,old_log_probs,rollout_log_probs,advantages,response_mask,hpt_is_sft,token_level_scores/rewards` + `uid,prompt_uid,min/max_global_steps` + meta(step,param_version) | 오프라인 정밀 분해 |
| run 정체성 | D0/A1/A5/A6c 각각 별도 run | arm 간 델타 |

### 12.1 공통 전처리 (de-confounding — 먼저 안 하면 오독)
1. **entropy_loss ≠ entropy**: 붕괴 판정에 `entropy_loss`(합계, RL 토큰 수에 오염) 금지 → `entropy_mean`/`entropy_top20_mean`(§11).
2. **SFT 희석 제거**: 집계 지표는 SFT 토큰(ratio≡1)이 분모에 섞임 → dump 파생은 `hpt_is_sft==False` 필터, 라이브는 이미 RL-only.
3. **커리큘럼 조성 통제**: RL 조성이 학습 중 0→다수로 이동 → 모든 추세를 `hpt/offline_data_ratio`·`hpt/p_success*`로 **층화/조건화**해 읽는다.
4. **arm 델타는 run 간**: D0(fused)와 A5(decoupled)의 `pg_clipfrac`은 이름만 같은 **다른 양** → 대시보드 구분(§8, DR-005 §8).

### 12.2 dump 파생 계산 (정확한 수식, `hpt_is_sft==False` 행만)
```
w_t = exp(old_log_probs − rollout_log_probs)   # A5: π_entry/π_roll ; D0: ≡1 (old=rollout)
r_t = exp(log_probs − old_log_probs)           # A5: π_θ/π_entry ; D0: π_θ/π_roll (fused)
clip_active_t = pg_clip_active_mask(r_t, advantages, ε_low, ε_high)   # §11 공유 predicate
staleness_row = param_version(meta) − max_global_steps(row)
resp_len_row  = response_mask.sum(-1) ;  correct_row = advantages>0 (또는 token_level_scores)
```
파생: `P(w>C_w)`(포화율)·위치별 `w_t`(partial 꼬리=앞토큰이 가장 낡음)·r-clip-frac·**low-prob death 밀도** `= mean( clip_active & (exp(log_probs)<τ_lowprob), A 부호별 )`.

### 12.3 arm별 레시피 (계산 → 판정; 서명 상세는 §3)
| arm | 계산 | 판정 규칙 |
|---|---|---|
| **A1** | dump에서 정답/오답(A 부호)별 `resp_len` 추세를 D0 vs A1. SFT 점유율 동일 확인(통제). | 오답↑·정답↓·간격 벌어짐 → sum-norm 실증. 간격 불변 → 이 레짐 길이게임 미발동(설계 불변). 성능 단독은 약증. |
| **A5 사전** | D0 wandb `pg_clipfrac` vs `staleness_row` 상관(binned scatter). | 무상관 → 낡음 없음 → A5 무의미(스킵). 상관 → 착수. |
| **A5** | A5 dump로 w/r 분해: `P(w>C_w)`, r-clip-frac vs D0 fused, 동결→흐름 전환된 stale 토큰. RL:SFT 유효비 불변, step-time 델타. | r-clip-frac≪D0 + `P(w>C_w)`>0 + 성능↑ → decoupling 이득. `P(w>C_w)`≈0 → 이 run 낡음 낮음. 성능↓ → 절단 편향 < 얼리기. |
| **A6c 전제** | A5 dump low-prob death 밀도 + 라이브 `clipfrac_top20entropy`(A5)>0 여부. | ≈0 → 살릴 것 없음 → A6c≈A5(DR-005 §4.1 검증). >0 → A6c 정보성 성립. |
| **A6c** | A6c dump death 밀도 재계산(≈0 기대) + grad-norm spike·발산 감시 + `ppo_kl` 의미 변경 유의. | death↓ + 성능↑ → g-슬롯 교체 정당. spike/발산 → vanilla 비관성이 안전장치였음. |

### 12.4 통합 라우터 (pivotal 해상도 = §11)
`clipfrac_top20entropy / pg_clipfrac` × staleness 상관:
- 비율 ≈1 → clip이 pivotal에 안 쏠림 → 집계 낮으면 종료(출구 ①).
- 비율 ≫1 & staleness **비례** → **A5(decoupling)**.
- 비율 ≫1 & staleness **무관** → **A6c(CISPO)**(movement 원인, CISPO 문헌 정합).
- `entropy_mean`/`top20` 하락(p_success 조건화 후) → 붕괴 서명, A6c 보강.

### 12.5 재현성·한계
- 파생 계산은 ad hoc 셀 금지 → dump+wandb export를 읽는 **분석 스크립트/노트북 1개**로 고정(입력=run dir, 출력=arm별 서명 표). arm 재실행 시 동일 스크립트로 재산출.
- **한계**: dump에 `entropy`가 없어 (a) 코호트 고정 entropy 붕괴 곡선, (b) high-entropy death 밀도는 **오프라인 불가**. (a)는 라이브 `entropy_top20_mean`(조성 조건화), (b)는 라이브 `clipfrac_top20entropy`로 대체. 정밀 오프라인이 필요하면 dump에 `entropy` 필드 추가가 유일 경로(§10 확장, entropy는 이미 계산되므로 forward 0).

---

## 부록: 런처 D0 검증 기록

`main_scripts/run_fully_async_policy_openr1_hpt_qwen25_math_1_5b_main.sh`를 §2 명세와 대조한 결과 **D0를 정확히 가리킴**(잘못 걸린 값 없음). 명시값(loss_agg_mode, clip 0.2/0.28/10, entropy 0.001, grpo+Dr.GRPO, gamma/beta 등)과 검증된 암묵 기본값(`rl_old_logprob_source="rollout"` Literal 강제, `ppo_epochs=1`, `rollout_is/rs=null`, `alpha=1.0`)이 모두 D0와 일치. §2의 3-pin(ppo_epochs·rl_old_logprob_source·k_max)을 명시로 반영해 앵커를 코드에 못박았다([각주 10]).

## 유지보수

이 문서는 실험 *설계·분석*의 단일 진실 출처다. arm이 추가/변경되면 §3·§4·§9와 함께 **분석 절차 §11·§12**도 갱신한다(지표·수식·판정 규칙이 arm과 어긋나면 안 됨). 수치 판정선(예: "오답 길이 +X%")이 정해지면 §3 각 arm에, `clipfrac_top20entropy/pg_clipfrac`의 "≫1" 임계 등 라우터 판정선이 정해지면 §12.4에 박는다. §12.5의 분석 스크립트가 이 절차의 실행 대응물이다. 결정의 *근거*는 여기서 재서술하지 말고 해당 DR을 인용한다(line 번호 회피, symbol 기준 — `Codemap_RL.md` 관례).
