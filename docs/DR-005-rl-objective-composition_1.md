# DR-005. RL Branch 목적함수의 결합 설계 — 통일 추정량, 슬롯 분리, 정준 조합

Status: **방향 문서** · 미구현 · 적용은 DR-004 §11 라우터 통과 후 · 기본값은 여전히 `rollout` anchor(main 불변)
범위: async-HPT **RL branch** 목적함수에서 staleness 처리와 trust-region 처리를 **어떻게 결합해도 되는가**의 수학적 판별과, 그 판별을 통과한 목표 조합의 고정. SFT branch(DR-003)·aggregation(DR-001)·auxiliary(DR-002)는 **불변 전제이자 보존 조건**이다.
관련 코드: `verl/workers/utils/losses.py::ppo_loss`, `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`(및 `rollout_is_weights` 적용점), `verl/trainer/ppo/rollout_corr_helper.py::compute_rollout_correction_and_rejection_mask`, `verl/experimental/fully_async_policy/hpt_training.py`, `verl/experimental/separation/ray_trainer.py::_compute_old_log_prob`
관련 문헌: HCS 2022 "Batch size-invariance"(arXiv:2110.00641) · AReaL(2505.24298) · A-3PO(2512.06547) · CISPO/MiniMax-M1(2506.13585) · SAPO(2511.20347) · GSPO · DAPO Clip-Higher / Dr.GRPO · dual-clip(1912.09729)
전제: GRPO advantage(Dr.GRPO 관례, std 정규화 없음), branch-blind loss(DR-001), fully-async + partial rollout, `ppo_epochs=1`에 fit_step당 소수(≈4)의 SGD minibatch.

---

## 구현 기록

- **미구현.** 이 문서는 코드 변경이 아니라 "무엇을 결합해도 수학적으로 성립하는가"의 판별 기록이다.
- 적용 순서는 DR-004 §11의 측정 라우터가 정한다. 이 문서의 역할은 라우터의 **각 출구에 사전 검증된 조합을 미리 배치**해서, 측정 결과가 나왔을 때 임기응변 패치가 아니라 검증된 결합으로 이동하게 하는 것이다.
- "최적"의 범위 한정: 여기서 최적은 §2의 제약(신선 극한 불편성, 계수 유계, 계약 보존, per-token 단위) **아래에서의** 조건부 최적이며, 전역 최적성 정리가 아니다. g-슬롯의 최종 선택은 레짐 의존이라 측정이 판정한다(fork의 measure-first 규율 유지).

---

## 0. 한 문단 요약

CISPO·SAPO·GSPO·Clip-Higher·decoupled+TIS는 서로 다른 알고리즘이 아니라, **하나의 per-token 추정량 족(family)의 두 슬롯에 대한 선택지**로 정리된다: `∇Ĵ = E[ w̄_t · g(r_t) · A_t · ∇log π_θ ]`. 여기서 **w-슬롯**(`w̄_t`, staleness 보정)은 θ-무관 상수 계수이고 **g-슬롯**(`g(r_t)`, trust-region gate)은 θ-의존 함수라, 두 슬롯 사이에 gradient 교차항이 없다(Lemma 1). 이 **슬롯 분리(disjointness)**가 "진짜 결합 가능"의 정확한 판별 기준이다 — 서로 다른 슬롯을 차지하면 결합 가능(decoupled×{hard-clip, Clip-Higher, dual-clip, CISPO, SAPO}), 같은 슬롯을 차지하면 결합이 아니라 대체(CISPO×SAPO), 슬롯의 단위 구조를 바꾸면 결합 불가(GSPO — token→sequence 단위 변경은 DR-001 aggregation 계약·Dr.GRPO 인센티브 제약과 충돌). w-슬롯에는 정준(canonical) 선택이 존재한다: `w̄ = min(w, C)`(per-token TIS)는 상한 C 아래에서 pointwise 편향 최소(Lemma 4)이며 verl이 이미 제공한다. g-슬롯에는 정준 선택이 없고 레짐이 정한다 — decoupling이 CISPO·SAPO의 gradient-死 레짐을 축소하는 정도는 **anchor 종류에 달렸다**(§1.1 주의): recent proximal(옵션 B)이면 r이 매 step 이동분이라 크게 축소되나, fork의 기존 MIS θ_sync(옵션 A)면 r이 sync 주기 내내 누적되어 축소가 제한적이다(§4.1). 따라서 목표 조합은 phase-1 = **decoupled + TIS-capped w + 기존 vanilla min-clip(+dual-clip)** (가장 보수적, verl 코드가 이미 정확한 적용점을 가짐 — Lemma 2), phase-2 = 라우터가 gradient-death/불연속을 실측할 때만 g-슬롯을 CISPO-sg 또는 SAPO gate로 교체. 모든 채택 조합은 `g(1)=1`을 만족하므로 SFT branch는 **구성상(by construction)** 불변이다(Lemma 3).

---

## 1. 통일 추정량 — 모든 후보는 한 식의 두 슬롯이다

### 1.1 참 추정량

정책 3개(DR-004 §3): `π_roll`(생성), `π_entry`(batch 진입 스냅샷), `π_θ`(inner SGD 중). per-token으로

```
w_t = π_entry/π_roll   (θ-무관: 두 스냅샷의 비)
r_t(θ) = π_θ/π_entry   (θ-의존)
ρ_t(θ) = w_t · r_t     (= π_θ/π_roll, 참 IS ratio)
```

rollout에서 뽑힌 데이터로 현재 정책의 gradient를 불편 추정하는 per-token(1차) 추정량은

```
∇J(θ) = E_{π_roll}[ ρ_t · A_t · ∇log π_θ ]        …(1)
```

(∇ρ = ρ∇log π_θ이므로 surrogate `E[ρA]`의 gradient가 정확히 (1)이다.)

> **주의 — `π_entry`의 실체 (DR-004 §1.5와 동일)**: 이 문서는 `π_entry`를 **이상적 "항상 최신" recent proximal**(AReaL식, 매 step 직전 정책 = DR-004 §9 옵션 B)로 둔다. 그러나 **fork의 기존 코드 경로(`_compute_old_log_prob`, MIS)는 이 자리에 sync 주기 첫 정책 θ_sync를 쓴다**(주기 내내 고정 = 옵션 A). 둘은 다르다: 옵션 B면 `r = π_θ/π_entry`가 **매 step 이동분**(작음), 옵션 A(MIS)면 `r = π_θ/θ_sync`가 **sync 주기 시작 이후 누적 이동**(주기 후반에 큼). 아래 **판별(§2)·결합(§5)은 anchor 종류와 무관하게 성립**하지만, **magnitude 논증(§4.1)과 목표 조합(§6)은 어느 anchor인지에 의존**한다.

### 1.2 통일족과 두 슬롯

검토한 모든 방법은 (1)의 계수 `ρ_t`를 **유계 계수로 교체**하는 것으로 표현된다:

```
∇Ĵ(θ) = E_{π_roll}[ w̄(w_t) · g(r_t; sign A_t) · A_t · ∇log π_θ ]        …(2)
         └ w-슬롯: θ-무관 prefactor   └ g-슬롯: θ-의존 gate의 유효 gradient 계수
```

각 방법의 슬롯 배치 (유효 gradient 계수 기준; sg = stop-gradient):

| 방법 | w-슬롯 `w̄(w)` | g-슬롯 `g(r)` | g의 성질 |
|---|---|---|---|
| coupled 단일 ratio (현행 HPT) | (슬롯 없음 — w가 g 안으로 흡수: `g(ρ)`) | min-clip on `ρ` | 낡음이 clip을 오발동 (DR-004 §5 World A) |
| **decoupled + TIS** (HCS/AReaL) | `min(w, C_w)` | min-clip on `r` | 한쪽(개선 방향 초과)에서 **정확히 0** (gradient 死) |
| + Clip-Higher (DAPO) | 〃 | 비대칭 밴드 `[1−ε_l, 1+ε_h]`, ε_h>ε_l | 死는 유지, 상방 여유 확대 |
| + dual-clip | 〃 | A<0에서 `min(·, c·A)` 바닥 | A<0·r 폭주 유계화 |
| CISPO | (원논문은 roll=entry라 슬롯 미분리) | `sg(min(r, 1+ε_h^IS))` | **절대 0이 안 됨** (계수 포화) |
| SAPO | (없음 — off-policy 보정 자체가 없음) | `4σ'(τ_±(r−1))·r` | 매끄럽게 감쇠, 0에 점근하나 도달 안 함 |
| GSPO | — | — | **per-token 슬롯 구조 이탈** (sequence 단위 ratio) |

세 gate 모두 `g(1)=1`을 만족한다(hard clip: r=1은 밴드 안; CISPO: min(1, 1+ε)=1; SAPO: 4σ'(0)·1=1). 이 성질이 §2의 계약 보존을 만든다.

---

## 2. 결합 가능성의 판별 기준 (Lemma 1–4)

"결합 가능"을 인상이 아니라 다음 네 개의 검증 가능한 명제로 정의한다.

**Lemma 1 (슬롯 분리 — 교차항 부재).** `w̄_t`는 θ-무관이므로 `∇_θ[ w̄_t · G(r_t) ] = w̄_t · ∇_θ G(r_t)`. w-슬롯의 어떤 절단/캡도 θ-gradient에 불연속을 만들지 않고, g-슬롯의 어떤 gate도 w의 통계에 영향을 주지 않는다. **서로 다른 슬롯의 선택은 독립적으로 교체 가능하다.** (참고: `sg(w̄)=w̄` — w는 애초에 gradient 경로가 없으므로 CISPO의 sg 안에 w를 넣든 밖에 두든 동일하다.)

**Lemma 2 (양수 상수의 clip-구조 가환성 — 적용점의 정당성).** `w̄_t > 0`이면 `w̄·min(x,y) = min(w̄x, w̄y)`이고, min/max/where로 구성된 vanilla loss의 분기 선택은 모든 분기가 같은 양수 `w̄_t`로 스케일될 때 불변이다. 따라서 **clip 로직이 끝난 뒤 `pg_losses * rollout_is_weights`로 곱하는 verl의 기존 적용점(`compute_policy_loss_vanilla`)은 결합 목적함수 (2)와 수학적으로 동치**다 — clip 분기 선택을 바꾸지 않으면서 w̄를 합성한다. (반대로 w를 ratio **안에** 넣으면(coupled) 분기 선택 자체가 바뀐다 — 이것이 현행의 문제였다.)

**Lemma 3 (계약 보존).** 채택 조합이 다음을 만족하면 기존 DR 계약이 구성상 유지된다:
- `g(1)=1`, SFT row에서 `w̄≡1`(G5 마스킹), `r≡1`(self-detach) ⇒ SFT token의 gradient 계수 = `1·1·A` ⇒ **β-스케일 NLL 그대로** — DR-003 계약이 gate 선택과 무관하게 유지된다.
- per-token 계수 수정만 있고 집계 구조 불변 ⇒ DR-001의 `seq-mean-token-sum-norm` + `L_max`와 Dr.GRPO 인센티브 제약(자기-길이 divisor 금지) 유지.
- 동기 극한(`π_roll = π_entry`, w≡1)에서 (2)가 vanilla PPO/GRPO로 정확히 환원.

**Lemma 4 (w-슬롯의 정준성).** 상한 `C`가 주어졌을 때 `min(w, C) = argmin_{0≤v≤C} |v − w|` (pointwise). 즉 TIS-cap은 **주어진 유계 제약 아래에서 pointwise 편향이 최소인 유일한 선택**이며, 편향–분산 트레이드오프는 스칼라 `C_w` 하나로 완전히 매개된다. w-슬롯에는 이 이상 논쟁할 것이 없다 — 남는 자유도는 `C_w` 값뿐.

**판별 규칙 요약**: (i) 두 방법이 **다른 슬롯**을 차지하고 Lemma 1~3을 통과하면 결합 가능. (ii) **같은 슬롯**이면 결합이 아니라 대체(둘 중 하나만). (iii) **슬롯의 단위 구조**(per-token)를 바꾸면 Lemma 3 위반으로 결합 불가.

---

## 3. w-슬롯: 정준 선택 — per-token TIS-capped w

- **선택**: `w̄_t = min(w_t, C_w)`, per-token, `C_w`는 verl `rollout_is_threshold` 기본 2.0에서 시작.
- **per-token인 이유 (per-sequence 곱 기각)**: (a) 시퀀스 곱 `Π_t w_t`는 분산이 길이에 지수적 — 절단해도 포화가 상시화된다. (b) partial rollout에서 한 응답의 토큰들이 **서로 다른 rollout 버전**에서 생성되므로(DR-004 §7), 시퀀스 곱은 서로 다른 behavior 정책의 ratio를 하나의 "likelihood ratio"인 척 곱하는 것 — per-token만이 정직한 grain이다. per-token `w_t`는 각 토큰을 실제로 뽑은 버전의 참 sampling logprob 대비라 pointwise로 잘 정의된다.
- **advantage staleness 우려의 해소 (GRPO 특수성)**: GAE와 달리 GRPO advantage는 생성된 시퀀스들의 보상 통계의 함수다. 고정된 시퀀스의 보상은 정책 버전과 무관하므로, critic-stale 문제가 없다. 남는 것은 분포 이동뿐이고 그것이 정확히 `w·r`이 1차 보정하는 대상이다. (DR-004의 "w는 advantage를 보정하지 않는다" 경고는 GAE 레짐용이며, 이 fork의 GRPO에서는 위협이 크게 완화된다 — 단 group 구성 자체가 π_roll 분포라는 잔여 사실은 남는다.)
- **rejection(`rollout_rs`)은 w-슬롯의 일부가 아니다**: 기본 OFF(DR-004 §8). 켜는 것은 support를 자르는 별도 결정이며 RL:SFT 조성을 왜곡한다(World B′).
- **k_max와의 관계**: `filter_hpt_stale_rollout_samples`(row 드롭)와 `C_w`(weight 캡)는 support 경계와 weight 경계로 **역할이 다르며 공존 가능**하나, 예산은 함께 조율할 것(§7-1).

---

## 4. g-슬롯: 정준 선택은 없다 — 레짐 논증과 보수적 기본값

### 4.0 현행 launcher g-slot 실태 (파리티 정정 — 코드 대조)

이 문서 곳곳(§0·§4.2·§5·§6)이 phase-1 g-slot을 "vanilla min-clip(대칭 ε)"으로 적었으나, **실제 main launcher(7B·1.5B)는 이미 Clip-Higher를 쓴다**: `clip_ratio_low=0.2, clip_ratio_high=0.28`(비대칭) + `clip_ratio_c=10`(dual-clip). 그리고 **비교 대상 UPT 레퍼런스(`Unify-Post-Training/exp_scripts/train.sh`)는 on-policy ratio clip을 아예 안 쓴다** — `loss_remove_clip=True`라 `-A·ratio`(unclipped), grad-norm(`max_grad_norm=80`)만 건다(코드: `mix_src/mix_actor.py`→`core_algos.compute_policy_loss`). 정리:

- fork가 clip을 **켜는 것 자체는 async staleness 때문에 강제**다(coupled에서 clip 없으면 stale ratio 폭발). 이건 UPT 대비 **피할 수 없는** 편차.
- 그 위의 **비대칭 0.28(clip-higher)·dual-clip 10은 "필수 최소 clip" 위에 얹은 DAPO 레시피**라 자유 선택.
- §5 판정상 **decoupled × Clip-Higher는 결합 가능(✅)**이므로 런처의 선택은 허용 집합 안이다.

**열린 결정 (baseline g-slot):** phase-1 g-slot을 (A) 런처 그대로 **Clip-Higher 유지**할지, (B) UPT 파리티/최소-clip 원칙에 맞춰 **대칭 0.2/0.2로 낮출지**는 아직 미정이다 — UPT **성능 비교**가 목적이면 (B)(또는 clip-higher를 async-forced 레시피로 명시), **decoupling ablation**만이 목적이면 (A)로 충분. **아래 §4.2~§6의 "vanilla min-clip" 표현은 (B) 방향의 이상형이며, 현행 코드는 (A)임을 유의.** (DR-004 §1-8도 이 실태를 기록.)

### 4.1 구조적 사실: decoupling이 g-슬롯의 문제 레짐을 축소한다

CISPO의 문제 레짐은 "한 batch에 다수의 inner update(원논문 16회) → r이 크게 표류 → 피벗 토큰이 clip 밴드 밖 → gradient 死"이다. 이 fork에서 그 레짐이 축소되는지는 **anchor 종류에 갈린다(§1.1 주의)** — 여기가 초안이 뭉갠 지점이다:

- **옵션 B (recent proximal, AReaL식):** r의 anchor가 **매 fit_step 리셋**되고 `ppo_epochs=1`·fit_step당 SGD≈4라, r 표류가 짧아 clip-frac이 구조적으로 작다. 현행 clip-frac을 부풀리던 staleness는 전부 w로 빠진다. → 이 경우엔 gradient-死 표면이 작아 CISPO/SAPO 유인이 약하다.
- **옵션 A (MIS θ_sync, 기존 코드 재연결로 얻는 것):** anchor가 **sync 주기마다만** 리셋되므로 r은 주기 전체(최대 `trigger_parameter_sync_step × fit_step당 SGD` ≈ 4×4 = 16 step)에 걸쳐 **누적**된다. within-cycle drift가 clip에 그대로 남아 **주기 후반 토큰은 fresh여도 死할 수 있고, decoupling(θ_sync)은 이를 못 고친다.** 이 死는 staleness가 아니라 gradient-death라 **CISPO 영역**이다(DR-004 §11 출구 ③).

따라서 "decoupling이 CISPO 레짐을 없앤다"는 **옵션 B에서만 강하게 성립**한다. 옵션 A(현행 재연결)에서는 축소가 제한적이다. 어느 쪽이든 예측이지 측정이 아니므로(§8), phase-2는 라우터가 판정한다 — 그리고 **어느 anchor를 쓰는지가 이 예측을 바꾸므로 §6에서 anchor를 명시**한다.

### 4.2 후보별 정밀 비교 (모두 Lemma 1–3 통과 — 결합은 전부 성립, 문제는 선택)

**(a) vanilla min-clip (+dual-clip) — phase-1 기본값.**
- 유지 근거: 전투 검증됨; verl native(코드 변화 0); min()의 **비관성(pessimism)** — clip은 "개선 방향으로 밴드 초과"에서만 gradient를 끊고, 악화 방향은 unclipped 항이 선택되어 gradient가 계속 흐른다. 이 부호-조건부 구조는 CISPO·SAPO에 없는 안전 장치다.
- 정확한 死 지점: `(A>0, r>1+ε_h)`와 `(A<0, r<1−ε_l)`. decoupling 후 이 영역이 실측으로 작으면 비용 없음.

**(b) Clip-Higher(ε_h>ε_l) — 기본값에 대한 저비용 옵션.** min() 비관성을 유지한 채 상방 여유만 확대(entropy 붕괴 완화). 파라미터 2개짜리 변경이라 phase-1 내에서 entropy 추이를 보고 결정.

**(c) CISPO-sg — phase-2 후보 (死 실측 시).**
```
L_t = − w̄_t · sg( min(r_t, 1+ε_h^IS) ) · A_t · log π_θ
```
- 결합의 이점: 원논문은 roll=entry(신선 batch 다회 갱신)라 caps가 하나뿐이지만, decoupled 합성에서는 **staleness 캡(C_w)과 이동 캡(ε_h^IS)이 분리**되어 각자의 의미가 깨끗해진다 — coupled 상태에서 CISPO화하면 두 캡이 하나의 ρ 위에서 충돌한다. 즉 CISPO는 decoupling과 결합할 때 오히려 원형보다 정합적이다.
- 정직한 비용: (i) min() 비관성 상실 — 계수가 부호와 무관하게 무조건 적용되므로, `(A>0, r 폭주)`에서 PPO는 0으로 멈추지만 CISPO는 캡 값으로 **계속 민다**. 반복 갱신 안전은 캡+작은 LR에 의존(원논문은 16회 갱신에서 실증). (ii) surrogate-objective에서 REINFORCE-with-coefficient로 추정량 클래스가 바뀐다 — `ppo_kl`류 모니터링 의미 변경.

**(d) SAPO gate — phase-2 후보 (불연속/고분산 실측 시).**
```
L_t = − w̄_t · f(r_t) · A_t,   f(x) = σ(τ_±(x−1))·4/τ_±   (유효 계수 g(r)=4σ'(τ_±(r−1))·r, g(1)=1)
```
- 이점: 死 대신 매끄러운 감쇠(양방향), A 부호별 온도(τ_neg>τ_pos)로 음수-advantage 안정화. 8k 장문 응답 레짐과 관련 가능.
- 정직한 비용: (i) **τ 하이퍼는 coupled 세팅에서 이식 불가** — gate의 인자가 ρ에서 r로 바뀌면 같은 τ가 전혀 다른 강도를 의미한다(§7-3). (ii) clip-frac 지표가 사라지고 감쇠 분포로 대체 — 모니터링 재설계. (iii) 신생 방법(2511), 검증 이력 짧음.

**(e) GSPO — 이 fork에서 기각.** (i) sequence 단위 ratio는 per-token 슬롯 구조를 이탈 → Lemma 3 위반(DR-001 aggregation 계약과 충돌). (ii) 길이-정규화 지수 `1/|y|`가 ratio에 길이 의존성을 재도입 → Dr.GRPO 인센티브 제약과 긴장. (iii) 동기 레짐(MoE 라우팅 변동성)이 부재 — 이 fork는 dense Qwen2.5-Math. (iv) sequence-level clip은 한 토큰의 이탈로 시퀀스 전체를 죽여 gradient 死를 오히려 악화(SAPO 논문의 GSPO 비판 지점). MoE로 갈 때만 재고.

---

## 5. 판별 결과 총괄

| 조합 | 판정 | 근거 |
|---|---|---|
| decoupled(TIS-w) × vanilla min-clip(+dual-clip) | ✅ **성립 (phase-1 목표)** | Lemma 1–3 통과, Lemma 2로 verl 적용점 그대로 정확 |
| decoupled × Clip-Higher | ✅ 성립 (phase-1 내 옵션) | 밴드 파라미터만 변경, 비관성 유지 |
| decoupled × CISPO-sg | ✅ 성립 (phase-2, 死 실측 시) | 슬롯 상이 + g(1)=1; 캡 2개의 의미 분리는 원형보다 정합적. 비관성 상실은 수용 비용 |
| decoupled × SAPO gate | ✅ 성립 (phase-2, 불연속 실측 시) | 슬롯 상이 + g(1)=1; τ 재튜닝 필수 |
| CISPO × SAPO 동시 | ❌ **대체 관계** | 같은 g-슬롯의 두 선택 — "결합"이 정의되지 않음. 둘 중 하나 |
| decoupled × GSPO | ❌ 기각 | 슬롯 단위 구조 이탈(Lemma 3 위반), Dr.GRPO 긴장, 동기 부재 |
| raw w (절단 없음) | ❌ 금지 | 분산 무계(DR-004 §4). Lemma 4가 대안을 정준화 |
| rejection(`rollout_rs`) 기본 ON | ❌ 기본 기각 | support 절단은 별도 결정, RL:SFT 조성 왜곡(World B′) |
| proximal-KL 추가 (r-clip 위에) | ❌ 중복 제동 | g-슬롯이 이미 trust region — 같은 축의 이중 기계. DR-003의 "기계는 제 값을 해야" 원칙 |
| A-3PO 폐형 근사 (π_entry forward 대체) | ⏸ 보류 | 정확한 anchor 대비 근사 오차를 사는 것 — entry forward 비용(≈actor fwd 1/3)이 실측으로 병목일 때만 |
| Dr.GRPO / branch-blind / β_r / aux 마스크 | ✅ 자동 유지 | advantage·aggregation·auxiliary 층은 (2)의 바깥 — Lemma 3 |

---

## 6. 목표 조합 (라우터 출구별 고정)

**Phase-1 (DR-004 §11 라우터 출구 ②: clip-frac 높고 staleness 비례):**

```
RL token:  L_t = − w̄_t · [ vanilla min-clip(r_t, A_t; ε_l, ε_h) with dual-clip(c) ]
           w̄_t = min(π_entry/π_roll, C_w)   per-token, C_w=2.0에서 시작
           r_t = π_θ/π_entry
           π_entry = ★recent proximal(옵션 B) 확정 (DR-004 §8-2)★ — 매 fit_step 현재 가중치 forward.
                     (옵션 A=MIS θ_sync는 B보다 이론·비용 열등이라 미채택; §1.1 주의)
SFT token: 불변 (w̄≡1, r≡1 → β-스케일 NLL)         [Lemma 3]
집계:       DR-001 그대로 (seq-mean-token-sum-norm, L_max)
rejection: OFF · k_max: 유지 · g-slot: **현행 launcher = Clip-Higher(0.2/0.28)+dual-clip(10)** (§4.0). UPT 파리티를 원하면 대칭 0.2/0.2로 낮추는 것도 §4.0의 열린 결정
```

이것은 HCS 2022의 decoupled objective + AReaL의 TIS를 verl의 기존 적용점(`rollout_is_weights`)에 실현한 것이다. **정정(초안 과장 교정)**: w-슬롯(`rollout_is_weights` 곱셈)은 기존 부품의 재배선이 맞지만, **`π_entry`(=old_log_probs)를 어떤 anchor로 계산하느냐는 신규 결정**이다(DR-004 §9 옵션 A/B/C). "기존 `_compute_old_log_prob` 재연결"은 옵션 A(MIS θ_sync)를 주며 **AReaL식 recent proximal이 아니다.** 진짜 AReaL식(옵션 B)은 old_log_probs를 매 fit_step 현재 가중치로 forward하는 것으로, 가중치 스왑이 없어 오히려 MIS보다 싸다. 즉 phase-1을 "그대로 재배선"으로 표현하면 안 되고, **anchor 선택이 phase-1의 핵심 결정**이다.

**Phase-2 (phase-1 가동 후 재측정으로 분기):**

- **r-clip 死가 피벗 토큰에 집중 실측** (§8의 death 밀도 지표) → g-슬롯을 CISPO-sg로 교체 (§4.2c 식). w-슬롯 불변.
- **감쇠 불연속/고분산 실측** (장문·급락 구간) → g-슬롯을 SAPO gate로 교체, τ_± 신규 탐색. w-슬롯 불변.
- **둘 다 미실측** → phase-1 유지. 기계를 늘리지 않는다.

**라우터 출구 ③ (clip-frac 높으나 staleness 무관 — DR-004 §11)**: decoupling 없이 g-슬롯만 교체(Clip-Higher → CISPO 순). 이 경우에도 본 문서의 판별은 유효하다 — coupled 상태의 CISPO는 캡 충돌(§4.2c)이 있으므로, g-슬롯 교체가 필요해지면 decoupling을 함께 켜는 쪽이 캡 의미론상 더 깨끗하다.

---

## 7. 이중 계수·중복 제동 위험 목록 (결합 시 반드시 점검)

1. **`C_w` × `k_max` 공동 조율**: 둘 다 staleness 축의 기계(weight 캡 vs support 컷). k_max를 조이면 w 분포의 꼬리가 잘려 `C_w` 포화율이 내려간다 — 한쪽을 바꾸면 다른 쪽 지표를 다시 읽을 것.
2. **캡 충돌**: coupled 상태에서 CISPO화하면 staleness 캡과 이동 캡이 하나의 ρ 위에서 충돌한다. decoupling이 이 충돌을 해소한다(§4.2c). 순서를 지킬 것: 분리 먼저, gate 교체는 그다음.
3. **하이퍼 이식 금지**: gate 인자가 ρ→r로 바뀌는 순간 ε·τ·임계값의 의미가 전부 바뀐다. coupled 시절 값(현행 ε 포함)을 "같은 이름이니 같은 값"으로 이식하지 말 것. phase-1이 기존 ε을 유지하는 것은 r-분포가 ρ-분포보다 좁아 보수적이기 때문이다(더 좁힐 이유가 생기면 측정 후).
4. **`rollout_is_batch_normalize`**: w̄를 배치 정규화하면 유효 LR이 배치 조성(RL:SFT 비율, staleness 분포)에 따라 흔들린다. 기본 OFF 유지, 켜려면 별도 결정.
5. **entropy(DR-002)와 g-슬롯**: entropy 항은 (2)의 곱 구조 바깥의 가산 항 — gate 교체와 독립이다. 단 CISPO/SAPO로 死가 사라지면 탐색 특성이 변해 entropy 계수의 체감 강도가 달라질 수 있다 — 지표로만 추적, 선제 조정 금지.

---

## 8. 계측 최소 세트 (phase 전환의 판정 근거)

| 지표 | 판정 대상 |
|---|---|
| `P(w > C_w)` (w 포화율), 토큰 위치별 w 분포 | C_w 적정성; partial 꼬리(`k_max+max_partial_span`)의 실재 |
| r-clip-frac (decoupling 후) | §4.1 구조 예상의 검증; phase-2 진입 여부 |
| death 밀도: clip된 토큰 중 low-prob(피벗형) 비율, A 부호별 | CISPO 전환의 직접 근거 (staleness와 무관한 死) |
| entropy 추이 | Clip-Higher 채택 여부 |
| RL:SFT 유효 row/token 비 (post-mask) | 조성 왜곡 감시 (rejection OFF 확인 포함) |
| step-time delta (entry forward) | A-3PO 재고 여부(§5 보류 항목) |

현행(coupled) clip-frac은 staleness와 이동이 뒤섞인 값이므로, **decoupling 전후의 clip-frac은 같은 이름의 다른 지표**다 — 대시보드에서 구분 표기할 것.

---

## 9. DR-001~004 정합 매트릭스

| 기존 결정 | 본 문서와의 관계 |
|---|---|
| DR-001 (branch-blind, sum-norm, L_max, β_r) | **불변 전제.** (2)는 per-token 계수만 수정 — 집계·β_r 채널 무접촉. 순수 RL·동기 극한에서 vanilla 환원(Lemma 3) |
| DR-002 (aux RL-only 마스크) | **불변.** entropy/KL은 곱 구조 바깥 가산 항. §7-5의 관측 주의만 |
| DR-003 (SFT self-detach, 보정·제동 불요) | **구성상 보존.** g(1)=1 + w̄≡1(G5)이면 어떤 채택 gate에서도 SFT는 β-NLL 그대로(Lemma 3). 벨트 불채택 결론도 유지 — g-슬롯은 RL 전용 |
| DR-004 (decoupling 결정, flag-off, §11 라우터) | **본 문서가 §4.5 격자의 정식화.** 라우터 출구마다 검증된 조합을 배치(§6). 귀속(HCS 2022) 일치 |

---

## 부록: 상태와 명명

§1–2(통일족·판별 기준)와 §5(판정표)는 수학적 사실 + 문헌·코드 검증 위에 있다. §4.1(레짐 축소)과 §6(목표 조합)은 구조 논증이며 최종 판정은 §8 계측이 한다. **미구현·방향 문서**이고, 코드 착수는 DR-004 §11 라우터와 §9 스코프를 따른다. 파일명 `DR-005`는 DR-004(decoupling 단일 결정)의 상위 결합 문법이다.

**정정 이력**: 초안은 `π_entry`를 AReaL식 recent proximal로 암묵 가정하고 "anchor는 fit_step마다 리셋", "기존 부품 재배선"으로 서술했다. 그러나 fork의 기존 `_compute_old_log_prob` 경로는 **MIS θ_sync**(sync 주기 고정, proximal age 톱니)로 이와 다르다. 정정 반영: §1.1 주의(π_entry 실체), §0·§4.1(anchor별 레짐 축소 차이), §6(anchor는 신규 결정). **판별부(§1.2 슬롯 분리, §2 Lemma, §5 결합/기각표)는 anchor 종류와 무관하게 유효**하므로 불변이다.

**정정 2 (파리티, 코드 대조)**: 초안이 phase-1 g-slot을 "vanilla min-clip(대칭)"으로 적었으나 실제 launcher는 **Clip-Higher(0.2/0.28)+dual-clip(10)**, UPT 레퍼런스는 **ratio clip 없음**(`loss_remove_clip=True`). §4.0 신설로 실태·async-forced 여부·열린 결정(clip-higher 유지 vs 대칭 0.2)을 기록하고, 이후 "vanilla min-clip" 표현은 (B) 이상형으로 재해석하도록 명시. §6 line도 정정.
