# DR-004. RL Branch의 Off-Policy 처리 — Staleness 보정과 Trust-Region Clip의 분리 (Decoupling)

_Last updated: 2026-07-10_

Status: 분석부 정리 · C1 config/routing/MIS-bypass 구현 완료(2026-07-04) · M-first ablation에서는 C1로 채택(entry-recent + TIS-w) · 기본값 `rollout` 유지 시 D0 경로 불변

> **실증 결과 노트 (2026-07-10, Ablation_RL.md §14.2)**: 본 문서 §6의 "이점은 조건부" 예측이 그대로 실증됐다 — 현행 main(nocispo, decoupled+vanilla)의 실측 w-포화율 `P(w>C_w=2)` 중앙값 **0.10%**(평온기 0.085%, w̄=0.954) → 이 레짐의 낡음이 낮아 **디커플링은 기계적으로 준-불활성**. C1 축은 이 통계로 무런(無run) 폐쇄됐고 D0′ 재실행은 취소됐다. 채택(entry+TIS-w)은 유지하되(비용 무해·보험 성격), 논문에서 단독 기여를 주장하지 않는다.
범위: mixed HPT batch에서 **RL branch**의 off-policy 처리 — clip anchor를 `rollout` vs `entry`로 둘 때의 semantics, 그리고 그에 딸려 활성화되는 off-policy correction. **SFT branch**의 off-policy 처리는 DR-003, aggregation/정규화는 DR-001, auxiliary 정칙화(entropy/KL)는 DR-002 소관.
관련 코드: `verl/workers/utils/losses.py::ppo_loss`, `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`, `verl/trainer/ppo/rollout_corr_helper.py::{_compute_hpt_rollout_correction_and_add_to_batch, compute_rollout_correction_and_rejection_mask}`, `verl/experimental/fully_async_policy/hpt_training.py::{apply_hpt_rollout_logprob_anchor, should_use_hpt_rollout_logprob_anchor, filter_hpt_stale_rollout_samples}`, `verl/experimental/fully_async_policy/hpt_config.py::AsyncHptConfig`, `verl/experimental/separation/ray_trainer.py::{_fit_compute_log_prob, _compute_old_log_prob}`
전제: multi-step minibatch 학습, fully-async + HPT, GRPO advantage.
개정 이력: 초안(phase-1) → 정정: (1) 현행 HPT = rollout-anchored **coupled**임을 명확화, (2) 기존 `_compute_old_log_prob` 경로 = **MIS θ_sync**(≠ AReaL recent proximal)임을 §1.5로 분리, (3) `entry` anchor를 A/B/C 옵션으로 재정의(§9), (4) launcher가 **Clip-Higher(0.2/0.28)+dual-clip**을 쓰고 UPT 레퍼런스는 **ratio clip 없음**(`loss_remove_clip=True`)이라는 파리티 사실 기록(§1-8). 코드 인용은 symbol 기준(line 번호 회피 — `Codemap_RL.md` 관례).

---

## 구현 기록

- **구현 상태(2026-07-04).** 현재 코드는 `rl_old_logprob_source=rollout|entry`를 허용한다. `rollout`은 기존 D0처럼 `old_log_probs=rollout_log_probs`를 사용하고, `entry`는 rollout anchor를 건너뛰어 trainer의 current/recent old-logprob recompute path를 탄다. `entry` 모드는 `rollout_is=token`, `rollout_rs=null`, correction bypass off를 fail-fast로 요구한다.
- 이 DR은 (a) 검증된 현행 동작(§1), (b) 이론 판정(§4), (c) flag-off 구현 스코프(§9)와 (d) M-first run 뒤의 사후 귀속 지표(§11)를 고정하기 위한 기록이다. 옛 "게이트 통과 후 착수" 표현은 `Ablation_RL.md`의 M-first 결정으로 폐기한다.
- `Overview_RL.md`의 G3, `Codemap_RL.md`의 anchor 서술은 **아직 사실이므로 건드리지 않는다**(구현 시점에 갱신).

---

## 0. 한 문단 요약

현행 HPT는 RL row의 `old_log_probs`를 rollout 로그확률로 덮어써(`rl_old_logprob_source=rollout` 강제) PPO ratio를 `π_current/π_rollout` **하나**로 만든다. 이 단일 ratio가 (a) 생성정책→현재정책의 off-policy 보정과 (b) clip의 신뢰영역 기준을 **동시에** 수행한다 — 즉 두 역할이 **융합(fused)**되어 있다. 문제는 rollout이 낡을수록 ratio가 낡음만으로 부풀고, 그 부푼 ratio가 clip에 걸려 **이번 업데이트가 과격하지 않은데도 낡았다는 이유로 학습이 동결**된다는 것이다. Decoupling은 `ρ = π_current/π_rollout = (π_entry/π_rollout)·(π_current/π_entry) = w·r`로 쪼개, clip을 **`r`(이번 업데이트 이동분)에만** 걸고 `w`(staleness)를 **절단된 IS 가중**으로 따로 곱한다. 대수적으로 unclipped surrogate는 불변(`w·r`은 여전히 참 IS ratio)이며, 바뀌는 것은 **clip이 어디서 무느냐**와 **w를 절단하느냐**뿐이다. 이 구조는 이론적으로 건전하고 on-policy PPO로 정확히 환원되지만, **이점은 보장되지 않는다** — 낡음이 작으면 0, 편향된 stale 학습이 동결보다 나쁘면 음수일 수 있고, 오직 "낡음이 유의미한 레짐"에서만 유효하다. M-first ablation에서는 이를 C1로 채택하고, 이득 여부는 M과 M−dec/M−cispo/D0의 **사후 귀속**으로 판정한다. 특히 decoupling에 필수인 것은 IS **재가중**뿐이며, RL row를 버리는 **rejection**은 별개 옵션이고 이번 M에는 넣지 않는다.

---

## 1. 검증된 현행 코드 동작 (baseline)

decoupling을 논하기 전에 지금 무엇이 도는지 확정한다. 아래는 소스 검증 사실이다.

1. **anchor 강제.** `apply_hpt_rollout_logprob_anchor`는 batch 전체에 대해 `old_log_probs = rollout_log_probs.clone()`을 수행한다. `should_use_hpt_rollout_logprob_anchor`는 `async_hpt.enabled`에서 참이고, `rl_old_logprob_source != "rollout"`이면 `ValueError`로 **hard-raise**한다. `AsyncHptConfig.rl_old_logprob_source`는 `Literal["rollout"]`이라 `entry`는 **config 레벨에서 표현 불가 + 런타임 거부**다.

2. **RL row는 융합 ratio.** `compute_policy_loss_vanilla`에서 `ratio = exp(log_prob − old_log_prob)`. RL row는 `old_log_probs == rollout_log_probs`이므로 `ratio = π_current/π_rollout`이고, 이 **같은 ratio가 clamp(clip)의 대상**이다. 즉 off-policy 보정과 trust-region이 한 양(量)이다.

3. **SFT row는 self-detach(불변, DR-003).** `ppo_loss`가 `hpt_sft_token_mask` 토큰에 대해 `old_log_prob = log_prob.detach()`로 되돌려 ratio≡1(NLL). RL row는 rollout anchor 유지. → 이 DR은 RL row만 다룬다.

4. **off-policy correction 경로는 존재하나 dormant.** `_compute_hpt_rollout_correction_and_add_to_batch`가 HPT batch에도 dispatch되어 `old_log_probs` vs `rollout_log_probs`로 IS 가중·rejection mask를 계산한다(SFT는 `rl_response_mask`로 제외 — G5 유지). 그러나 `old == rollout`이라 **IS ratio가 항등적으로 1** → 재가중 무효, rejection no-op. **코드는 돌지만 효과 0(inert).** loss 경로에서도 `rollout_is_weights`는 None이다.

5. **staleness는 ratio 밖에서도 bound.** `filter_hpt_stale_rollout_samples`가 `current_param_version − max_generation_step > k_max`인 RL row를 학습 전 드롭한다(`max_global_steps` 기준, SFT 면제, `k_max=None`이면 no-op).

6. **cost 구조.** `_fit_compute_log_prob`는 if/elif/else 상호배타 체인이다 — HPT anchor 분기는 `rollout_log_probs`를 clone할 뿐이고, `π_entry` forward(`_compute_old_log_prob`)를 수행하는 else 분기는 **건너뛴다.** 즉 현행 HPT는 entry-policy forward를 아예 안 한다.

7. **관측 지표.** `hpt/old_logprob_from_rollout`는 `1.0` 하드코딩이며, 소비처는 `tests/special_RL/test_hpt_trainer_queue_contract.py`의 계약 assert(4개)뿐이다.

8. **launcher clip 실태 + UPT 파리티 (참고, 코드 대조).** main launcher(7B·1.5B)의 g-slot은 `clip_ratio_low=0.2, clip_ratio_high=0.28`(Clip-Higher) + `clip_ratio_c=10`(dual-clip)이다. 비교 대상 UPT 레퍼런스(`Unify-Post-Training/exp_scripts/train.sh`)는 `loss_remove_clip=True`로 **on-policy ratio clip을 쓰지 않는다**(`-A·ratio` unclipped, grad-norm 80만; `mix_src/mix_actor.py`→`core_algos.compute_policy_loss`). fork의 clip은 async staleness 때문에 필요하나, clip-higher·dual-clip은 그 위의 레시피 선택이다. baseline g-slot을 clip-higher로 둘지 대칭 0.2로 낮출지는 **열린 결정**(DR-005 §4.0) — §11 게이트도 이 clip 설정 위에서 clip-frac을 잰다는 점 유의.

---

## 1.5 현재 상태 vs 구현 경로 — 세 층위를 혼동하지 말 것 (정정)

이 문서 초안이 뭉갠 세 가지를 분리한다:

**(가) 현재 HPT main = rollout-anchored coupled.** `old = rollout`, `ratio = π_current/π_rollout`. decoupled도 MIS도 **아니다.** §1.1~1.6이 이 상태다.

**(나) 일반 fully-async decoupled 경로(`_compute_old_log_prob`, else 분기) = MIS θ_sync.** 이 경로는 **HPT batch가 도달하지 않는다**(§1.6 상호배타). 동작:
- `local_trigger_step==1`에 현재 가중치를 CPU slot 1에 저장 → 그 뒤 fit_step들은 slot 1(θ_sync)을 restore해서 old_log_prob 계산.
- 즉 **proximal anchor = sync 주기 첫 정책 θ_sync, 주기 내내 고정.** proximal age가 **톱니(0→N→리셋)**.

**(다) AReaL/HCS decoupled-PPO = 항상 최신 proximal.** AReaL은 매 training step recent snapshot, HCS는 EWMA로 age 일정. 그래서 fresh 샘플에도 `π_prox ≠ π_behav`라 decoupling이 실제로 작동한다.

**결정적 함의 (§9와 직결):**
- (나)의 MIS는 **(다)가 아니다.** fresh(주기 내) 샘플엔 `π_prox = π_behav = θ_sync` → `w=1` → **coupled로 붕괴**하고, proximal age가 톱니라 주기 후반엔 clip이 within-cycle drift까지 흡수한다. decoupling은 cross-cycle 낙오분에만 부분 작동.
- 따라서 **"기존 경로 재연결 = AReaL decoupled"는 틀렸다.** 재연결은 MIS(§9 옵션 A)를 줄 뿐이다. 이 문서의 §3·§5에서 `π_entry`라 부른 것은 **이상화된 recent proximal(§9 옵션 B)**이며, fork의 기존 코드가 주는 것(θ_sync)과 다르다.

---

## 2. 두 장치의 정확한 정의

혼동을 막기 위해 clip과 correction의 **작동 방식 차이**를 명시한다.

**장치 ① — clip (신뢰영역 gate).** `compute_policy_loss_vanilla`에서
`pg_losses2 = −A · clamp(ratio, 1−ε_low, 1+ε_high)`, `clip_pg = max(−A·ratio, pg_losses2)` (A<0은 dual-clip).
ratio가 밴드 **밖(개선 방향)**이면 `clamp`가 경계 상수로 포화 → θ에 대한 미분 0 → **해당 토큰 gradient가 끊긴다.** 값을 부드럽게 줄이는 것이 아니라 **상수로 고정해 학습을 멈추는 gate**다. 단 한쪽만 막는다(엉뚱한 방향으로 간 경우 unclipped 항이 선택되어 gradient가 흐름).
근거: PPO surrogate는 π_old 근처에서만 유효한 국소 근사이므로, 한 update에서 너무 멀리 가면 밀기를 멈춰 파괴적 업데이트를 방지.

**장치 ② — off-policy correction (IS 재가중 + 선택적 rejection).** `pg_losses = pg_losses * rollout_is_weights` (곱셈; gradient를 끊지 않고 **크기를 스케일**). `compute_rollout_correction_and_rejection_mask`가 두 손잡이로 분리 제공한다.
- `rollout_is` (IS 재가중): `rollout_is_threshold`(기본 2.0)로 **절단(truncated IS)**. verl은 raw가 아니라 절단해서 준다.
- `rollout_rs` (rejection): `response_mask`를 수정해 토큰/샘플을 **버림.**
docstring 명시: rejection은 response_mask를 항상 갱신, IS 가중은 `rollout_is` 설정 시에만 추가 — "적용 전 지표만 관측" 조합이 가능하도록 **의도적으로 분리**되어 있다.

---

## 3. 융합 vs 분리 — crux 대수

**On-policy PPO**: 생성정책 = old이라 IS 역할은 자명히 1. **신뢰영역 역할만** 활성. 긴장 없음.

**Off-policy 단일 ratio (현행 HPT)**: `ρ = π_current/π_rollout`가 동시에 rollout→current의 참 IS 보정값이자 clip 대상. **장치 ①이 장치 ②의 양에 그대로 적용됨 → 융합.** clip이 신뢰영역과 IS 분산억제를 거칠게 한꺼번에 함.

**Decoupled**: `π_entry`에서 한 번 쪼갠다.
```
ρ = π_current/π_rollout = (π_entry/π_rollout) · (π_current/π_entry) = w · r
    w = π_entry/π_rollout   → 장치 ②: IS 보정 (절단)
    r = π_current/π_entry    → 장치 ①: 신뢰영역 clip (이번 update 이동분만)
```
**crux(대수적으로 정확):** `w·r = π_current/π_rollout` — unclipped surrogate는 **하나도 안 바뀐다**(여전히 참 IS ratio). 바뀌는 것은 오직 **(a) clip이 어디서 무느냐**(전체 stale ρ가 아니라 update 내 이동 `r`에) **(b) w를 따로 절단하느냐**뿐이다. 그러므로 변화의 크기는 **π_entry가 π_rollout에서 얼마나 벌어졌나(=staleness)**에 정확히 비례한다 — 신선하면 `w≈1`이라 decoupled≈coupled, 낡을수록 차이가 커진다.

> **π_entry 표기 주의**: 이 문서에서 `π_entry`는 **이상화된 "항상 최신" proximal**(§9 옵션 B, AReaL식)을 뜻한다. fork의 기존 코드 경로가 주는 것은 MIS θ_sync(§1.5)로 이 이상화와 다르며, 그 경우 위 crux의 `r`은 update-내 이동이 아니라 **sync 주기 시작 이후 누적 이동**이 된다.

---

## 4. 이론 판정

- **w 방향**: `π_entry/π_rollout`은 target/proposal 방향이 맞다(rollout에서 뽑아 entry로 보정). 단 이는 per-step 1차 보정이고, 시퀀스 전체 보정은 곱, advantage 자체의 낡음은 별도 문제.
- **clip을 r에만**: async off-policy RL의 표준 형태다. 전체 stale ratio를 clip하면 stale 데이터가 step 0부터 밴드 밖이라 신뢰영역이 무력화되므로, `r`에 clip하는 것이 옳다.
- **on-policy 환원**: `π_rollout=π_entry=π_current`면 `w=r=1`, loss = A → 표준 PPO와 일치. ✔
- **w 절단은 필수(유일한 실질 결함 지점)**: raw likelihood-ratio w는 분산 무계이고, clip 안 걸린 `r` 항에 큰 w가 곱해지면 gradient가 폭주한다. AReaL/TIS는 전부 절단한다. **verl의 `rollout_is`가 임계값(기본 2.0)으로 이미 절단**하므로, 이 헬퍼를 경유하면 방지된다(직접 raw 구현만 위험).
- **문헌 귀속**: behavior-vs-proximal decoupling의 출처는 **Hilton, Cobbe & Schulman 2022 ("Batch size-invariance for policy optimization", arXiv:2110.00641; PPO-EWMA)** 그 자체다 — 즉 이 구성이 바로 그것이다(DR-003 §1과 일치). **AReaL**(arXiv:2505.24298)이 async LLM RL에 적용(π_prox = 업데이트 직전 스냅샷, EWMA 대신)하며 **TIS**를 직교적 분산 상한으로 얹고, **A-3PO**(arXiv:2512.06547)는 π_prox를 폐형 근사한 특수 사례다. **PPG**(Cobbe et al. 2021, arXiv:2009.04416)는 policy-vs-value로 **다른 축**이라 제외가 맞다. (SAPO/CISPO는 이 계보가 아니라 trust-region 축의 방법 — §4.5.)

**요지**: 구조는 건전하고 on-policy로 환원되며 async 레짐에서 clip 위치 분리는 정당하다. 단 (i) w는 절단해야 하고(verl 기본 제공), (ii) 계보는 Hilton/Cobbe/Schulman 2022(behavior-vs-proximal)이며 AReaL이 async에 적용한 것이다. **주의**: 여기서 정당하다고 한 "recent proximal"은 AReaL식(항상 최신)을 말하며, fork의 기존 MIS θ_sync 경로는 이것과 **다르다**(§1.5) — MIS는 recent proximal의 정의적 조건("proximal이 behavior보다 최신")을 fresh 샘플에서 위반한다.

---

## 4.5 다른 clipping/off-policy 계열과의 관계 — 경쟁이 아니라 다른 축

"decoupled가 CISPO/SAPO/GSPO보다 낫냐"는 대체로 **범주 오류**다. 이들은 같은 문제의 경쟁 해법이 아니라 **PPO ratio 과부하의 서로 다른 축**을 건드린다. 2축 격자로 봐야 한다:

- **Staleness(off-policy) 축**: coupled(단일 ρ) vs **decoupled + TIS**. ← 이 fork가 고른 축.
- **Trust-region 축**: hard-clip / Clip-Higher(DAPO) / **CISPO**(sg한 clipped IS, gradient 안 죽임) / **SAPO**(sigmoid soft gate) / **GSPO**(sequence 단위 ratio).

| 방법 | 겨냥하는 실패 모드 | decoupled와의 관계 |
|---|---|---|
| decoupled + TIS | staleness가 신뢰영역 오염 | (본인) |
| AReaL / A-3PO | 동일 문제(staleness) | 같은 계열/구현(경쟁 아님) |
| CISPO | pivotal 토큰 gradient 死 (staleness 0에서도 발생) | 다른 축, 결합 가능 |
| SAPO | hard-clip 불연속(고분산·MoE·장문) | 다른 축, 결합 가능 |
| GSPO | token vs sequence 단위 불일치 | 다른 축, 결합 가능 |
| DAPO / Dr.GRPO | clip 모양·정규화 편향 | 다른 축, 결합 가능 |

핵심 세 가지:

1. **다른 게 이기는 지점.** clip-frac이 높은데 **staleness와 무관**하면 문제는 gradient 死(pivotal 토큰 과클립)이지 staleness가 아니다 → 이땐 **CISPO/Clip-Higher가 정답, decoupled는 헛돈다.** decoupled는 `r`에 여전히 hard-clip을 걸므로, 진짜 큰 update-내 이동은 그대로 얼려 CISPO/SAPO가 고치는 gradient 死를 못 고친다.
2. **결합 가능 = 거짓 이분법.** decoupled(staleness 축)와 CISPO/SAPO(trust-region 축)는 목적함수의 서로 다른 인자라 **쌓인다.** 측정이 gradient 死 + staleness 공존을 보이면 decoupled 위에 CISPO식 sg 계수나 SAPO gate를 `r`-clip 자리에 얹으면 된다. flag-off decoupled 레버는 이 선택지를 막지 않는다.
3. **정당화 범위.** 그러므로 우월성 주장은 하지 않는다. decoupled는 **문제 적합성**(§6: 이 fork가 구조적으로 RL-branch staleness를 만든다 = decoupled의 정확한 영역)과 **공학 비용**(§9: verl이 flag-off로 제공)으로만 정당화되며, 순이득은 §11 지표로 사후 판정한다. CISPO/SAPO/GSPO는 지금 기각하는 대안이 아니라 **결합 가능한 trust-region 축**으로 남긴다.

결합의 엄밀한 판별(무엇이 진짜 결합 가능하고 무엇이 대체/기각인가)과 M-first 2×2의 목표 조합은 `DR-005-rl-objective-composition_1.md`가 정식화한다.

---

## 5. 구체 예시 (RL 응답 하나, 토큰 4개)

세팅: advantage 전부 `+1`, clip 밴드 `[0.8, 1.3]`, IS 절단 임계 `2.0`.

| 토큰 | π_rollout | π_entry | π_current | 낡음 |
|---|---|---|---|---|
| t1 | 0.50 | 0.50 | 0.50 | 없음 |
| t2 | 0.10 | 0.20 | 0.25 | 조금 |
| t3 | **0.02** | **0.40** | 0.50 | **매우** |
| t4 | 0.30 | 0.30 | 0.35 | 없음 |

t3: 생성 땐 2%, 업데이트 시작 땐 이미 40%(20배 드리프트). 이번 update 실제 이동은 40%→50%(1.25배)뿐.

**World A — 현행(융합, `old=rollout`)**, `ρ = π_current/π_rollout`:

| 토큰 | ρ | 밴드 | 결과 | 장치② |
|---|---|---|---|---|
| t1 | 1.00 | 안 | 흐름 | IS=1 |
| t2 | 2.50 | 밖 | **동결** | IS=1 |
| t3 | **25.0** | 밖 | **동결** | IS=1 |
| t4 | 1.17 | 안 | 흐름 | IS=1 |

→ 낡은 t2·t3이 **동결**. t3은 실제 1.25배만 움직였는데 stale-부푼 ρ=25 때문에 학습 신호가 사라짐. 장치②는 IS≡1로 dormant. **융합 방식은 낡은 토큰을 골라 얼린다**(안전하지만 stale 신호 폐기).

**World B — decoupled(`old=entry`, `rollout_is` ON/절단, `rollout_rs` OFF)**, `r = π_current/π_entry`, `w = π_entry/π_rollout`:

| 토큰 | r | 밴드 | 결과 | w(raw) | w(절단 후) |
|---|---|---|---|---|---|
| t1 | 1.00 | 안 | 흐름 | 1.0 | 1.0 |
| t2 | 1.25 | 안 | 흐름 | 2.0 | 2.0 |
| t3 | **1.25** | 안 | **흐름** | **20** | **2.0(절단)** |
| t4 | 1.17 | 안 | 흐름 | 1.0 | 1.0 |

→ 네 토큰 모두 gradient 흐름. 낡은 t3도 학습되되(clip이 이동분 r=1.25만 봄), 낡음은 절단된 w로 보정(20→2). `rollout_rs` OFF라 RL 토큰 수 불변 → SFT 쏠림 없음. **여기서 깨어난 장치②(w)는 부작용이 아니라 decoupling의 의도된 절반**이다(w를 안 켜면 낡음을 무시하는 편향 estimator가 됨).

**World B′ — decoupled + rejection(`rollout_rs` ON)**: 예컨대 raw ratio>10이면 버림 → t3 삭제 → 여러 배치에 걸쳐 낡은 RL 토큰이 계속 버려짐 → **배치가 SFT-heavy로 기움.** 이것이 SFT-heavy 위험의 진짜 자리이며, **decoupling 필수가 아닌 선택 손잡이**다.

---

## 6. ★이점은 조건부다 (냉정 포인트)★

decoupling은 "더 좋은 것"이 아니라 "다른 estimator"다. 순이득이 0/음수일 수 있다.

- **낡음이 작으면 이점 ≈ 0.** 대부분 토큰이 t1·t4처럼 안 낡으면 `w≈1`, `r≈ρ`라 clip 동작이 World A와 거의 같다. 이때 decoupling은 **§9의 추가 forward 비용만 내고 이득 없는 순손해.** 배치에 t3 같은 토큰이 얼마나 많은지가 전부이며, 그 값이 지금 미측정.
- **음수 가능.** World A의 동결은 일종의 분산 통제/안전장치다. World B는 낡은 데이터를 **절단(=편향) IS 가중**으로 학습에 넣는다. "편향 보정으로 stale 데이터 쓰기"가 "얼리기"를 이기는지는 off-policy RL의 bias/variance/표본효율 트레이드오프이지 정리가 아니다. 순이득 ≈ (동결 안 해 얻는 표본효율) − (절단 편향 + throughput 비용).
- **GRPO라 stale-advantage는 완화.** advantage가 그룹 내 보상 통계 기반이라 critic 기반 GAE보다 정책 낡음에 덜 민감. 최악은 아니나 "보장된 이점"은 아니다.

**함의**: "일단 구현"은 정당(레버 확보). "켜면 좋아진다"는 미확정 — §11의 측정이 그 판정.

---

## 7. Partial rollout 상호작용

`async_training.partial_rollout=True`면 한 응답의 토큰이 여러 파라미터 버전에 걸쳐 생성된다. `rollout_log_probs`는 각 토큰을 뽑은 버전의 참 sampling logprob를 per-token으로 flat concat하므로, **per-token w = π_entry(t)/π_rollout(t)는 pointwise로 잘 정의**된다(값 계산엔 문제 없음). 단 "the rollout policy"가 하나의 분포가 아니라 per-token-span 혼합이라는 해석상 주의가 있다.

중요한 정량 사실: staleness 필터는 `max_global_steps`(가장 **최신** chunk) 기준이라, 유지된 RL row의 **가장 오래된 토큰은 `k_max + max_partial_span`만큼** 낡을 수 있다(`k_max`가 아니라). 즉 앞부분 토큰의 raw w가 특히 크게 나올 수 있어 **§4의 w 절단 필요성을 강화**한다. `max_partial_span`을 w 분포와 함께 보고할 것.

---

## 8. 결정

1. **decoupling(`entry` 모드)을 flag-off로 구현**한다. 기본값 `rollout` 유지 → main HPT 경로는 비트 단위로 불변(테스트로 증명).
2. **anchor = 옵션 B(recent proximal) 확정**(§9): 매 fit_step 현재 가중치로 old_log_prob forward. 옵션 A(MIS θ_sync 재연결)는 항상-최신 proximal이 아니고 가중치 스왑까지 들어 **B보다 이론·비용 양쪽에서 열등**하므로 기본 채택하지 않는다(기존 코드 참고로만). 옵션 C(A-3PO 근사)는 entry forward가 실측 병목일 때의 비용 최적화로 유보(§11 step-time 지표).
3. `entry` 모드에서 **`rollout_is` ON(절단 유지), `rollout_rs` OFF**를 기본으로 한다(§5-B). rejection은 별도 결정 사항.
4. **켜서 이득 여부는 §11 지표로 사후 판정**한다. M-first 결정 이후 §11은 착수 게이트가 아니라 M/M−dec/M−cispo/D0 델타를 해석하는 진단 장치다.
5. `Overview_RL.md` G3 / `Codemap_RL.md` anchor 서술은 **구현·기본값 전환 시점에** 갱신(지금은 rollout이 사실).

---

## 9. 구현 스코프 (flag-off, main 불변)

| # | 파일 / symbol | 변경 |
|---|---|---|
| 1 | `hpt_config.py::AsyncHptConfig` | `rl_old_logprob_source: Literal["rollout"]` → `Literal["rollout","entry"]`, 기본 `"rollout"`. `validate`에 entry 케이스 |
| 2 | `hpt_training.py::should_use_hpt_rollout_logprob_anchor` | `!="rollout"` raise 대신 분기: `entry`면 anchor 스킵하고 **옵션 B(매 fit_step 현재 가중치 forward)**로 proximal 계산, `rollout_log_probs`는 behavior weight로 보존 (A/B/C 비교는 §9 하단) |
| 3 | correction config | `entry` 모드에서 `rollout_corr.rollout_is` ON(절단), `rollout_rs` OFF 기본 |
| 4 | `hpt_training.py` 지표 | `hpt/old_logprob_from_rollout`(하드코딩 1.0)를 source 반영(rollout=1.0/entry=0.0) 또는 `hpt/old_logprob_source` 신설 |
| 5 | `tests/special_RL/test_hpt_trainer_queue_contract.py` | anchor==rollout / metric==1.0 assert 4개에 entry 변형 + entry 경로에서 `old≠rollout → IS≠1`이고 SFT가 올바로 마스킹됨을 검증하는 신규 테스트 |
| 6 | docs | 본 DR. (G3/Codemap/Overview는 구현 전환 시점에) |

### anchor 선택 — 별도 결정이다 (§1.5)

`entry` 모드의 proximal anchor는 **자동으로 정해지지 않는다.** 세 옵션이 있고, "기존 경로 재연결"은 그중 옵션 A일 뿐이다:

| 옵션 | anchor | 구현 | HCS/AReaL 정합 | 비용 |
|---|---|---|---|---|
| **A. θ_sync (MIS 재사용)** | sync 주기 첫 정책, 주기 내내 고정 | 기존 `_compute_old_log_prob` else 분기로 재연결 (신규 로직 0) | **아님** — proximal age 톱니, fresh 샘플은 coupled로 붕괴 | forward 1 + **가중치 스왑 2회/fit_step** |
| **B. recent proximal (AReaL식)** | 매 fit_step/batch-entry **직전** 정책 | old_log_prob를 **현재 가중치로 forward**(스왑 불필요) | **맞음** | forward 1회/fit_step (**스왑 없어 A보다 쌈**) |
| **C. A-3PO 근사** | `log π_prox = α·log π_behav + (1−α)·log π_θ`, α=1/d | forward 생략, 기존 텐서 산술 | 근사이나 recent 취지 부합 | ≈0 (근사오차·신규 검증 필요) |

**권장 결정 — anchor = 옵션 B (recent proximal) 확정.** 편하다는 이유로 옵션 A(기존 재연결)를 고르면 **AReaL decoupled가 아니라 MIS**를 얻는다. 역설적으로 **옵션 B가 A보다 이론 정합적이면서 비용도 낮다**(가중치 스왑이 없음) — A를 고를 유일한 이유는 "코드가 이미 있다"뿐이므로 **기본 채택하지 않는다.** 따라서 C1(M·M−cispo) anchor는 **B로 확정**(§8-2), forward 비용이 실측 병목일 때만 C(A-3PO)를 유보 옵션으로 연다. SFT self-detach와 G5(SFT correction 제외)는 세 옵션 모두에서 loss/헬퍼가 보존한다.

**세 옵션은 서로 다른 학습 역학을 낳으므로 §11 게이트·§10 ablation의 arm 정의에 어느 anchor인지 반드시 명시**한다(예: "B0(옵션 B) vs B0'(옵션 A)"는 그 자체로 별개 비교다).

---

## 10. 실험 설계 (M-first 재배향 후)

`A` vs `B` 한 방은 교란된다 — `A`는 "no correction"이 아니라 "융합(clip이 전체 ρ) + 장치② dormant"이다. 이 사실은 유지하되, 실행 설계의 기준은 `Ablation_RL.md`의 2×2로 옮긴다.

- **D0/A**: 현행 HPT (융합, rollout anchor, correction dormant). HPT-objective 비교 baseline, 그대로 유지.
- **M−cispo/B0**: decoupled(**anchor=옵션 B**, clip on `r`) + 절단 w 적용 + **rejection OFF**. → `(M−cispo)−D0` = decoupling 단독 효과.
- **M**: B0 위에 CISPO g-slot을 얹은 full stack. → `M−D0` = 헤드라인 총이득, `M−(M−cispo)` = CISPO 한계 기여.
- **M−dec**: D0 위에 CISPO만 켠 coupled+CISPO. → `M−(M−dec)` = CISPO 위에서 decoupling 한계 기여.
- **B1**: B0 + rejection ON은 이번 M에서 제외하고 후순위 ablation으로 둔다.

(anchor A vs B를 굳이 비교하려면 **별도 arm `B0'=옵션 A(MIS θ_sync)`**를 두고 `B0−B0'`로 측정 — clip anchor 종류는 decoupling on/off와 **별개 축**이다, §1.5. 단 §8-2대로 기본은 B이므로 이 비교는 선택.)

`k_max`, cliprange, mini-epochs, RL:SFT 목표비율을 arm 간 고정. **계측**: w 분포 및 w-clip-frac, r-clip-frac, post-rejection **RL:SFT 유효비율**, rows dropped, `max_partial_span`, step-time delta. RL:SFT 비율 지표 없이는 "안정성 개선"이 진짜인지 배치가 조용히 SFT-지배가 된 건지 구분 불가.

---

## 11. 사후 귀속 지표 (거의 공짜, M run에서 판독)

기존 게이트 로직은 유지하되, 지위가 바뀐다. 전면 구현의 선행 조건이 아니라 M-first run 이후 "어느 축이 실제로 값을 했는가"를 읽는 사후 귀속 지표다.

- `actor/pg_clipfrac` (`compute_policy_loss_vanilla`) — RL clip 비율(SFT는 ratio≡1이라 clip 안 됨 → 사실상 RL clip-frac).
- staleness 메타 `current_param_version − max_generation_step`.

판정(**사후 라우터로 읽을 것** — §4.5):
- **① clip-frac 낮음** → M≈D0이면 두 축 모두 이 레짐에서 조건 미충족.
- **② clip-frac 높고 staleness에 비례** → decoupling(C1)의 영역. M−dec 대비 M이 좋아지는지 확인.
- **③ clip-frac 높으나 staleness와 무관** → 문제는 gradient 死/과클립. CISPO(C2)의 영역. M−cispo 대비 M이 좋아지는지 확인.

이 지표는 §6의 "이점 존재 여부"뿐 아니라 M의 개선이 C1/C2 중 어디서 왔는지 함께 답한다.

---

## 12. DR-003과의 관계

- **DR-003 (SFT branch)**: supervised 신호는 rollout provenance가 없어 **보정도 통계적 trust region도 불요** — self-detach(ρ≡1)가 정답. 기계를 **더하지 않는다**.
- **DR-004 (RL branch)**: RL 신호는 rollout provenance가 있어 off-policy 보정이 의미를 가지며, 융합된 단일 ratio를 **decoupling으로 정련할 수 있다**. 단 이점은 조건부라 **M-first run의 사후 귀속으로 판정**한다.
- 두 DR은 provenance 원리(rollout에서 왔는가)로 대칭을 이룬다: SFT는 provenance 부재라 기계 불요, RL은 provenance 존재라 기계가 유의미하되 레짐 의존. auxiliary(entropy/KL)의 RL-only 마스킹(DR-002)도 같은 provenance 원리의 사례.

---

## 부록: 상태와 명명

§1(현행 동작)·§2~4(장치 정의·crux·이론)는 소스/문헌 검증으로 확정. §6(이점 조건부)는 미측정 사실의 정직한 진술이며 §11이 그 판정이다. **C1(config/routing/MIS-bypass)은 구현·테스트 완료**이며(2026-07-04), `Ablation_RL.md`의 M-first 결정에 따라 M full-stack의 C1로 채택되고 §11은 사후 진단으로 쓰인다. 이점 자체는 여전히 미검증(사후 귀속 대기). 파일명 `DR-004`는 DR-003(supervised branch)의 RL-branch 짝이다.
