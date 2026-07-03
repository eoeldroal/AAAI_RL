# DR-003. Off-Policy Treatment of the Supervised Branch — Why It Needs Neither Correction nor a Statistical Trust Region

Status: 분석부 확정 · self-detach 유지 구현됨 (`feat/async-hpt-branch-blind-loss`) · 안정성 가설 미검증(사전 계측 후 ablation 대상)
범위: mixed batch에서 supervised branch(τ*)의 off-policy 처리 — 보정(IS)과 제동(trust region)의 필요 여부. aggregation/정규화는 DR-001, auxiliary 정칙화는 DR-002 소관.
관련 코드: `verl/workers/utils/losses.py::ppo_loss`, `verl/trainer/ppo/rollout_corr_helper.py`
전제: multi-step minibatch 학습.
개정 이력: 본 DR의 초판은 SFT에 PPO-style clip("벨트")을 채택했으나, 그 정당화가 오류임이 확인되어 재작성됨(§6).

---

## 구현 기록

- 완료: DR-001 refactor 후에도 SFT row의 `old_log_prob = log_prob.detach()` self-detach 경로 유지.
- 완료: SFT row는 rollout correction/staleness의 대상이 아니라는 기존 계약을 유지하는 테스트 보존.
- 완료: obsolete B_eff 계열 loss field가 들어오면 loss/monitoring 경로에서 fail-fast.
- 미구현: belt / entry snapshot. 사전 계측에서 SFT-induced drift가 확인되기 전까지 config surface에도 넣지 않는다.
- 검증: `/home/sogang_nlpy/miniconda3/envs/RL/bin/python -m pytest tests/special_RL -q` 통과.

---

## 0. 한 문단 요약

RL의 clip은 두 근거 위에 선다 — importance weight 분산 억제와, π_old 하에서 추정된 advantage의 국소 유효성 유지. supervised branch(τ*)에는 이 두 근거가 **모두 부재**한다. τ*는 생성 정책이 없어 importance sampling 자체가 없고(→ correction 공집합), advantage가 추정값이 아니라 부여된 상수 β라 정책이 멀어져도 낡지 않는다(→ 통계적 trust region 불요). 따라서 SFT의 올바른 처리는 self-detach(ρ≡1, 순수 cross-entropy)이며, 이는 결함이 아니라 A.4가 도출한 정확한 설계다. **RL과의 이 비대칭(보정도 통계적 제동도 불요)이 이 DR의 분석적 기여다.** 단, 이와 독립적으로 최적화 안정성 우려가 있다 — τ*의 dense-coherent gradient가 batch당 파라미터 이동을 크게 만들어 비동기 rollout을 낡게 할 수 있다. 이 우려에 대한 가능한 처치로 벨트(entry snapshot + clip)를 **drift-pacing ablation 후보**로 남긴다(미검증 가설). 벨트는 "놓친 trust region의 복원"이 아니라 "측정으로 판정할 안정화 처치"이며, 사전 계측으로 전제가 확인되기 전에는 현재 구현 범위에 넣지 않는다.

---

## 1. 배경: decoupled view와 기여 경계

off-policy 처리의 기계는 이 DR의 것이 아니다. 경계를 먼저 못박는다.

- **TRPO(2015) → PPO(2017)**: clip은 태생이 trust region(KL 제약)의 값싼 근사이지 IS 보정 장치가 아니다. 동기에서 생성=직전 정책이라 하나의 ratio가 두 역할을 겸했을 뿐.
- **Decoupled Loss(Hilton, Cobbe, Schulman 2022)**: π_old의 두 역할(보정=behavior policy, 제동=proximal policy)을 명시적으로 분리. AReaL이 LLM 비동기에 적용.
- **A-3PO(arXiv 2512.06547)**: proximal policy를 log-linear 보간으로 근사(`α=1/s`, staleness-aware).
- **SAPO(arXiv 2511.20347)**: hard clip을 smooth gate로. 순수 RL의 IS variance 겨냥.

이 문헌들은 전부 **순수 RL**이며, off-policy를 version lag로 다룬다. 이 DR은 그 decoupled view를 분석 도구로 **채택**하되, 기여는 supervised branch를 그 틀에 넣었을 때 드러나는 **비대칭**(§3–4)과, 그로부터 갈라지는 안정성 가설(§5)이다.

---

## 2. decoupled view: 하나의 ratio, 두 역할 (채택)

정책 경사 loss의 ratio $\rho = \pi_{\text{now}}/\pi_{\text{ref}}$에서 분모 π_ref가 **무엇이냐**에 따라 두 역할이 갈린다.

**역할 A · 보정(correction).** 데이터를 만든 정책과 지금 정책의 빈도 차이를 곱셈 가중치로 교정. 기준 = 생성 정책. 빈도 교정이므로 (1) batch 시작에 한 번 동결하고 (2) 자르면 편향이다.

**역할 B · 제동(trust region).** 이번 학습에서 정책이 너무 멀리 못 가게 제한. 기준 = batch 진입 시점 정책. clip 대상이다.

동기에서는 생성=직전=entry라 하나로 충분했다. 비동기·multi-step에서 갈라진다. 이 분해가 decoupled loss이고, RL에는 유효하다. 이하는 이 두 역할을 SFT에 각각 물었을 때 무슨 일이 일어나는지다.

---

## 3. supervised branch를 두 역할 앞에 세우면 — 이중 부재

### 3.1 역할 A(보정) — 공집합

τ*는 우리 정책의 rollout이 아니라 고정 target(정답 혹은 teacher 산출물)이다. "이 데이터를 만든 우리 정책"이 없으므로 빈도 교정의 대상이 아니다. 따라서 **보정 가중치 = 1**(곱해도 무변). 이것이 A.5의 "SFT는 IS correction 면제"의 근거이며, 특혜가 아니라 대상의 부재다. 코드에 이미 반영되어 있다. — **확정, 구현됨.**

### 3.2 역할 B(통계적 trust region) — 근거 부재

RL이 제동(clip)을 거는 통계적 근거는 둘이다.

1. **IS 분산.** rollout을 딴 정책이 만들었으니 importance weight가 튀면 분산이 폭발 → clip으로 억제.
2. **advantage staleness.** RL의 advantage는 π_old 하에서 *추정*한 값이라, 정책이 멀어지면 추정이 낡음 → clip이 정책을 π_old 근처에 묶어 추정을 유효하게 유지.

τ*에는 둘 다 없다. correction이 공집합이라 **importance weight 자체가 없고(근거 1 부재)**, advantage가 추정값이 아니라 부여한 상수 β라 **정책이 멀어져도 낡지 않는다(근거 2 부재)**. 즉 RL에서 clip을 정당화하는 두 기둥이 SFT에는 모두 이전되지 않는다. "RL이 clip을 쓰니 SFT도"는 논증이 아니다.

이 관찰이 **"off-policy but not stale"**의 정확한 의미다. τ*는 version lag=0인데 분포거리는 크다. 그러나 분포거리가 크다는 것이 "통계적 제동이 필요하다"를 뜻하지 않는다 — 통계적 제동은 추정의 낡음을 막는 장치인데, 부여 상수 β는 낡지 않기 때문이다. 따라서 lag로 keying하는 기계(A-3PO 등)가 SFT에 안 맞는 것은 물론이고, 분포거리로 keying하는 것조차 통계적으로는 불필요하다.

### 3.3 결론: self-detach는 결함이 아니라 정답

`losses.py::ppo_loss`의 self-detach는 SFT의 ρ를 항상 1로 만든다. gradient는 $\beta\nabla\log\pi_\theta$ — 순수 cross-entropy의 β배. 이는 A.4가 도출한 "SFT를 정확한 CE로 만드는 설계"이며, 정답을 그대로 모방한다는 SFT 목적의 달성 상태다. ρ≡1은 "clip이 안 걸리는 고장"이 아니라 **의도된 정상 동작**이다.

| | 역할 A (보정) | 역할 B (통계적 제동) |
|---|---|---|
| RL rollout | 생성 정책 (필요) | entry snapshot (필요: IS분산 + staleness) |
| SFT τ* | **공집합 (weight 1)** | **불요 (두 근거 모두 부재)** |

SFT 행은 두 칸이 다 비어 있다. RL과의 이 이중 비대칭이 분석적 기여이며, 순수 RL 문헌은 다룰 수 없는 자리다(SFT branch가 없으므로).

---

## 4. 그럼 원래의 "SFT가 IS를 키운다" 우려는? — 층위 정정

최초 우려는 "SFT가 파라미터를 움직여 rollout IS가 커진다"였다. 이는 실재하지만 **RL rollout 측의 문제**이고 **RL 측 correction이 처리**한다. 무엇이 drift를 일으켰든(SFT든 RL이든), 낡는 것은 rollout provenance를 가진 RL 데이터이고 그 보정은 RL 축이 한다. 여기에 "SFT 자체에 벨트를 걸어 drift를 예방"하는 것은 RL 처리와 중복이며, SFT 학습을 깎는 방식이다. drift가 실제로 문제라면 직접 손잡이(β, lr, sync 주기, γ로 SFT 비율)로 다루는 것이 우선이다. "SFT가 RL을 stale하게 만든다"(참)에서 "SFT가 자기 통계적 벨트를 필요로 한다"(거짓)로의 비약을 경계한다.

---

## 5. 독립 경로: 최적화 안정성과 drift-pacing 가설 (미검증)

§3은 **통계적** 근거로 벨트를 기각했다. 그러나 이와 **독립적인** 우려가 하나 남는다 — 최적화 안정성. 이는 통계적 유효성과 다른 명제이며, 벨트의 유일하게 정당한 가능 근거다.

**메커니즘.** τ*의 토큰당 gradient는 유계(CE)지만, **전 토큰이 같은 방향으로 조밀하게** 밀기 때문에 batch당 순 파라미터 이동이 branch 중 가장 크다. 공유 파라미터라 이 이동은 (i) 다른 능력에 간섭하고, (ii) 비동기에서 in-flight rollout을 낡게 만들어 IS 팽창 → 유효 배치 축소의 외부효과를 낸다. "통계적으로 유효(not stale)"와 "최적화적으로 안전"은 다른 문제다.

**벨트의 정확한 성격 — 감쇠가 아니라 pacing.** 이 우려에 대한 도구로서 벨트(entry snapshot 기준 + clip)는 원조 reshape와 다르다. reshape(π/(π+c))는 먼 토큰의 학습 force를 **영구 감쇠**시켜 HPT 목적(실패 영역 강학습)을 훼손한다. 벨트는 방향과 힘을 CE 그대로 두되 **batch당 이동에만 예산**을 둬서 여러 batch에 분산시킨다(pacing). 먼 target도 결국 학습되며, batch당 drift만 상한된다. source-side 개입 중 목적 훼손이 가장 적은 형태다.

**그러나 미검증이며 기본값이 아니다.** 더 강하게 말하면, 벨트는 현재 phase의 구현 대상이 아니라 **사전 계측 이후에만 열 수 있는 ablation 후보**다. 세 가지 이유로 self-detach가 기본, 벨트가 도전자다.
- **입증 책임은 기계를 추가하는 쪽에.** 벨트는 entry snapshot forward 비용, A.4의 CE 등가성 서사 포기, ρ 증폭(1~1+ε 구간 CE 크기 최대 ε 왜곡)을 요구하는데, 사는 것(batch당 이동 상한)의 효용은 미측정이다(DR-001의 "복잡성은 제 값을 해야 한다" 원칙).
- **벨트가 무는 레짐이 불확실.** ppo_epochs=1 + 소수 minibatch에서는 SFT 토큰이 entry에서 벌어질 기회가 적어 벨트가 거의 inert하다. 반복 갱신 상한은 ppo_epochs>1에서만 실질적이다.
- **더 직접적 손잡이 존재.** β, lr, grad-clip, sync 주기, γ. 벨트의 고유 능력(토큰별 적응적 이동 예산)이 필요하다는 것이 관측되기 전에는 도입 이유가 없다.

따라서 구현 순서는 다음으로 고정한다.

```text
1. 기본 경로는 self-detach CE로 유지한다.
2. main/self-detach run에서 SFT drift와 RL IS 팽창을 먼저 계측한다.
3. SFT가 실제로 async drift를 크게 만든다는 증거가 있을 때만
   entry snapshot + clip 옵션을 별도 ablation 구현 대상으로 승격한다.
```

---

## 6. 개정 사유 — 기각된 정당화들의 정정

본 DR 초판은 SFT에 벨트를 채택하며 다음 정당화를 썼고, 모두 오류다.

- **"ρ≡1은 놓친 trust region이다" → 오진.** ρ≡1은 A.4가 도출한 순수 CE 설계이지 결함이 아니다(§3.3). 정상 동작을 고장으로 오진하고 그 위에 치료법을 세웠다.
- **"B축 = RL 복제" → 근거 이전 실패.** RL clip의 두 근거(IS 분산·advantage staleness)가 SFT엔 둘 다 없다(§3.2). 유비만으로 벨트를 이식했다.
- **"off-policy but not stale"을 상수 ε의 근거로 사용 → 정반대.** 이 사실의 올바른 귀결은 "통계적 trust region 불요"이지 "벨트의 ε을 상수로"가 아니다. 논거가 결론을 반박하고 있었다.
- **asynchrony validity(A.5) 프레임에 편입 → 분류 오류.** 벨트가 무는 지점은 batch 재사용 시 반복 갱신이지 비동기 staleness가 아니다.

정정의 뿌리는 하나다 — **RL 직관을 supervised 신호에 근거 없이 이식.** 이는 이 프로젝트에서 반복된 실수 패턴(gradient 폭주 오진, reshape 채택, scale 동등 요구)의 또 다른 인스턴스였다. supervised 신호는 off-policy RL의 관습을 자동 상속하지 않는다는 것이 교훈이다.

기각된 그 외 대안:
- **teacher를 behavior policy로.** τ* 생성 teacher의 logprob를 보정 기준으로 쓰는 안. 기각 — teacher 분포 모방이 아니라 τ* 결과물을 배우는 것이고, 데이터셋 정답인 경우 teacher 자체가 없다.

---

## 7. 후속 실험: 벨트 on/off ablation (drift-pacing 가설 판정)

§5의 가설은 실험으로만 판정된다. 다만 이 실험은 즉시 수행하는 기본 실험이 아니라, **사전 계측에서 drift-pacing 전제가 보였을 때만 여는 후속 ablation**이다. 벨트 on/off 비교는 어느 결과든 기여가 된다 — 벨트가 이기면 "hybrid에 supervised drift pacing이 필요"라는 발견, 무승부/CE승이면 "순수 CE로 충분, 통계적·최적화적 제동 모두 불요"라는 §3 비대칭의 실증.

**공정성 조건(필수).** 벨트가 발동 가능한 레짐에서 비교해야 한다. 다수 minibatch(require_batches>1), 가능하면 ppo_epochs>1 축 포함. **벨트 발동률(ε 포화 비율)을 반드시 함께 보고** — "벨트가 살아 있었는데도" 차이가 있었나/없었나를 보여야 실험이 유효하다. inert한 세팅의 무승부는 증거가 아니다.

**사전 관문 측정치.** 벨트 구현 전에 먼저 봐야 할 값은 SFT row 비율, SFT-heavy batch 직후 grad-norm spike, rollout IS 분포(SFT 비율로 층화), 유효 배치 크기, stale/drop 증가, batch당 SFT drift 점유율이다. 이 값들이 SFT 기인 drift를 가리키지 않으면 벨트 ablation은 열지 않는다.

**ablation 측정치.** 전제가 관측된 뒤 벨트를 구현해 비교할 때는 최종 성능 + rollout IS 분포 + 유효 배치 크기 + grad-norm 스파이크 빈도 + batch당 SFT drift 점유율 + 벨트 발동률을 함께 보고한다.

**우선순위(컴퓨트 유한).** main(self-detach) → DR-001 β_r 축 → **벨트 on/off** → DR-001 aggregation mode 축. 벨트를 aggregation 재현 실험보다 앞에 두는 이유는 이 논문 고유 질문이기 때문이고, β_r보다 뒤인 이유는 β_r가 main 설정을 정하는 선행 실험이기 때문이다.

**서사 포지션.** 기본 설계 = self-detach(§3 원리 분석이 뒷받침). 벨트 = "놓친 안전장치"가 아니라 "drift-pacing 가설의 처치군". 이렇게 두면 어느 결과가 나와도 방법이 무너지지 않는다 — 분석(비대칭)과 실험(pacing 효용)이 서로 독립적으로 선다.

---

## 8. 재사용 과적합 (운영 노트, 잔여)

dataset epoch 수준에서 같은 τ*를 반복 노출하면 과적합. 이는 벨트(batch 내부 도구, 매 entry 리셋)의 관할 밖이고, SFT/imitation의 일반 상식이라 논문 기여가 아니다. 처방은 dataset 수준 노출 제어 — `remove_sfted_data`(원조 config에 존재) 또는 τ* 노출 상한. PPO inner-epoch 수준은 논점이 아님(batch 내부는 서로 다른 데이터, 바퀴 간 반복은 PPO 표준이며 RL도 동일, ε 상한 존재).

---

## 9. 후속 작업 / 전제

- **현재 코드 변경 없음.** self-detach 유지. 벨트는 지금 구현하지 않는다.
- **옵션 구현 조건.** 사전 계측에서 SFT drift / RL IS 팽창 / 유효 배치 축소가 실제로 보일 때만, 벨트를 ablation용 옵션으로 별도 구현한다. 그때의 기계는 entry snapshot = SFT old_log_prob를 첫 forward에서 캐시해 동결하는 방식이며, 기본값은 off다.
- **multi-step 전제.** 단일 step에서는 벨트가 무력(진입=현재, ρ≡1)이라 ablation 자체가 무의미. 벨트 실험은 multi-minibatch 레짐에서만.
- **계측이 우선.** §7 측정치는 벨트 ablation 이전에도 main run에서 수집 — SFT drift 점유율·IS 팽창이 애초에 관측되지 않으면 벨트 가설의 전제(§5 메커니즘)부터 성립 안 함. 즉 계측이 ablation의 사전 관문이다.
- **DR-001/002 정합.** self-detach 유지이므로 DR-001 §5의 "살아남는 유일한 커스텀 한 줄 = self-detach"가 그대로 유효(초판이 예고한 교차참조 수정 불요). DR-002의 provenance 원리는 §3.1이 본체.

---

## 부록: 상태와 명명

§3(이중 비대칭 분석)은 확정. §5(drift-pacing)는 미검증 가설이며 사전 계측과, 필요할 경우에만 열리는 §7 ablation으로 판정된다. 이 DR의 핵심 산출물은 **"supervised branch는 보정도 통계적 trust region도 필요로 하지 않는다"는 비대칭**이며, 벨트는 그와 독립된 최적화 안정성 가설의 후속 실험 처치로 강등·재정의되었다. 현재 구현 범위에서는 self-detach만 유지하고 벨트는 구현하지 않는다.
