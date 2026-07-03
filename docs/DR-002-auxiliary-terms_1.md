# DR-002. Auxiliary Terms (Entropy / KL) on the Supervised Branch

Status: 확정 (phase-1) · 미검증 (관련 코드 현재까지 미실행)
범위: Async-HPT actor loss의 auxiliary 정칙화 항(entropy, KL)이 branch별로 어떻게 적용되는가. off-policy 보정/trust region은 DR-003 소관.
관련 코드: `verl/workers/utils/losses.py` (entropy/kl 계산 277–296), `verl/experimental/fully_async_policy/hpt_config.py`
선행 예고: DR-001 §7에서 "KL/entropy가 SFT row에 균일 적용되는 문제는 이 결정과 독립"으로 분리 예고됨.

---

## 0. 한 문단 요약

현재 loss는 entropy와 KL 항을 `response_mask` 전체에 적용하므로 SFT row도 이 정칙화를 받는다(losses.py:277–296, detach 바깥). 결정은 **auxiliary 항을 RL row에만 적용하고 SFT row는 마스킹**하는 것이다. 근거는 provenance 원리(A.5, DR-003 §): entropy·KL은 rollout provenance를 가진 데이터에만 의미를 갖는 정칙화이며, SFT의 목적(고정 target을 확신 있게 모방)과 상충한다. entropy는 RL에 표준값(0.001), SFT에는 끈다. KL은 현 config가 reference model 자체를 띄우지 않아(`ref.use_ref=False`) 이 설정에서는 애초에 non-issue이나, 개념적으로도 SFT에 anchor-KL을 거는 것은 학습 방해라 면제가 맞다. 처방은 새 기제가 아니라 correction 면제(rollout_corr_helper:1061)와 동형의 branch 마스킹이다.

---

## 1. 두 종류의 KL을 먼저 구분한다

"KL을 SFT에 거느냐"는 질문은 두 개의 다른 KL을 뭉치면 답이 안 나온다.

- **anchor-KL** `D_KL(π_θ ‖ π_init)`, 고정 reference. 정책을 초기값에 묶는 정칙화. → SFT의 목적(실패 영역에서 정책을 **크게 끌어냄**)과 정면 충돌. 실패 영역일수록 τ*는 정책에서 멀고, anchor-KL은 바로 그 큰 이동을 가장 세게 억제한다. **SFT에 유해, 면제.**
- **proximal-KL** `D_KL(π_θ ‖ π_batch-entry)`, 매 update 갱신되는 reference. 정책이 한 번에 너무 못 가게 하는 trust region. → 학습 **방향**은 안 막고 per-update **보폭**만 제한. 이것은 정칙화가 아니라 trust region이며 **DR-003의 소관**이다. DR-003은 SFT에 대해 이를 기본 채택하지 않는다(self-detach 기본, 벨트는 미검증 ablation 옵션). 어느 쪽이든 anchor-KL과는 별개 항이다.

이 DR이 다루는 KL은 anchor-KL이다. proximal-KL(=trust region)의 처리는 DR-003 소관이며, 갱신된 DR-003은 SFT에 이를 기본으로 부여하지 않는다(self-detach 기본, 벨트는 미검증 ablation 옵션). "anchor-KL을 끈다"와 "proximal trust region"은 모순이 아니라 서로 다른 항이며, 후자의 채택 여부는 DR-003에서 별도로 판정된다.

현 config 실측: `use_kl_loss=False`, `kl_loss_coef=0.00`, `kl_ctrl.kl_coef=0.000`, `ref.use_ref=False`. reference model이 없으므로 anchor-KL은 계산 대상 자체가 없다. 즉 **이 config에서 KL은 non-issue**이며, 이 결정은 "reference를 켜는 config로 갈 때"의 개념적 방침으로 유효하다.

---

## 2. entropy: RL에만, SFT에는 끈다

원조 HPT는 `entropy_coeff=0.001`을 쓴다. 그러나 이것을 "SFT에도 켜자"의 근거로 삼는 것은 오독이다. 네 논거:

1. **원조도 SFT에는 안 걸었다.** 원조 HPT는 `α·L_RL + β·L_SFT`로 두 loss를 **분리**하는 구조라, entropy 항은 policy-gradient(RL) 항에 붙는다. SFT cross-entropy에 entropy 보너스를 더하지 않는다. 우리 구현은 두 loss를 하나의 경로로 통합했으므로, `response_mask` 전체에 entropy를 걸면 원조에 없던 일(SFT token에 entropy)이 생긴다. 즉 SFT에 entropy를 켜는 것은 원조 계승이 아니라 통합 구현의 부작용이다.
2. **config 규율.** HPT run과 baseline run이 동일한 entropy_coeff를 써야 순수 RL batch의 bit-identity가 성립하며, 이는 SFT 마스킹과 독립이다(순수 RL batch엔 SFT row가 없으므로 SFT 처리가 bit-identity에 영향을 주지 않는다). 이 repo의 baseline은 현재 entropy_coeff=0이므로, RL entropy를 켜려면 baseline도 함께 맞춰야 한다.
3. **SFT 목적과 상충.** entropy 보너스는 분포를 평평하게 만든다. 그런데 τ*를 SFT하는 목적은 "이 토큰을 확신 있게 내라"이다(뾰족하게). 게다가 τ*는 실패 영역의 소수 샘플이라, entropy로 학습을 무르게 하면 부족한 supervised 신호가 더 약해진다.
4. **β 커플링.** SFT의 학습 크기는 β_r(pseudo reward)로 조절되는데, entropy 항은 advantage 경로가 아니라 별도 항(−c_ent·H)이라 β로 스케일되지 않는다. 따라서 β를 키우면 entropy의 상대적 영향이 작아지고 β를 줄이면 커진다. SFT에 entropy를 켜면 β 튜닝이 의도치 않게 entropy 상대강도를 흔든다.

결론: **entropy는 RL row에 0.001, SFT row에 0.** 로컬 코드엔 entropy 설정이 아예 없으므로, RL 쪽 0.001 배선과 SFT 마스킹을 함께 가져간다.

예비: SFT-heavy 레짐(gate가 SFT에 고착되는 OSWorld류)에서 τ* 과적합이 실측되면, SFT entropy를 **ablation 축**으로 열 수 있다. 단 기본값이 아니라 실험 항목이다.

---

## 3. 구현: auxiliary 항의 SFT row 마스킹

losses.py:277–296이 entropy·kl을 `response_mask`로 계산한다. 이를 `response_mask & ~sft_mask`로 바꾸면 RL row에만 걸리고 SFT는 면제된다. 이는 correction 면제(rollout_corr_helper:1061의 `torch.where(sft_token_mask, ones, weights)`)와 **동형 패턴**이다.

원리적 정당화는 provenance(A.5, DR-003 §3): correction·staleness가 rollout provenance를 가진 RL row에만 의미를 갖듯, entropy(탐색 정칙화)·anchor-KL(앵커링)도 rollout policy 기준의 항이라 rollout provenance가 있는 RL row에만 의미가 있다. 즉 이 마스킹은 새 예외가 아니라 **provenance 원리가 auxiliary 항으로 확장되는 사례**다. DR-003이 이 원리의 본체(보정·staleness 면제의 이중 부재)를 세우고, 이 DR이 그것을 인용한다.

---

## 4. config 정합과 재현성

- RL: `entropy_coeff=0.001`, `use_kl_loss=False`(현 phase-1). KL을 켜는 config로 갈 경우 anchor-KL은 RL에만.
- SFT: entropy·KL 마스킹.
- **HPT 전용 example config 부재(별건, 재현성 결함).** 현재 shell/의 것들은 base async RL이라 `async_hpt` 블록이 없다. 의도된 auxiliary 설정을 보여주는 실행 가능한 config를 최소 하나 ship해야 한다. 마스킹을 넣으면 config에서 KL/entropy를 억지로 끌 이유가 사라져 더 깔끔해진다.

---

## 5. 정정 이력

- **validator 강제 → 폐기.** 초기에 "HPT면 use_kl_loss/entropy_coeff를 validator로 강제"를 제안했으나, 그것은 전역 스위치라 RL의 정당한 auxiliary까지 함께 죽인다. 개념적으로 옳은 해법은 config 강제가 아니라 **branch 마스킹**이다(RL은 유지, SFT만 면제).
- **"entropy를 비교 위해 SFT에도 켠다" → 기각.** 비교 가능성이야말로 SFT에서 끄라고 말한다(원조 미적용 + SFT 목적 상충). §2-1,3 참조.

---

## 부록: 상태

이 결정은 설계 방침이며 미검증이다(코드 미실행). KL은 현 config에서 non-issue라 즉시 검증 대상은 entropy 마스킹뿐이다. entropy on/off(SFT)는 §2 예비대로 SFT-heavy 레짐에서 ablation 후보로 남긴다.
