# DR-001. Loss Normalization, the Removal of B_eff, and Branch-Blind Reduction

Status: 확정 (phase-1) · 미검증 (관련 코드 현재까지 미실행)
범위: Async-HPT actor loss의 reduction 계층. routing / learner contract / reference 계약은 불변.
관련 코드: `verl/workers/utils/losses.py`, `verl/experimental/fully_async_policy/hpt_assembler.py`, `verl/trainer/ppo/core_algos.py`, `verl/workers/engine/*/transformer_impl.py`

---

## 0. 한 문단 요약

Async-HPT의 초기 구현은 mixed RL/SFT batch의 loss를 `B_eff = Σ w_r`라는 **batch 단위 별도 분모**로 정규화했다. 이 한 결정이 서로 무관해 보이는 네 가지 복잡성(minibatch 정규화 왜곡, surrogate 재구현, 방어 코드, 정규화 철학 분열)의 단일 근원이었다. 결정은 **별도 분모를 제거하고 동기 verl의 표준 정규화(`loss_agg_mode`)로 회귀**하는 것이다. aggregation mode는 `seq-mean-token-sum-norm` + `loss_scale_factor = L_max`로 두되, 이 선택은 **RL branch의 인센티브 제약(Dr.GRPO)이 단독으로 결정**한다. supervised branch의 trajectory별 학습 예산은 aggregation이 아니라 **per-row pseudo reward `β_r`(reward 채널)**로 표현된다. 그 결과 reduction은 branch의 존재를 모르는 **branch-blind** 계층이 되고, branch semantics는 advantage와 reference 두 지점으로 완결된다. 정확한 prompt-equal은 방법의 요구가 아니라 구현이 스스로 부과한 요구였으므로 포기하고 근사 균형을 채택한다. 특히 gradient 스케일 동등은 목표가 아니다 — SFT는 실패 영역의 off-policy 감독이라 큰 gradient가 의도된 동작이며, β는 그것을 RL에 맞추는 보정기가 아니라 개입 강도를 정하는 손잡이다.

---

## 1. 배경: 무엇을 풀려던 결정이었나

HPT batch는 한 prompt가 branch에 따라 다른 row 수로 펼쳐진다.

- RL prompt → n개 rollout row
- SFT prompt → 1개 verified-trajectory row

정규화 없이 두면 RL prompt가 SFT prompt보다 n배 크게 기여한다. 초기 구현은 이 비대칭을 "prompt 단위 동등 기여(prompt-equal)"로 바로잡으려 했고, 그 수단으로 per-row weight와 batch 분모를 도입했다.

- per-row weight: `w_r = α/n` (RL), `w_r = 1` (SFT)
- batch 분모: `B_eff = Σ w_r` (`finalize_loss_denominator`)
- 한 prompt의 총 기여 = RL이면 `n·(α/n) = α`, SFT면 `1`. α=1이면 `B_eff = batch 내 prompt 수`.

**발상 자체는 옳았다.** prompt 단위 균형은 깔끔하고 납득 가능한 목표다. 문제는 목표가 아니라, 그것을 **별도 분모라는 기계**로 구현한 선택에 있었다. 이하는 그 선택이 왜 문제였는지(§2), 그 하나가 어떻게 네 겹의 복잡성을 낳았는지(§3), 그래서 어떤 결정에 이르렀는지(§4)의 순서다.

---

## 2. 결함: batch-level 분모와 step-level 학습의 grain 불일치

### 2.1 동기 verl의 정규화 관례

먼저 기준선을 확정한다. 비동기 trainer는 동기 경로를 **상속**하므로(`FullyAsyncTrainer(SeparateRayPPOTrainer)`), 업데이트 정규화도 동기 관례를 따라야 한다. 비동기가 더하는 것은 per-token IS correction과 rollout-anchored old-logprob뿐, 별도 업데이트 체계가 아니다.

동기 경로의 모든 분모는 **optimizer step이 실제로 소비하는 데이터의 크기**로 잡히며, "global"은 batch 전체가 아니라 DP rank 전체를 뜻한다.

- `global_batch_size`: driver가 `ppo_mini_batch_size`로 설정 (이름과 달리 minibatch 크기).
- `batch_num_tokens`: engine이 minibatch 진입 시점에 `loss_mask.sum()` 후 DP all-reduce (해당 step의 토큰 수).
- `dp_size` 곱: FSDP의 gradient DP 평균을 상쇄해 "DP 전체 합 ÷ DP 전체 분모"를 만든다.

즉 동기 verl은 "조각을 그 조각 크기로 나눈다". `ppo_mini_batch_size`는 batch당 정상 크기 step을 몇 번 밟을지 정하는 손잡이이고, `ppo_epochs`는 그 step들을 몇 바퀴 반복할지 정한다.

### 2.2 B_eff의 이탈

`hpt_loss_denominator`(B_eff)만이 **batch 전체 단위로, driver에서, sharding/split 이전에** 계산되는 유일한 분모다. loss는 각 minibatch를 이 batch-level 값으로 나눈다. batch를 M개 minibatch로 쪼개면 각 step은 자기 몫(1/M)만 담고도 전체로 나누므로, **step 크기가 1/M로 축소**된다.

이는 가정이 아니라 설계상 반드시 발현된다. 비동기 trainer는 라운드마다 minibatch의 `require_batches`배를 모아 조립한다(`required_samples = ppo_mini_batch_size * require_batches`). `require_batches > 1`이면 조립 batch는 정의상 여러 minibatch로 쪼개진다.

### 2.3 수치 예시

batch = prompt 4개(P1~P4), α=1, 각 prompt의 loss 기여를 8·4·6·2라 하자. `B_eff = 4`.

| 방식 | step 구성 | step 크기 | step당 총량 |
|---|---|---|---|
| ① 단일 minibatch (부록의 정의) | 1 step, ÷4 | 5 | 5 |
| ② 코드: 2 minibatch, 항상 ÷4 | 2 step | 3, 2 | 5 (= ①을 2회로 분할) |
| ③ 표준: 2 minibatch, 각자 ÷2 | 2 step | 6, 4 | 10 (= ①의 2배) |

②의 각 step은 ③의 절반이다. 결과적으로 `ppo_mini_batch_size`를 줄여도 batch당 학습량이 늘지 않으며(손잡이 무력화), 동일 config에서 HPT가 non-HPT baseline보다 조용히 M배 적게 학습한다(baseline 비교 오염).

`losses.py`의 `hpt_loss_denominator.max()` + allclose 검사는 microbatch 내 균일성만 보증할 뿐, 분모가 해당 step의 실제 weight 합과 일치하는지는 검사하지 않아 이 불일치를 잡지 못한다. 코드가 한 번도 실행된 적이 없어(모든 커밋 "no tests run") 표면화되지 않았다.

---

## 3. 진단: 하나의 결정이 낳은 네 겹의 복잡성

§2의 grain 불일치는 증상 하나에 불과하다. "정규화를 별도 분모로 구현한다"는 단일 결정이, 서로 무관해 보이는 네 가지 복잡성의 공통 근원이다.

1. **grain 불일치** (§2). 분모가 batch-level이라 minibatch step마다 어긋난다.
2. **surrogate 재구현**. prompt-equal reduction은 per-row weight/divisor를 적용하기 위해 **집계 이전의 token-level loss**를 손에 쥐어야 한다. 그런데 표준 `get_policy_loss_fn("vanilla")`은 내부에서 이미 loss를 집계해 스칼라로 돌려준다. 그래서 HPT는 동일한 PPO surrogate를 `_compute_vanilla_token_losses`로 **따로 재구현**할 수밖에 없었다(앞선 검토의 φ_PPO 지적). method의 핵심 명제("SFT가 RL 코드 경로를 그대로 탄다")가 코드 층위에서는 "별도 재구현본을 함께 탄다"로 약화됐다.
3. **방어 코드**. batch-level 분모를 per-row로 broadcast했으므로, microbatch마다 그 값이 균일한지 확인하는 `hpt_loss_denominator.max()` + allclose가 필요해졌다. 분모가 애초에 step-grain이면 이 확인 대상 자체가 없다.
4. **정규화 철학 분열**. 별도 divisor를 branch별로 다르게 잡으면서 SFT=token-mean(자기 길이) vs RL=상수 divisor의 비대칭이 생겼다. 두 branch가 서로 다른 길이 정규화 철학을 갖게 된 것이다.

이 넷은 별개의 결함이 아니라 한 결정의 그림자다. 그러므로 각각을 개별 패치하는 대신 **근원(별도 분모)을 제거**하면 네 갈래가 동시에 닫힌다 — 이것이 결정의 논리적 출발점이다.

---

## 4. 결정: 별도 분모를 제거하고 표준 정규화로 회귀

**`loss_agg_mode = "seq-mean-token-sum-norm"`, `loss_scale_factor = L_max`. mode는 RL이 결정, SFT 배분은 `β_r`가 담당, reduction은 branch-blind.**

이 결정에 이르는 논증은 네 단계다: 무엇을 포기하는가(§4.1) → mode는 무엇이 정하는가(§4.2) → 그 강제가 SFT에 비용인가(§4.3) → 그래서 무엇이 사라지는가(§4.4).

### 4.1 정확한 prompt-equal을 포기한다

B_eff를 제거하려면 먼저 그것이 지키려던 "정확한 prompt 단위 균형"을 포기해야 한다. 근거는 넷이다.

- 부록 어디에도 정확한 prompt-equal이 invariant로 선언돼 있지 않다. 핵심은 gate / contract / objective 세 invariant뿐이다.
- 정확한 1:1은 RL advantage 크기와 길이에 의존하므로 어떤 전역 상수로도 달성 불가하다.
- 원조 HPT의 `α·L_RL + β·L_SFT`도 정확 균형이 아니라 상수 계수로 균형을 잡았다. 근사 균형은 방법의 본성에 부합한다.
- **gradient 스케일 동등은 애초에 추구할 대상이 아니다** (아래).

**"동등 기여"가 겨눈 것이 무엇인지 정확히 해야 한다.** prompt-equal은 row-count 축의 회계일 뿐, 두 branch의 gradient 크기를 같게 만드는 것이 아니며 그럴 이유도 없다. SFT row가 나르는 것은 실질적으로 NLL이고, gate가 그 prompt를 SFT로 부른 이유는 정책이 거기서 전멸했기 때문이다. 즉 τ*는 정의상 정책 분포에서 먼 off-policy 감독이고, 멀수록 NLL과 gradient가 크다 — 이것은 보정할 불일치가 아니라 **실패 영역에 강한 교정을 넣는다는 HPT의 목적 그 자체**다. 이를 RL 신호의 O(1) 스케일에 억지로 맞추면 가장 배워야 할 곳의 신호를 깎는 셈이 된다. "모든 신호를 O(1)로 정규화"는 on-policy RL(mean-center + std)의 관습이지 supervised 신호의 요구가 아니다. 하나의 estimator로 통합했다는 것은 **코드 경로의 통합이지 신호 크기의 균질화가 아니다.**

따라서 β의 역할도 "SFT를 RL 스케일에 맞추는 보정기"가 아니라 **"실패 영역에 얼마나 강하게 개입할지 정하는 정책 손잡이"**로 읽어야 한다(방향이 반대다 — 맞추는 것이 아니라 정하는 것). 균형은 "gradient 크기를 같게"가 아니라 "두 신호를 원하는 비율로 섞기"이며, 그 비율이 1:1일 이유도 없다.

원래 우려는 "정확히 1:1"이 아니라 "RL로 과쏠림"이었고, 그것은 reward 채널(§4.3)로 충분히 답해진다.

**부록 워딩 함의**: A.4의 "row 수와 무관하게 동등하게 기여"는 스케일 균질화를 함의하므로 부정확하다. 대신 "각 row가 한 단위로 계상되고, 한 RL prompt는 결정 수 n에 비례해 n row-unit으로, 한 SFT prompt는 1 row-unit으로 기여한다"처럼 weight 회계의 사실만 진술하도록 교정한다. branch 간 상대 강도는 row 수가 아니라 β_r가 정한다(§4.2 기여 편향, §6).

### 4.2 aggregation mode는 RL branch가 단독으로 결정한다

"RL로 쏠린다"는 우려에는 독립인 두 차원이 섞여 있다. **인센티브**(토큰당 가중이 응답 자신의 길이에 의존하는가 → mode의 관할)와 **배분**(branch·row 간 상대 학습량 → reward 채널의 관할)이다. mode는 배분을 못 정하고, 배분은 mode에 의존하지 않는다. 이 분리가 이하 논증의 축이다.

mode 선택은 인센티브 축에서 결정되고, 그 축의 제약은 RL branch에서만 발생한다.

"길이 편향"을 둘로 나눈다.
- **기여 편향**: 이번 update에서 누가 더 큰 몫을 갖는가(예: 10.7:1 vs 8:1). 무해하다 — unbiased policy gradient가 원래 처방하는, 결정 수에 비례한 신호일 뿐이며 정책이 exploit할 수 없다.
- **인센티브 편향**: 토큰당 gradient가 그 응답 자신의 길이에 의존해, 정책이 길이를 조작해 loss를 회피할 수 있는가. Dr.GRPO가 문제 삼은 것은 오직 이것.

seq-mean-token-mean은 각 row를 자기 길이 |o|로 나눈다(토큰당 가중 = A/|o|). 이때 row 총기여가 길이 무관 상수가 되어 8:1이 나오고, **동시에** 토큰당 가중이 1/|o|가 되어 인센티브 편향이 생긴다. row 동등 기여(총기여 = 길이 × 토큰당가중 = 상수)는 토큰당 가중 = 1/|o|를 요구하는데, 그것이 정확히 Dr.GRPO가 금지한 형태다. **즉 8:1을 택하는 것과 Dr.GRPO 위반은 같은 하나의 선택이다.**

이 위험은 **RL branch 고유**다. RL rollout의 길이는 정책이 생성하므로 게임이 실재한다(틀린 응답: 자기-길이 divisor면 총벌점이 길이 무관 상수 → 길게 틀려도 공짜 → 길이 폭주). 부호 상쇄 반론("A>0도 보면 균형 아니냐")은 성립하지 않는다 — 자기-길이 divisor의 토큰당 가중은 "맞으면 짧을수록 강화 세짐, 틀리면 짧을수록 처벌 세짐"이라, 정책은 조건부로 "맞을 땐 짧게, 틀릴 땐 길게"를 동시에 학습한다(실증 서명: 정답 길이 정체/감소, 오답 길이 증가). Async-HPT의 어려운-prompt 레짐은 A<0 경험이 다수라 왜곡이 길이 폭주 쪽으로 기운다.

따라서 aggregation mode는 RL이 단독으로 sum-norm(상수 divisor)을 강제한다. 이는 이 코드베이스의 순정 `sft_loss`(token-equal, `masked_sum/batch_num_tokens`)와도 정합하며, §3-(4)의 정규화 철학 분열을 소멸시킨다.

### 4.3 그 강제가 SFT에 비용인가 — 아니다 (branch-blind reduction)

§4.2가 mode를 RL 사정으로 못박았으니 자연히 묻게 된다: "그러면 SFT가 손해 보지 않는가?" 답은 아니오이며, 그 이유가 이 결정의 우아함의 핵심이다.

**단일 trajectory 관찰**: τ*가 하나뿐이면 두 mode의 차이는 스칼라 하나(|o|/L_max)이고 reward로 흡수된다. 차이가 실체를 갖는 것은 **길이가 다른 τ*들이 함께 학습될 때**의 배분뿐이다. 두 mode는 "τ* 한 편당 예산 1" vs "토큰 한 개당 정액"이라는 서로 다른 배분 규칙일 뿐이다.

**배분은 mode가 아니라 β_r로 표현된다**: SFT row의 총 기여는 `β_r · |o_r| / L_max`이다. 따라서
- `β_r = β` (상수) → "토큰당 정액" 배분.
- `β_r = β · L_max / |o_r|` → row 총기여 = β로 고정 → "한 편당 1(자기-길이 정규화)" 배분을 sum-norm 위에서 **정확히 재현**.

즉 자기-길이 mode가 SFT에 주던 배분은 sum-norm + β_r로 커스텀 divisor 0줄로 복원 가능하다. **mode는 SFT 배분에 대해 중립**이며, 어떤 배분도 미리 박지 않는다는 그 중립성이 sum-norm의 SFT 측 이점이다 — "정액이 최선이라서"가 아니라 "실험 뒤 β_r로 정할 수 있어서".

이로써 다음 사슬이 완성된다.
1. 테제(UPGE)는 "SFT와 RL은 하나의 estimator"다. 하나의 estimator면 aggregation도 하나여야 한다. branch별로 mode를 달리 주면 loss 계층에서 branch를 섞는 것 = 원조 HPT의 loss mixing = 방금 걷어낸 branch-aware reduction의 재발명.
2. 그 단일 mode는 RL이 단독 결정한다(§4.2, 협상 불가).
3. 이 강제가 SFT에 비용을 지우는가? → 아니다. SFT 배분은 β_r로 완전히 표현되므로 비용 0.
4. 결론: **hybrid 결합은 reduction 계층에 새 요구를 부과하지 않는다. reduction은 branch-blind이며, branch semantics는 advantage(β_r)와 reference(self-detach) 두 지점으로 완결된다.**

이는 A.1(queue가 branch를 해석 안 함, transport-blind), A.6(실행이 의미론을 모름, execution-blind)에 이은 **aggregation-blind**로, "이질성이 문제되는 계층마다 blindness를 확보한다"는 논문 모티프의 세 번째 실현이다.

과장 경계: branch-blind 주장은 "HPT가 aggregation에 요구를 추가하지 않는다"이지 "어떤 mode든 똑같이 좋다"가 아니다. mode 간 우열은 존재하되 그것이 **순수 RL 문제로 환원**된다는 것이 주장의 정확한 범위다.

### 4.4 그래서 무엇이 사라지고 무엇이 남는가

§3의 네 갈래가 근원 제거로 동시에 닫힌다.

| §3의 복잡성 | 이 결정에서의 해소 |
|---|---|
| (1) grain 불일치 | 분모가 동기의 step-grain이 되어 정의상 소멸 |
| (2) surrogate 재구현 | 표준 `get_policy_loss_fn("vanilla")`를 그대로 호출 |
| (3) 방어 코드 | per-row broadcast 분모가 없어 확인 대상 소멸 |
| (4) 정규화 철학 분열 | 두 branch가 단일 mode 공유 |

**삭제되는 코드**: `B_eff`(`hpt_loss_denominator`), `hpt_seq_weight`, `hpt_length_divisor`, `finalize_loss_denominator`, `_compute_vanilla_token_losses` 재구현, `.max()`+allclose 방어 코드.

**남는 유일한 HPT 커스텀 코드**: effective old log-prob의 `torch.where(sft_mask, log_prob.detach(), old_log_prob)` 한 줄. 이것은 방법의 기여 자체이므로 남는 것이 옳다. §3-(2)에서 약화됐던 명제("SFT가 RL 코드 경로를 그대로 탄다")가 reduction까지 순정이 되어 원래의 강한 형태로 복원된다.

**부수 효과 (순수 이득)**: 순수 RL batch가 baseline GRPO와 비트 단위로 동일해져, HPT on/off 비교가 나눗셈 관례가 아닌 알고리즘만을 비교하게 된다.

---

## 5. 간결성 가설 (검증 대상, 미승격)

강한 모델로 뽑은 τ*는 헤매는 행동과 중복 action이 적어 토큰 밀도가 높다는 가설. **주의: 이것은 균형 논거가 아니다** — sum-norm에서 간결한 τ*는 오히려 기여가 작아지므로, 균형에 붙이면 논리가 뒤집힌다. 올바른 자리는 "token-grain 학습의 노출 품질"이다: sum-norm은 모든 τ* 토큰을 정가로 batch에 주입하므로, 주입되는 것이 신호냐 잡음이냐는 τ*의 품질 밀도가 정한다. 밀도가 높으면(간결) 정가 주입이 순장점, 낮으면(장황) 잡음이 선형 주입된다.

- **측정**: 같은 prompt에서 τ* action 수 ÷ RL rollout 평균 action 수. task horizon이 통제되므로 남는 차이가 순수 장황함. (토큰 수가 아니라 **action/turn 수** — 강한 reasoning 모델은 step당 텍스트가 길 수 있어 토큰 기준은 오판.)
- **반증 시 처방**: 가설이 깨지면 지우는 것은 "시너지" 문장 하나뿐이다. 설계는 무너지지 않는다 — sum-norm의 정당화는 Dr.GRPO와 동기 관례라는 독립 근거 위에 있고, 대응은 estimator 회귀가 아니라 데이터 측(τ* 큐레이션, response mask 내부 세분화, 또는 β_r 길이-보정)이다.
- **배치**: 검증 전에는 부록/논문에 넣지 않는다. 검증 후에만 승격.

---

## 6. 부록(A.4) 반영 방침

부록의 수식 `L = (1/B_eff) Σ w_r L_seq^r`, `B_eff = Σ w_r`는 합의 범위를 명시하지 않으므로, "r이 도는 범위 = 하나의 update가 보는 batch"로 읽으면 표준 aggregation과 일치한다. 수식 변경 없이 Reduction 절을 다음 취지로 다시 쓴다. **여기서 §2–§3의 실패 서사(복잡성의 근원)는 부록에 싣지 않는다 — 부록은 완성된 설계를 보이는 자리이므로 긍정형만 남긴다.**

- 별도 분모를 정의하지 않고 표준 policy-gradient aggregation을 그대로 사용한다.
- aggregation mode는 RL branch의 인센티브 제약(Dr.GRPO식 상수 divisor)이 결정한다.
- supervised branch의 trajectory별 학습 예산은 aggregation이 아니라 pseudo reward `β_r`가 담당한다.
- "동등하게 기여"(스케일 균질화 함의)를 "각 row가 한 단위로 계상되고, RL prompt는 그 결정 수 n에 비례해 n row-unit으로, SFT prompt는 1 row-unit으로 기여한다"(weight 회계)로 교정한다. 이 row-비례 가중은 unbiased policy gradient가 처방하는 바이며(§4.2 기여 편향), branch 간 상대 강도는 row 수가 아니라 β_r가 정한다. gradient 스케일 동등은 목표가 아니며, off-policy 감독의 큰 gradient는 의도된 동작임을 명시한다(§4.1).
- 따라서 reduction은 branch-blind이며, branch semantics는 advantage와 reference 두 지점으로 수렴한다(A.1/A.6과 같은 사상).

A.4 thesis 후보: **"외부 신호의 크기는 β_r가, 내용은 τ*가, 인센티브는 길이-중립 estimator가 담당한다."**

---

## 7. 후속 작업 (코드)

- [ ] `losses.py`에서 HPT 분기를 표준 vanilla 경로 + `loss_agg_mode` 배선으로 교체.
- [ ] assembler에서 B_eff/weight/divisor 필드 생성 제거. SFT pseudo reward를 `β_r` 인터페이스로 노출(상수 기본값, 길이-반비례 옵션).
- [ ] `loss_scale_factor = L_max` 설정 경로 확인 및 config 노출.
- [ ] 등가성 테스트: (a) 1-minibatch vs M-minibatch에서 각 step loss가 "그 step의 표준 정규화"와 일치, (b) all-RL·α=1 batch가 baseline GRPO와 스케일 일치, (c) `β_r = β·L_max/|o|` 설정이 자기-길이 정규화 배분을 재현(§4.3 lemma의 실행 검증).
- [ ] 관측 지표 추가: round별 branch별 gradient-mass 점유율, `hpt/tau_action_ratio`(§5 측정).
- [ ] 미해결(별건, DR 분리 예정): KL/entropy가 SFT row에 균일 적용되는 문제는 이 결정과 독립.

---

## 8. 후속 작업 (실험 / ablation)

branch-blind 덕분에 aggregation ablation이 **config 한 줄 교체**로 가능하다(HPT 코드 무변경). 이 실행 가능성 자체가 §4.3 주장의 구성적 증거이므로 실험 절 도입에 명시한다.

**2축 격자 (축을 분리해 교란 제거)**:
- **축 1 — aggregation mode (RL 인센티브)**: sum-norm vs seq-mean-token-mean. 예측: 후자에서 오답 길이 인플레이션(Dr.GRPO 재현), HPT의 어려운-prompt 레짐에서 증폭. **통제**: mode 간 SFT 실효 예산을 β_r로 동일 고정(§4.3 lemma를 실험 도구로 사용).
- **축 2 — β_r 정책 (SFT 배분)**: 상수 β vs 길이-반비례 β_r. "토큰당 정액 vs 한 편당 1" 및 간결성 가설(군더더기 레짐 vs horizon 레짐)을 판정. 이 논문 고유 질문.

**우선순위**(컴퓨트 제약 시): main(sum-norm+상수β) → 축 2 → 축 1. 축 1은 재현 성격이라 가장 먼저 잘릴 후보.

**서술 원칙**: "여러 mode를 시도해 sum-norm이 최선이었다"(경험적 튜닝, 방어 취약)가 아니라 "테제에서 도출 → RL이 mode 결정 → β_r로 SFT 비용 0 → branch-blind → ablation이 도출의 각 고리를 검증"(도출 후 검증). ablation은 선택을 낳은 것이 아니라 도출을 확인하는 위치에 둔다.

---

## 부록: 이 문서의 상태와 명명

이 기록은 설계 결정을 담은 것이지 검증된 동작이 아니다(관련 코드 현재까지 미실행). §7의 등가성 테스트가 통과해야 결정이 실측으로도 확인된다. 파일명 `DR-001`은 후속 결정 기록(τ* 무결성, KL/entropy 처리, 실패-원자성 등)을 위한 번호 체계의 시작이다.
