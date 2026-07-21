# StreamWeave 전체 논문 초안 (한국어)

> **Working title:** *StreamWeave: Reconciling Off-Policy Expert Supervision with
> Fully-Asynchronous Policy Learning*

## 1. Introduction

Reinforcement learning with verifiable rewards (RLVR)는 자동으로 검증 가능한 성공을 강화하여
LLM의 reasoning 능력을 향상시키지만, policy가 아직 해결하지 못한 어려운 문제에서는 rollout generation에
상대적으로 많은 계산을 쓰면서도 유효한 learning signal은 가장 적게 얻는 역설에 직면한다. 널리 사용되는
group-based RLVR에서는 policy가 정답을 탐색하기 위해 다수의 긴 autoregressive rollout을
생성하므로, rollout collection이 강화학습 wall-clock의 지배적인 병목이 된다. 그러나 그렇게
비싸게 수집한 group이 전부 실패하면 group 내 보상 차이도 사라져
relative learning signal을 얻지 못한다. 이를 해결하기 위해 서로 다른 두 연구 방향이 발전해 왔다.
먼저 실행 효율의 관점에서 fully-asynchronous RL은 generation과 learning을 분리하고 중첩하여
synchronization idle을 줄인다. 이 계열은 policy-generated rollout이 생성 시점보다 늦게 소비되는
temporal mismatch를 다루지만, 학습 신호의 source 자체는 policy experience로 유지된다. 반면 학습
신호 보강의 관점에서는 expert-provided trajectories가 policy가 아직 성공하지 못한 문제에 유효한
해결 경로를 제공한다. 두 병목은 독립적이지 않다. 동일한 어려운 문제가 긴 rollout과 all-failure group을 함께
유발하기 때문이다. 따라서 scalable RLVR에는 asynchronous
execution과 heterogeneous supervision이 동시에 필요하다. 그럼에도 기존 방법들은 두 역량을 서로
다른 training regime에서 다루어 왔으며, heterogeneous supervision의 learning semantics를
fully-asynchronous policy learning 안에서 어떻게 보존할지는 여전히 열린 문제다.

두 방향의 결합이 단순하지 않다는 사실은 외부 expert trajectory의 사용 여부를 policy-generated
outcomes에 따라 결정할 때 선명해진다. Group-based RLVR에서 이 결정은 완성된 prompt group을
요구한다. 충분한 성공 신호가 있는 group은 policy-generated rollouts로 학습하고, 그렇지 않은 group은
expert trajectory를 사용한다. 같은 group은 policy branch에서 rollout 간 reward 차이로 relative
advantage를 구성하는 단위이기도 하다. 따라서 complete group은 training source와 policy-side learning
contribution을 함께 결정한다. 동기식 실행에서는 source decision 전에 group이 완성되지만,
fully-asynchronous execution은 개별 trajectory를 독립적으로 전진시켜 효율을 얻는다. Group을 실행
단위로 유지하면 가장 느린 trajectory가 다시 장벽이 되고, group이 완성되기 전에 판단하면 source와
relative signal이 달라진다. 또한 group을 복원하는 것만으로 결합이 끝나지는 않는다. Source decision은
학습 data뿐 아니라 update rule도 바꾸기 때문이다. Stale rollout은 learner policy의 과거 버전이
생성하여 policy mismatch를 정의할 reference를 유지하지만, expert trajectory는 policy lineage 밖에서
제공되어 동일한 correction에 필요한 recorded behavior-policy reference를 갖지 않는다. 따라서 두
source에는 서로 다른 learning rule이 필요하다. 이러한 충돌에서 이 논문의 질문이 나온다.
**Trajectory-level execution의 효율을 유지하면서, 완성된 group이 결정하는 training source와 각
source의 고유한 학습 역할을 하나의 fully-asynchronous stream에서 보존할 수 있는가?** 이는 scheduler나
loss 하나를 고르는 문제가 아니라, execution granularity와 learning boundary를 함께 설계하는
algorithm-system co-design 문제다.

우리는 이 충돌을 **StreamWeave**로 해결한다. StreamWeave의 핵심은 학습 판단에 필요한 group
boundary와 runtime이 작업을 진행시키는 execution unit을 분리하는 것이다. Complete group은 어떤
source를 사용할지 결정하고 learning rule을 확정하는 경계로 유지하되, 각 trajectory는 독립적으로
실행된다. Runtime은 다른 작업을 막지 않으면서 group을 복원하고, group이 완성되면 source를 확정해
learner가 해당 update를 적용하도록 한다. 따라서 StreamWeave는 complete-group decision과
source-specific update를 보존하면서도, 그 boundary를 전체 pipeline의 synchronization barrier로
확장하지 않고 generation과 learning을 실질적으로 중첩한다.

그러나 pipeline을 계속 전진시키는 것만으로 이 결합이 올바른 것은 아니다. 신호가 생성된 맥락이
learner까지 보존되지 않으면, 비동기 runtime의 완료 순서와 batching이 어떤 source를 training
stream에 받아들일지, 어떤 update를 적용할지, 그리고 두 source가 얼마나 기여할지를 다시 쓸 수 있다.
이 공동설계를 관통하는 기준은 간결하다. 비동기는 신호가 **도착하는 시점과 순서**를 바꿀 수 있지만,
그 신호가 **의미하는 것**을 바꾸어서는 안 된다. StreamWeave는 이를 위해 각 신호의 provenance, 즉
어느 group과 source에서 왔으며 rollout이라면 어떤 policy가 생성했는지를 learner까지 유지한다. 우리는
이를 세 조건의 learner contract로 명시한다. Source는 complete group에서만 결정되고, 각 source에는
자신에게 정의된 update가 적용되며, 두 source는 사전에 선언된 weighting과 reduction으로만 결합된다.
이 contract는 특정한 동기식 control flow가 아니라 execution optimization이 보존해야 할 boundary를
규정한다. 동일한 counterfactual audit를 단순 결합과 StreamWeave에 적용하여 이 조건들의 유지 여부를
판정한다.

이러한 의미 보존은 비동기 효율과의 타협을 요구하지 않는다. Math-reasoning 실험에서
StreamWeave는 동일한 source-selection policy를 사용하는 synchronous counterpart와 대등한 품질을
유지하면서 strongest competing hybrid baseline을 **[ΔQ points]** 상회한다. 동시에 동일한
하드웨어에서 synchronous counterpart 대비 **[T×]**의 처리량을 달성하고 trainer idle을
**[I_sync%]에서 [I_async%]로** 낮춘다. 이로써 StreamWeave는 같은 learning semantics를 유지한 채
quality-versus-wall-clock을 개선한다.

이 논문의 기여는 다음과 같다.

1. **Policy rollout과 expert supervision의 learning composition.** Complete group이 training
   source와 policy-side relative signal을 함께 결정하는 환경에서, policy rollout과 expert
   supervision을 각자의 reference와 learning rule을 유지한 채 하나의 학습 과정으로 결합한다.
   Learner contract는 fully-asynchronous execution이 보존해야 할 이 composition의 경계를 명시한다.
2. **Fully-asynchronous execution architecture.** StreamWeave는 이 learning composition을 trajectory
   attempt의 독립 실행, nonblocking group reconstruction, 그리고 분리된 rollouter와 trainer를 잇는
   bounded stream으로 end-to-end 실현하여, group-level barrier를 복원하지 않고 비동기 실행의
   효율 원천인 generation–learning overlap을 유지한다.
3. **학습 효과와 실행 효율의 공동 실측.** 프로토콜을 통일한 benchmark 평가와 동일 하드웨어
   비교를 통해 model quality, quality-versus-wall-clock, throughput, trainer idle을 함께 측정한다.

## 2. Related Work

관련 선행연구는 fully-asynchronous policy learning과 expert trajectory를 활용하는 policy learning의
두 계보로 나뉜다. 전자는 실행 효율을 높였고, 후자는 policy-generated experience만으로 학습 신호가
부족할 때 이를 보완했다. 아래에서는 각 계보의 성과와 두 접근을 함께 사용할 때 남는 문제를 정리한다.

**Fully-asynchronous policy learning.** 시스템 계층에서 HybridFlow (EuroSys 2025)는 RLHF의
computation과 data dependency를 분리하여 다양한 알고리즘과 resource mapping을 유연하게 구성할 수
있는 실행 기반을 제공했다. Asynchronous RLHF (ICLR 2025)는 generation과 learning을 직접 분리하고,
이전 policy가 생성한 sample로 학습할 때 발생하는 policy lag와 실행 효율 사이의 trade-off를 분석했다.
AReaL (NeurIPS 2025)은 rollout worker와 training worker를 완전히 분리하고 workload balancing과
staleness-aware optimization을 결합하여 fully-asynchronous LLM RL을 end-to-end로 실증했으며, TBA
(NeurIPS 2025)는 asynchronous actor가 수집한 replay experience를 off-policy objective로 학습한다.
이 연구들은 policy rollout의 생성과 소비를 비동기화하고 그에 따른 policy lag를 관리한다. 그러나
learning stream은 current 또는 past policy가 생성한 rollout을 중심으로 하며, all-failure group을
expert trajectory로 보완하는 문제는 연구 대상이 아니었다.

**Learning from policy rollouts and expert trajectories.** 다른 연구 방향은 demonstration, offline
data, expert trajectory를 활용하여 학습을 policy가 스스로 생성한 experience에만 의존하지 않도록
확장해 왔다. 일반 RL에서는 DQfD (AAAI 2018)가 demonstration을 temporal-difference update와 supervised
loss에 사용했고, RLPD (ICML 2023)는 offline data를 online RL에 지속적으로 결합했다. LLM
post-training에서는 InstructGPT (NeurIPS 2022)가 demonstration-based supervised learning과 RLHF를
단계적으로 연결했고, SimpleMix (ICML 2025)는 preference learning에서 on-policy와 off-policy data를
직접 혼합했다. Reasoning post-training의 LUFFY (NeurIPS 2025)는 off-policy reasoning trace와 policy
rollout을 mixed-policy learning으로 결합하고, CHORD (ICLR 2026)는 SFT를 on-policy exploration과 함께
최적화되는 dynamically weighted auxiliary objective로 재구성한다. 이 연구들은 외부 data를 어떤
objective와 weighting으로 학습에 기여시킬지를 다루지만, policy outcome에 따른 expert source
decision을 trajectory-level fully-asynchronous execution에서 유지하는 문제는 연구 대상이 아니었다.

**Composing asynchronous execution with heterogeneous supervision.** 기존 fully-asynchronous methods는
policy-generated stream을 중심으로 하고, expert trajectory를 활용하는 methods는 group-dependent
source decision을 trajectory-level asynchrony에서 유지하는 문제를 다루지 않는다. 전자가 다루는 stale
rollout은 learner policy의 과거 버전이 생성하지만, expert trajectory는 learner policy 밖에서 별도의
학습 신호로 들어온다. Yao et al. (ICLR 2026)은 group-relative REINFORCE의 off-policy 해석을
제시했지만, objective가 off-policy data를 학습할 수 있다는 사실만으로 runtime이 source decision과
source-specific update를 보존하는 것은 아니다. 특히 group-based RLVR에서는 source decision과
relative signal이 complete group에 의존하는 반면, fully-asynchronous runtime은 trajectory를
독립적으로 실행한다. StreamWeave는 complete group을 학습 판단의 경계로 유지하면서도 이를 실행
장벽으로 만들지 않아 이 간극을 메운다.

## 3. StreamWeave

그림 2는 StreamWeave가 완성된 rollout group에서 학습 source와 update rule을 정하고, 이를
fully-asynchronous pipeline에서 실행하는 전체 과정을 보여준다. 먼저 complete group의 결과에 따라
policy rollout group이나 이에 대응하는 expert trajectory를 선택한다. 이 선택은 각 sample에 적용할
reference policy, objective, correction rule도 함께 정한다. Execution architecture에서는 각 trajectory
attempt를 독립적으로 생성하고, source를 선택하기 직전에 complete group을 복원한다. 선택된 data는
source와 생성 policy에 관한 정보와 함께 bounded queue를 거쳐 trainer로 전달된다. Source를 선택하는
단계만 해당 group이 완성되기를 기다리고, 다른 trajectory의 generation과 learner update는 계속 진행된다.
§3.1은 complete group을 learner update로 바꾸는 규칙을 정의하고, §3.2는 그 규칙을 global
synchronization barrier 없이 실행하는 방법을 설명한다.

### 3.1 Learning Composition

여기서 learning composition은 complete prompt group을 하나의 learner update로 바꾸는 규칙을
뜻한다. Complete group에서 학습 source를 정하고, source별 gradient contribution을 계산한 뒤,
미리 정한 방식으로 이들을 합친다. 이 규칙은 sample이 runtime에 도착하는 순서와 무관하게 정의된다.

**Source selection from a completed group.** Prompt $x$에 대해 policy가 생성한 $n$개의 rollout을
$G_x=\{\tau_{x,1},\ldots,\tau_{x,n}\}$이라 하자. 각 rollout은 verifier score
$R(\tau_{x,i})$를 가지며, complete group의 success rate는 다음과 같다.

$$
P_x=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\!\left[R(\tau_{x,i})>\delta\right],
\qquad
z_x=S_\gamma(G_x)=
\begin{cases}
\mathrm{expert}, & P_x\le\gamma,\\
\mathrm{policy}, & P_x>\gamma.
\end{cases}
$$

$z_x=\mathrm{policy}$이면 생성된 group $G_x$를 학습에 사용하고,
$z_x=\mathrm{expert}$이면 같은 prompt에 대응하는 expert trajectory $\tau_x^\star$를 사용한다.
StreamWeave는 $n$개의 score가 모두 모인 뒤에 source를 선택한다. 실험에서는 HPT가 제안한
success-rate threshold rule을 $S_\gamma$로 사용한다. Threshold와 대응하는 expert trajectory의
가용 범위는 Experimental Setting에서 명시한다.

**Source-specific gradient contributions.** Source가 정해졌다고 해서 gradient 계산까지 자동으로
정해지는 것은 아니다.
Policy rollout은 같은 prompt의 다른 rollout과 비교되어 relative learning signal을 얻으며, 자신을
생성한 policy의 확률과 version 정보를 가진다. 반면 expert trajectory는 policy가 생성한 sample이
아니라 학습할 target으로 주어지므로 group-relative signal을 사용하지 않으며, learner가 rollout
correction에 사용할 생성 확률도 기록하지 않는다. 따라서 source selection은 사용할 data와 함께
reference policy, objective, correction rule을 정한다.

Source가 확정된 뒤 learner batch의 각 sample $r$은 source label $z_r=z_{x(r)}$와 원래 prompt
$x(r)$를 유지한다. Policy sample은 생성 policy의 확률과 version 정보 $p_r$도 유지한다. 각 sample의
gradient contribution과 batch $B$의 gradient를 다음과 같이 정의한다.

$$
g(r)=
\begin{cases}
g_{\mathrm{policy}}\!\left(r;G_{x(r)},p_r\right),
& z_r=\mathrm{policy},\\[1mm]
\beta_r g_{\mathrm{expert}}(r),
& z_r=\mathrm{expert},
\end{cases}
\qquad
g(B)=\operatorname{Aggregate}_{r\in B} g(r).
$$

여기서 $g_{\mathrm{policy}}$는 complete group에서 계산되는 policy gradient,
$g_{\mathrm{expert}}$는 expert trajectory에서 계산되는 supervised gradient, $\beta_r$는 그 강도를
정하는 계수다. $\operatorname{Aggregate}$는 미리 정한 weighting과 batch averaging을 나타낸다.
먼저 두 gradient를 각자의 규칙으로 계산하고, 그 뒤에만 하나의 learner update로 합친다.

**Where asynchronous correction applies.** Policy rollout에는 각 token을 생성한 behavior policy의
확률이 기록되어 있으므로, rollout과 learner policy의 차이를 측정하고 importance weighting으로
보정할 수 있다. Expert trajectory에는 이 보정에 필요한 생성 당시의 policy probability가 기록되어
있지 않다. 따라서 expert sample은 더 오래된 rollout이 아니며, rollout용 importance weighting을
적용할 대상도 아니다. 임의의 dummy probability를 대신 사용하면 expert gradient가 그 값에 따라
달라진다.

현재 구현에서 policy branch는 $G_x$로부터 group-relative advantage를 계산하고 vanilla clipped
PPO로 학습한다. PPO reference는 batch가 learner update에 들어올 때의 policy snapshot이다. Rollout을
생성한 policy와 이 snapshot 사이의 차이는 token-level truncated importance weight로 보정한다.
Expert branch는 $\beta_r$로 조절한 supervised log-likelihood gradient를 사용하며, 이 importance
weight는 1로 고정한다. 두 branch를 같은 policy-gradient 구현에서 처리하기 위해 expert sample의
reference에는 current-policy log-probability의 stop-gradient copy를 사용한다. 그러면 forward
probability ratio는 1이지만 supervised gradient는 남는다. 이 계산의 자세한 미분은 Appendix에서
제시한다.

**Combining policy and expert gradients.** 현재 구현은 source별 gradient를 계산한 뒤 두 branch에
동일한 batch aggregation rule을 적용하며, expert gradient의 크기는 $\beta_r$로 명시한다. 두 source의
실제 상대 기여는 $\beta_r$뿐 아니라 선택된 policy와 expert sample의 수, 학습 token의 수, weighting,
batch averaging에 의해 함께 정해진다. StreamWeave는 이 결합 규칙을 먼저 고정한다. Queue의
도착 순서나 batch 구성은 여기에 별도의 normalization이나 weight를 추가하지 않는다.

우리는 이 learning composition을 세 가지 요구사항으로 요약하고 이를 learner contract라 부른다.
첫째, source는 complete group이 갖추어진 뒤에만 정한다. 둘째, policy와 expert sample에는 각각
정의된 reference, objective, correction만 적용한다. 셋째, 두 gradient는 미리 정한 weighting과
batch aggregation으로만 결합한다. 이 contract는 fully-asynchronous execution이 보존할 학습 규칙을
정의하지만, 동기식 control flow를 요구하지 않는다. 다음 절은 trajectory가 서로 다른 시점에
완성되는 동안에도 이 규칙을 global synchronization barrier 없이 실행하는 방법을 설명한다.

---

## 내부 편집 메모 (본문 아님)

### 1. 논문 헌법

이 헌법은 Related Work만의 포지셔닝 메모가 아니라, 논문 전체의 논증 순서와 증거 위계를 정하는
최상위 기준이다. 모든 주요 문단, 그림, 기여, 실험은 아래 어느 층위를 전진시키는지 설명할 수 있어야
한다. 어느 층위에도 대응하지 않는 구현 세부는 Appendix로 내리고, 여러 층위를 반복하는 문단은
압축한다.

| 논증 층위 | 논문 전체의 핵심 판단 |
|---|---|
| **필드의 궤적** | RLVR은 실행 확장성을 위해 generation과 learning의 시간적 결합을 풀어 왔고, 신호 부족을 극복하기 위해 학습 source를 policy rollout 바깥으로 넓혀 왔다. 어려운 reasoning regime은 두 방향을 동시에 요구한다. |
| **숨은 충돌** | Fully-asynchronous execution은 작업을 연속적인 stream으로 해체하지만, source-conditioned learning은 어떤 data와 update를 사용할지 결정하는 맥락을 요구한다. |
| **연구 문제** | 실행 시점의 자유와 학습 source의 자유를, 서로의 의미를 바꾸지 않고 함께 실현할 수 있는가? |
| **핵심 통찰** | 학습 결정을 위해 필요한 boundary가 pipeline 전체의 synchronization barrier가 될 필요는 없다. |
| **우리의 방법** | StreamWeave는 필요한 학습 맥락을 국소적으로 복원하면서 trajectory-level execution과 generation–learning overlap을 유지하는 algorithm-system architecture다. |
| **판정 기준** | Learner contract와 counterfactual audit는 비동기 실행이 source admission, source-native contribution, declared composition을 바꾸지 않았는지 판정한다. |
| **실증과 범위** | Group-conditioned policy/expert learning을 사용하는 RLVR에서 정합성, model quality, quality-versus-wall-clock, execution efficiency를 함께 보인다. |

**Canonical thesis paragraph:** RLVR 연구는 서로 다른 두 결합을 독립적으로 풀어 왔다.
Fully-asynchronous RL은 rollout generation과 policy learning을 분리하여 실행의 시간적 제약을
완화하고, expert-guided learning은 학습을 policy 자신의 성공에만 의존하지 않도록 supervision의
source를 넓힌다. 그러나 두 자유도를 동시에 허용하면 runtime의 도착 순서와 batching이 어떤 source를
선택하고 각 source가 어떻게 기여하는지를 바꿀 수 있다. StreamWeave의 중심 통찰은
source-conditioned learning에 필요한 boundary가 global synchronization barrier로 되돌아올 필요가
없다는 것이다. StreamWeave는 필요한 맥락을 국소적으로 복원하고 source별 학습 역할을 유지하면서도
fully-asynchronous pipeline을 계속 전진시킨다.

**소유권과 위계:** 필드의 궤적은 배경에 대한 우리의 해석이지 novelty가 아니다. 논문이 소유하는
지점은 두 방향의 교차점에서 발생하는 composition problem, learning boundary와 execution barrier의
분리, 그리고 이를 end-to-end로 실현하는 StreamWeave다. Learner contract와 audit는 이 합성을
판정하는 supporting apparatus이며, group-conditioned two-source RLVR은 보편성의 증명이 아니라
핵심 주장을 실물로 보이는 empirical witness다.

### 2. 최상위 작문 원칙

- **Interpretation first, mechanism backed.** 논문의 1차 산출물은 사실 목록이 아니라, 연구가
  제시하는 철학과 판단이다. 각 문단은 하나의 interpretive claim으로 시작하며, 사실은 그 판단을
  정당화하고 반증 가능하게 만드는 인과적 근거로만 사용한다.
- **Benefit before mechanism.** StreamWeave를 소개할 때 `attempt → group → update`를 먼저 열거하지
  않는다. “Complete-group·source-specific semantics는 보존하되 그 경계를 execution barrier로 만들지
  않는다”는 공동 이점을 먼저 선언하고, architecture는 그 판단을 가능하게 하는 근거로 배치한다.
- **Define and scope before abstracting.** `adaptive heterogeneous learning`, `source-selection rule`
  같은 비표준 포괄어를 정의 없이 사용하지 않는다. 먼저 policy-generated rollout, expert-provided
  trajectory, 완성된 group에 따른 source choice를 설명하고, 모든 외부-trajectory 방법의 보편적
  성질로 확대하지 않는다.
- **One primary home per claim.** Introduction은 문제와 design judgment를, Related Work는 attribution과
  scope boundary를, Method는 정확한 mechanism을 소유한다. 다른 섹션에서 같은 주장을 회수할 때는
  다시 전개하지 않고 해당 섹션의 역할에 필요한 한 문장만 남긴다.
- **Strength through structure.** 주장의 힘은 `fundamental`, `critical`, `inevitable` 같은 수식어가
  아니라 구조적 긴장과 그 결과에서 만든다. 강한 문장은 hook, research question, design principle처럼
  방향을 바꾸는 지점에만 두고, 나머지는 차분하고 정확하게 기전을 뒷받침한다.
- **Explicit recovery.** 이름 붙인 문제·원리·기여는 장식으로 남기지 않고, Method의 설계와
  Experiment의 evidence에서 명시적으로 회수한다.

### 3. Introduction 서사와 기여 회수

아래 표는 논문 헌법을 세 개의 공개 contribution으로 투영한다. Learning composition은 무엇을
보존해야 하는지를, execution architecture는 그 조건을 full asynchrony 아래에서 어떻게 실현하는지를,
공동 실측은 두 목표가 실제로 함께 달성되었는지를 담당한다.

| 기여 축 | 핵심 주장 | 본문 회수 | 주된 evidence |
|---|---|---|---|
| **1. Learning composition** | Policy rollout과 expert supervision이 각자의 reference와 learning rule을 유지한 채 하나의 학습 과정에서 결합되도록 구성하고, learner contract로 fully-asynchronous execution이 보존할 경계를 명시 | Method의 source decision, source-native objective, mixed learner formulation | Source-native formulation, three-clause contract와 scoped counterfactual audit |
| **2. Execution architecture** | 위 composition을 독립 trajectory 실행, nonblocking group reconstruction, bounded rollouter-trainer stream으로 실현함 | Method의 Execution Architecture | Throughput, trainer idle, quality-versus-wall-clock |
| **3. 공동 실측** | 학습 효과와 실행 효율을 protocol-matched evaluation으로 함께 측정함 | Learning Effectiveness와 Execution Efficiency | Fixed-checkpoint quality와 same-hardware efficiency |

Learner contract와 audit는 독립 기여나 전체 올바름의 증명이 아니라, 기여 1·2를 뒷받침하는
supporting apparatus로 둔다. 공개 핵심 약속은
`preserve the intended learning composition without turning its boundaries into execution barriers`로
고정하며, 최적성을 요구하는 `maximize` 대신 `retain`, `realize`, `without giving back`을 사용한다.

| 문단 | 서사적 역할 | 독자가 가져갈 판단 |
|---|---|---|
| **1문단** | Compute–signal double bottleneck | 어려운 RLVR일수록 비싼 rollout과 부족한 성공 신호가 함께 심해지므로 두 연구 방향을 함께 다뤄야 한다. |
| **2문단** | 단일 composition gap과 research question | Complete group은 source admission과 RL contribution을 함께 정의하며, 이를 execution barrier 없이 source-specific learning까지 연결해야 한다. |
| **3문단** | Execution architecture | Learning boundary를 보존하면서도 group 대기를 전역 장벽으로 만들지 않아 full asynchrony를 유지한다. |
| **4문단** | Learning-composition boundary와 conformance | Contract는 runtime이 다시 쓰면 안 되는 조건을 명시하고, audit는 그 제한된 조건의 유지 여부를 점검한다. |
| **5문단** | Empirical payoff | 같은 learning semantics 아래에서 quality-versus-wall-clock이 개선됨을 보인다. |

### 4. 핵심 충돌 지도와 실현 원장

**역할:** 아래 지도는 Introduction의 공개 목차가 아니라, 기여 1·2가 Method에서 빠짐없이 회수되는지
확인하는 내부 coverage ledger다. 공개 Introduction은 네 경계를 각각 열거하지 않고, **의미적 경계를
그대로 실행 장벽으로 두면 효율을 잃고, 경계를 지우면 runtime이 learning composition을 다시 쓴다**는
하나의 composition gap으로 추상화한다.

| 설계 경계 | 보존할 learning semantics | 유지할 execution property | 나이브 결합의 실패 | StreamWeave의 해법 | 소유 범위 |
|---|---|---|---|---|---|
| **Decision: group ↔ attempt** | Complete group이 source admission을 결정하고 policy branch의 relative signal을 정의 | 느린 attempt가 다른 작업을 막지 않는 독립 실행 | Groupwise control flow는 synchronization idle을 복원하고, premature admission은 source decision과 relative signal을 선취 | Attempt-level scheduling, nonblocking group reconstruction, route-before-admission | StreamWeave의 핵심 bridge. Reconstruction만을 독립 novelty로 주장하지 않음 |
| **Update: source ↔ unified learner** | Rollout과 expert가 서로 다른 reference와 operator domain을 유지 | Source별 동기 phase 없이 하나의 연속 stream을 소비 | 동기 phase는 overlap을 분절하고, branch-blind operator는 expert contribution을 변형 | Provenance-keyed unified consumption, source-native update, rollout-only correction | Learning composition의 핵심 |
| **Composition: variable output ↔ fixed-grain learner** | Whole-group selection과 declared weighting·reduction을 유지 | 가변 cardinality를 stall·무한 과수집 없이 learner에 공급 | 대기는 backlog와 staleness를 만들고, row 단위 폐기·평균은 selection과 effective mixture를 바꿈 | Bounded semantics-preserving assembly, whole-group deferral, explicit accounting | 일반 interface invariant는 Method, 정확한 verl realization은 Appendix |
| **Temporal flow: freshness ↔ overlap** | Versioned rollout의 temporal validity를 관리하고 expert를 stale rollout처럼 취급하지 않음 | Rollouter와 trainer의 continuous overlap을 유지 | Queue drain은 idle을, 무제한 backlog는 staleness와 batch explosion을 유발 | Bounded queue, live backpressure, partial rollout, incremental sync | 기존 async substrate와 StreamWeave integration의 일부이며 독립 novelty가 아님 |

**세 계약 조항의 구현 witness:** 세 조항은 위 설계를 학습 관점에서 닫는 명세다. 실제 수행한 조치를
Introduction에 열거하지 않고, Method에서는 각 조치가 보존하는 성질을 먼저 제시한다.

| 계약 조항 | 실제 수행한 조치 | 공개 추상화 |
|---|---|---|
| **Complete-context admission** | `group_uid`와 attempt 순서를 보존하고 complete group에서만 routing하며, 실패 group은 fail-closed 처리 | 필요한 맥락이 완성되기 전에는 어떤 source도 training stream에 admit하지 않음 |
| **Source-native contribution** | RL은 group-relative advantage와 rollout-policy context를 유지하고, expert는 singleton `β_r` 신호와 self-detached CE를 사용하며 rollout-only operator에서 제외 | 각 신호에는 그 source에서 정의되는 reference와 operator만 적용 |
| **Declared composition** | 표준 branch-blind reduction과 명시적 `β_r`를 사용하고, variable-cardinality output은 whole-group 단위로 조립·이월하며 예외를 계측 | 정의된 contribution은 사전에 선언된 weighting과 reduction으로만 결합 |

**2문단 논증 순서:** 조건부 expert source를 먼저 설명하고, complete group의 source-admission·RL-signal
이중 역할을 밝힌 뒤, trajectory-level execution과의 double bind를 제시한다. Group reconstruction에서
멈추지 말고 source choice가 data와 update rule까지 바꾼다는 결과로 전진한 다음 research question을
제시한다. HPT, `n:1`, reducer, deferred materialization은 이 문단에 넣지 않는다.

**사실 앵커와 회수 위치:**

| 주장 | 사실 앵커 | 회수 위치 |
|---|---|---|
| Fully-asynchronous separation과 trajectory-level execution | AReaL (NeurIPS 2025)은 generation-learning 분리, Laminar (EuroSys 2026)는 trajectory-level independent execution; `fully_async_rollouter.py::_submit_hpt_trajectory_attempts` | Intro의 효율 원리; Method의 Execution Architecture |
| Complete-group source choice | `hpt_gate.py::HptRolloutGate.route` | Intro의 판단 경계; Method의 selector와 attribution |
| Group-relative RL signal | `ray_trainer.py::compute_advantage`; `core_algos.py::compute_grpo_outcome_advantage` | Intro의 group 역할; Method의 objective |
| Nonblocking group reconstruction | `hpt_rollout_accumulator.py::HptPromptGroupAccumulator`; `fully_async_rollouter.py::_record_hpt_trajectory_attempt_result` | Method의 Execution Architecture |
| Source-dependent materialization | `hpt_gate.py::route_rollout_sample`; `hpt_assembler.py::materialize_training_batch` | Method의 learning composition과 stream interface |
| Variable-cardinality assembly | `fully_async_trainer.py::_get_samples_from_queue`; `_plan_row_alignment_deferral` | Method에는 bounded assembly; Appendix에는 trim-and-carryover와 예외 회계 |
| Partial-group routing | G4 admission replay의 나이브 counterfactual; 현행 runtime은 gate 전에 group을 복원 | Conformance Analysis에서만 사용; 현행 장애나 Intro headline으로 서술 금지 |

**본문 승격 필터:** 본문급 설계는 `nonblocking group reconstruction`, `provenance-keyed unified
consumption`, `bounded semantics-preserving assembly`다. Accumulator 자료구조, queue 크기, 정확한
batch divisor, subset-sum, trim-and-carryover 절차·예외, tensor field는 Appendix로 내린다. Bounded
assembly는 네 번째 계약이 아니라 declared composition을 learner interface까지 닫는 supporting
mechanism이다. 실현된 mixture는 `β`만이 아니라 routing 빈도, cardinality, token volume에도 의존한다.

### 5. 포지셔닝과 정보 계층

- **Research object and instantiation:** 논문의 주인공은 두 연구 방향의 composition problem을
  해결하는 algorithm-system architecture인 **StreamWeave**다. HPT는 group-success source-selection
  policy의 실험적 instantiation일 뿐 방법의 정체성이 아니다.
- **Working title:** **StreamWeave: Reconciling Off-Policy Expert Supervision with
  Fully-Asynchronous Policy Learning**으로 고정한다. 부제의 두 절은 아래 Core distinction이
  구별하는 두 off-policy에 대응한다 — 앞절 = 외부 trajectory인 expert supervision, 뒷절 =
  stale rollout을 낳는 fully-asynchronous 실행 — 이라 제목 자체가 둘의 화해를 압축한다.
  `Off-Policy`는 유지한다.
- **Core distinction:** 논문을 `on-policy/off-policy mixing`으로 요약하지 않는다. 대신 두
  가지 off-policy를 정확히 구별한다: **stale rollout**은 우리 정책의 과거 버전이 생성했고
  생성 시점의 확률이 기록되어 있어 IS ratio 보정이 **정의되는** off-policyness이고, **expert
  trajectory**는 정책 계보 밖에서 온 외부 trajectory이며 동일한 correction에 필요한 recorded
  behavior-policy reference가 제공되지 않아 IS ratio가 **정의되지 않는다**. 따라서 보정이 아니라
  별도의 학습 규칙이 맞는 처방이다. 후자를 "더 심한
  전자"로 취급하는 서술을 금지한다(CHORD의 expert-행 IS 기각이 이 고장의 야생 실증). 단
  "보정이 정의된다"≠"보정이면 충분하다" — 적용 가능성과 효용은 별개다(CISPO·decoupling
  관측). 제목의 `Off-Policy`는 2문단 삽입 문장이 이 구별을 본문에서 받친다는 전제로 유지한다.
- **Prior-art posture:** AReaL이 exact GRPO group-completeness를 명시적으로 정식화하지 않았다는
  부재에 기대지 않는다. PPO/RLOO를 포함한 homogeneous asynchronous policy learning을 충분히
  인정한 뒤, complete group이 training source와 update semantics까지 바꾸는 경우를 구획한다.
- **Related Work argument and paragraph grammar:** 선행연구를 결함 목록이나 census로 제시하지 않는다.
  각 계보는 `분야명 → 해결한 구체적 문제 → 대표 accepted-conference 연구 → 범위 밖에 남은 문제`
  순서로 쓴다. 공개 본문에서는 `실행 효율`과 `학습 신호 보완`을 직접 말하고, `두 자유`, `유지된
  전제`, `frontier` 같은 분석어는 내부 메모에만 사용한다. 앞의 두 계보에서는 StreamWeave를 반복해
  호출하지 않고, 마지막 composition 문단에서만 group-dependent learning과 trajectory-level
  asynchrony의 결합 문제를 도출한 뒤 StreamWeave의 위치를 한 번 회수한다. 대표 연구는 앞선 판단의
  증거이며, 범위 밖에 남은 문제는 선행연구의 결함으로 쓰지 않는다. 공개 지면에는 축마다 대표작만
  남기고 세부 novelty audit는 Appendix로 보낸다.
- **Literature rule:** 포지셔닝과 Related Work의 근거는 accepted-conference 논문만 사용한다.
  공개 Introduction에서는 HPT를 언급하지 않으며, Method에서 selector의 출처로만 attribution한다.

| 위치 | 남길 내용 |
|---|---|
| **Introduction** | Group-based RLVR의 compute–signal double bottleneck, complete group의 source-admission·RL-contribution 이중 역할, 하나의 composition gap, StreamWeave의 design judgment, contract와 audit의 역할, headline empirical payoff |
| **Related Work** | 동기식 RL의 실행 효율을 개선한 연구와 policy-generated signal을 외부 data로 보완한 accepted-conference 연구, 그리고 group-dependent learning을 fully-asynchronous execution에 결합할 때 남는 문제 |
| **Method: Learning Composition** | HPT selector attribution과 정확한 routing rule, source-native objective, behavior-policy reference, weighting과 reduction |
| **Method: Execution Architecture** | Independent attempt scheduling, nonblocking group reconstruction, bounded rollouter-trainer stream, end-to-end realization |
| **Conformance Analysis** | 같은 three-clause probe를 나이브 결합과 StreamWeave에 대칭 적용하되, 명시한 조건의 유지 여부로 범위를 제한 |
| **Experiments** | Protocol-matched quality, quality-versus-wall-clock, throughput, trainer idle, 학습 동역학 해석 |
| **Appendix** | Queue, partial rollout, trim+carryover, schema, placeholder, 개별 operator·incident와 보조 분석 |

Aliasing lemma, n-source 일반화, necessity/sufficiency/selectivity 서사, CISPO·decoupling의 부정
결과는 Introduction과 공개 contribution에서 제외한다.

### 6. 증거 게이트와 주장 규율

- **Quality:** `[ΔQ points]`는 protocol-fair fixed-checkpoint mean@32와 문항 단위 paired
  hierarchical bootstrap이 닫힌 뒤 확정한다.
- **Efficiency:** `[T×]`와 `[I_sync%]→[I_async%]`는 matched `nocispo` 재측정이 닫힌 뒤 확정한다.
- **Learning dynamics:** RL-only의 초반 동등·후반 정체, 교사 채널 `+3.4`, cold-start 역할의 역전,
  장기 안정성은 Introduction이 아니라 Learning Effectiveness에서 곡선과 함께 제시한다.
- **Conformance:** Counterfactual audit는 three-clause contract의 유지 여부만 판정한다. 전체
  correctness의 증명, 보편적 necessity, 독립 contribution으로 서술하지 않는다. 기여 1의
  "결합됨을 보인다"와 4문단의 audit 문장은 G4(audit 실행) 완료에 결속된 주장이다. Audit는
  검증 도구이므로 기여 bullet에 이름을 올리지 않고 본문(4문단·Conformance)에서만 등장한다.
- **금지 표현:** `maximize`, `provably necessary`, `zero-waste`, 무조건적 data preservation,
  “mixture strength는 β만으로 결정된다”를 사용하지 않는다. 대신 실제 보장 범위와 예외 회계를
  직접 명시한다.

### 7. Method §3 작성 헌장

#### 7.1 두 subsection의 단일 역할

§3은 `Learning Composition → Fully-Asynchronous Execution` 순서로 고정한다. 두 subsection은
동등한 추상화 수준의 병렬 구성요소가 아니다. §3.1은 실행 순서와 무관하게 **무엇이 유효한 mixed
update인지** 정의하고, §3.2는 그 학습 구성을 **global group barrier 없이 어떻게 실현하는지**
설명한다. 즉 §3.1이 학습 구성을 정의하고 §3.2가 그 정의를 만족하는 시스템 실현을 제공한다. 두 절의
경계를 다음처럼 잠근다.

| 절 | 독자의 질문 | 이 절이 소유하는 답 | 넘기지 않을 내용 |
|---|---|---|---|
| **3.1 Learning Composition** | Complete group은 어떤 source와 update contribution으로 변환되는가? | Group-conditioned source decision, source-native contribution, provenance에 따른 operator domain, declared composition | Scheduler, accumulator, queue, backpressure, row alignment |
| **3.2 Fully-Asynchronous Execution** | 앞서 정의한 composition을 보존하면서 어떻게 pipeline을 계속 전진시키는가? | Independent attempts, nonblocking group reconstruction, routed-group transport, bounded trainer consumption, parameter refresh | Loss 유도, self-detach 증명, 개별 mask와 tensor field |

§3 전체의 opening judgment는 다음으로 고정한다.

> StreamWeave는 먼저 complete group이 source-native contribution으로 변환되는 learning composition을
> 정의한다. 그런 다음 trajectory를 독립적으로 실행하면서도 그 composition을 global group barrier
> 없이 실현하는 fully-asynchronous execution architecture를 구성한다.

#### 7.2 Method 주장 위계: invariant → design → realization

§3.1과 §3.2의 `정의 → 실현` 관계는 유지하되, Method 안의 개별 주장은 아래 세 수준으로 구별한다.
이는 subsection을 하나 더 만드는 구성이 아니라, **무엇을 일반적 조건으로 주장하고 무엇을
StreamWeave의 설계로 소유하며 무엇을 현행 시스템의 구현으로 제시할지** 정하는 지위 체계다.

| 수준 | 핵심 질문 | 논문에서의 역할 |
|---|---|---|
| **Learning invariants** | Fully-asynchronous execution 아래에서도 무엇이 바뀌면 안 되는가? | Learning-composition 수식과 three-clause contract가 보존 대상을 정의 |
| **StreamWeave design** | 그 조건을 full asynchrony의 효율을 반납하지 않고 어떻게 만족시키는가? | 학습 경계와 실행 장벽을 분리하는 architecture와 설계 판단 |
| **Concrete realization** | 현재 실험과 코드에서 그 설계를 어떤 objective와 mechanism으로 구현했는가? | Main-run 명세와 Appendix의 구현 세부 |

여기서 learning invariant는 모든 heterogeneous learning에 대한 보편 공리가 아니라, 이 논문이
정의하는 **group-conditioned policy/expert composition의 정체성 조건**이다. StreamWeave가 소유하는
주장은 임의의 hybrid algorithm에 대한 유일 해법이 아니라, 이 composition을 full asynchrony 아래에서
변형하지 않고 실현하는 원리와 architecture다.

전체 논증은 다음 순서를 따른다.

```text
Learning invariants
  무엇이 유지되어야 하는가
        ↓
StreamWeave design
  왜 이 architecture를 선택했는가
        ↓
Concrete realization
  현행 시스템에서 어떻게 구현했는가
        ↓
Counterfactual audit
  제한된 invariant가 실제 실행에서 유지되는가
```

Contract는 첫 수준을 압축하고, counterfactual audit는 마지막 단계에서 그 contract에 대한
conformance만 점검한다. 중간의 architecture는 수학적으로 유일한 해법으로 주장하지 않지만,
불변 조건과 비동기 효율을 함께 달성하기 위해 StreamWeave가 선택하고 소유하는 핵심 기여다.

| 요소 | 지위 | 본문에서의 취급 |
|---|---|---|
| Complete group에서만 source 결정 | **Learning invariant** | Selector의 정의역과 수식으로 제시 |
| Policy와 expert의 source-native contribution | **Learning invariant** | Branch-defined contribution으로 제시 |
| Rollout operator의 behavior-provenance 조건 | **Learning invariant** | Operator가 유효한 domain으로 설명 |
| 명시된 weighting과 reducer에서만 source 결합 | **Learning invariant** | Declared composition으로 제시 |
| Nonblocking group reconstruction | **Core StreamWeave design** | Learning boundary를 execution barrier와 분리하는 중심 architecture |
| Independent trajectory attempts | **Execution strategy** | Group context를 복원하면서도 보존해야 할 attempt-level execution freedom |
| Routed-group queue와 deferred materialization | **StreamWeave design** | Source를 먼저 확정하고 transport와 batching이 결정을 다시 쓰지 않게 하는 경계 |
| Bounded queue와 backpressure | **Efficiency/freshness design** | Pipeline utilization과 stale backlog를 함께 제어하는 end-to-end closure |
| HPT success-conditioned hard switch | **Concrete instantiation** | 실험에서 사용한 group-conditioned selector로 한 번 attribution |
| Entry-proximal anchor, token IS, vanilla clipped PPO | **Main-run realization** | 현행 policy branch의 정확한 명세이며 StreamWeave 자체의 일반 원리는 아님 |
| Self-detached expert reference | **Implementation mechanism** | Shared learner path에서 supervised contribution을 복원하는 현행 방식 |
| Trim-and-carryover | **Framework-specific realization** | Appendix에서 fixed-grain learner와의 정렬 방법으로 설명 |

Self-detach는 이 위계를 보여주는 대표 사례다. 보존해야 할 invariant는 expert branch가 의도한
supervised contribution을 유지하는 것이다. StreamWeave는 policy와 expert를 하나의 shared learner
path에서 처리하고, 현행 realization은 expert reference를 self-detach하여 그 contribution을 복원한다.
Appendix의 미분은 self-detach 자체가 유일하게 필수임을 보이는 것이 아니라, **선택한 realization이
목표 invariant를 만족함을 보이는 formal witness**다.

#### 7.3 §3.1 Learning Composition

§3.1은 **mechanism-first, contract-last**로 쓴다. Contract-first는 Introduction을 반복하고,
objective-first는 HPT loss의 재서술처럼 보이므로 사용하지 않는다. 문단 순서는 아래 네 개로 고정한다.

| 문단 | 역할 | 반드시 전달할 판단 | 이후 회수 |
|---|---|---|---|
| **Source selection from a completed group** | 무엇을 결합하는지 정의 | Complete rollout group이 policy 또는 expert source를 선택하며, 이 결정은 개별 trajectory가 아니라 group 결과에 의존 | §3.2 reconstruction, admission audit |
| **Source-specific gradient contributions** | 선택된 source가 어떻게 학습하는지 정의 | Source selection은 data와 함께 reference policy, objective, correction rule을 결정 | source-fidelity audit, quality 결과 |
| **Where asynchronous correction applies** | 비동기 correction의 적용 범위 설명 | Importance weighting은 생성 policy probability가 기록된 rollout에만 적용되고 expert trajectory에는 적용되지 않음 | conformance, Appendix의 stop-gradient 유도 |
| **Combining policy and expert gradients** | 하나의 learner update로 닫기 | 각 gradient를 먼저 계산한 뒤 미리 정한 weighting과 batch aggregation으로 결합 | mixture audit, §3.2 bridge |

§3.1의 reader takeaway는 하나다.

> **Source selection은 사용할 data와 함께 reference policy, objective, correction rule을 정한다.
> StreamWeave는 policy와 expert gradient를 각자의 규칙으로 계산한 뒤, 미리 정한 방식으로 하나의
> update에 결합한다.**

본문의 display 수식은 두 개만 둔다. 첫째는 complete group의 success rate와 source-selection rule이다.
실험에서는 HPT의 success-rate threshold rule을 사용한다고 Method에서 한 번만 attribution한다.
`gamma=0`, `n=8`은 Experimental Setting으로 보낸다.
Matched expert가 필요한데 없는 경우는 main과 동일하게 fail-closed이며, 예외 절차는 Appendix로 보낸다.

둘째는 구체적인 optimizer 선택보다 앞선 수준에서 **source-specific gradient contribution**을 정의한다.

$$
g(r)=
\begin{cases}
g_{\mathrm{policy}}\!\left(r;G_{x(r)},p_r\right), & z_r=\mathrm{policy},\\
\beta_r g_{\mathrm{expert}}(r), & z_r=\mathrm{expert},
\end{cases}
\qquad
g(B)=\operatorname{Aggregate}_{r\in B}g(r).
$$

여기서 `G_x`는 policy gradient에 필요한 complete group, `p_r`는 생성 policy의 확률과 version 정보,
`beta_r`는 expert gradient의 명시적 강도, `Aggregate`는 미리 정한 weighting과 batch averaging이다.
이 식은 StreamWeave가 보존하는 learning composition을 정의하며 특정 PPO variant나 correction
mechanism을 일반 원리로 격상하지 않는다.

현재 구현은 그 뒤 산문으로 분리한다. Policy branch는 group-relative advantage, learner-entry
policy snapshot, rollout-to-entry token IS, vanilla clipped PPO를 사용한다. Expert branch는
`beta_r`로 조절한 supervised gradient를 사용하고 importance weight를 1로 둔다. Current-policy
log-probability의 stop-gradient copy로 두 branch를 같은 policy-gradient 구현에서 처리한다는 사실은
본문 한 문장으로만 쓰고, 정확한 미분, mask와 tensor-level 구현은 Appendix로 내린다.

Contract는 출발점이나 별도 장문의 Definition이 아니라 위 구성을 압축하는 결론이다. 세 조건을
`(i) complete-group source selection`, `(ii) source-specific objective and correction`,
`(iii) pre-specified weighting and batch aggregation` 순서로 한 번만 제시하고, Conformance의 세
probe가 같은 순서로 회수한다. §3.1은 다음 bridge로 닫는다.

> 이 절이 유효한 mixed update가 무엇인지 정의했다면, 다음 절은 trajectory가 독립적으로 완성되는
> 동안에도 같은 update를 synchronization barrier 없이 구성하는 방법을 설명한다.

#### 7.4 Canonical main objective

공개 Method와 모든 내부 요약은 아래 현행 main을 단일 기준으로 사용한다.

> **Main = decoupled policy correction + vanilla clipped PPO.**

| 축 | 현행 main (`M5abl_nocispo`) |
|---|---|
| Policy objective | `vanilla` clipped PPO; lower `0.2`, upper `0.28` |
| RL signal | GRPO group-relative advantage; std normalization 활성 |
| Proximal reference | learner-entry policy (`rl_old_logprob_source=entry`) |
| Behavior correction | rollout-to-entry token-level truncated IS, `C_w=2.0` |
| Rejection / learner stale-drop | rejection 비활성, `k_max=null` |
| Expert contribution | constant `beta=0.3`, self-detached current-policy reference, IS identity |
| Auxiliary | expert entropy 제외; KL은 main 전체에서 비활성 |
| Composition | branch-blind reduction; effective mixture는 routing, cardinality, token volume, `beta`, reducer가 함께 결정 |

**CISPO는 Method 구성요소가 아니다.** 기각된 ablation이자 Appendix의 secondary diagnosis로만 다룬다.
Decoupling은 main에 활성인 realization이지만 StreamWeave의 novelty가 아니며, 이 레짐에서 효과가 거의
비활성이었다는 결과 역시 Appendix에 둔다. 기존 CISPO arm에서 잰 efficiency 수치는 matched
`nocispo` 재측정 전까지 headline에 사용하지 않는다. `entropy/KL 제외`라고 뭉뚱그리지 않고,
main에서는 entropy exclusion만 활성이고 KL은 전역 비활성이라고 쓴다.

#### 7.5 §3.2와 Generator–Trainer authoritative flow

코드 기준 실행 순서는 아래가 단일 진실이다. Accumulator는 Generator 내부에서 Gate보다 먼저
동작하고, Queue에는 learner row가 아니라 source가 확정된 prompt-group record가 들어간다.

```text
Prompt group
  -> independent attempt scheduling
  -> parallel trajectory generation
  -> complete-group reconstruction
  -> group-conditioned source decision  <- expert trajectory store
  -> routed group + provenance
  -> bounded group queue
  -> trainer-side materialization
  -> bounded batch assembly
  -> source-native update
  -> updated policy
  -> parameter refresh back to Generator
```

§3.2의 최상위 설계 판단은 **learning boundary를 보존하되 그것을 execution barrier로 만들지
않는다**는 것이다. 이 절은 장치 목록이 아니라, 각 invariant를 효율적으로 실현하는 이유를 보여주는
논증으로 쓴다. 핵심 문단은 `invariant의 요구 → 단순 구현의 효율 손실 또는 의미 왜곡 → StreamWeave의
설계 → 의미와 효율을 함께 유지하는 결과`의 인과를 갖는다.

| 논증 단위 | 단순 결합의 긴장 | StreamWeave design | 함께 유지되는 것 |
|---|---|---|---|
| **Complete-group decision** | Group을 blocking execution unit으로 두면 tail latency와 idle이 돌아오고, 미완성 맥락에서 판단하면 source decision이 달라짐 | Attempt를 독립 실행하고 source decision 직전에만 group을 nonblocking reconstruction | Complete context와 attempt-level asynchrony |
| **Source-native contribution** | 도착 즉시 공통 row로 변환하면 transport와 batching이 source의 reference·operator를 암묵적으로 결정 | Source가 확정된 routed group과 provenance를 운반하고 trainer에서 deferred materialization | Source-native semantics와 continuous transport |
| **Declared composition** | 이질적인 cardinality를 무제한으로 기다리거나 row 단위로 버리고 평균하면 backlog 또는 hidden reweighting 발생 | Bounded assembly에서 contribution을 구성한 뒤 선언된 weighting과 reducer로만 결합 | Bounded consumption과 declared mixture |
| **End-to-end overlap** | 무제한 queue는 stale backlog를 만들고 지나치게 작은 capacity는 producer를 반복적으로 정지 | Bounded queue, backpressure, parameter refresh를 하나의 control loop로 닫음 | Pipeline utilization과 freshness control |

전체 서술은 `inherited asynchronous freedom → added group dependency → local reconstruction → routed
group transport → bounded stream closure → end-to-end overlap` 순서로 진행한다. AReaL의
generation-learning separation과 Laminar의 trajectory-level execution을 인정한 뒤, StreamWeave의
소유 지점은 **complete group이 source와 update rule까지 결정하는 환경에서 필요한 context만 국소적으로
복원하고, 그 결정을 learner까지 보존하는 bridge**로 세운다. 마지막 takeaway는 다음이다.

> **Group-dependent semantics만 group completion을 기다리며, pipeline 전체는 기다리지 않는다.**

`n:1`, required multiple, subset-sum, trim-and-carryover, queue 크기와 예외적 discard는 Appendix다.
본문에는 `bounded semantics-preserving assembly`만 남기며 `zero-waste`, `crash-free`, 무조건적 보존을
주장하지 않는다.

#### 7.6 Runtime figure contract

Method의 간단한 runtime 그림은 Figure 1의 문제 서사를 반복하지 않고, §3.1에서 정의한 learning
composition의 요구가 §3.2의 runtime에서 어디에 실현되는지를 보여준다. 하나의 왼쪽-오른쪽 pipeline을
그리고, 그 위에 세 requirement를 callout으로 대응시킨다.

```text
Learning composition:        [1. Complete-group decision]  [2. Source-native handling]  [3. Declared composition]
                                           |                          |                            |
Runtime architecture: Prompt -> attempts -> reconstruct -> route -> queue -> materialize/assemble -> update -> refresh
```

이 그림의 상단은 별도의 물리적 pipeline이 아니라 §3.1이 §3.2에 부과하는 요구사항이다. 큰 물리 영역은
`Generator`, `Bounded group queue`, `Trainer` 세 개로 제한한다.

| Learning Composition의 요구 | Runtime에서의 실현 |
|---|---|
| **Complete-group decision** | Attempt를 독립 실행한 뒤 gate 직전에 complete group을 복원하여 source를 결정 |
| **Source-native handling** | Routed group에 provenance를 유지하고 trainer에서 source에 맞게 materialize·update |
| **Declared composition** | Source별 contribution을 정의한 뒤 bounded assembly와 명시된 reducer에서만 결합 |

| 내부 객체 | 공개 그림의 명칭 | 표시할 의미 |
|---|---|---|
| trajectory scheduler + rollout workers | **Independent attempts** | Group을 실행에서만 분해 |
| `HptPromptGroupAccumulator` | **Complete-group reconstruction** | Gate 전에 semantic context 복원 |
| `HptRolloutGate` + expert store | **Source decision** | Policy group 또는 expert trajectory 확정 |
| `MessageQueue` | **Bounded group queue** | Routed group과 provenance 운반; learner row가 아님 |
| `HptBatchAssembler` | **Materialize & assemble** | Policy group은 `n` rows, expert는 `1` row로 변환 |
| Trainer objective | **Source-native update** | Policy와 expert에 유효한 contribution을 정의한 뒤 결합 |
| checkpoint/queue budget | **Parameter refresh / backpressure** | Dashed reverse control path |

Solid arrow는 data plane, dashed reverse arrow는 parameter refresh와 capacity feedback을 나타낸다.
Provenance는 별도 계산 모듈이 아니라 routed group에 붙는 `group/source/policy-version` metadata로
표현한다. 그림 내부에는 CISPO, self-detach 미분, token IS, `beta=0.3`, queue 크기, carryover 절차를
넣지 않는다. 구체적인 policy label이 필요하면 `Policy update` 또는 `GRPO + vanilla PPO`를 사용한다.

그림의 단 하나의 takeaway는 다음이다.

> StreamWeave는 의도한 learning composition을 먼저 정의하고, 각 요구사항을 fully-asynchronous
> runtime의 구체적인 경계에 대응시킨다. Source decision에 필요한 group completion만 국소적으로
> 기다리며, 나머지 pipeline은 계속 전진한다.
