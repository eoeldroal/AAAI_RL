# StreamWeave 전체 논문 초안 (한국어)

> **Working title:** *StreamWeave: Reconciling Off-Policy Expert Supervision with
> Fully-Asynchronous Policy Learning*

## Abstract

Reinforcement learning with verifiable rewards (RLVR)는 policy가 풀기 어려운 문제에서 긴 rollout에
많은 계산을 쓰고도, 생성한 응답이 모두 실패하면 group 내 보상이 같아져 relative learning signal을
얻지 못한다. 이 계산 병목과 신호 부족은 서로 다른 방향에서 다뤄져 왔다. Fully-asynchronous RL은
rollout generation과 model training을 병렬로 진행해 실행 시간을 줄이고, expert trajectory를 활용하는
방법은 policy가 풀지 못한 문제에 정답 경로를 제공한다. 우리는 이 두 접근이 만나는 group-conditioned
setting을 다룬다. 이 setting에서는 완성된 rollout group의 결과에 따라 policy rollout과 expert
trajectory 중 어느 데이터를 학습할지 결정한다. Fully-asynchronous execution은 각 rollout을 독립적으로
처리해야 효율을 얻지만, 학습할 데이터는 group이 완성된 뒤에야 정할 수 있다. 완성된 뒤에도 두 데이터는
같은 학습 입력이 아니다. Policy rollout은 group-relative advantage와 이를 생성한 policy 정보를 바탕으로
update되지만, expert trajectory는 정답 sequence 자체를 supervised target으로 사용하며 동일한
rollout-policy reference를 제공하지 않는다. 따라서 policy rollout을 위한 비동기 보정을 두 데이터에
일괄 적용하면 expert trajectory가 의도한 supervised gradient가 달라진다. **StreamWeave**는 trajectory
단위 실행과 rollout generation·model training의 병렬화를 유지하면서, 데이터를 선택하는 순간에만
다른 작업을 막지 않고 완성된 group을 복원한다. 이후 선택 결과와 생성 policy 정보를 trainer까지
유지하여 policy rollout에는 RL objective와 비동기 보정을, expert trajectory에는 supervised objective를
적용한다. 수학 추론 실험에서 StreamWeave는 비교한 방법 중 가장 높은 평균 성능 38.5를 기록하고, 동일한
데이터 선택 기준과 학습 목적을 사용하는 동기식 pipeline(37.7)과 대등한 품질을 달성한다. 동일한
8$\times$B200에서 128개 prompt group의 처리 시간을 46초에서 28초로 줄여 throughput을 1.64$\times$
높이고, trainer가 다음 update에 필요한 데이터를 기다리며 멈추는 시간의 비율을 54.7\%에서 3.2\%로
낮춤으로써 complete-group decision을 전체 pipeline의 동기화 병목으로 되돌리지 않는다.

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
이를 세 조건의 learner contract로 명시한다. **Complete-group decision**은 source를 group이 완성된
뒤에만 정하고, **source-specific update**는 각 source에 정의된 reference, objective, correction을
적용하며, **declared composition**은 두 source를 사전에 정한 weighting과 reduction으로만 결합한다.
이 contract는 특정한 동기식 control flow가 아니라 execution optimization이 보존해야 할 boundary를
규정한다. §3.1은 실제 mixed objective가 이 세 조건으로 환원됨을 구성적으로 보이고, §3.2는 같은
조건을 global synchronization barrier 없이 실현하는 architecture를 제시한다.

이러한 의미 보존은 비동기 효율과의 타협을 요구하지 않는다. Math-reasoning 실험에서 StreamWeave는
비교한 방법 중 가장 높은 평균 성능 38.5를 기록하고, 동일한 source-selection policy와 objective를
사용하는 synchronous counterpart(37.7)와 대등한 품질을 달성한다. 동일한 8$\times$B200에서는 128개
prompt group의 처리 시간을 46초에서 28초로 줄여 throughput을 1.64$\times$ 높이고, trainer가 다음
update에 필요한 data를 기다리며 멈추는 시간의 비율을 54.7\%에서 3.2\%로 낮춘다. 이로써
StreamWeave는 같은 learning composition을 유지한 채 complete-group decision이 pipeline 전체의
실행 장벽으로 되돌아오는 것을 막는다.

이 논문의 기여는 다음과 같다.

1. **Policy rollout과 expert supervision의 learning composition.** Complete group이 training
   source와 policy-side relative signal을 함께 결정하는 환경에서, policy rollout과 expert
   supervision을 각자의 reference와 learning rule을 유지한 채 하나의 학습 과정으로 결합한다.
   Learner contract는 fully-asynchronous execution이 보존해야 할 이 composition의 경계를 명시한다.
2. **Fully-asynchronous execution architecture.** StreamWeave는 trajectory가 독립적으로 전진하는
   fully-asynchronous pipeline 안에서 complete group을 source decision 직전에만 복원하고, 확정된
   source와 생성 맥락을 learner까지 보존한다. 이로써 group-level barrier를 복원하지 않고 비동기
   실행의 효율 원천인 generation–learning overlap을 유지한다.
3. **학습 효과와 실행 효율의 공동 실측.** 프로토콜을 통일한 benchmark 평가와 동일 하드웨어
   비교를 통해 model quality, throughput, trainer idle을 함께 측정한다.

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
source와 생성 policy에 관한 정보와 함께 trainer로 전달된다. Source를 선택하는
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
**Complete-group decision**은 source를 complete group이 갖추어진 뒤에만 정한다.
**Source-specific update**는 policy와 expert sample에 각각 정의된 reference, objective, correction만
적용한다. **Declared composition**은 두 gradient를 미리 정한 weighting과 batch aggregation으로만
결합한다. 이 contract는 fully-asynchronous execution이 보존할 학습 규칙을 정의하지만, 동기식
control flow를 요구하지 않는다. 다음 절은 trajectory가 서로 다른 시점에 완성되는 동안에도 이
규칙을 global synchronization barrier 없이 실행하는 방법을 설명한다.

### 3.2 Fully-Asynchronous Execution

§3.1의 규칙은 complete group을 요구하지만, group 전체를 하나의 실행 단위로 묶을 필요는 없다.
Group을 하나의 blocking task로 실행하면 가장 늦게 끝나는 trajectory까지 실행 자원이 묶여 tail
latency와 pipeline idle이 다시 커진다. StreamWeave는 rollout generator와 trainer가 독립적으로
동작하는 fully-asynchronous 구조를 유지하고, complete group이 필요한 source selection만 해당
group의 완료를 기다리게 한다.

**Independent generation with complete-group reconstruction.** 하나의 prompt group을 이루는 trajectory
attempt들은 서로의 완료를 기다리지 않고 생성된다. 각 attempt는 자신이 속한 group을 식별할 정보를
유지하므로, 먼저 끝난 작업은 실행 자원을 비우고 다른 prompt의 generation도 계속될 수 있다.
StreamWeave는 이 흐름을 다시 group 단위 실행으로 되돌리지 않는다. 대신 source를 정해야 하는 순간에만
완료된 attempt들을 원래 group으로 복원하고, 모든 결과가 모인 뒤 §3.1의 source-selection rule을
적용한다. 불완전한 group은 learner로 전달되지 않지만, 이를 기다리는 동안 다른 group의 generation과
이미 준비된 data의 학습은 계속된다. 따라서 complete group은 **학습 결정을 위한 경계**로 남되
**pipeline 전체의 실행 장벽**이 되지는 않는다.

**Choosing the source before transport.** Group이 완성되면 policy rollout group을 사용할지, 이에
대응하는 expert trajectory를 사용할지 먼저 확정한다. 선택된 data에는 prompt-group identity, source,
그리고 policy rollout에만 존재하는 생성 policy 정보가 함께 유지된다. 이 정보는 중간 stream에서
재해석되지 않은 채 trainer로 전달되며, trainer는 source가 확정된 group을 해당 학습 sample로
변환한다. Batch 구성 역시 group membership, source별 gradient 계산, §3.1에서 정한 weighting을
바꾸지 않는다. 따라서 data의 도착 순서와 trainer의 처리 시점은 이미 내려진 source decision이나
각 source의 학습 역할을 다시 정하지 않는다.

Generator와 trainer는 동일한 iteration boundary를 기다리지 않으며, 현행 시스템은 bounded queue,
periodic parameter refresh, backpressure와 같은 fully-asynchronous runtime의 flow control을 유지한다.
이 실행 기반 위에서 StreamWeave는 group-dependent learning이 추가된 뒤에도 연속적인
generation–learning overlap이 무너지지 않도록, 필요한 기다림을 source decision에만 국소화한다.

StreamWeave는 §3.1의 학습 규칙을 단순화해서 비동기 효율을 얻지 않는다. Trajectory의 생성과 소비는
서로 다른 시점에 진행되지만, source는 complete group에서 정해지고, 선택된 data에는 source에 맞는
gradient가 적용되며, 두 gradient는 미리 정한 방식으로만 결합된다. 기다림이 필요한 것은 group에
의존하는 학습 결정뿐이며, pipeline 전체는 계속 전진한다.

---

## 내부 편집 메모 (본문 아님)

이 메모의 권한은 절별로 나눈다. §1은 thesis와 claim boundary, §2는 공통 작문 원칙, §3은
Introduction과 contribution의 회수, §4는 design ledger, §5는 positioning과 공개 정보의 층위,
§6은 evidence ledger, §7은 Method 작성 계획을 소유한다. 뒤 절은 앞 절의 결정을 다시 정의하지 않고,
자기 역할에 필요한 적용 방식과 세부 명세만 덧붙인다.

### 1. 논문 헌법

이 헌법은 Related Work만의 포지셔닝 메모가 아니라, 논문 전체의 논증 순서와 증거 위계를 정하는
최상위 기준이다. 모든 주요 문단, 그림, 기여, 실험은 아래 어느 층위를 전진시키는지 설명할 수 있어야
한다. 어느 층위에도 대응하지 않는 구현 세부는 Appendix로 내리고, 여러 층위를 반복하는 문단은
압축한다.

| 논증 층위 | 논문 전체의 핵심 판단 |
|---|---|
| **필드의 궤적** | RLVR은 실행 확장성을 위해 generation과 learning의 시간적 결합을 풀어 왔고, 신호 부족을 극복하기 위해 학습 source를 policy rollout 바깥으로 넓혀 왔다. 어려운 reasoning regime은 두 방향을 동시에 요구한다. |
| **숨은 충돌** | Fully-asynchronous execution은 작업을 연속적인 stream으로 해체하지만, 우리가 다루는 group-conditioned policy/expert learning은 complete group을 바탕으로 어떤 data와 update를 사용할지 결정한다. |
| **연구 문제** | 실행 시점의 자유와 학습 source의 자유를, 서로의 의미를 바꾸지 않고 함께 실현할 수 있는가? |
| **핵심 통찰** | 학습 결정을 위해 필요한 boundary가 pipeline 전체의 synchronization barrier가 될 필요는 없다. |
| **우리의 방법** | StreamWeave는 필요한 학습 맥락을 국소적으로 복원하면서 trajectory-level execution과 generation–learning overlap을 유지하는 algorithm-system architecture다. |
| **정합 근거** | §3.1은 source별 contribution과 aggregation을 명시하고, 실제 mixed objective가 이 composition으로 환원됨을 구성적으로 보인다. §3.2는 complete-group decision을 보존하면서도 이를 global synchronization barrier로 만들지 않는 실행 구조를 제시한다. |
| **실증과 범위** | Group-conditioned policy/expert learning을 사용하는 RLVR에서 선언한 learning composition, model quality, same-hardware execution efficiency를 함께 보인다. |

**Canonical thesis와 novelty kernel:** Fully-asynchronous RL은 실행의 시간적 제약을 완화하고,
expert-guided learning은 supervision의 source를 policy experience 밖으로 넓힌다. 이 논문이 다루는
group-conditioned setting에서는 complete group이 policy와 expert 중 사용할 data와 update를 함께
결정하므로, group completion이 단순한 data-availability condition을 넘어 control dependency가 된다.
StreamWeave의 novelty는 비동기 실행, expert trajectory, source selector 중 어느 부품에도 단독으로
있지 않다. 이 dependency를 global barrier로 확장하지 않고, 필요한 맥락과 source별 학습 역할을
보존하면서 full asynchrony 안에서 실현하는 composition architecture가 소유 지점이다.

| 지위 | 해당 내용 |
|---|---|
| **소유하는 주장** | Group-conditioned heterogeneous learning과 trajectory-level asynchrony의 composition problem, learning decision boundary와 execution barrier의 분리, 이를 end-to-end로 실현하는 StreamWeave |
| **주장을 지지하는 명세와 증거** | Learner contract, source별 contribution의 구성적 유도, matched synchronous control, fixed-checkpoint quality와 execution breakdown |
| **독립 novelty로 주장하지 않음** | Fully-asynchronous RL 자체, success-conditioned selector, expert trajectory 사용, PPO/GRPO/IS, accumulator·queue·backpressure, self-detach·trim-and-carryover |

Learner contract는 보존할 composition을 압축하는 specification이고, group-conditioned two-source RLVR은
보편성의 증명이 아니라 핵심 주장을 실물로 보이는 empirical witness다. 공개 브랜드는 StreamWeave
하나만 유지하며 별도의 named principle이나 audit를 novelty와 동급으로 세우지 않는다.

### 2. 최상위 작문 원칙

- **Interpretation first, mechanism backed.** 논문의 1차 산출물은 사실 목록이 아니라 연구가 제시하는
  철학과 판단이다. 다만 판단은 첫 독자가 이해할 수 있는 관찰에서 출발해야 한다. 대상과 원인이 생략된
  평가를 먼저 선언하지 않고, 무엇이 어떤 조건에서 왜 문제인지 보인 뒤 그 구조를 해석한다. 사실과
  mechanism은 이 판단을 정당화하고 반증 가능하게 만드는 근거로 사용한다.
- **Claim before component list.** StreamWeave를 소개할 때 scheduler, accumulator, queue를 순서대로
  열거하지 않는다. 먼저 학습 효과와 full-asynchronous efficiency를 함께 유지한다는 목표를 밝히고,
  이어서 이를 가능하게 하는 architecture의 핵심 동작과 마지막으로 보존되는 learning condition을
  설명한다. Abstract는 독자의 직관을 위해 `trajectory-level execution → local group reconstruction →
  source-appropriate update` 순서로 설명할 수 있지만, Method의 논증 순서는 `§3.1 definition → §3.2
  realization`으로 유지한다.
- **Define and scope before abstracting.** `adaptive heterogeneous learning`, `source-selection rule`
  같은 포괄어를 정의 없이 사용하지 않는다. 먼저 policy-generated rollout, expert-provided trajectory,
  완성된 group의 결과가 학습 data를 정하는 **group-conditioned setting**을 설명한다. 이 조건을 모든
  expert-guided learning의 보편적 성질로 확대하지 않으며, HPT는 이 setting에서 사용하는 구체적인
  selector로만 attribution한다. 자체 용어와 slogan은 설명을 대신하지 않고 이미 설명한 내용을
  압축하는 표지로만 사용한다.
- **One primary home per claim.** Introduction은 문제와 design judgment를, Related Work는 attribution과
  scope boundary를, Method는 정확한 mechanism을 소유한다. 다른 섹션에서 같은 주장을 회수할 때는
  다시 전개하지 않고 해당 섹션의 역할에 필요한 한 문장만 남긴다.
- **Strength through structure.** 주장의 힘은 `fundamental`, `critical`, `inevitable` 같은 수식어가
  아니라 구조적 긴장과 그 결과에서 만든다. 강한 문장은 hook, research question, design principle처럼
  방향을 바꾸는 지점에만 두고, 나머지는 차분하고 정확하게 기전을 뒷받침한다. 서로 다른 충돌을 한
  문장에 압축하지 않는다. 특히 `complete group을 요구하는 source decision ↔ 독립 trajectory 실행`의
  control dependency와 `rollout-policy context를 가진 RL update ↔ supervised expert update`의 학습
  차이를 분리하고, 일반적인 group-relative RL의 특징을 결합 문제 자체처럼 제시하지 않는다.
- **Explicit recovery.** 이름 붙인 문제·원리·기여는 장식으로 남기지 않고, Method의 설계와
  Experiment의 evidence에서 명시적으로 회수한다. 결론은 부정형 가능성 주장보다 StreamWeave가
  group-conditioned expert use를 새로운 global barrier 없이 full asynchrony에 통합했다는 달성
  사실로 닫는다.

**Public-draft conformance checklist:** 공개 원고를 갱신할 때 아래 네 항목을 먼저 확인한다.

| 점검 항목 | 현행 규율 |
|---|---|
| **Orphan evidence** | `counterfactual audit`처럼 Method나 Experiments에서 회수되지 않는 별도 evidence를 약속하지 않으며, 정합 근거는 §3.1의 구성적 유도와 §3.2의 architecture로 회수한다. |
| **Placeholders** | Abstract와 Introduction에는 `[ΔQ]`, `[T×]`, `[I_sync%]`를 남기지 않고 §6 evidence ledger의 현재 승인값만 사용한다. |
| **Claim hierarchy** | Primary headline은 matched learning composition 아래의 execution efficiency이고, quality는 synchronous counterpart와의 대등성 및 비교 방법 전반의 경쟁력으로 제시한다. |
| **Canonical vocabulary** | 공개 명칭은 `complete-group decision`, `source-specific update`, `declared composition`으로 통일한다. |

### 3. Introduction 서사와 기여 회수

아래 표는 논문 헌법을 세 개의 공개 contribution으로 투영한다. Learning composition은 무엇을
보존해야 하는지를, execution architecture는 그 조건을 full asynchrony 아래에서 어떻게 실현하는지를,
공동 실측은 두 목표가 실제로 함께 달성되었는지를 담당한다.

| 기여 축 | 핵심 주장 | 본문 회수 | 주된 evidence |
|---|---|---|---|
| **1. Heterogeneous learning composition** | Complete group이 source와 policy-side relative signal을 함께 결정하는 setting에서 policy와 expert contribution을 각각 정의한 뒤, 선언된 방식으로 하나의 update에 결합한다 | §3.1의 complete-group decision, source-specific update, declared composition | Source-specific formulation, expert-gradient reduction, explicit aggregation |
| **2. Decision-localized asynchronous architecture** | Group은 학습 결정 경계로 유지하되 source decision 직전에만 국소적으로 복원하여 pipeline 전체의 execution barrier가 되지 않게 한다 | §3.2의 nonblocking reconstruction, source-before-transport, context-preserving handoff | Same-hardware throughput과 trainer stall |
| **3. 학습 효과와 실행 효율의 공동 실측** | 동일 selector·objective를 사용하는 synchronous counterpart와 protocol-matched hybrid baselines를 통해 composition과 architecture의 payoff를 함께 측정한다 | Learning Effectiveness와 Execution Efficiency | Matched quality, hybrid competitiveness, throughput과 stall breakdown |

Learner contract는 독립 기여나 정리가 아니라 기여 1의 learning composition을 압축하는 명세다.
§3.1의 구성적 유도와 §3.2의 architecture가 각각 그 명세와 실행을 회수하며, unit·contract test는
implementation QA로만 남긴다. 공개 핵심 약속은 `preserve the intended learning composition without
turning its boundaries into execution barriers`로 고정하고, 최적성을 요구하는 `maximize` 대신
`retain`, `realize`, `without giving back`을 사용한다.
Headline은 quality SOTA가 아니라 matched quality를 전제로 한 실행 효율이다. `1.64x` throughput은
end-to-end payoff를, `54.7% -> 3.2%` stall은 그 기전을 설명하며, `+0.8` points는 통계적 근거가
닫히기 전까지 경쟁력의 보조 증거로만 둔다.

| 문단 | 서사적 역할 | 독자가 가져갈 판단 |
|---|---|---|
| **1문단** | Compute–signal double bottleneck | 어려운 RLVR일수록 비싼 rollout과 부족한 성공 신호가 함께 심해지므로 두 연구 방향을 함께 다뤄야 한다. |
| **2문단** | 단일 composition gap과 research question | Complete group은 source admission과 RL contribution을 함께 정의하며, 이를 execution barrier 없이 source-specific learning까지 연결해야 한다. |
| **3문단** | Execution architecture | Learning boundary를 보존하면서도 group 대기를 전역 장벽으로 만들지 않아 full asynchrony를 유지한다. |
| **4문단** | Learning-composition boundary | Contract는 runtime이 다시 쓰면 안 되는 complete-group decision, source-specific update, declared composition의 경계를 압축한다. 정확한 구성과 유도는 Method가 소유한다. |
| **5문단** | Empirical payoff | Matched quality와 same-hardware throughput·stall을 함께 제시하여 학습 효과와 실행 효율이 동시에 유지됨을 보인다. |

### 4. 핵심 충돌 지도와 실현 원장

**역할:** 아래 표는 기여 1·2, learner contract, §3.1·§3.2, runtime figure가 공유하는 **단일
authoritative design ledger**다. 이후 절은 이 세 경계를 다시 정의하지 않고, 해당 절에서 필요한
realization과 공개 표현만 덧붙인다. 공개 Introduction은 세 경계를 열거하지 않고, **의미적 경계를
그대로 실행 장벽으로 두면 효율을 잃고, 경계를 지우면 runtime이 learning composition을 다시 쓴다**는
하나의 composition gap으로 추상화한다.

| Canonical boundary | 보존할 요구 | 나이브 결합의 실패 | StreamWeave design | Implementation witness | 소유 범위와 공개 위치 |
|---|---|---|---|---|---|
| **Complete-group decision** | Complete group이 source와 policy-side relative signal을 결정하되, trajectory attempt는 서로를 막지 않고 실행 | Groupwise control flow는 synchronization idle을 복원하고, premature admission은 source와 relative signal을 선취 | Attempt-level execution을 유지하고 source decision 직전에만 group을 nonblocking reconstruction한 뒤 route-before-admission | `group_uid`·attempt 순서, `HptPromptGroupAccumulator`, `HptRolloutGate`; 실패 group의 fail-closed 처리는 Appendix | StreamWeave의 핵심 bridge이며 §3.2에서 회수. Reconstruction만을 독립 novelty로 주장하지 않음 |
| **Source-specific update** | Policy와 expert가 각자 정의된 reference, objective, correction을 유지하면서 하나의 continuous stream에서 소비 | Source별 동기 phase는 overlap을 분절하고, branch-blind correction은 expert gradient를 변형 | Source를 transport 전에 확정하고 group/source/generation-policy context를 trainer까지 유지하며 rollout-only correction의 적용 범위를 제한 | `route_rollout_sample`, `materialize_training_batch`, policy rollout context, expert self-detached CE와 IS identity | Learning composition의 핵심이며 §3.1의 유도와 §3.2의 handoff에서 회수 |
| **Declared composition** | Whole-group membership과 사전에 정한 weighting·reduction을 유지하면서 도착 순서와 무관하게 batch를 구성 | Group 분할이나 추가 normalization은 data selection과 effective mixture를 암묵적으로 변경 | Source가 확정된 group을 보존하고 batching이 §3.1의 weighting과 reduction을 다시 쓰지 않도록 interface를 제한 | 명시적 `beta`, branch-blind reduction; fixed-grain alignment와 예외 회계는 Appendix | Method에는 interface requirement만 남기고 framework-specific alignment는 Appendix |

**상속된 비동기 기반과 framework-specific 실현:** 아래 요소는 end-to-end 실행에 필요하지만
StreamWeave의 독립 기여와 동급으로 세우지 않는다. 본문에서는 완전한 시스템의 작동을 닫는 데 필요한
만큼만 언급하고, 정확한 제어와 정렬 절차는 Appendix에서 설명한다.

| 요소 | 현행 역할 | 공개 지위 |
|---|---|---|
| **Independent attempt scheduling** | Trajectory-level 실행 자유를 제공 | Fully-asynchronous runtime에서 상속한 효율 전제. StreamWeave의 novelty는 complete-group decision과의 결합 |
| **Bounded queue, backpressure, parameter refresh** | Backlog와 policy freshness를 제어하며 generator–trainer overlap 유지 | 상속된 async substrate. §3.2에서 한 문장으로만 인정 |
| **Variable-cardinality batch alignment** | Source에 따라 달라지는 sample 수를 현행 fixed-grain learner에 연결 | `verl`-specific realization. `n:1`, divisor, deferral, trim-and-carryover, 예외 회계는 Appendix |

Learner contract는 위 세 canonical boundary를 학습 관점에서 압축하는 specification이다. 실제 수행한
조치와 공개 위치는 같은 표의 implementation witness와 마지막 열을 따르며, Introduction에서는 이를
다시 열거하지 않는다. Method는 각 장치보다 그것이 보존하는 boundary를 먼저 설명한다.

**2문단 논증 순서:** 조건부 expert source를 먼저 설명하고, complete group의 source-admission·RL-signal
이중 역할을 밝힌 뒤, trajectory-level execution과의 double bind를 제시한다. Group reconstruction에서
멈추지 말고 source choice가 data와 update rule까지 바꾼다는 결과로 전진한 다음 research question을
제시한다. HPT, `n:1`, reducer, deferred materialization은 이 문단에 넣지 않는다.

**사실 앵커와 회수 위치:**

| 주장 | 사실 앵커 | 회수 위치 |
|---|---|---|
| Fully-asynchronous separation과 trajectory-level execution | AReaL (NeurIPS 2025)은 generation-learning 분리, Laminar (EuroSys 2026)는 trajectory-level independent execution; `fully_async_rollouter.py::_submit_hpt_trajectory_attempts` | Intro의 효율 원리; Method의 Execution Architecture |
| Complete-group decision | `hpt_gate.py::HptRolloutGate.route` | Intro의 판단 경계; Method의 selector와 attribution |
| Group-relative RL signal | `ray_trainer.py::compute_advantage`; `core_algos.py::compute_grpo_outcome_advantage` | Intro의 group 역할; Method의 objective |
| Nonblocking group reconstruction | `hpt_rollout_accumulator.py::HptPromptGroupAccumulator`; `fully_async_rollouter.py::_record_hpt_trajectory_attempt_result` | Method의 Execution Architecture |
| Source-dependent materialization | `hpt_gate.py::route_rollout_sample`; `hpt_assembler.py::materialize_training_batch` | Method의 learning composition과 stream interface |
| Variable-cardinality assembly | `fully_async_trainer.py::_get_samples_from_queue`; `_plan_row_alignment_deferral` | Method에는 batching이 group membership과 weighting을 바꾸지 않는다는 조건만 남기고, trim-and-carryover와 예외 회계는 Appendix |
| Partial-group routing | Complete-group selector가 미완성 맥락에서는 정의되지 않음을 보이는 counterexample; 현행 runtime은 gate 전에 group을 복원 | Complete-group decision의 동기로만 사용하며 현행 장애나 Intro headline으로 서술 금지. 실제 dump에서 비자명한 빈도와 효과가 확인될 때만 failure analysis 후보로 승격 |

**본문 승격 필터:** 본문급 설계는 `nonblocking group reconstruction`과 source를 확정한 뒤 그 맥락을
learner까지 보존하는 transport boundary다. Batching은 §3.1의 group membership과 weighting을 다시
쓰지 않아야 한다는 interface condition만 한 문장으로 남긴다. Accumulator 자료구조, queue 크기,
정확한 batch divisor, subset-sum, trim-and-carryover 절차·예외, tensor field는 Appendix로 내린다.
실현된 mixture는 `β`만이 아니라 routing 빈도, cardinality, token volume에도 의존한다.

### 5. 포지셔닝과 정보 계층

**Research object and title:** 논문의 주인공은 complete-group 결과가 policy와 expert 중 사용할 data와
update를 결정하는 group-conditioned setting을 fully-asynchronous execution과 결합하는
algorithm-system architecture인 **StreamWeave**다. 제목은 **StreamWeave: Reconciling Off-Policy
Expert Supervision with Fully-Asynchronous Policy Learning**으로 고정한다. HPT는 group-success
selector의 concrete instantiation일 뿐 방법의 정체성이나 포지셔닝 근거가 아니다.

| Accepted-conference 계보 | 해결한 제약 | 이 논문의 범위 밖에 남긴 문제 |
|---|---|---|
| **Fully-asynchronous policy learning**: Asynchronous RLHF, AReaL, TBA | Policy rollout의 생성과 학습을 비동기화하고 policy lag를 관리 | Learning stream의 source가 policy experience 밖으로 바뀌며 complete group이 source와 update까지 결정하는 경우 |
| **Policy/expert learning**: LUFFY, CHORD, ReLIFT, SRFT | Expert signal을 선택·가중하여 policy-generated signal의 한계를 보완 | 그 learning decision을 trajectory-level full asynchrony에서 유지하는 실행 구조 |
| **StreamWeave** | 두 계보의 기능을 나열하는 대신, complete-group decision과 source-specific update를 global execution clock 없이 함께 실현 | 실증 범위는 현행 two-source, group-conditioned RLVR setting으로 제한 |

**Reviewer classification을 잠그는 규칙:** StreamWeave를 새로운 hybrid objective나 범용 async framework로
분류시키지 않는다. 올바른 분류는 **heterogeneous learning을 full asynchrony 아래에서 실현하는
composition architecture**다. AReaL이 group-relative policy learning을 다룬다는 사실은 충분히
인정하되, StreamWeave에서는 group completion이 source와 유효한 update까지 바꾸는 추가 control
dependency라는 차이를 세운다. `AReaL + HPT` 공격에는 부품의 새로움이 아니라 §4의 세 비합성성,
즉 groupwise blocking, premature decision, branch-blind update로 답한다.

**Off-policy distinction:** stale rollout은 learner policy의 과거 버전이 생성하고 behavior-policy
context를 가지므로 mismatch correction이 정의된다. Expert trajectory는 policy lineage 밖의 supervised
target이며 같은 correction의 reference를 갖지 않는다. 후자를 더 오래된 rollout로 취급하지 않으며,
이 구별의 정확한 objective와 유도는 Introduction이 아니라 §3.1이 소유한다.

**Related Work와 literature rule:** 각 계보는 `분야명 -> 해결한 문제 -> 대표 accepted-conference
연구 -> 범위 밖에 남은 문제` 순서로 쓴다. 선행연구의 결함 목록을 만들지 않고 마지막 composition
문단에서만 StreamWeave의 위치를 회수한다. 포지셔닝 근거에는 accepted-conference 논문만 사용하며,
HPT는 Introduction과 Related Work에서 제외하고 Method에서 selector의 출처로 한 번 attribution한다.

| 위치 | 남길 내용 |
|---|---|
| **Abstract** | 관찰 가능한 compute–signal bottleneck, group-conditioned setting의 범위, 실행 충돌과 학습 차이, `trajectory-level execution → local group reconstruction → source-appropriate update`, 비교 방법 전반과 matched synchronous quality, same-hardware 작업 시간과 trainer 대기 비율. Accumulator, queue, mask, self-detach는 명명하지 않고 결과는 quality와 efficiency 한 문장씩으로 닫음 |
| **Introduction** | Group-based RLVR의 compute–signal double bottleneck, complete group의 source-admission·RL-contribution 이중 역할, 하나의 composition gap, StreamWeave의 design judgment, learning-composition boundary, headline empirical payoff |
| **Related Work** | 동기식 RL의 실행 효율을 개선한 연구와 policy-generated signal을 외부 data로 보완한 accepted-conference 연구, 그리고 group-dependent learning을 fully-asynchronous execution에 결합할 때 남는 문제 |
| **Method: Learning Composition** | HPT selector attribution과 정확한 routing rule, source-specific objectives, behavior-policy reference, weighting과 reduction |
| **Method: Execution Architecture** | Trajectory-level execution 안의 nonblocking group reconstruction, source-before-transport, source와 생성 맥락의 learner-side 보존; queue와 flow control은 end-to-end realization으로만 언급 |
| **Experiments** | Protocol-matched quality, throughput, trainer idle, 학습 동역학 해석. Quality-versus-wall-clock은 §6의 evidence gate를 충족할 때만 승격 |
| **Appendix** | Source-faithful composition의 전체 미분과 scope, queue configuration, partial rollout, trim+carryover, schema, 개별 operator와 보조 분석 |

Aliasing lemma, n-source 일반화, necessity/sufficiency/selectivity 서사, CISPO·decoupling의 부정
결과는 Introduction과 공개 contribution에서 제외한다.

### 6. 증거 게이트와 주장 규율

아래 표가 논문 주장과 수치의 단일 evidence ledger다. `LOCKED`는 공개 본문에 사용할 수 있는 결과,
`DERIVED`는 내부 분석은 끝났지만 공개 asset과 함께 제시해야 하는 결과, `PENDING`은 분석이 닫히기 전까지
headline에 사용할 수 없는 결과, `APPENDIX`는 본문 논증을 보조하는 구현·진단 자료를 뜻한다.

| Claim | Status | Source | Public home | 허용 문구 | Caveat |
|---|---|---|---|---|---|
| **Fixed-checkpoint quality** | `LOCKED` | §6.1의 반올림 전 score 원장과 고정 checkpoint 평가 | Table 1, Abstract, Introduction | 비교한 protocol-matched 방법 중 평균 38.5; 동일 selector·objective의 synchronous counterpart 37.7과 대등한 품질 | `\dagger` 외부 인용 행은 동일 protocol ranking의 근거에서 제외. Main/sync의 정확한 checkpoint와 raw evaluation artifact ID는 Table 1 확정 전에 provenance manifest에 등록 |
| **LUFFY 대비 +0.8 points** | `PENDING` | Main 38.4910, LUFFY 37.6678의 문항별 결과 | Experiments 본문만 | Paired uncertainty analysis가 닫힌 뒤 제한적으로 해석 | Abstract·Introduction headline 금지 |
| **Same-hardware execution efficiency** | `LOCKED` | W&B full history: sync `v96fvd0p`, main `oki4kv8u` | Abstract, Introduction, Execution Efficiency | 동일 8×B200에서 `2.78→4.58 groups/s`(`1.64×`), 128 groups `46→28초`, trainer wait `54.7%→3.2%` | 기존 CISPO run의 `1.54×`와 섞지 않음 |
| **Quality-versus-wall-clock** | `PENDING` | Main·control checkpoint와 W&B timestamp 정렬 | Learning Effectiveness figure | 완성된 curve와 checkpoint protocol이 함께 있을 때만 wall-clock 우위 주장 | 현재 headline과 contribution evidence에서 확정형 사용 금지 |
| **Learning dynamics** | `DERIVED` | RL-only와 expert-channel checkpoint/window 분석 | Learning Effectiveness figure | 초반 유사, 후반 RL-only 정체, expert channel의 후반 기여 | `+3.4`, cold-start 역전, 장기 안정성은 곡선과 함께만 제시 |
| **Source-faithful composition** | `LOCKED` | §3.1의 mixed-objective 구성적 유도 | Method와 Appendix proof | 고정된 complete group, learner parameter, source별 objective·aggregation 아래에서 실제 objective가 선언된 contribution으로 환원 | 전체 system correctness, 보편적 necessity, optimizer-trajectory equivalence는 주장하지 않음 |
| **Implementation QA** | `APPENDIX` | Unit·contract test, gradient/reducer equality check | 저장소와 필요시 Appendix | 선택한 구현이 명세를 따르는지 확인 | 논문의 독립 evidence나 section으로 사용하지 않음 |

Quality protocol은 AIME24·AIME25·AMC에 `mean@32`, MATH500·Minerva·Olympiad에 `mean@8`을
사용한다. 실제 training dump에서 비자명한 failure의 빈도와 효과 크기가 확인된 분석만 Experiments
승격을 검토한다. `maximize`, `provably necessary`, `zero-waste`, 무조건적 data preservation,
“mixture strength는 β만으로 결정된다”는 표현을 사용하지 않고 실제 보장 범위와 예외를 직접 명시한다.

**Evaluation provenance manifest:** Table 1을 최종 배치하기 전에 자체 평가한 각 행에 대해
`model artifact/checkpoint`, grader와 decoding config, 공유 evaluation-seed manifest, raw result
artifact의 경로 또는 ID를 한 원장에 기록한다. 현행 score와 macro-average는 잠겼지만 main과 sync의
정확한 checkpoint/artifact 식별자는 이 문서에 아직 등록되지 않았다. 이를 추정해서 채우지 않고,
Experimental Setup 또는 Appendix의 reproducibility block에서 확정한다.

#### 6.1 최종 benchmark 표 산술 원장

최종 quality 표는 모든 행을 **소수점 한 자리**로 통일한다. 자체 평가와 통일 grader로 재평가한
행은 현재 표시된 두 자리 수를 다시 반올림하지 않고, 문항별 binary correctness의 원시 합계에서
benchmark score를 계산한 뒤 한 번만 반올림한다. `AVG`는 여섯 benchmark의 **반올림 전 score를
동일 가중한 macro-average**로 계산한 뒤 한 번만 반올림하며, 표에 표시된 한 자리 수를 다시 평균하지
않는다. 이 규칙은 `16.1458... -> 16.1`을 `16.15 -> 16.2`로 잘못 바꾸는 이중 반올림을 막는다.

| Benchmark | 문항 수 | Sampling | Binary 판정 수 | 최소 score 간격 (percentage points) |
|---|---:|---:|---:|---:|
| AIME24 | 30 | `mean@32` | 960 | 0.10417 |
| AIME25 | 30 | `mean@32` | 960 | 0.10417 |
| AMC | 83 | `mean@32` | 2,656 | 0.03765 |
| MATH500 | 500 | `mean@8` | 4,000 | 0.02500 |
| Minerva | 272 | `mean@8` | 2,176 | 0.04596 |
| Olympiad | 674 | `mean@8` | 5,392 | 0.01855 |

`mean@32`는 32개 stochastic generation의 평균 pass@1이며 `pass@32`가 아니다. Experimental Setting에는
AMC 평가본이 83문항이고 Olympiad 평가본이 674문항임을 명시한다. SFT와 RL-only의 외부 인용값은
`\dagger`로 구별하고 출처와 원 평가 protocol을 함께 밝힌다. 외부 protocol이 다르면 통일 protocol로
재평가한 행처럼 서술하지 않는다.

| Model | AIME24 | AIME25 | AMC (83) | MATH500 | Minerva | Olympiad | **AVG** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 6.6 | 3.5 | 31.2 | 43.3 | 10.9 | 24.9 | **20.1** |
| Instruct | 10.6 | 9.4 | 47.0 | 75.5 | 29.5 | 40.4 | **35.4** |
| SFT$^{\dagger}$ | 11.7 | 13.2 | 37.8 | 70.6 | 26.8 | 31.3 | **31.9** |
| RL-only$^{\dagger}$ | 11.8 | 7.7 | 40.2 | 61.8 | 26.8 | 32.0 | **30.1** |
| Async RL (SFT + RL) | 12.9 | 7.9 | 44.9 | 75.8 | 28.8 | 39.5 | **35.0** |
| HPT (sync) | 15.4 | 12.6 | 45.8 | 78.0 | 31.3 | 43.2 | **37.7** |
| SRFT | 12.3 | 10.4 | 43.0 | 71.6 | 26.1 | 38.4 | **33.7** |
| ReLIFT | 12.6 | 8.1 | 40.3 | 74.6 | 28.6 | 39.4 | **34.0** |
| Oat-Zero | 17.2 | 12.6 | 49.6 | 73.7 | 30.1 | 38.1 | **36.9** |
| LUFFY | 15.1 | 14.0 | 46.0 | 77.5 | 30.0 | 43.5 | **37.7** |
| CISPO (ours, ablation) | 13.2 | 13.1 | 43.9 | 77.2 | 31.8 | 41.3 | **36.8** |
| No-CISPO (ours, main) | 16.1 | 13.0 | 47.0 | 78.5 | 33.0 | 43.2 | **38.5** |

표시값만 다시 평균하면 SRFT와 ReLIFT는 각각 33.6과 33.9로 보이지만, 반올림 전 score의 macro-average는
각각 33.6542와 33.9593이므로 공개 `AVG`는 33.7과 34.0이 맞다. 이 오해를 막기 위해 표 각주는 다음으로
고정한다.

> Scores are rounded to one decimal place. Avg. is the unweighted macro-average computed from
> unrounded benchmark scores. $^{\dagger}$ Results reported by the original source.

현재 원점수 기준 main의 macro-average는 38.4910, LUFFY는 37.6678이며 격차는 0.8233 points다. 공개
표에는 반올림된 두 값을 그대로 제시하되, Abstract에서는 이 작은 margin을 headline으로 삼지 않는다.
대신 비교한 RL·expert-trajectory 활용 방법들보다 높은 평균 성능과 matched synchronous counterpart에
준하는 품질을 함께 제시한다. `+0.8 points`의 직접 해석은 두 행이 같은 grader, decoding, sampling
budget을 사용하고 paired uncertainty analysis까지 닫힌 경우에만 Experiments에서 허용한다.

### 7. Method §3 작성 헌장

#### 7.1 두 subsection의 단일 역할

§3은 `Learning Composition → Fully-Asynchronous Execution` 순서로 고정한다. 두 subsection은
동등한 추상화 수준의 병렬 구성요소가 아니다. §3.1은 실행 순서와 무관하게 **무엇이 유효한 mixed
update인지** 정의하고, §3.2는 그 학습 구성을 **global group barrier 없이 어떻게 실현하는지**
설명한다. 즉 §3.1이 학습 구성을 정의하고 §3.2가 그 정의를 만족하는 시스템 실현을 제공한다. 두 절의
경계를 다음처럼 잠근다.

| 절 | 독자의 질문 | 이 절이 소유하는 답 | 넘기지 않을 내용 |
|---|---|---|---|
| **3.1 Learning Composition** | Complete group은 어떤 source와 update contribution으로 변환되는가? | Complete-group decision, source-specific update, provenance에 따른 correction domain, declared composition과 그 구성적 유도 | Scheduler, accumulator, queue, backpressure, row alignment |
| **3.2 Fully-Asynchronous Execution** | 앞서 정의한 composition을 보존하면서 어떻게 pipeline을 계속 전진시키는가? | Trajectory-level execution과 complete-group reconstruction의 결합, source-before-transport, source와 생성 맥락의 learner-side 보존 | Loss 유도, self-detach 증명, fixed-grain batch alignment, 개별 mask와 tensor field |

#### 7.2 Method 주장 위계: definition → derivation → design → realization

§3.1과 §3.2의 `정의 → 실현` 관계는 유지하되, Method 안의 개별 주장은 아래 네 수준으로 구별한다.
이는 subsection을 하나 더 만드는 구성이 아니라, **무엇을 일반적 조건으로 주장하고 무엇을
StreamWeave의 설계로 소유하며 무엇을 현행 시스템의 구현으로 제시할지** 정하는 지위 체계다.

| 수준 | 핵심 질문 | 논문에서의 역할 |
|---|---|---|
| **Learning definition** | Fully-asynchronous execution 아래에서도 어떤 mixed update를 유지해야 하는가? | Source별 contribution, operator 적용 범위, aggregation을 명시하고 contract로 압축 |
| **Constructive derivation** | 실제 StreamWeave objective가 그 정의를 만족하는가? | Policy contribution의 보존, expert-gradient 환원, declared composition을 직접 유도 |
| **StreamWeave design** | 그 조건을 full asynchrony의 효율을 반납하지 않고 어떻게 만족시키는가? | 학습 경계와 실행 장벽을 분리하는 architecture와 설계 판단 |
| **Concrete realization** | 현재 실험과 코드에서 그 설계를 어떤 objective와 mechanism으로 구현했는가? | Main-run 명세와 Appendix의 구현 세부 |

Learning definition은 모든 heterogeneous learning에 대한 보편 공리가 아니라 이 논문이 다루는
**group-conditioned policy/expert composition의 정체성 조건**이다. Contract는 이 정의를 압축하고,
구성적 유도는 실제 objective를 대입해 각 branch가 선언한 contribution으로 환원됨을 보인다.
Architecture는 수학적으로 유일한 해법이 아니라, 이 composition과 비동기 효율을 함께 달성하기 위해
StreamWeave가 선택하고 소유하는 design이다. 구현 테스트는 코드 품질을 관리할 뿐 수학적 유도나
과학적 실험을 대신하지 않는다. Composition requirement, core design, inherited substrate의 지위는
§4가 소유하며, 이 절은 현행 시스템의 concrete realization만 기록한다.

| Concrete element | 지위 | 본문에서의 취급 |
|---|---|---|
| Routed-group queue와 trainer-side conversion | **Concrete realization** | §4의 source-before-transport boundary를 현행 runtime에서 구현하는 방식 |
| HPT success-conditioned hard switch | **Concrete instantiation** | 실험에서 사용한 group-conditioned selector로 한 번 attribution |
| Entry-proximal anchor, token IS, vanilla clipped PPO | **Main-run realization** | 현행 policy branch의 정확한 명세이며 StreamWeave 자체의 일반 원리는 아님 |
| Self-detached expert reference | **Implementation mechanism and formal witness** | Shared learner path에서 supervised contribution을 복원하며 §3.1에서 핵심 환원을 짧게 보임 |
| Trim-and-carryover | **Framework-specific realization** | Appendix에서 fixed-grain learner와의 정렬 방법으로 설명 |

Self-detach는 이 위계를 보여주는 대표 사례다. 먼저 expert branch가 의도한 supervised contribution을
정의하고, 현행 realization이 expert reference를 self-detach하여 그 contribution으로 환원됨을 보인다.
§3.1에는 ratio가 1이면서 supervised gradient가 남는 핵심 유도를 제시하고, Appendix에는 전체 미분과
정확한 적용 범위를 둔다. 이 유도는 self-detach가 유일한 해법임을 보이는 것이 아니라, **선택한
realization이 목표 contribution을 만족함을 보이는 formal witness**다.

#### 7.3 §3.1 Learning Composition

§3.1은 **mechanism-first, contract-last**로 쓴다. Contract-first는 Introduction을 반복하고,
objective-first는 HPT loss의 재서술처럼 보이므로 사용하지 않는다. 문단 순서는 아래 네 개로 고정한다.

| 문단 | 역할 | 반드시 전달할 판단 | 이후 회수 |
|---|---|---|---|
| **Source selection from a completed group** | 무엇을 결합하는지 정의 | Complete rollout group이 policy 또는 expert source를 선택하며, 이 결정은 개별 trajectory가 아니라 group 결과에 의존 | §3.2의 complete-group reconstruction과 arrival-order independence |
| **Source-specific gradient contributions** | 선택된 source가 어떻게 학습하는지 정의 | Source selection은 data와 함께 reference policy, objective, correction rule을 결정 | Source-faithful composition proposition과 quality 결과 |
| **Where asynchronous correction applies** | 비동기 correction의 적용 범위 설명 | Importance weighting은 생성 policy probability가 기록된 rollout에만 적용되고 expert trajectory에는 적용되지 않음 | Expert-gradient 환원과 Appendix의 전체 미분 |
| **Combining policy and expert gradients** | 하나의 learner update로 닫기 | 각 gradient를 먼저 계산한 뒤 미리 정한 weighting과 batch aggregation으로 결합 | Explicit aggregation equation과 §3.2 bridge |

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
mechanism을 일반 원리로 격상하지 않는다. 이어서 **Source-faithful composition**을 좁은 명제로
제시한다. Complete group, learner parameter, source별 objective와 aggregation을 고정하면 policy row는
선언된 policy contribution을 유지하고, expert row는 self-detached reference를 통해
`beta`-weighted supervised contribution으로 환원되며, 두 contribution은 위의 `Aggregate`에서만
결합된다. 이는 전체 optimizer trajectory의 동등성이나 임의의 SFT implementation과의 동일성을
주장하지 않는다.

현재 구현은 그 뒤 산문으로 분리한다. Policy branch는 group-relative advantage, learner-entry
policy snapshot, rollout-to-entry token IS, vanilla clipped PPO를 사용한다. Expert branch는
`beta_r`로 조절한 supervised gradient를 사용하고 importance weight를 1로 둔다. Current-policy
log-probability의 stop-gradient copy가 forward ratio를 1로 만들면서 supervised gradient를 남긴다는
핵심은 proof sketch로 직접 보이고, 전체 미분과 mask·tensor-level 구현은 Appendix로 내린다.

Contract는 출발점이나 별도 장문의 Definition이 아니라 위 구성과 유도를 압축하는 결론이다. 세 조건을
`(i) Complete-group decision`, `(ii) Source-specific update`, `(iii) Declared composition` 순서로
한 번만 제시한다. 첫째는 §3.2의
complete-group reconstruction이, 둘째와 셋째는 §3.1의 source-faithful derivation이 회수한다. §3.1은
다음 bridge로 닫는다.

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
비활성이었다는 결과 역시 Appendix에 둔다. 효율 headline은 §6에 잠근 main-run W&B 원장의
`1.64x`를 사용하며, 기존 CISPO arm의 `1.54x`와 혼용하지 않는다. `entropy/KL 제외`라고
뭉뚱그리지 않고, main에서는 entropy exclusion만 활성이고 KL은 전역 비활성이라고 쓴다.

#### 7.5 §3.2와 Generator–Trainer authoritative flow

코드 기준 실행 순서는 아래가 단일 진실이다. Accumulator는 Generator 내부에서 Gate보다 먼저
동작하고, Queue에는 learner row가 아니라 source가 확정된 prompt-group record가 들어간다.

```text
Prompt group
  -> independent attempt scheduling                    [inherited async substrate]
  -> parallel trajectory generation
  -> complete-group reconstruction                     [core StreamWeave design]
  -> group-conditioned source decision  <- expert trajectory store
  -> routed group + source/generation context          [core StreamWeave boundary]
  -> bounded group queue                               [inherited async substrate]
  -> trainer-side materialization                      [concrete realization]
  -> fixed-grain batch alignment                       [framework-specific realization]
  -> source-specific update
  -> updated policy
  -> parameter refresh back to Generator               [inherited async substrate]
```

§3.2는 §4 authoritative design ledger의 세 boundary를 같은 순서와 명칭으로 회수하고, 각 문단을
`requirement → naive failure → StreamWeave design → 보존되는 성질`의 인과로 쓴다. 전체 서술은
`inherited asynchronous freedom → added group dependency → local reconstruction →
source-before-transport → context-preserving handoff → end-to-end overlap` 순서를 따르며 별도의 충돌
분류나 contract 명칭을 추가하지 않는다. Independent attempt scheduling, bounded queue, backpressure,
parameter refresh는 상속한 실행 기반으로 인정하되
별도 기여로 나열하지 않는다. StreamWeave의 소유 지점은 complete group이 source와 update rule까지
결정하는 환경에서 필요한 context만 국소적으로 복원하고 그 결정을 learner까지 보존하는 bridge다.

`n:1`, required multiple, subset-sum, trim-and-carryover, queue 크기와 예외적 discard는 Appendix다.
본문에는 batching이 group membership이나 §3.1의 weighting을 바꾸지 않는다는 조건만 남기며
`zero-waste`, `crash-free`, 무조건적 보존을 주장하지 않는다.

#### 7.6 Runtime figure contract

Method의 간단한 runtime 그림은 Figure 1의 문제 서사를 반복하지 않고, §3.1에서 정의한 learning
composition의 요구가 §3.2의 runtime에서 어디에 실현되는지를 보여준다. 하나의 왼쪽-오른쪽 pipeline을
그리고, 그 위에 세 requirement를 callout으로 대응시킨다.

```text
Learning composition:        [1. Complete-group decision]  [2. Source-specific update]  [3. Declared composition]
                                           |                            |                              |
Runtime architecture: Prompt -> attempts -> reconstruct -> route -> routed-group stream -> trainer conversion -> update
```

이 그림의 상단은 별도의 물리적 pipeline이 아니라 §3.1이 §3.2에 부과하는 요구사항이다. 큰 물리 영역은
`Generator`, `Routed-group stream`, `Trainer` 세 개로 제한한다.

| Learning Composition의 요구 | Runtime에서의 실현 |
|---|---|
| **Complete-group decision** | Attempt를 독립 실행한 뒤 gate 직전에 complete group을 복원하여 source를 결정 |
| **Source-specific update** | Routed group에 source와 생성 맥락을 유지하고 trainer에서 source에 맞게 변환·update |
| **Declared composition** | Batch 구성이 group membership과 미리 정한 weighting·reduction을 바꾸지 않게 함 |

| 내부 객체 | 공개 그림의 명칭 | 표시할 의미 |
|---|---|---|
| trajectory scheduler + rollout workers | **Independent attempts** | Group을 실행에서만 분해 |
| `HptPromptGroupAccumulator` | **Complete-group reconstruction** | Gate 전에 semantic context 복원 |
| `HptRolloutGate` + expert store | **Source decision** | Policy group 또는 expert trajectory 확정 |
| `MessageQueue` | **Routed-group stream** | Source가 확정된 group과 필요한 맥락을 운반; learner row가 아님 |
| `HptBatchAssembler` | **Trainer-side conversion** | Source가 확정된 group을 해당 training sample로 변환 |
| Trainer objective | **Source-specific update** | Policy와 expert에 유효한 contribution을 정의한 뒤 결합 |
| checkpoint/queue budget | **Runtime flow control** | 필요할 때만 낮은 강조도의 dashed path로 표시하며 핵심 mechanism 번호를 부여하지 않음 |

Solid arrow는 data plane, dashed reverse arrow는 parameter refresh와 capacity feedback을 나타낸다.
Provenance는 별도 계산 모듈이 아니라 routed group에 붙는 `group/source/policy-version` metadata로
표현한다. 그림 내부에는 CISPO, self-detach 미분, token IS, `beta=0.3`, queue 크기, carryover 절차를
넣지 않는다. 구체적인 policy label이 필요하면 `Policy update` 또는 `GRPO + vanilla PPO`를 사용한다.

그림은 §3.1의 learning composition 요구를 §3.2 runtime의 구체적인 경계에 대응시키며, source
decision에 필요한 group completion만 국소적으로 기다리고 나머지 pipeline은 계속 진행됨을 보여준다.
