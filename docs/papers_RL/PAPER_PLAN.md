# Paper Plan — StreamWeave (AAAI)

> **Tracked canonical plan.** 이 파일은 저장소 안에서 논문 실행 계획을 소유한다. 공개 문안과
> claim boundary는 `Full_Paper_Draft_ko.md`가 우선한다.

_Last updated: 2026-07-28_

## 0. 문서 위계와 현재 상태

논문 작업의 전체 문서 지도는
`AAAI_RL/docs/papers_RL/README.md`가 소유한다. 이 문서는 **현재 원고를 완성하기 위한 실행
계획**이며, 논문의 실제 문안과 주장 범위는
`AAAI_RL/docs/papers_RL/Full_Paper_Draft_ko.md`가 소유한다. 충돌할 경우 아래 순서를 따른다.

1. **Full_Paper_Draft_ko.md** — 공개 원고, 논문 헌법, authoritative design ledger,
   evidence ledger
2. **PAPER_PLAN.md** — 남은 집필 순서와 section/asset 계획
3. **Paper_writing/TASKS.md** — workspace에 있을 때만 사용하는 선택적 dashboard mirror
4. **W&B 원장 및 Efficiency.tex** — run 원자료와 현행 main-run 효율 Appendix ledger
5. **Ablation_RL.md** — `REFERENCE ONLY`; run 계보와 학습 결과의 원자료
6. **DR-001~005** — `REFERENCE ONLY`; 개별 구현 결정의 기술적 근거
7. **Draft.tex, Improvement_RL.md** — `LEGACY/REFERENCE ONLY`; 구현과 디버깅의 역사적 기록

하위 문서의 과거 수치나 용어를 공개 원고로 직접 올리지 않는다. 하위 문서에서 유효한 자산을
가져올 때도 `Full_Paper_Draft_ko.md`의 claim boundary와 evidence status를 먼저 적용한다.
`TASKS.md`는 이 문서의 P0--P2를 짧게 투영할 뿐 독립적인 연구 결정이나 완료 상태를 만들지 않는다.

| 항목 | 현행 결정 |
|---|---|
| Working title | **StreamWeave: Reconciling Off-Policy Expert Supervision with Fully-Asynchronous Policy Learning** |
| 공개 브랜드 | `StreamWeave` 하나만 사용 |
| 연구 대상 | Group-conditioned policy/expert learning을 fully-asynchronous execution에 결합하는 algorithm-system architecture |
| Canonical main | Decoupled policy correction + vanilla clipped PPO (`M5abl_nocispo`, W&B `oki4kv8u`) |
| HPT의 지위 | Policy rollout과 expert trajectory를 선택적으로 사용하는 learning program의 출처; threshold 자체는 StreamWeave의 novelty가 아님 |
| CISPO의 지위 | Main에서 기각된 ablation; Appendix의 secondary diagnosis |
| 핵심 실증 | Fixed-checkpoint quality, expert-off learning dynamics, expert-supervised generation workload 변화, resource-matched execution efficiency |
| 신규 학습 | 현재 핵심 주장을 위해 요구하지 않음 |
| 공개 원고 상태 | English v2의 Abstract, Introduction, Related Work와 §3.1 개정을 완료했다. 현재 §3.2 opening과 산문을 정제하면서, §3.1의 learning contract를 명시적 제약으로 받되 C2의 architecture를 하위 구현으로 낮추지 않는 관계를 닫고 있다. §3의 core dataflow, §4의 evidence order와 현행 Figure 1--3의 역할은 유지한다. Appendix A.3은 두 source에 같은 loss averaging을 적용하는 식과 이유를 소유하며 framework-specific alignment와 carryover는 공개 본문과 Appendix에서 제외한다. |
| 온보딩 상태 | Full draft §0, project memory, TASKS, code-facing Overview를 2026-07-23 기준으로 동기화; 저장소 `AGENTS.md`는 코드 전용으로 유지 |

### 0.1 본문 추상화 게이트

공개 본문이 소유할 내용은 아래 세 축으로 제한한다.

1. **Composition problem:** Complete group이 필요한 학습 판단과 trajectory-level fully-asynchronous
   execution의 충돌
2. **One-path learning composition:** Group이 정한 source를 signal·reference·correction이 명시된
   training input으로 변환하고, 하나의 policy objective와 같은 averaging rule로 소비
3. **Decision-localized execution:** Complete group을 학습 판단 경계로 유지하되, 그 completion을
   serialized critical-path barrier로 만들지 않는 execution architecture

본문의 장치와 수식은 반드시 이 셋 중 하나를 직접 전진시켜야 한다. 그렇지 않은 queue, tensor와
개별 correction mechanism은 Appendix 또는 reproducibility record가 소유한다. Framework-specific
batch alignment와 carryover는 공개 본문과 Appendix에서 제외한다. §3.2의 세 번째 축은 §4에서
`공통 prompt-group work unit → critical-path overlap과
rollout-capacity recovery → 1.64× end-to-end payoff`로 직접 회수한다.

## 1. 논문 전체를 관통하는 포지셔닝

### 1.1 필드 수준의 판단

RLVR은 서로 다른 두 제약을 독립적으로 완화해 왔다. Fully-asynchronous RL은 rollout
generation과 model training의 동기식 결합을 풀어 실행 효율을 높였고, expert-guided learning은
policy-generated rollout만으로 성공 신호를 얻지 못하는 문제에 외부의 성공 경로를 제공했다.
어려운 reasoning problem에서는 긴 rollout의 계산 비용과 all-failure group의 신호 부족이 함께
커지므로 두 역량이 동시에 필요하다.

그러나 두 역량은 단순히 같은 pipeline에 넣는다고 결합되지 않는다. 이 논문이 다루는
group-conditioned setting에서는 complete group이 policy rollout과 expert trajectory 중 사용할
source를 정하고, policy-side relative signal의 맥락도 제공한다. 반면 trajectory-level
fully-asynchronous execution은 같은 group의 rollout을 독립적인 작업으로 분해해 효율을 얻는다.
두 source의 차이는 별도의 learner나 objective로 표현할 필요가 없다. Source decision에 따라
signal, reference와 correction이 다른 training input을 구성할 수 있다. 따라서 핵심은 source별 경로를
병치하는 것이 아니라, complete-group decision을 하나의 update가 바로 소비할 입력으로 바꾸는 데 있다.

### 1.2 소유하는 통찰

> **Complete group은 학습 판단의 boundary로 남아야 하지만, 그 completion이 pipeline 전체의
> serialized critical-path barrier가 될 필요는 없다.**

StreamWeave는 complete group을 필요한 순간에만 국소적으로 복원하고, source와 생성 맥락을
training input으로 변환한다. 이후 두 source는 하나의 policy objective와 같은 averaging rule을
통과한다. 그 결과 group-conditioned learning을 보존하면서도 trajectory-level execution을 유지하고,
complete-group generation을 learner의 serialized critical path 밖에서 계속 수행하며, group-tail
waiting에 묶이던 유효 rollout capacity도 회수한다.

### 1.3 주장 범위

**소유하는 주장**

- Group-conditioned heterogeneous learning과 trajectory-level asynchrony가 만날 때 생기는
  composition problem
- Complete-group decision을 source별 training input과 one primary update로 닫는 learning composition
- Learning decision boundary와 serialized critical-path barrier의 분리
- 이를 end-to-end fully-asynchronous system으로 실현한 StreamWeave
- 선언한 learning composition을 fully-asynchronous execution에서 실현하며 얻는 resource-matched efficiency

**독립 novelty로 주장하지 않는 것**

- Fully-asynchronous RL 자체
- Expert trajectory 사용 자체
- HPT의 selective expert-trajectory learning과 unified policy-gradient formulation
- PPO, GRPO, importance sampling, decoupling
- Accumulator, queue, backpressure와 self-detach 각각
- 임의 router, 다중 source, 모든 hybrid algorithm에 대한 보편적 지원

## 2. 공개 기여와 회수 구조

| 기여 | 핵심 주장 | 본문 회수 | 주된 evidence |
|---|---|---|---|
| **1. Asynchrony-compatible learning composition** | Complete-group decision이 사용할 sequence와 signal·reference·policy-lag correction을 정한다. Policy와 expert input은 하나의 policy objective와 같은 averaging rule을 통과한다 | §3.1 | Policy input과 expert input이 각각 의도한 contribution으로 환원되는 구성적 유도 |
| **2. Boundary-localized asynchronous architecture** | Queue 앞에서 complete group을 복원해 source를 확정하고, queue 뒤에서 training input을 구성한다. Generator와 learner의 기본 역할 및 기존 asynchronous flow는 유지한다 | §3.2 | Nonblocking reconstruction, engine-preserving handoff, §4의 two-part critical-path 해석과 resource-matched throughput |
| **3. Composition-aware empirical study** | Compute--signal pressure, 지속적인 expert 수요와 변화한 generation workload를 측정하고, competitive reasoning performance와 동일 GPU 예산의 1.64× prompt-group throughput으로 공동 payoff를 회수한다 | §4 | Reasoning-performance table, learning-dynamics analysis, prompt-group throughput과 critical-path breakdown |

Learner contract는 별도 기여가 아니라 기여 1의 learning composition을 압축하는 명세다. 별도의
`counterfactual audit` section이나 evidence category를 만들지 않는다. 정합 근거는 §3.1의
구성적 유도와 §3.2의 architecture에서 회수한다. Unit/contract test는 implementation QA이며
논문의 독립 실증으로 사용하지 않는다.

공개 본문에서는 오해를 부르는 `Source-specific update`를 핵심 명칭으로 사용하지 않는다. Source는
각 sample의 signal·reference·correction을 정하고, 그렇게 준비된 input은 같은 policy objective와
averaging rule을 통과한다고 평이하게 쓴다. 이는 RL update와 SFT update를 별도로 실행한 뒤 더하는
구조가 아니다.

### 2.1 Canonical boundaries

아래 세 명칭을 Introduction, Method, figure, caption에서 동일하게 사용한다.

| Boundary | 보존할 요구 | StreamWeave의 실현 |
|---|---|---|
| **Complete-group decision** | Source는 complete group이 갖추어진 뒤에만 결정 | Attempt는 독립 실행하고 source decision 직전에만 group을 복원 |
| **Training-input construction** | Policy와 expert가 의도한 학습 역할을 signal·reference·correction 조건으로 보존 | Queue 전에 source를 확정하고 trainer 경계에서 source와 생성 맥락을 각 sample의 training input으로 변환 |
| **One primary update** | Source 의미가 정해진 sample을 하나의 policy objective와 같은 averaging rule로 소비 | 별도 learner branch나 source별 loss·denominator를 두지 않음 |

## 3. 원고 구조

| Section | 한 가지 역할 | 반드시 남길 내용 | 넣지 않을 내용 |
|---|---|---|---|
| **Abstract** | 문제, 통찰, 방법, reasoning-performance/efficiency payoff를 한 문단으로 압축 | Compute-signal bottleneck, group-conditioned conflict, local decision boundary와 one primary update, 38.5/37.7, 1.64× | HPT, CISPO, contract 조항명, queue, self-detach, mechanism breakdown, 구체적인 hardware 구성 |
| **1. Introduction** | 필드의 두 흐름이 만나는 composition problem과 해결 판단을 세움 | Double bottleneck, complete-group dependency, source별 training input과 one primary update, decision boundary/critical-path barrier 분리, 세 기여 | 구현 chronology, framework-specific batching |
| **2. Related Work** | 두 연구 계보가 해결한 제약과 남긴 교차점을 구획 | Fully-asynchronous policy learning, expert-guided learning, 마지막 composition paragraph | 논문별 결함 목록, 비채택 논문에 의존한 포지셔닝 |
| **3.1 Learning Composition** | 선택된 learning role을 재현하기 위한 contract를 확립 | Complete-group selector, source별 training input, 하나의 policy objective와 같은 averaging rule, 좁은 구성적 유도 | Scheduler, queue, tensor field, 개별 correction 구현 |
| **3.2 Fully-Asynchronous Execution** | Contract를 지키면서 complete-group waiting의 실행 범위를 제한 | Independent attempts, local group reconstruction, source-fixed record, queue 뒤의 input construction과 continuous generation-training overlap | Loss 유도, exact queue protocol과 framework-specific batching |
| **4. Experiments** | 학습 효과와 실행 효율이 함께 성립하는지 평가 | Setup, quality, learning dynamics, resource-matched efficiency | Implementation QA를 실험으로 포장 |
| **5. Conclusion** | 필드 수준의 판단을 회수하고 검증 범위를 문단 안에서 명확히 함 | Boundary/barrier 분리, two-source/group-conditioned scope | 임의 source에 대한 미검증 일반화 |

### 3.1 Related Work 규율

- 각 계보는 `분야명 → 해결한 제약 → 대표 accepted-conference 연구 → 남은 범위` 순서로 쓴다.
- 포지셔닝에는 **채택된 컨퍼런스 논문만** 사용한다.
- Laminar의 trajectory-level GRPO는 fully-asynchronous 계보 안에 한 문장으로 배치한다. 이를
  별도의 방어 논점으로 확대하지 않는다.
- HPT는 Introduction과 Related Work에서 논문의 출발점처럼 다루지 않는다.
- HPT는 §3.1에서 selective expert-trajectory learning의 출처로 한 번 attribution한다.
- 마지막 composition 문단에서만 complete-group outcome이 정한 source를 asynchronous stream이
  소비할 training record로 변환하는 문제를 제시한다.

## 4. Method 작성 잠금

### 4.1 §3 도입부

§3은 서로 무관한 두 layer를 병렬로 소개하지도, §3.2를 §3.1의 하위 구현으로 소개하지도 않는다.
§3.1은 complete-group decision이 선택한 learning role을 재현하기 위해 training input이 만족해야 할
learning contract를 확립한다. §3.2는 그 contract를 제약으로 받아들이면서 complete-group waiting을
source decision에 국소화하는 별도의 systems problem을 푼다. §3.1이 먼저 오는 것은 논리적 독해
순서이며 contribution hierarchy가 아니다.

> StreamWeave는 complete group이 정한 learning role을 source-appropriate training input으로
> 재현한다. 이 의미적 제약을 지키면서도 trajectory attempt는 독립적으로 실행하고,
> complete-group waiting은 source가 정해지는 지점에서 끝낸다.

### 4.2 §3.1 Learning Composition

1. **Complete-group decision.** Complete rollout group에서 source를 결정한다. HPT는
   policy rollout과 expert trajectory를 선택적으로 사용하는 learning program의 출처로 한 번만 밝힌다.
2. **Source-specific training inputs.** Source decision은 사용할 data와 함께 signal, reference,
   correction 조건을 정한다.
3. **One policy objective.** 이렇게 준비된 policy와 expert sample은 별도 learner branch로
   갈라지지 않고 하나의 policy objective와 같은 averaging rule을 통과한다.
4. **Endpoint consistency.** Unified estimator가 policy-only 경우에는 표준 policy update로, expert
   경우에는 의도한 supervised contribution으로 환원됨을 본문에서 짧게 구성적으로 보인다.
5. 절 마지막에서 `complete-group decision → source별 training input → one primary update`를 한 문장으로
   압축하고 §3.2로 넘긴다.

본문 수식은 selector, source별 training input과 one primary update를 이해하는 데 필요한 최소 수만 둔다. Exact
singleton construction, pseudo-reward placement, self-detached reference, token-level correction,
auxiliary mask와 tensor schema는 Appendix A.2가, exact `seq-mean-token-sum-norm` 식과 선택 이유는
Appendix A.3이 소유한다. 특정 PPO 설정은 reproducibility block으로 보낸다.
이 장치들이 아니라 **source difference를 하나의 policy objective에 들어갈 input 조건으로 국소화했다는 구조**를 전면에
둔다. Unified estimator 자체도 StreamWeave의 독립 novelty로 격상하지 않는다.

### 4.3 §3.2 Fully-Asynchronous Execution

최상위 설계 판단은 다음과 같다.

> **Learning boundary를 보존하되 그것을 execution barrier로 만들지 않는다.**

각 문단은 `requirement → naive implementation의 손실 → StreamWeave design → 함께 유지되는 성질`
순서로 쓴다.

```text
Prompt group
  -> independent trajectory attempts              [inherited async substrate]
  -> complete-group reconstruction                 [core StreamWeave design]
  -> group-conditioned source decision
  -> source-fixed group transport                   [core StreamWeave boundary]
  -> trainer-side input construction                [core StreamWeave boundary]
  -> one policy update
```

이 절의 종점은 “queue가 동작한다”가 아니라, **complete-group generation이 serialized learner critical
path에서 분리되어도 §3.1의 learning composition이 유지된다**는 것이다. 이 설계 결과는 §4에서 routing
이전 prompt group을 공통 work unit으로 삼아 `46→28초`, `1.64×`로 회수한다. Independent scheduling,
bounded queue, backpressure, parameter refresh, accumulator 자료구조와 exact handoff protocol은 완성된
system의 기반이지만 독립 기여로 세우지 않는다. `n:1`, required multiple, subset-sum,
trim-and-carryover와 예외적 discard는 framework-specific realization이므로 공개 본문과 Appendix에서
제외하고 코드 문서에만 둔다.

§3.2의 추가적인 핵심 판단은 **source는 transport 전에 확정하고 training input은 transport 뒤에
구성한다**는 것이다. 이 비대칭은 arrival order가 source decision을 바꾸지 못하게 하면서 inference
engine에는 expert tokenization과 training tensor construction을 요구하지 않는다. Queue가 운반하는
단위는 training row가 아니라 source-fixed prompt group이며, trainer boundary에서만 training
sample로 변환된다. 따라서 trajectory attempt, routed group, training sample은 각각 실행, 판단·transport,
optimization에 맞는 서로 다른 물리적 단위가 된다.

이 구조는 composition logic을 backend-specific inference engine 내부에 hard-code하지 않고 기존 async
generation과 policy-update interface를 보존한다. 다만 이를 모든 inference/trainer backend 또는 임의의
selector·다중 source에서 검증했다는 범용성 주장으로 확대하지 않는다.

### 4.4 Canonical realization boundary

현행 main은 vanilla clipped PPO와 asynchronous policy correction을 사용한다.
이는 StreamWeave를 실증하는 한 realization이지 논문의 개념적 정체성이 아니다. Method와 Experimental
Setup은 objective family, selector와 source별 learning role만 명세한다. Main run ID, clipping, anchor,
token-level correction, $\beta$, auxiliary 설정과 stale/drop knob의 정확한 값은 Appendix의
reproducibility block이 소유한다. Method 본문은 특정 PPO variant가 아니라 complete-group decision,
source별 training input과 one primary update를 설명한다.

CISPO는 Method 구성요소가 아니다. Decoupling도 main의 realization일 뿐 독립 novelty가 아니며,
두 결과는 필요할 경우 Appendix의 secondary diagnosis로만 둔다.

## 5. Experiments 작성 잠금

공개 §4의 `Experimental Setup → Learning Effectiveness → Execution Efficiency with Expert Supervision` 구조와 evidence
role은 잠겼다. §4.1은 §4.2·§4.3이 실제로 사용하는 control과 분석 단위에 맞춰 갱신했다. 세 절의
중심 논증은 유지하고, 현재 문안 검토는 §4 도입부가 그 결과를 정확히 예고하는 데 필요한 범위로
제한한다. 새로운 evidence category나 별도 composition-fidelity 실험은 추가하지 않는다.

### 5.1 실험이 답할 세 질문

1. **Learning effectiveness:** Complete-group outcome이 정한 policy/expert source를 one primary
   update에 반영하면서 경쟁력 있는 reasoning performance를 달성하는가?
2. **Role of expert supervision:** Expert source가 선택되는 all-failure 영역에 signal scarcity와
   generation workload가 함께 집중되며, 그 수요가 학습 중에도 지속되는가?
3. **Execution efficiency with expert supervision:** Expert input을 사용하는 동안 generation의 길이와
   인접한 policy version의 rollout을 함께 포함하는 group 비율이 어떻게 달라지며, complete-group decision에 필요한 waiting을 source decision에
   국소화하여 이 workload를 resource-matched concurrent execution과 prompt-group throughput으로
   처리하는가?

### 5.2 Experimental Setup

- Base model은 `Qwen2.5-Math-1.5B`, training data는
  `Elliott/Openr1-Math-46k-8192`를 현재 prompt/tau contract로 전처리한
  `openr1_hpt_main_v2`다.
- 한 prompt에서 `n=8` rollout을 생성한다. Source selector는 `gamma=0`인 complete-group rule로,
  `0/8` all-failure group만 matched expert trajectory로 보낸다. Expert가 필요한데 없는 경우
  main은 fail-closed다. Canonical main의 expert contribution은 constant `beta=0.3`이지만, 공개
  §4.1은 이를 source-specific supervised signal로만 설명하고 exact coefficient는 Appendix에서 명세한다.
- Training과 자체 평가는 통일된 Math-Verify grader를 사용한다. 정확한 decoding config와
  evaluation seed는 Table 1 provenance manifest에 기록한다.
- Quality 평가는 AIME24, AIME25, AMC(83)에 `mean@32`, MATH500, Minerva, Olympiad에
  `mean@8`을 사용한다.
- `mean@32`는 32 stochastic generations의 평균 pass@1이며 `pass@32`가 아니다.
- AVG는 반올림 전 여섯 benchmark score의 동일가중 macro-average를 계산한 뒤 한 번만 반올림한다.
- Table 1과 cross-domain 표의 모든 checkpoint는 공통 evaluation protocol로 직접 평가한다.
  SRFT, ReLIFT, Oat-Zero와 LUFFY에는 각 연구가 공개한 공식 checkpoint를 사용한다.
- 자체 평가한 각 행은 model checkpoint, grader/decoding config, evaluation-seed manifest,
  raw result artifact ID를 기록한 뒤 Table 1에 올린다. 현재 main/sync checkpoint metadata는
  최종 원장에 아직 입력되지 않았으므로 추정하지 않고 제출 전에 확정한다.
- Efficiency의 공통 work unit은 routing 이전의 prompt group이다. RL group은 8 learner rows,
  expert group은 1 row를 만들기 때문에 rows/s, samples/s, steps/s를 headline으로 사용하지 않는다.
- Throughput은 cycle별 비율의 평균이 아니라 `∑ consumed prompt groups / ∑ non-evaluation
  training-loop time`으로 계산한다. Sync cycle은 128 groups, async cycle은 261–489 groups를
  담으므로 cycle 수나 cycle별 throughput을 직접 비교하지 않는다.
- Sync inline-generation share `54.7%`와 async learner-side acquisition·assembly share
  `3.25%`는 서로 다른 기전 지표로 정의하며 직접 차감하지 않는다. 본문의 더 직관적인
  critical-path 해석은 sync generation-only `25.1초`와 StreamWeave 전체 pipeline
  `28.0초`의 근접성이다.
- 정확한 `8×B200` 구성, sync colocated topology와 async trainer–rollouter partition,
  timing scope와 run ID는 Appendix에서 명시한다.

### 5.3 Evidence ledger

이 표가 현재 주장 가능 범위다. 상세 수치 원장은
`Full_Paper_Draft_ko.md` §6과 §6.1을 따른다.

| Claim | Status | 공개 사용 |
|---|---|---|
| Fixed-checkpoint quality: main 38.5, same-selector synchronous quality reference 37.7 | **LOCKED** | Abstract, Introduction, Table 1 |
| Resource-matched efficiency: 2.78→4.58 groups/s, 1.64×, 46→28초 | **LOCKED** | Abstract, Introduction, §4 efficiency; full-history `∑groups/∑time`, exact hardware는 Appendix |
| Source별 training input과 one primary update의 구성적 유도 | **LOCKED** | §3.1과 Appendix A.2--A.3 |
| Learning dynamics: 초반 유사, 후반 RL-only 정체, expert channel의 후반 기여 | **LOCKED / INTEGRATED** | §4.2 본문과 실증 그림; 단일 training seed의 보편적 인과로 확대하지 않음 |
| LUFFY 대비 +0.8 points | **PENDING** | Paired uncertainty analysis 전 headline 금지 |
| Unit/contract tests | **INTERNAL ONLY** | Implementation QA로만 사용하며 공개 원고나 Appendix evidence로 올리지 않음 |

과거 `Efficiency.tex`의 CISPO-arm `1.54×`와 main-run `1.64×`를 혼용하지 않는다. 공개
headline은 main W&B 원장 `sync=v96fvd0p`, `main=oki4kv8u`만 사용한다.
Full history는 각각 13,312 groups / 4,780.2초와 86,174 groups / 18,828.4초다. Async를 sync와
같은 첫 13,312-group budget으로 제한해도 1.67×이며, 마지막 동일 budget에서는 1.76×이므로 unequal
run length가 headline을 부풀리지 않는다. 이 equal-work 분석은 Appendix의 robustness check로 둔다.

### 5.4 본문 실증 자산

| Asset | 역할 | 상태 |
|---|---|---|
| **Table 1: Fixed-checkpoint reasoning performance** | Async execution과 external expert 사용 여부를 함께 표시하여 StreamWeave, HPT (sync), RL/expert baselines 비교 | 수치와 본문 잠김; checkpoint/artifact provenance만 확정 필요 |
| **Hard-region table (§4.2)** | All-failure 영역의 signal scarcity와 generation workload·completion spread 집중 | **PUBLIC CONTENT INTEGRATED**; 최종 LaTeX 번호·폭만 미정 |
| **Learning-dynamics figure (§4.2)** | Expert 수요의 지속성과 expert-off learning consequence | **LOCKED MAIN; RENDER AND VISUAL QA COMPLETE**; normalized progress와 scale 비노출 export 완료, 최종 번호만 남음 |
| **Cross-domain reasoning table (§4.2)** | 수학 중심 학습과 cross-domain reasoning 유지의 동시 성립 | **PUBLIC CONTENT INTEGRATED**; benchmark-level 결과를 본문에 두고 CISPO는 Appendix 유지 |
| **Integrated efficiency figure (§4.3)** | 공통 prompt-group work unit → serialized critical path의 분리 → end-to-end payoff | **LOCKED MAIN; RENDER AND VISUAL QA COMPLETE**; §4.3 본문·caption 연결과 export 완료, 최종 번호·LaTeX 배치 확인만 남음 |
| **Integrated execution table (§4.3)** | Expert-off 대비 generation workload 변화와 synchronous 대비 end-to-end payoff를 분리된 두 reference block으로 제시 | **PUBLIC CONTENT INTEGRATED**; response length 1.59$\times$, group별 최장 rollout 1.94$\times$, adjacent-version group 3.46$\times$, throughput 1.64$\times$ |

Efficiency 본문은 다음 네 단계를 하나의 실행 논증으로 전개한다.

1. **Workload change:** 같은 asynchronous stack의 expert-off control과 비교해 expert input을 사용한
   run에서 response length, group별 최장 rollout과 adjacent-version group 비율이 증가했음을 보인다.
   이는 data source뿐 아니라 asynchronous generation의 완료 양상도 달라짐을 뜻한다. Run-level
   관측을 expert supervision의 보편적 causal effect로 확대하지 않는다.
2. **Completion pressure:** Expert source가 필요한 all-failure group에서 version-span 비율이 가장
   높고, synchronous execution은 가장 늦은 sibling rollout이 끝날 때까지 먼저 빈 실행 슬롯을
   후속 작업에 사용하지 못한다.
3. **Mechanism:** StreamWeave는 complete-group decision을 제거하지 않고, group completion이 필요한
   지점만 국소적으로 기다리게 한다. 그 결과 이미 준비된 group의 training을 generation과 겹치고,
   먼저 끝난 rollout worker가 group tail을 기다리지 않고 다음 attempt를 처리하여 유효 rollout
   capacity도 회수한다. Sync는 128 groups의 generation에만 25.1초를 쓰지만 StreamWeave는 generation과
   training을 포함한 전체 pipeline을 28.0초에 끝낸다. Appendix의 resource-normalized check는
   `0.637→0.763 groups/(GPU·s)`와 same-rate counterfactual `33.5초`로 overlap-only 설명의 잔차를 닫는다.
4. **Payoff:** Full-history aggregate에서 throughput이 `2.78→4.58 groups/s`, 즉 `1.64×`
   증가하고 128-group-equivalent time은 46초에서 28초로 줄어든다. 두 표현은 하나의 결과다.

Route별 exact version-span, absolute group count와 completeness는 Appendix C.5가 소유한다. Partial
trajectory, queue cap, parameter-sync 상각과 MFU는 Appendix의 보조 cost ledger로 보낸다.
Framework-specific fixed-grain alignment와 carryover는 논문의 learning composition이나 execution
claim을 설명하지 않으므로 공개 Appendix에도 넣지 않는다. Startup 제외 1.59×, transient 제외 1.65×,
마지막 quarter 4.95 groups/s, equal-work 1.67×는 robustness 자산이며 본문 숫자를 늘리는 용도로
사용하지 않는다.

## 6. 본문과 Appendix의 경계

| 본문 | Appendix |
|---|---|
| Composition problem과 boundary/barrier 분리 | 과거 incident와 debugging chronology |
| Complete-group decision, source별 training input, one policy objective와 같은 averaging rule | Exact input construction, self-detach derivation, mask·tensor schema와 `seq-mean-token-sum-norm`의 식·선택 이유 |
| Expert-off 대비 generation workload 변화와 decision-localized execution의 핵심 해석 | Route별 version-span, absolute count, completeness, partial rollout과 exact handoff protocol |
| Canonical realization의 최소 명세 | 세부 PPO/correction knob, CISPO/decoupling secondary diagnosis |
| Fixed-checkpoint quality와 resource-matched efficiency | Telemetry scope, completion-tail timing과 robustness 분석 |
| Learning dynamics의 핵심 해석 | 추가 benchmark별·window별 세부 분석 |

본문은 framework-specific mechanism을 나열하지 않고, 그 mechanism이 실현하는 설계 판단을 먼저
제시한다. Fixed-grain alignment와 carryover는 코드 문서에만 남기며, Appendix도 구현 확인을 별도
논문 evidence로 포장하지 않는다.

## 7. 그림과 표의 역할

| 번호 | 목적 | 핵심 메시지 |
|---|---|---|
| **Figure 1** | Queue handoff의 data-state backbone | Complete-group decision → one-of-two source-fixed records → shared queue → source-appropriate inputs → one primary update |
| **Table 1** | 최종 reasoning performance | StreamWeave 38.5, HPT (sync) 37.7, 주요 baseline landscape |
| **Hard-region table (§4.2, 번호 미정)** | Compute--signal concentration | All-failure 영역에 response length와 completion spread가 함께 집중됨 |
| **Figure 2** | Learning dynamics와 expert routing | Residual hard region이 후반에도 남고 expert channel이 지속적으로 작동함 |
| **Cross-domain table (§4.2, 번호 미정)** | Cross-domain reasoning scope | 수학 benchmark와 분리된 ARC-C·GPQA-D·MMLU-Pro 결과로 target specialization과 retention을 확인 |
| **Figure 3** | Resource-matched execution efficiency | 더 지속적인 concurrent GPU activity가 동일 시간의 더 많은 prompt-group work로 이어짐 |
| **Integrated execution table (§4.3, 번호 미정)** | Expert supervision을 포함한 async workload와 end-to-end payoff | Expert-off 대비 response length·group별 최장 rollout·adjacent-version group 변화와 synchronous 대비 46→28초·1.64×를 서로 다른 reference block으로 제시 |

그림 안 텍스트는 최소화하고, 정확한 field 의미와 endpoint 유도는 §3.1과 caption이 담당한다.
Figure 1은 policy/expert record의 cardinality와 provenance 차이, 하나의 shared queue, 같은 input
schema와 shared learner tail을 도형으로 보여준다. Local waiting과 slot reuse는 §3.2 산문과
Algorithm 1이 소유한다.

기존 timeline figure는 source routing을 현행 Figure 1과 중복하고, 중복을 제외하면 상속한
asynchronous scheduling만 남으므로 2026-07-28 본문에서 제거했다. 새롭고 비중복적인 claim이
생기지 않는 한 이 자산을 본문이나 Appendix에 복원하지 않는다.

## 8. 남은 집필 순서

### P0 — 현재 원고를 닫는 작업

핵심 evidence order와 현행 Figure 1--3의 역할은 안정화됐고 English v2 이관도 진행됐다. 이 절이 남은
작업의 **유일한 실행 큐**이며, historical memo의 과거 `P0`나 `LOCKED` 표시는 진행 상태를 다시
정의하지 않는다.

1. **현재:** §3.2 opening과 산문을 systems-problem-first로 정제한다. §3.1의 learning contract를
   명시적 제약으로 받되, §3.2를 그 contract의 하위 구현으로 낮추지 않는다.
2. §3 전체에서 C1·C2 contribution promise, Figure 1의 data-state backbone과 Algorithm 1의 concurrent
   control flow가 중복 없이 각각 회수되는지 검사한다.
3. English v2를 렌더하여 수식, algorithm float, Figure 1 배치와 page flow를 확인한다.
4. Method가 닫힌 뒤 Conclusion과 Abstract까지 terminology가 역행하지 않는지 전역 검사한다.
5. 제출 전 bibliography, Appendix, provenance manifest와 reproducibility material을 연결하고 마지막
   page-budget compression을 수행한다. Thesis, 두 dependency boundary와 quality/dynamics/efficiency의
   evidence role은 압축 과정에서도 제거하지 않는다.

Appendix A--D는 exact learning realization, asynchronous realization, evaluation protocol, secondary
diagnostics와 scope를 소유한다. 2026-07-25에 승인된 matched process census는 Appendix C.3에
통합했다. 이후 Appendix 수정은 citation, provenance, page budget 또는 실제 LaTeX 표기 정합에 필요한
범위로 제한한다.

### P1 — 기존 로그와 평가 결과로 가능한 보강

- Main–LUFFY 문항별 paired uncertainty analysis
- Efficiency Appendix의 equal-work, boundedness, utilization, parameter-sync cost ledger 정리

P1이 닫히지 않아도 핵심 headline은 fixed-checkpoint quality와 resource-matched efficiency로
성립한다. 새로운 RL training run을 핵심 주장이나 제출의 필수 gate로 두지 않는다.

### P2 — 자원이 생길 때만 고려

- 다른 backbone 또는 scale
- 현재 세 평가를 넘어서는 추가 cross-domain reasoning benchmark
- Sequential SFT→RL 자체 학습 baseline

P2는 현재 논문의 정체성을 바꾸지 않는 선택 보강이다. 이를 기다리느라 본문 구조를 잠정 상태로
두지 않는다.

## 9. 주요 리뷰 리스크와 방어

| 예상 공격 | 본문의 답 |
|---|---|
| “AReaL과 HPT를 합친 engineering 아닌가?” | StreamWeave는 별도 RL/SFT learner를 나란히 붙이지 않는다. Complete-group source decision을 queue 앞에서 확정하고, source와 generation context를 queue 뒤의 training-input construction까지 보존한 뒤 하나의 policy objective로 닫는다. 이 경계 배치는 Generator와 learner의 기본 역할을 유지하면서 complete-group generation을 serialized critical path에서 분리한다. §4.3은 expert input이 generation workload와 policy-version span을 실제로 바꾸는 상황에서도 이 architecture가 resource-matched throughput을 회수함을 보인다 |
| “Group reconstruction은 기존 async RL에도 있는 것 아닌가?” | Reconstruction 자체를 novelty로 주장하지 않고, source와 update까지 결정하는 additional control dependency를 localize한 bridge를 소유 |
| “Source별 처리가 자명하다” | Source decision이 data choice뿐 아니라 signal·reference·correction을 정하며, 두 input이 의도한 policy/supervised contribution으로 환원되고 averaging까지 공통임을 구성적으로 제시 |
| “빠르지만 다른 학습 문제를 푼 것 아닌가?” | 같은 group-conditioned source-selection policy를 사용하는 synchronous quality reference와 대등한 fixed-checkpoint quality를 먼저 제시하고 resource-matched efficiency를 뒤에 결합 |
| “1.64×가 서로 다른 cycle 수나 source mixture에서 나온 착시 아닌가?” | 본문에서는 routing 이전의 공통 prompt-group work unit과 work-weighted full-history prompt-group throughput을 보고한다. 통합 효율 그림은 동일한 79.7분에서 누적 작업량이 1.00×에서 1.67×로 벌어지는 과정을 보여주며, exact hardware budget·topology·timing scope·source mix는 Appendix에서 공개 |
| “6-GPU rollouter인데 28초라면 overlap만으로 산술이 닫히지 않는 것 아닌가?” | 맞다. 이것이 두 번째 시스템 효과다. Appendix는 synchronous 0.637 대비 StreamWeave가 최소 0.763 groups/(GPU·s)를 공급해야 함을 보이고, attempt-level scheduling이 complete-group tail waiting에서 유효 rollout capacity를 회수한다는 해석을 제시한다. 다만 exact 20%를 token-normalized causal decomposition으로 확대하지 않는다 |
| “단일 모델·수학 domain이라 일반성이 약하다” | 검증 범위를 two-source group-conditioned RLVR로 명시하고 architecture의 구성 가능성과 empirical scope를 구별 |
| “외부 baseline protocol이 섞였다” | 외부 인용 행을 각주로 분리하고 protocol-matched 주장에는 자체 통일 평가 행만 사용 |
| “보편적으로 올바르다고 과장한다” | Fixed group/parameter/objective/aggregation 아래의 source-specific input composition만 주장하고 optimizer trajectory equivalence나 universal necessity는 주장하지 않음 |

## 10. 보조 문서 사용 규율

문서별 역할과 링크는
`AAAI_RL/docs/papers_RL/README.md`의 **Document Authority**와
**Reference-Only Code Documents**가 단일하게 소유한다. 이 계획서는 그 지도를 복제하지 않는다.

- Evidence와 코드 참고자료는 현재 원고의 문안·novelty·우선순위를 독립적으로 바꾸지 않는다.
- 구현과 원고가 충돌하면 실제 코드를 먼저 확인하되, 공개 claim 변경은
  `Full_Paper_Draft_ko.md`에서 결정한다.
- 역사 문서는 삭제하지 않지만, 초기 M/CISPO 지시나 `Async-HPT` 용어를 현행 논문으로 복원하지 않는다.

## 11. 금지된 회귀

- `Async-HPT`를 논문 방법명이나 제목으로 복원하지 않는다.
- `Semantics Follow Provenance`, aliasing lemma, necessity/sufficiency/selectivity를 공개
  contribution으로 복원하지 않는다.
- `counterfactual audit`을 별도 section, contribution, headline evidence로 복원하지 않는다.
- CISPO를 main objective처럼 서술하지 않는다.
- RL과 SFT를 별도 objective 또는 별도 learner update로 실행한 뒤 사후 합산한다고 서술하지 않는다.
- Exact pseudo-reward, self-detach, tensor construction 순서를 본문 핵심 설명으로 복원하지 않는다.
  필요할 경우 Appendix에서 실제 realization 경계를 정확히 명시한다.
- Unified policy-gradient estimator 자체를 StreamWeave의 독립 novelty로 주장하지 않는다. 소유 지점은
  이 single-path composition을 fully-asynchronous runtime에서 보존하고 실현하는 architecture다.
- 과거 `1.54×`, `54%→4.6%`, 70% tail wait, 82% rollouter utilization,
  response-length-conditioned 41%, MFU headline을 main-run 근거로 복원하지 않는다.
- `54.7%→3.25%`를 하나의 trainer-idle 또는 stall 감소율로 쓰지 않는다.
- `46→28초`와 `1.64×`를 서로 독립적인 두 증거처럼 세지 않는다.
- `0.637→0.763 groups/(GPU·s)`를 별도 speedup으로 더하거나 exact 19.8%를 barrier 제거 하나의
  인과 효과로 분해하지 않는다. 이는 1.64×의 두 번째 기전을 지지하는 consistency check다.
- Cycle들을 i.i.d. sample로 취급한 표준오차나 유의확률, architecture-isolated speedup,
  time-to-quality를 현행 efficiency 원장에서 주장하지 않는다.
- LUFFY +0.8을 paired uncertainty analysis 전에 확정형 headline으로 쓰지 않는다.
- `zero-waste`, `crash-free`, 보편적 guarantee, optimizer-trajectory equivalence를 주장하지 않는다.
- Framework-specific alignment와 carryover를 공개 본문이나 Appendix의 논문 evidence로 복원하지 않는다.

## 12. 완료 기준

첫 독자가 다음처럼 요약하면 개정이 성공한 것이다.

> **StreamWeave separates trajectory execution, complete-group decision, and learner optimization.
> It reconstructs groups only where source selection requires them, carries the resolved learning
> meaning into one update, and preserves the rest of fully-asynchronous execution.**

다음처럼 요약된다면 개정이 미완료다.

- “AReaL에 HPT를 붙였다.”
- “Accumulator로 GRPO group을 모았다.”
- “RL loss와 SFT loss를 같이 썼다.”
- “Async로 만들어 1.64배 빨라졌다.”
