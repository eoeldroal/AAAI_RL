# Paper Plan — StreamWeave (AAAI)

> **Tracked canonical plan.** 이 파일은 저장소 안에서 논문 실행 계획을 소유한다. 공개 문안과
> claim boundary는 `Full_Paper_Draft_ko.md`가 우선한다.

_Last updated: 2026-07-24_

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
| HPT의 지위 | 실험에서 사용하는 success-conditioned selector의 출처; 논문의 정체성이나 방법명은 아님 |
| CISPO의 지위 | Main에서 기각된 ablation; Appendix의 secondary diagnosis |
| 핵심 실증 | Fixed-checkpoint quality, expert-off learning dynamics, resource-matched execution efficiency |
| 신규 학습 | 현재 핵심 주장을 위해 요구하지 않음 |
| 공개 원고 상태 | 한국어 Abstract--Conclusion과 Appendix A--D의 구조·주장 잠금 완료 |
| 온보딩 상태 | Full draft §0, project memory, TASKS, code-facing Overview를 2026-07-23 기준으로 동기화; 저장소 `AGENTS.md`는 코드 전용으로 유지 |

### 0.1 본문 추상화 게이트

공개 본문이 소유할 내용은 아래 세 축으로 제한한다.

1. **Composition problem:** Complete group이 필요한 학습 판단과 trajectory-level fully-asynchronous
   execution의 충돌
2. **Source-conditioned shared learning:** Group이 정한 source를 shared learner가 해석할 학습
   조건으로 변환하고, 하나의 objective와 reduction으로 소비
3. **Decision-localized execution:** Complete group을 학습 판단 경계로 유지하되, 그 completion을
   serialized critical-path barrier로 만들지 않는 execution architecture

본문의 장치와 수식은 반드시 이 셋 중 하나를 직접 전진시켜야 한다. 그렇지 않은 queue, tensor,
alignment, exception-handling, 개별 correction mechanism은 Appendix 또는 reproducibility record가
소유한다. §3.2의 세 번째 축은 §4에서 `공통 prompt-group work unit → critical-path overlap과
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
또한 두 source의 차이는 별도의 learner나 objective를 요구하는 것이 아니라, 하나의 shared learner가
해석할 advantage, reference, correction 조건으로 표현될 수 있다. 따라서 핵심은 source별 경로를
병치하는 것이 아니라, complete-group decision을 source-conditioned learner input으로 바꾸는 데 있다.

### 1.2 소유하는 통찰

> **Complete group은 학습 판단의 boundary로 남아야 하지만, 그 completion이 pipeline 전체의
> serialized critical-path barrier가 될 필요는 없다.**

StreamWeave는 complete group을 필요한 순간에만 국소적으로 복원하고, source와 생성 맥락을
shared learner가 해석할 학습 조건으로 변환한다. 이후 두 source는 하나의 objective와 reduction을
통과한다. 그 결과 group-conditioned learning을 보존하면서도 trajectory-level execution을 유지하고,
complete-group generation을 learner의 serialized critical path 밖에서 계속 수행하며, group-tail
waiting에 묶이던 유효 rollout capacity도 회수한다.

### 1.3 주장 범위

**소유하는 주장**

- Group-conditioned heterogeneous learning과 trajectory-level asynchrony가 만날 때 생기는
  composition problem
- Complete-group decision을 source-conditioned input으로 변환하는 shared learning composition
- Learning decision boundary와 serialized critical-path barrier의 분리
- 이를 end-to-end fully-asynchronous system으로 실현한 StreamWeave
- 선언한 learning composition을 fully-asynchronous execution에서 실현하며 얻는 resource-matched efficiency

**독립 novelty로 주장하지 않는 것**

- Fully-asynchronous RL 자체
- Expert trajectory 사용 자체
- HPT의 success-conditioned selector와 unified policy-gradient formulation
- PPO, GRPO, importance sampling, decoupling
- Accumulator, queue, backpressure, self-detach, trim-and-carryover 각각
- 임의 router, 다중 source, 모든 hybrid algorithm에 대한 보편적 지원

## 2. 공개 기여와 회수 구조

| 기여 | 핵심 주장 | 본문 회수 | 주된 evidence |
|---|---|---|---|
| **1. Group-conditioned learning composition** | Complete group이 결정한 source를 shared learner가 해석할 advantage·reference·correction 조건으로 변환하고, policy와 expert를 하나의 objective와 공통 reduction으로 학습 | §3.1 | Unified estimator가 policy와 expert의 의도한 contribution으로 환원되는 구성적 유도 |
| **2. Decision-localized asynchronous architecture** | Group을 학습 판단 경계로 유지하되 필요한 순간에만 국소적으로 복원한다. Source는 transport 전에 확정하고 final learner input은 transport 뒤에 구성하여 이질성을 engine 경계에 국소화하고, complete-group dependency가 training serialization이나 rollout-side capacity loss로 번지지 않게 함 | §3.2 | Nonblocking reconstruction, engine-preserving handoff, §4의 two-part critical-path 해석과 resource-matched throughput |
| **3. 학습 효과와 실행 효율의 공동 실측** | Fixed-checkpoint quality, expert-off dynamics, resource-matched throughput을 통해 practical utility와 execution payoff를 함께 측정 | §4 | Quality table, learning-dynamics analysis, prompt-group throughput과 critical-path breakdown |

Learner contract는 별도 기여가 아니라 기여 1의 learning composition을 압축하는 명세다. 별도의
`counterfactual audit` section이나 evidence category를 만들지 않는다. 정합 근거는 §3.1의
구성적 유도와 §3.2의 architecture에서 회수한다. Unit/contract test는 implementation QA이며
논문의 독립 실증으로 사용하지 않는다.

공개 본문에서는 오해를 부르는 `Source-specific update`를 핵심 명칭으로 사용하지 않는다. 대신 source가
각 sample의 학습 조건을 정하는 **source-conditioned input construction**과, 그렇게 정의된 sample이 같은
objective와 reduction을 통과하는 **shared primary update**를 구별한다. 이는 RL update와 SFT update를 별도로
실행한 뒤 더하는 구조가 아니다.

### 2.1 Canonical boundaries

아래 세 명칭을 Introduction, Method, figure, caption에서 동일하게 사용한다.

| Boundary | 보존할 요구 | StreamWeave의 실현 |
|---|---|---|
| **Complete-group decision** | Source는 complete group이 갖추어진 뒤에만 결정 | Attempt는 독립 실행하고 source decision 직전에만 group을 복원 |
| **Source-conditioned input construction** | Policy와 expert가 의도한 학습 역할을 advantage·reference·correction 조건으로 보존 | Transport 전에 source를 확정하고 trainer 경계에서 source와 생성 맥락을 각 sample의 유효한 학습 조건으로 변환 |
| **Shared primary update** | Source 의미가 정해진 sample을 하나의 objective와 사전 선언된 reduction으로 소비 | 별도 learner branch나 branch-specific reducer를 두지 않고 transport와 batching이 effective mixture를 다시 쓰지 않게 함 |

## 3. 원고 구조

| Section | 한 가지 역할 | 반드시 남길 내용 | 넣지 않을 내용 |
|---|---|---|---|
| **Abstract** | 문제, 통찰, 방법, quality/efficiency payoff를 한 문단으로 압축 | Compute-signal bottleneck, group-conditioned conflict, local decision boundary와 shared update, 38.5/37.7, 1.64× | HPT, CISPO, contract 조항명, queue, self-detach, mechanism breakdown, 구체적인 hardware 구성 |
| **1. Introduction** | 필드의 두 흐름이 만나는 composition problem과 해결 판단을 세움 | Double bottleneck, complete-group dependency, source-conditioned input과 shared update, decision boundary/critical-path barrier 분리, 세 기여 | 구현 chronology, framework-specific alignment |
| **2. Related Work** | 두 연구 계보가 해결한 제약과 남긴 교차점을 구획 | Fully-asynchronous policy learning, expert-guided learning, 마지막 composition paragraph | 논문별 결함 목록, 비채택 논문에 의존한 포지셔닝 |
| **3.1 Learning Composition** | 실행 순서와 무관한 group-conditioned shared update를 정의 | Complete-group selector, source-conditioned input construction, shared objective와 공통 reduction, 좁은 구성적 유도 | Scheduler, queue, tensor field, 개별 correction 구현 |
| **3.2 Fully-Asynchronous Execution** | §3.1의 composition을 serialized group barrier 없이 실현 | Independent attempts, local group reconstruction, source/context preservation, continuous generation-training overlap | Loss 유도, queue protocol, batch alignment, framework-specific flow control |
| **4. Experiments** | 학습 효과와 실행 효율이 함께 성립하는지 평가 | Setup, quality, learning dynamics, resource-matched efficiency | Implementation QA를 실험으로 포장 |
| **5. Conclusion** | 필드 수준의 판단을 회수하고 검증 범위를 문단 안에서 명확히 함 | Boundary/barrier 분리, two-source/group-conditioned scope | 임의 source에 대한 미검증 일반화 |

### 3.1 Related Work 규율

- 각 계보는 `분야명 → 해결한 제약 → 대표 accepted-conference 연구 → 남은 범위` 순서로 쓴다.
- 포지셔닝에는 **채택된 컨퍼런스 논문만** 사용한다.
- HPT는 Introduction과 Related Work에서 논문의 출발점처럼 다루지 않는다.
- HPT의 routing rule은 §3.1에서 실험적 selector로 한 번 attribution한다.
- AReaL 등 기존 async system의 generation-training separation과 trajectory-level freedom을
  인정하고, StreamWeave의 소유 지점은 group-conditioned source decision과의 결합으로 제한한다.

## 4. Method 작성 잠금

### 4.1 §3 도입부

§3은 동등한 두 layer를 병렬로 소개하지 않는다. 먼저 learning composition을 정의하고, 이어서
그 정의를 fully-asynchronous runtime에서 실현하는 순서로 쓴다.

> StreamWeave는 먼저 complete group이 정한 source를 shared learner가 해석할 학습 조건으로 변환하고,
> 하나의 objective와 reduction으로 소비하는 learning composition을 정의한다. 그런 다음 trajectory를
> 독립적으로 실행하면서도 그 composition을 serialized group barrier 없이 실현하는 fully-asynchronous
> execution architecture를 구성한다.

### 4.2 §3.1 Learning Composition

1. **Complete-group decision.** Complete rollout group에서 source를 결정한다. 실험에서는 HPT의
   success-conditioned hard switch를 사용한다고 한 번만 밝힌다.
2. **Source-conditioned input construction.** Source decision은 사용할 data만 고르는 것이 아니라, shared learner가
   그 sample을 해석하는 advantage, reference, correction 조건을 함께 정한다.
3. **Shared primary update.** 이렇게 의미가 정해진 policy와 expert sample은 별도 learner branch로
   갈라지지 않고 하나의 policy-gradient objective와 사전 선언된 reduction을 통과한다.
4. **Endpoint consistency.** Unified estimator가 policy-only 경우에는 표준 policy update로, expert
   경우에는 의도한 supervised contribution으로 환원됨을 본문에서 짧게 구성적으로 보인다.
5. 절 마지막에서 세 canonical boundary를 learner contract로 압축하고 §3.2로 넘긴다.

본문 수식은 selector와 source-conditioned shared update를 이해하는 데 필요한 최소 수만 둔다. Exact
singleton construction, pseudo-reward placement, self-detached reference, token-level correction,
auxiliary mask, tensor schema와 특정 PPO 설정은 Appendix 또는 reproducibility block이 소유한다.
이 장치들이 아니라 **source difference를 shared update의 입력 조건으로 국소화했다는 구조**를 전면에
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
  -> source-resolved group transport                [core StreamWeave boundary]
  -> trainer-side learner materialization
  -> source-conditioned input construction          [core StreamWeave boundary]
  -> shared primary update
```

이 절의 종점은 “queue가 동작한다”가 아니라, **complete-group generation이 serialized learner critical
path에서 분리되어도 §3.1의 learning composition이 유지된다**는 것이다. 이 설계 결과는 §4에서 routing
이전 prompt group을 공통 work unit으로 삼아 `46→28초`, `1.64×`로 회수한다. Independent scheduling,
bounded queue, backpressure, parameter refresh, accumulator 자료구조와 exact handoff protocol은 완성된
system의 기반이지만 독립 기여로 세우지 않는다. `n:1`, required multiple, subset-sum,
trim-and-carryover, 예외적 discard는 framework-specific realization이므로 Appendix로 보낸다.

§3.2의 추가적인 핵심 판단은 **source는 transport 전에 확정하고 final learner input은 transport 뒤에
구성한다**는 것이다. 이 비대칭은 arrival order가 source decision을 바꾸지 못하게 하면서 inference
engine에는 expert tokenization과 training tensor construction을 요구하지 않는다. Queue가 운반하는
단위는 learner row가 아니라 source-resolved prompt group이며, trainer boundary에서만 source-conditioned
sample로 변환된다. 따라서 trajectory attempt, routed group, learner sample은 각각 실행, 판단·transport,
optimization에 맞는 서로 다른 물리적 단위가 된다.

이 구조는 composition logic을 backend-specific inference engine 내부에 hard-code하지 않고 기존 async
generation과 policy-update interface를 보존한다. 다만 이를 모든 inference/trainer backend 또는 임의의
selector·다중 source에서 검증했다는 범용성 주장으로 확대하지 않는다.

### 4.4 Canonical realization boundary

현행 main은 vanilla clipped PPO 기반의 shared learner와 asynchronous policy correction을 사용한다.
이는 StreamWeave를 실증하는 한 realization이지 논문의 개념적 정체성이 아니다. Main run ID, clipping,
anchor, token-level correction, $\beta$, auxiliary 설정과 stale/drop knob의 정확한 값은 Experimental Setup과
Appendix의 reproducibility block에서만 명세한다. Method 본문은 특정 PPO variant가 아니라
complete-group decision, source-conditioned input construction, shared primary update를 설명한다.

CISPO는 Method 구성요소가 아니다. Decoupling도 main의 realization일 뿐 독립 novelty가 아니며,
두 결과는 필요할 경우 Appendix의 secondary diagnosis로만 둔다.

## 5. Experiments 작성 잠금

공개 §4의 구조와 문안은 `Experimental Setup → Learning Effectiveness → Execution Efficiency`로
잠겼다. 이후 수정은 provenance와 figure/caption 확정에 필요한 범위로 제한하며, 새로운 evidence
category나 별도 composition-fidelity 실험을 추가하지 않는다.

### 5.1 실험이 답할 세 질문

1. **Learning effectiveness:** StreamWeave가 policy/expert learning의 품질 이점을 유지하는가?
2. **Role of expert supervision:** Expert channel은 self-generated RL signal이 부족해지는 구간에서
   어떤 역할을 하는가?
3. **Execution efficiency:** Complete-group decision을 유지하면서 generation을 serialized critical
   path에서 분리하고 resource-matched prompt-group throughput을 높이는가?

### 5.2 Experimental Setup

- Base model은 `Qwen2.5-Math-1.5B`, training data는
  `Elliott/Openr1-Math-46k-8192`를 현재 prompt/tau contract로 전처리한
  `openr1_hpt_main_v2`다.
- 한 prompt에서 `n=8` rollout을 생성한다. 실험적 source selector는 `gamma=0`인 HPT rule로,
  `0/8` all-failure group만 matched expert trajectory로 보낸다. Expert가 필요한데 없는 경우
  main은 fail-closed이며, expert contribution은 constant `beta=0.3`을 사용한다.
- Training과 자체 평가는 통일된 Math-Verify grader를 사용한다. 정확한 decoding config와
  evaluation seed는 Table 1 provenance manifest에 기록한다.
- Quality 평가는 AIME24, AIME25, AMC(83)에 `mean@32`, MATH500, Minerva, Olympiad에
  `mean@8`을 사용한다.
- `mean@32`는 32 stochastic generations의 평균 pass@1이며 `pass@32`가 아니다.
- AVG는 반올림 전 여섯 benchmark score의 동일가중 macro-average를 계산한 뒤 한 번만 반올림한다.
- 외부 출처의 SFT/RL-only 행은 각주로 구별하고 동일 protocol ranking의 근거로 사용하지 않는다.
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
| Fixed-checkpoint quality: main 38.5, synchronous counterpart 37.7 | **LOCKED** | Abstract, Introduction, Table 1 |
| Resource-matched efficiency: 2.78→4.58 groups/s, 1.64×, 46→28초 | **LOCKED** | Abstract, Introduction, §4 efficiency; full-history `∑groups/∑time`, exact hardware는 Appendix |
| Source-conditioned composition의 구성적 유도 | **LOCKED** | §3.1과 Appendix |
| Learning dynamics: 초반 유사, 후반 RL-only 정체, expert channel의 후반 기여 | **DERIVED** | 곡선과 protocol이 함께 있을 때 §4 |
| LUFFY 대비 +0.8 points | **PENDING** | Paired uncertainty analysis 전 headline 금지 |
| Unit/contract tests | **APPENDIX** | Implementation QA로만 사용 |

과거 `Efficiency.tex`의 CISPO-arm `1.54×`와 main-run `1.64×`를 혼용하지 않는다. 공개
headline은 main W&B 원장 `sync=v96fvd0p`, `main=oki4kv8u`만 사용한다.
Full history는 각각 13,312 groups / 4,780.2초와 86,174 groups / 18,828.4초다. Async를 sync와
같은 첫 13,312-group budget으로 제한해도 1.67×이며, 마지막 동일 budget에서는 1.76×이므로 unequal
run length가 headline을 부풀리지 않는다. 이 equal-work 분석은 Appendix의 robustness check로 둔다.

### 5.4 본문 실증 자산

| Asset | 역할 | 상태 |
|---|---|---|
| **Table 1: Fixed-checkpoint quality** | Main, 같은 selector의 synchronous reference, RL/expert baselines 비교 | 수치와 본문 잠김; checkpoint/artifact provenance만 확정 필요 |
| **Learning-dynamics figure** | Expert channel의 후반 보완 역할을 해석 | 분석과 본문 잠김; 최종 caption과 공개 asset 배치 필요 |
| **Efficiency figure** | 공통 prompt-group work unit → serialized critical path의 분리 → end-to-end payoff | Main-run 수치와 본문 잠김; 최종 caption과 panel 배치 필요 |

Efficiency 본문은 다음 세 단계만 전면에 둔다.

1. **Metric:** Source mixture에 따라 learner row 수가 달라지므로 routing 이전의 prompt group을
   공통 work unit으로 사용한다. Learning boundary와 measurement unit을 같은 단위에 맞춘다.
2. **Mechanism:** StreamWeave는 complete-group decision을 제거하지 않고, group completion이 필요한
   지점만 국소적으로 기다리게 한다. 그 결과 이미 준비된 group의 training을 generation과 겹치고,
   먼저 끝난 rollout worker가 group tail을 기다리지 않고 다음 attempt를 처리하여 유효 rollout
   capacity도 회수한다. Sync는 128 groups의 generation에만 25.1초를 쓰지만 StreamWeave는 generation과
   training을 포함한 전체 pipeline을 28.0초에 끝낸다. Appendix의 resource-normalized check는
   `0.637→0.763 groups/(GPU·s)`와 same-rate counterfactual `33.5초`로 overlap-only 설명의 잔차를 닫는다.
3. **Payoff:** Full-history aggregate에서 throughput이 `2.78→4.58 groups/s`, 즉 `1.64×`
   증가하고 128-group-equivalent time은 46초에서 28초로 줄어든다. 두 표현은 하나의 결과다.

Queue cap, partial trajectory, parameter-sync 상각, MFU, carryover는 Appendix의 기전·cost
ledger로 보낸다. Startup 제외 1.59×, transient 제외 1.65×, 마지막 quarter 4.95 groups/s,
equal-work 1.67×는 robustness 자산이며 본문 숫자를 늘리는 용도로 사용하지 않는다.

## 6. 본문과 Appendix의 경계

| 본문 | Appendix |
|---|---|
| Composition problem과 boundary/barrier 분리 | 과거 incident와 debugging chronology |
| Complete-group decision, source-conditioned input construction, shared primary update | Exact estimator construction, self-detach derivation, mask와 tensor schema |
| Decision-localized execution과 critical-path 분리 | Queue size, backpressure 세부, partial rollout, exact handoff protocol |
| Canonical realization의 최소 명세 | 세부 PPO/correction knob, CISPO/decoupling secondary diagnosis |
| Fixed-checkpoint quality와 resource-matched efficiency | Trim-and-carryover, fixed-grain alignment, 예외 회계 |
| Learning dynamics의 핵심 해석 | 추가 benchmark별·window별 세부 분석 |

본문은 framework-specific mechanism을 나열하지 않고, 그 mechanism이 실현하는 설계 판단을 먼저
제시한다. Appendix도 구현 확인을 별도 논문 evidence로 포장하지 않는다.

## 7. 그림과 표의 역할

| 번호 | 목적 | 핵심 메시지 |
|---|---|---|
| **Figure 1** | 문제와 StreamWeave의 통찰 | Group은 decision boundary로 남지만 execution barrier일 필요는 없음 |
| **Figure 2** | §3.1과 §3.2의 대응 | Independent attempts → local reconstruction → source-conditioned input construction → shared primary update |
| **Table 1** | 최종 quality | Main 38.5, same-selector sync reference 37.7, 주요 baseline landscape |
| **Figure 3** | 실험의 인과 구조 | Learning dynamics와 resource-matched efficiency를 제한된 panel로 제시 |

그림 안 텍스트는 최소화하고, 정확한 수치와 해석은 caption이 담당한다. Figure 2는 물리적인
Generator–Trainer pipeline 위에 세 canonical boundary를 callout으로 대응시키되, learning
composition과 runtime을 동등한 두 layer처럼 그리지 않는다.

## 8. 남은 집필 순서

### P0 — 현재 원고를 닫는 작업

핵심 공개 한국어 원고인 Abstract–§5 Conclusion은 2026-07-23 기준 구조를 잠갔다. 남은 P0는 아래처럼
재정의한다.

1. Main/sync checkpoint, grader·decoding config, evaluation seed, raw result artifact를 provenance
   manifest에 등록하고 Table 1 각주를 확정한다.
2. Figure 1–3의 최종 panel 역할과 caption을 공개 본문의 claim 순서에 맞춰 잠근다. Learning dynamics와
   efficiency figure는 이미 잠긴 수치 원장을 다시 계산하지 않는다.
3. 잠긴 한국어 원고를 AAAI-27 LaTeX 구조로 옮기고, 최종 영어 표현을 같은 canonical vocabulary로
   통일한다.
4. Accepted-conference citation과 BibTeX를 정리하고 AAAI-27 reproducibility checklist의 근거를
   provenance manifest와 Appendix에 연결한다.
5. 마지막으로 page budget에 맞춰 중복만 압축한다. Thesis, 두 dependency boundary, quality/dynamics/
   efficiency의 세 evidence role은 압축 과정에서도 제거하지 않는다.

Appendix A--D는 exact learning realization, asynchronous realization, evaluation protocol, secondary
diagnostics와 scope까지 2026-07-23 기준으로 이관을 마쳤다. 이후 Appendix 수정은 citation, provenance,
page budget 또는 실제 LaTeX 표기 정합에 필요한 범위로 제한한다.

### P1 — 기존 로그와 평가 결과로 가능한 보강

- Main–LUFFY 문항별 paired uncertainty analysis
- Efficiency Appendix의 equal-work, boundedness, utilization, parameter-sync cost ledger 정리

P1이 닫히지 않아도 핵심 headline은 fixed-checkpoint quality와 resource-matched efficiency로
성립한다. 새로운 RL training run을 핵심 주장이나 제출의 필수 gate로 두지 않는다.

### P2 — 자원이 생길 때만 고려

- 다른 backbone 또는 scale
- 추가 OOD benchmark
- Sequential SFT→RL 자체 학습 baseline

P2는 현재 논문의 정체성을 바꾸지 않는 선택 보강이다. 이를 기다리느라 본문 구조를 잠정 상태로
두지 않는다.

## 9. 주요 리뷰 리스크와 방어

| 예상 공격 | 본문의 답 |
|---|---|
| “AReaL과 HPT를 합친 engineering 아닌가?” | StreamWeave는 별도 RL/SFT learner를 나란히 붙이지 않는다. Group-conditioned source decision을 shared learner가 해석할 조건으로 변환하고, single-path composition을 유지한 채 complete-group generation을 serialized critical path에서 분리하는 architecture가 소유 지점 |
| “Group reconstruction은 기존 async RL에도 있는 것 아닌가?” | Reconstruction 자체를 novelty로 주장하지 않고, source와 update까지 결정하는 additional control dependency를 localize한 bridge를 소유 |
| “Source별 처리가 자명하다” | Source decision이 data choice뿐 아니라 shared learner의 해석 조건을 정하며, 두 endpoint가 의도한 policy/supervised contribution으로 환원되고 reduction까지 공통임을 구성적으로 제시 |
| “빠르지만 다른 학습 문제를 푼 것 아닌가?” | 같은 group-conditioned source-selection policy를 사용하는 synchronous counterpart와 대등한 fixed-checkpoint quality를 먼저 제시하고 resource-matched efficiency를 뒤에 결합 |
| “1.64×가 서로 다른 cycle 수나 source mixture에서 나온 착시 아닌가?” | 본문에서는 routing 이전의 공통 prompt-group work unit과 work-weighted full-history throughput을 보고한다. Appendix에서는 같은 13,312-group budget에서도 1.67×임을 보이고, exact hardware budget·topology·timing scope·source mix를 공개 |
| “6-GPU rollouter인데 28초라면 overlap만으로 산술이 닫히지 않는 것 아닌가?” | 맞다. 이것이 두 번째 시스템 효과다. Appendix는 synchronous 0.637 대비 StreamWeave가 최소 0.763 groups/(GPU·s)를 공급해야 함을 보이고, attempt-level scheduling이 complete-group tail waiting에서 유효 rollout capacity를 회수한다는 해석을 제시한다. 다만 exact 20%를 token-normalized causal decomposition으로 확대하지 않는다 |
| “단일 모델·수학 domain이라 일반성이 약하다” | 검증 범위를 two-source group-conditioned RLVR로 명시하고 architecture의 구성 가능성과 empirical scope를 구별 |
| “외부 baseline protocol이 섞였다” | 외부 인용 행을 각주로 분리하고 protocol-matched 주장에는 자체 통일 평가 행만 사용 |
| “보편적으로 올바르다고 과장한다” | Fixed group/parameter/objective/aggregation 아래의 source-conditioned composition만 주장하고 optimizer trajectory equivalence나 universal necessity는 주장하지 않음 |

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
- Framework-specific alignment를 Method의 핵심 기여로 올리지 않는다.
