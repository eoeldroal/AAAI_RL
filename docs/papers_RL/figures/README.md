# StreamWeave Figure Index

그림은 `SVG`를 원본으로 사용한다. `PNG`는 빠른 검토용, `PDF`는 LaTeX 삽입용이다. 수치 그림은
`data/*.json`과 분리되어 있어 측정값이 확정되면 데이터를 교체하고 다시 생성할 수 있다.
공개 도식은 flat, line-first academic schematic을 따른다. Rounded container는 실제 계산 경계나
자료 객체를 나타낼 때만 사용하고, section 구분은 여백과 얇은 rule로 처리한다.
한 의미는 한 위치에서만 표현한다. Complete-group 상태는 reconstruction에서, source identity는
분기 색과 형태에서, update semantics는 operator label에서만 보여 준다. 의미 있는 합류점은 반드시
명시적 junction으로 표시하고, 장식용 outer container와 반복 아이콘은 두지 않는다.

## 공개 그림 세트

| 자산 | 역할 | 상태 | 캡션이 맡아야 할 내용 |
|---|---|---|---|
| `figure1_streamweave_overview` | 나이브 결합의 double bind와 StreamWeave의 execute-decide-learn 분리 | **본문 사용 가능** | complete-group decision, nonblocking execution, source-native learning path, 색·기호 의미 |
| `figure2_training_pipeline` | rollout, provenance-aware composition, source-aware learning을 잇는 end-to-end architecture | **본문 사용 가능** | 색·기호 의미, policy context는 rollout에만 적용됨, parameter refresh는 control-plane임 |

두 그림은 AAAI `figure*` 전폭 배치를 전제로 한다. Figure 1은 문제와 핵심 설계 판단을, Figure 2는
그 판단이 전체 학습 파이프라인에서 어떻게 닫히는지를 담당한다. SVG 내부의 각 단계는 편집 가능한
그룹으로 유지한다.

## 보류한 실증 자산

| 자산 | 역할 | 현재 지위 |
|---|---|---|
| `figure2_learning_effect` | main과 RL-only의 early/late window 비교 | 수치가 확정될 때까지 본문 서사에서 제외하고 Appendix 후보로 보관 |
| `figure3_execution_efficiency` | 동일 8xB200·128 group의 시간축과 처리량/idle/MFU | 공정한 최종 재측정 전까지 본문 서사에서 제외 |

## 아직 잠긴 본문 그림

최종 `quality-versus-wall-clock`은 `data/figure2_quality_wallclock.schema.json`의 계약을 따르며,
대상 run과 현재 차단 사유는 `data/figure2_quality_wallclock.pending.json`에 고정했다. 현재 문서에는
step-window 요약만 있고 원시 W&B `_runtime` 이력이 없으므로 연속 곡선을 보간하지 않았다.
main-vs-RL-only 원시 이력이 복구되면 같은 grader와 evaluation multiplicity를 확인한 뒤 최소 8개
실측점을 넣는다. async-vs-sync 품질 곡선은 synchronous arm의 공정화 전까지 만들지 않는다.

## 제외한 그림

- Three-clause audit는 그림보다 표가 더 직접적이므로 figure로 중복하지 않는다.
- Queue 크기, subset-sum, trim-and-carryover, event-loop 개선 배수는 구현 공정에 과도하게 시선을
  끌어 본문의 추상화를 훼손하므로 독립 그림으로 만들지 않는다. 필요하면 Appendix 표로 회수한다.
- `asynchpt_efficiency.*`와 `figure1_streamweave_draft.*`는 이전 iteration이며 새 자산으로 대체되었다.

## 재생성

```bash
node src/generate_paper_figures.cjs
NODE_PATH=/Users/baghyeonbin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  node src/export_paper_figures.cjs
```

반복 편집 중에는 필요한 자산 이름만 넘겨 생성과 export를 제한할 수 있다.

```bash
node src/generate_paper_figures.cjs figure1_streamweave_overview figure2_training_pipeline
NODE_PATH=/Users/baghyeonbin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  node src/export_paper_figures.cjs figure1_streamweave_overview figure2_training_pipeline
```

가시 텍스트는 패널 표시, 단계·축·단위, 최소 범례로 제한한다. 제목과 해석 문장은 LaTeX caption에만
둔다.

현재 권장 배치는 `figure1_streamweave_overview`와 `figure2_training_pipeline`을 `figure*` 전폭으로
두는 것이다. 실증 자산의 최종 배치는 수치와 본문 구조가 확정된 뒤 결정한다.
