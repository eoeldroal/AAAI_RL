# StreamWeave Paper Workspace

이 디렉터리는 StreamWeave AAAI 원고, 증거 원장과 그림 자산의 단일 진입점이다. 코드 구현 당시의
설계·실험 문서는 사실 확인을 위한 참고자료이며, 현재 논문의 주장·용어·서사를 결정하지 않는다.

## Start Here

논문 작업은 다음 순서로 시작한다.

1. [`Full_Paper_Draft_ko.md`](Full_Paper_Draft_ko.md)에서 현재 문안과 내부 편집 메모의
   cold-start capsule, 논문 헌법, claim boundary, evidence status를 확인한다.
2. [`PAPER_PLAN.md`](PAPER_PLAN.md)에서 남은 집필 순서와 asset 작업만 확인한다.
3. 필요한 증거가 있을 때만 아래 evidence 문서와 bundle로 내려간다.
4. 구현 사실이 필요한 경우에만 코드용 참고 문서를 연다. 참고 문서의 과거 용어나 판정은
   `Full_Paper_Draft_ko.md`를 덮어쓰지 않는다.

## Document Authority

| 등급 | 문서 | 소유하는 것 | 소유하지 않는 것 |
|---|---|---|---|
| **CANONICAL** | [`Full_Paper_Draft_ko.md`](Full_Paper_Draft_ko.md) | 공개 원고, 논문 헌법, claim boundary, design/evidence 결정 | 단기 작업 dashboard |
| **PLAN** | [`PAPER_PLAN.md`](PAPER_PLAN.md) | 남은 집필 순서, section·asset 계획, 금지된 회귀 | 공개 문안과 새로운 claim |
| **EVIDENCE** | [`Efficiency.tex`](Efficiency.tex), [`figures/figures_README.md`](figures/figures_README.md), [`figures/execution_efficiency/`](figures/execution_efficiency/) | 수치 정의, provenance, 그림 상태와 재생 경로 | 논문의 thesis와 novelty |
| **LEGACY SOURCE** | [`Draft.tex`](Draft.tex) | 구현 세부, Appendix 후보, 과거 유도 | 현재 방법명·포지셔닝·main-run 결론 |

충돌할 경우 `CANONICAL -> PLAN -> EVIDENCE -> REFERENCE ONLY` 순서를 따른다. 코드와 canonical
문서가 다르면 먼저 실제 구현을 확인하고, 그 사실을 어떤 공개 주장으로 사용할지는 canonical 문서에서
다시 결정한다.

상위 CoWork workspace의 `Paper_writing/TASKS.md`는 dashboard를 위한 선택적 mirror다. 저장소의
필수 문서가 아니며, 별도의 우선순위·완료 상태나 논문 결정을 소유하지 않는다.

## Reference-Only Code Documents

다음 문서는 async-HPT 구현, 실행, 디버깅과 실험 역사를 이해하기 위한 자료다. 논문을 쓸 때는
필요한 사실만 가져오고, 내부 명칭·과거 main·구현 장치를 novelty framing으로 직접 복사하지 않는다.

| 문서 | 참고 용도 |
|---|---|
| [`../Codebase_Onboarding_RL.md`](../Codebase_Onboarding_RL.md) | verl fork와 프로젝트 코드의 온보딩 |
| [`../Overview_RL.md`](../Overview_RL.md) | 구현의 목적과 전체 구조 |
| [`../Codemap_RL.md`](../Codemap_RL.md) | 코드 위치, payload 흐름, failure boundary |
| [`../Readme_RL.md`](../Readme_RL.md) | 환경, launcher, 로그 점검 |
| [`../AsyncBudget_RL.md`](../AsyncBudget_RL.md) | queue·staleness·batch 운용 기록 |
| [`../Debug_RL.md`](../Debug_RL.md) | 디버깅·프로파일링 절차 |
| [`../MIGRATION.md`](../MIGRATION.md) | 평가 자산의 이전·복원 기록 |
| [`../Ablation_RL.md`](../Ablation_RL.md) | run과 ablation 역사; 현행 main 사실은 §14 |
| [`../Improvement_RL.md`](../Improvement_RL.md) | 병리 분석과 개선 과정의 역사 |
| [`../DR-001-loss-normalization_1.md`](../DR-001-loss-normalization_1.md), [`../DR-002-auxiliary-terms_1.md`](../DR-002-auxiliary-terms_1.md), [`../DR-003-offpolicy-supervised-branch_1.md`](../DR-003-offpolicy-supervised-branch_1.md), [`../DR-004-offpolicy-rl-branch_1.md`](../DR-004-offpolicy-rl-branch_1.md), [`../DR-005-rl-objective-composition_1.md`](../DR-005-rl-objective-composition_1.md) | 구현 당시 목적함수 결정과 수학적 근거 |

`AAAI_RL/AGENTS.md`와 일반 `docs/`는 코드 작업 및 verl 문서용이다. 논문 작성 규율이나 현재
StreamWeave claim의 출처로 사용하지 않는다.

## Figure And Evidence Entry Points

- 그림 상태와 공개 세트: [`figures/figures_README.md`](figures/figures_README.md)
- 실행 효율 그림·스크립트·데이터:
  [`figures/execution_efficiency/README.md`](figures/execution_efficiency/README.md)
- 효율 수치와 estimator ledger: [`Efficiency.tex`](Efficiency.tex)
- AAAI 형식과 제출 템플릿: [`../../AuthorKit27/`](../../AuthorKit27/)

[`figures/README.md`](figures/README.md)는 기존 링크 호환을 위한 포인터다. 그림 결정은
`figures_README.md`만 수정한다.

## Maintenance Rules

- 현재 주장이나 용어를 바꾸면 먼저 `Full_Paper_Draft_ko.md`를 갱신한다.
- 실행 순서만 바뀌면 `PAPER_PLAN.md`를 갱신한다.
- 수치·모집단·estimator가 바뀌면 해당 evidence ledger와 canonical 원고를 함께 검토한다.
- 코드 참고 문서에는 paper-facing 결론을 중복해서 쌓지 않고 이 인덱스와 canonical 원고를 가리킨다.
- 역사 기록은 삭제하지 않는다. 현재 판단과 충돌하는 경우 `REFERENCE ONLY`, `LEGACY`, 또는
  해당 절의 historical status를 명시한다.
