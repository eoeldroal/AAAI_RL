# StreamWeave Figure Index

그림은 `SVG`를 원본으로 사용한다. `PNG`는 빠른 검토용, `PDF`는 LaTeX 삽입용이다. 실행 효율
그림·스크립트·원장은 `execution_efficiency/` bundle이 함께 소유한다.

공개 도식은 flat, line-first academic schematic을 따른다. Rounded container는 실제 계산 경계나
자료 객체를 나타낼 때만 사용하고, section 구분은 여백과 얇은 rule로 처리한다.
한 의미는 한 위치에서만 표현한다. Complete-group 상태는 reconstruction에서, source identity는
분기 색과 형태에서, update semantics는 operator label에서만 보여 준다. 의미 있는 합류점은 반드시
명시적 junction으로 표시하고, 장식용 outer container와 반복 아이콘은 두지 않는다.

## 색·기호 규약 (2026-07-24 개정)

구 규약(blue policy / orange expert / teal provenance / coral failure)은 **mono+gold**로 대체한다.

- **neutral(먹·회색)** = policy 계열 전부 (rollout, policy record, RL 경로)
- **gold(#b68235)** = expert supervision **이자 학습 신호의 존재.** τ* 카드·주입 화살표·expert record·
  train bar의 gold 구간에만 사용. gold의 부재가 곧 "학습 신호 없음"의 시각적 진술이다.
- **✗ 글리프** = attempt 실패 (색 아님 — 흑백 인쇄 안전)
- **provenance** = metadata 칩(`group · source · policy-version`)의 형태로 표현 (색 아님)
- **회색 dashed** = control plane (parameter refresh)
- 세로 눈금 문법: 실선 = barrier(기다려 얻은 경계), 점선 = barrier 없는 boundary(완성 눈금).
  두 그림이 공유하는 핵심 어휘.

## 공개 그림 세트 (2026-07-24 개정)

| 자산 | 역할 | 상태 | 캡션이 맡아야 할 내용 |
|---|---|---|---|
| Figure 1 — 문제 서사 ([`plan-figure1-draft.md`](plan-figure1-draft.md)) | 두 병목(compute·signal) × 두 부분 처방: sync+expert=신호만, fully-async RL=compute만, StreamWeave=둘 다 | **DESIGN LOCKED / EXPORT PENDING**; 구 `figure1_streamweave_overview` 대체 | 세 행이 같은 all-✗ hard group을 비교한다는 무대 설명, row C 분기의 γ-threshold 정의, row B의 train이 다른 group 신호로는 계속 돈다는 사실 |
| Figure 2 — runtime pipeline ([`plan-diagram-set.md`](plan-diagram-set.md)) | 5단계 파이프라인(Generator / Reconstruction·source decision / Routed-group stream / Conversion / Trainer)과 단일 branch-blind update | **DESIGN LOCKED / EXPORT PENDING**; 구 `figure2_training_pipeline` 대체 | 경계 단 2개가 각각 generator/trainer 프로세스 내부에서 실행된다는 사실, π_g 칩의 per-sample 단위, mixed batch의 branch-blind reduction |

Figure 1·2는 AAAI `figure*` 전폭 배치를 전제로 한다. Figure 1은 "왜"(두 병목, 두 부분 처방)를,
Figure 2는 "어떻게"(경계 장치와 단일 update)를 담당한다. §4.2와 §4.3의 실증 그림은 아래에서
별도로 관리하며, 최종 번호는 전체 AAAI 배치가 잠길 때 정한다.

### 구 승인본의 지위

- `figure1_streamweave_overview.*` — **폐기.** "Wait→idle / Decide early→wrong source" 프레임은
  사실 오류(fully-async RL은 group을 조기 판정하지 않으며, 실패의 정체는 학습 신호의 부재다).
  기각 이력은 [`plan-figure1-draft.md`](plan-figure1-draft.md) 참조.
- `figure2_training_pipeline.*` — **폐기.** "RL operators / SFT update 두 박스 + ⊕" 구도는
  코드 사실(단일 branch-blind loss, per-branch 합산 불가)과 어긋나고 "SFT"는 내부 용어다.
- `asynchpt_efficiency.*`, `figure1_streamweave_draft.*` — 이전 iteration, 이미 대체됨.

## 실증·보조 자산

| 자산 | 역할 | 현재 지위 |
|---|---|---|
| `learning_effect_dynamics.*` | Normalized progress 위의 StreamWeave·Async RL (expert-off) reasoning performance와 expert-route dynamics | **LOCKED MAIN; RENDER/QA COMPLETE.** 절대 cycle과 training scale을 노출하지 않는 §4.2 canonical asset; 최종 번호만 미정 |
| `learning_effect_single_panel_draft.*` | Cycle 축을 사용한 이전 learning-effect draft | 신규 normalized-progress 자산으로 대체된 source asset |
| `execution_efficiency/outputs/execution_gpu_activity_overview.*` | Full-history GPU activity, active-GPU coverage, matched-wall-clock relative cumulative work | **LOCKED MAIN; RENDER/VISUAL QA COMPLETE.** §4.3 본문·caption 통합과 SVG/PDF/PNG export 완료; 최종 번호와 LaTeX 배치 확인만 남음 |
| `execution_efficiency/outputs/execution_activity_active_gpu` | Standalone active-GPU distribution | 통합본에 흡수된 source asset |
| `execution_efficiency/outputs/execution_efficiency_cumulative_work` | Standalone cumulative work | 통합본에 흡수된 source asset |
| `execution_efficiency/outputs/execution_energy_candidate` | Prompt-group-normalized GPU energy | **STRONG APPENDIX.** Work-weighted energy/group ECDF |
| `execution_efficiency/outputs/execution_efficiency_completion_tail` | Synchronous request-completion tail | **APPENDIX.** 핵심 관측만 §4.3 산문에서 회수 |

## 제외한 그림

- Three-clause audit는 그림보다 표가 더 직접적이므로 figure로 중복하지 않는다.
- Queue 크기, subset-sum, trim-and-carryover, event-loop 개선 배수는 구현 공정에 과도하게 시선을
  끌어 본문의 추상화를 훼손하므로 독립 그림으로 만들지 않는다. 필요하면 Appendix 표로 회수한다.

## 재생성

Figure 1·2의 편집 가능한 DC HTML은 외부 디자인 세션에 있으며 저장소 파일 경로가 아니다.
저장소 안에서는 위 두 plan이 의도와 기각 이력을 보존한다. 확정 시 SVG/PDF/PNG를 이 디렉터리에
반입하고 본 표를 `LOCKED MAIN`으로 갱신한다.

`src/generate_paper_figures.cjs`와 `src/export_paper_figures.cjs`는 폐기된 구 Figure 1·2와 초기
learning-effect draft를 재생성하는 **legacy pipeline**이다. 기각 이력을 재현할 때만 사용하며,
현재 공개 그림을 생성하거나 갱신하는 명령으로 사용하지 않는다.

```bash
# Legacy assets only.
node src/generate_paper_figures.cjs
NODE_PATH=... node src/export_paper_figures.cjs
```

현행 learning-dynamics figure는 다음 명령으로 재생성한다.

```bash
uv run --no-project --with matplotlib --with numpy python \
  src/plot_learning_effect_dynamics.py \
  --input-snapshot data/learning_effect_single_panel_draft.json \
  --output-base learning_effect_dynamics
```

가시 텍스트는 패널 표시, 단계·축·단위, 최소 범례로 제한한다. 제목과 해석 문장은 LaTeX caption에만
둔다.

현재 배치 방향: Figure 1(문제 서사)은 Intro/§1의 `figure*` 전폭, Figure 2(pipeline)는 Method의
`figure*` 전폭으로 둔다. Learning-dynamics figure와 통합 efficiency figure의 렌더·시각적 QA 및
각 본문 연결은 완료됐다. 두 실증 그림의 최종 번호와 LaTeX 배치 확인만 남아 있다. 새로운 efficiency
후보 탐색은 종료한다. Activity telemetry의 15초 interval을 독립 실험 반복으로 해석하지 않으며,
`zero active`를 idle이나 stall로 바꾸어 부르지 않는다. Energy panel은 같은 8-GPU
device-power telemetry의 sample-based estimate로서 Appendix에서만 사용하며, node 전체 에너지나
독립 전력계 측정으로 바꾸어 부르지 않는다.
