# StreamWeave Figure Index

현행 제출본은 `final/figures/figure2.pdf`, `figure3.pdf`, `figure4.pdf`를 각각 논문의 Figure
1--3으로 사용한다. Source asset `figure1.pdf`의 timeline은 2026-07-28 본문에서 제거했으며,
provenance를 위해서만 보존한다. 해당 timeline은 source routing을 현행 Figure 1과 중복하고,
나머지는 상속한 asynchronous scheduling을 재도식화하므로 새롭고 비중복적인 claim 없이는
복원하지 않는다.

현행 세 figure는 `final/figures/tikz/`의 TikZ source를 원본으로 사용하고, 미리 생성한 `PDF`만
LaTeX 제출본에 삽입한다. 기존 plotting source와 `execution_efficiency/` bundle은 수치·변환
원장으로 유지한다. `PNG`는 빠른 검토용, `PDF`는 LaTeX 삽입용이다.

공개 도식은 flat, line-first academic schematic을 따른다. Rounded container는 실제 계산 경계나
자료 객체를 나타낼 때만 사용하고, section 구분은 여백과 얇은 rule로 처리한다.
한 의미는 한 위치에서만 표현한다. Complete-group 상태는 reconstruction에서, source identity는
분기 색과 형태에서, update semantics는 operator label에서만 보여 준다. 의미 있는 합류점은 반드시
명시적 junction으로 표시하고, 장식용 outer container와 반복 아이콘은 두지 않는다.

## 색·기호 규약 (2026-07-27 개정)

- **deep teal (`figowned`)** = StreamWeave가 국소화한 decision/input 경계와 handoff
- **deep amber (`figdecide`)** = complete-group source decision
- **먹·회색 (`figink`, `figsub`)** = data glyph와 상속받은 asynchronous substrate
- **solid bundle / hatch singleton** = policy group / expert trajectory; 색 없이도 source를 구분
- **회색 dashed** = parameter refresh와 낮은 강조도의 control-plane relation
- **✗/✓ 글리프** = attempt outcome; fallback font를 쓰지 않도록 TikZ path로 직접 그린다.

두 accent color는 WCAG 4.5:1을 넘고 CMYK로 출력한다. 모든 텍스트는 최종 배치 크기에서 최소
9pt, 모든 선은 최소 0.5pt이며, 현행 Figure 1--3은 동일한 Times 계열 활자와 glyph 문법을 공유한다.

## 공개 그림 세트 (2026-07-27 개정)

| 자산 | 역할 | 상태 | 캡션이 맡아야 할 내용 |
|---|---|---|---|
| Retired timeline ([`tikz/figure1.tex`](tikz/figure1.tex)) | Local waiting과 계속되는 work의 timeline | **RETIRED; NOT INCLUDED** | Source routing 중복과 inherited async scheduling 재도식화 때문에 복원하지 않음 |
| Figure 1 — data-state boundaries ([`tikz/figure2.tex`](tikz/figure2.tex)) | Complete-group decision, source-specific training-input construction, one primary update에서 객체가 어떻게 바뀌는지 시각화 | **LOCKED MAIN; TIKZ SOURCE / PDF RENDER / QA COMPLETE** | Policy group과 expert singleton의 cardinality·provenance 차이, shared queue, source별 correction과 common update endpoint |
| Figure 2 — learning dynamics ([`../final/figures/tikz/figure3.tex`](../final/figures/tikz/figure3.tex)) | Normalized progress 위의 quality와 expert-routing dynamics | **LOCKED MAIN; TIKZ SOURCE / PDF RENDER / QA COMPLETE** | Expert demand가 감소하지만 후반에도 남고 expert-on/off behavior가 이후 분리됨 |
| Figure 3 — execution efficiency ([`../final/figures/tikz/figure4.tex`](../final/figures/tikz/figure4.tex)) | GPU activity, active-GPU coverage와 matched-wall-clock cumulative work | **LOCKED MAIN; TIKZ SOURCE / PDF RENDER / QA COMPLETE** | Local waiting과 overlap이 sustained concurrent activity와 더 많은 누적 work로 나타남 |

Figure 1과 Figure 3은 AAAI `figure*` 전폭 자연 크기, Figure 2는 단일 열 배치를 사용한다.
Local waiting과 slot reuse는 별도 timeline이 아니라 §3.2 산문과 Algorithm 1이 소유한다.

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
| `learning_effect_dynamics.*` | Normalized progress 위의 StreamWeave·Async RL (expert-off) reasoning performance와 expert-route dynamics | **LOCKED CONTENT; TIKZ RENDER / QA COMPLETE.** 절대 cycle과 training scale을 노출하지 않는 §4.2 canonical asset; 기존 PDF/PNG는 data·layout reference |
| `learning_effect_single_panel_draft.*` | Cycle 축을 사용한 이전 learning-effect draft | 신규 normalized-progress 자산으로 대체된 source asset |
| `execution_efficiency/outputs/execution_gpu_activity_overview.*` | Full-history GPU activity, active-GPU coverage, matched-wall-clock relative cumulative work | **LOCKED CONTENT; TIKZ RENDER / QA COMPLETE.** §4.3의 세 패널과 비교 범위는 고정; 기존 PDF/PNG는 data·layout reference |
| `execution_efficiency/outputs/execution_activity_active_gpu` | Standalone active-GPU distribution | 통합본에 흡수된 source asset |
| `execution_efficiency/outputs/execution_efficiency_cumulative_work` | Standalone cumulative work | 통합본에 흡수된 source asset |
| `execution_efficiency/outputs/execution_energy_candidate` | Prompt-group-normalized GPU energy | **STRONG APPENDIX.** Work-weighted energy/group ECDF |
| `execution_efficiency/outputs/execution_efficiency_completion_tail` | Synchronous request-completion tail | **APPENDIX.** 핵심 관측만 §4.3 산문에서 회수 |

## 제외한 그림

- Three-clause audit는 그림보다 표가 더 직접적이므로 figure로 중복하지 않는다.
- Queue 크기, subset-sum, trim-and-carryover, event-loop 개선 배수는 구현 공정에 과도하게 시선을
  끌어 본문의 추상화를 훼손하므로 독립 그림으로 만들지 않는다. 필요하면 Appendix 표로 회수한다.

## 재생성

현행 Figure 1--3의 canonical source와 standalone wrapper, 공통 style은
[`../final/figures/tikz/`](../final/figures/tikz/)가 소유한다. 해당 README의 명령으로 외부 PDF를
생성하고, 제출본은 생성 PDF만 `\includegraphics`로 삽입한다. `plan-figure1-draft.md`,
`plan-diagram-set.md`, DC HTML은 이전
설계 이력이며 현행 figure를 재생성하는 원장이 아니다.

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

현재 배치: Figure 1은 Introduction에서 미리 선언해 **2페이지 최상단**의 `figure*`로 고정하고
§3이 해석하며, Figure 2는 §4.2의 단일 열 `figure`, Figure 3은 §4.3의 `figure*`에 삽입한다.
세 figure의 data·계산·본문 연결과 시각 QA는 완료됐다. 새로운
efficiency 후보 탐색은 종료한다. Activity telemetry의 15초 interval을 독립
실험 반복으로 해석하지 않으며, `zero active`를 idle이나 stall로 바꾸어 부르지 않는다. Energy
panel은 같은 8-GPU device-power telemetry의 sample-based estimate로서 Appendix에서만 사용하며,
node 전체 에너지나 독립 전력계 측정으로 바꾸어 부르지 않는다.
