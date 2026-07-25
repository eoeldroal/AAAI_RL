# Execution-Efficiency Figure Bundle

StreamWeave의 실행 효율 로그에서 만든 그림, 생성 스크립트, 재현 데이터를 한곳에 보관한다.
본문·Appendix 채택 여부와 무관하게 효율성 관련 수치 그림은 이 폴더에서만 관리한다.

## Structure

| 경로 | 내용 |
|---|---|
| `outputs/` | 검토용 PNG, 편집 가능한 SVG, LaTeX 삽입용 PDF |
| `scripts/` | W&B 원장 갱신과 각 후보 그림의 생성 스크립트 |
| [`data/`](data/README.md) | Raw telemetry, 파생 CSV, manifest, 공개 efficiency figure 입력 JSON과 provenance |

## Outputs

모든 효율성 그림은 `outputs/` 한곳에 둔다. 현재 지위와 재현 여부만 아래 표에서 구분한다.

| 자산 | 역할 | 상태 |
|---|---|---|
| `execution_gpu_activity_overview.*` | Full-history GPU activity + active-GPU coverage + matched-wall-clock cumulative work | **SELECTED MAIN; COMPOSITION/SCOPE LOCKED; POLISH IN PROGRESS**, §4.3의 선택된 전폭 그림, 본문·caption 통합과 번호 확정 진행 중, 재생 가능 |
| `execution_activity_active_gpu.*` | Active-GPU count distribution | 통합본에 흡수된 source asset, 재생 가능 |
| `execution_efficiency_completion_tail.*` | Synchronous completion tail | **Appendix 전용**, 재생 가능 |
| `execution_efficiency_cumulative_work.*` | Cumulative work standalone source panel | 통합본에 흡수된 진단 자산, 재생 가능 |
| `execution_efficiency_throughput_stability.*` | Equal-work throughput stability | Appendix robustness, 재생 가능 |
| `execution_energy_candidate.*` | Work-weighted GPU-energy ECDF | **STRONG APPENDIX**, 재생 가능 |
| `streamweave_gpu_activity_exploration.png` | 전체 학습 이력의 per-GPU SM-activity heatmap | 통합본으로 대체된 탐색본; 입력 telemetry 보존, exact one-off script 없음 |

## Locked Publication Scope

`execution_gpu_activity_overview.*`의 시간 범위와 공개 수치 소유권은 다음으로 고정한다.

- `(a)--(b)`는 각 run의 전체 non-validation telemetry를 해당 run 안에서 독립적으로
  `0--100%` progress로 정규화한다. 같은 x 좌표를 동일 wall-clock 시각으로 해석하지 않는다.
- `(c)`는 synchronous complete non-validation history인 `79.7 min`과 StreamWeave의 동일 시간
  prefix를 비교한다. x축은 실제 wall-clock 분, y축은 synchronous endpoint를 `1.0x`로 둔 상대
  누적 prompt-group work다. 그림에는 총 prompt-group 수를 노출하지 않는다.
- Full-history `2.78 -> 4.58 groups/s`와 `1.64x`는 Table 2가 소유한다. Panel (c)의 endpoint
  ratio로 이를 대체하거나, 그림 안에서 Table 2의 scalar result를 반복하지 않는다.
- 최종 caption은 `(a)--(b)`의 run별 전체 이력과 `(c)`의 동일 wall-clock 비교를 명시한다.
  Table 2의 estimator를 caption에서 별도로 해명하지 않는다. 이 규칙을 바꾸려면 figure data,
  caption과 Table 2의 estimator scope를 함께 재검토해야 한다.

## Rebuild

저장된 raw CSV에서 효율성 원장과 공개 efficiency figure 입력을 다시 계산한다.

```bash
python docs/papers_RL/figures/execution_efficiency/scripts/refresh_execution_efficiency.py
```

W&B API에서 raw history까지 갱신하려면 `--fetch`를 추가한다.

```bash
uv run --no-project --with "wandb>=0.19,<0.24" \
  python docs/papers_RL/figures/execution_efficiency/scripts/refresh_execution_efficiency.py --fetch
```

각 그림은 bundle 내부의 frozen snapshot에서 다시 생성한다.

```bash
python docs/papers_RL/figures/execution_efficiency/scripts/plot_execution_active_gpu.py

python docs/papers_RL/figures/execution_efficiency/scripts/plot_execution_gpu_activity_overview.py

python docs/papers_RL/figures/execution_efficiency/scripts/plot_execution_efficiency_evidence_draft.py \
  --input-snapshot docs/papers_RL/figures/execution_efficiency/data/figure_evidence_draft.json \
  --output-dir docs/papers_RL/figures/execution_efficiency/outputs

python docs/papers_RL/figures/execution_efficiency/scripts/plot_execution_efficiency_throughput_stability.py \
  --input-snapshot docs/papers_RL/figures/execution_efficiency/data/figure_throughput_stability.json \
  --output-dir docs/papers_RL/figures/execution_efficiency/outputs
```

`plot_execution_energy_candidate.py`는 local W&B binary 두 개를 입력받는 진단용 스크립트다.
재현 가능한 공개 energy 자산은 `refresh_execution_efficiency.py`가 생성하는 CSV와
`execution_energy_candidate.*`가 소유한다.

## Claim Boundary

- `zero active`를 idle, stall, 또는 0% GPU utilization로 바꾸어 부르지 않는다.
- 15초 telemetry row를 독립 실험 반복으로 해석하지 않는다.
- Completion-tail 그림은 본문 그림으로 사용하지 않는다. 핵심 관측은 §4.3의 기전 설명에
  사용할 수 있지만, 전체 cycle 곡선은 Appendix에서만 제시한다.
- Energy 값은 8-GPU device-power telemetry의 sample-based estimate이며 node-total energy나
  외부 전력계 측정이 아니다.
- 공개 수치와 금지 해석의 기준은 `data/manifest.json`과 `data/verified_snapshot.json`을 따른다.
- §4.3의 통합본으로 main evidence chain이 닫혔으므로 새로운 efficiency figure 탐색은 종료한다.
