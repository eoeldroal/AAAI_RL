# Execution-Efficiency Data Package

이 디렉터리는 StreamWeave 실행 효율 주장의 데이터 계보를 고정한다. 공개 headline은
`2.78 → 4.58 prompt groups/s`, `46.0 → 28.0 s / 128 groups`, `1.64×`다.
Cycle 수나 learner row 수를 공통 work unit으로 사용하지 않는다.

## 파일 지도

| 파일 | 역할 |
|---|---|
| `manifest.json` | W&B entity/project/run ID, metric key, 집계 규칙 |
| `verified_snapshot.json` | 2026-07-23 full-history 분석에서 잠근 모든 공개·Appendix 수치 |
| `computed_snapshot.json` | Raw CSV에서 refresh script가 다시 계산하는 검토용 snapshot |
| `run_aggregates.csv` | Headline throughput의 최소 원장 |
| `equal_work_windows.csv` | Unequal run length에 대한 동일 group-budget 검증 |
| `block_throughput.csv` | StreamWeave 전 구간의 13,312-group block 안정성 |
| `gpu_activity_distribution.csv` | 20% SM-active 기준으로 동시에 활성인 GPU 수의 training-interval 분포 |
| `gpu_activity_threshold_sensitivity.csv` | SM-active 문턱 10–50%에 대한 분포 결론의 민감도 |
| `gpu_activity_population_sensitivity.csv` | 첫 training cycle 포함 여부에 대한 분포 결론의 민감도 |
| `gpu_energy_summary.csv` | Non-validation cycle의 8-GPU power와 prompt-group-normalized energy 요약 |
| `gpu_energy_cycle_points.csv` | Power--throughput와 work-weighted distribution의 cycle-level 입력 |
| `gpu_energy_sensitivity.csv` | Pooled, cycle-weighted, edge-trimmed, startup·validation population 검산 |
| `execution_summary_table.csv` | §4.3 일반 표의 prompt-group-normalized 실행 요약 |
| `mechanism_metrics.csv` | Critical-path, boundedness, utilization의 보조 수치와 공개 범위 |
| `raw/*_history.csv` | 인증된 refresh가 생성하는 cycle-level default history |
| `raw/*_system.csv` | 인증된 refresh가 생성하는 15초 system telemetry |
| `../scripts/refresh_execution_efficiency.py` | W&B full history export와 파생 자산 재생성 |

`figure3_execution_efficiency.json`은 이 디렉터리의 snapshot에서 active-GPU와 energy 단일
패널에 필요한 값만 얇게 투영한 파일이다. 기존 원장 호환성을 위해 파일명은 유지한다. 논문 문안은
`Efficiency.tex`이, 계산 가능한 숫자 원장은 이 디렉터리가 소유한다.

## 데이터 계보

1. Synchronous run: `eoeldroal-sogang-university/async-hpt-openr1/v96fvd0p`
2. StreamWeave main: `eoeldroal-sogang-university/async-hpt-openr1/oki4kv8u`
3. Numerator: cycle별 `hpt/onpolicy_num_groups`의 합
4. Denominator: evaluation을 제외한 cycle별 `timing_s/step`의 합
5. Estimator: `sum(groups) / sum(time)`
6. Equal-work sensitivity: boundary cycle의 time을 group 비율로 선형 배분
7. GPU activity population: validation timer가 없는 cycle의
   `[history timestamp - timing_s/step, history timestamp]` 안에 들어오는 system row
8. GPU activity metric: 여덟 GPU의 `system.gpu.<i>.smActive`가 모두 존재하는 15초 row에서
   20%를 초과한 GPU 수
9. GPU energy estimate: 같은 interval의 `system.gpu.<i>.powerWatts` 합을 cycle duration과
   consumed prompt groups로 정규화. Cycle을 독립 반복으로 취급하지 않음

`hpt/onpolicy_num_groups`는 `group_uid`를 중복 제거해 learner가 소비한 고유 prompt
group을 센다. RL-routed group은 8 rows, expert-routed group은 1 row를 만들기 때문에 rows/s,
samples/s, steps/s는 headline에 사용하지 않는다.

## Refresh

W&B API 인증이 설정된 환경에서 다음 명령이 default/system raw CSV, aggregate,
equal-work window, GPU-activity distribution과 Figure 3 입력 JSON을 함께 갱신한다.

```bash
uv run --no-project --with "wandb>=0.19,<0.24" \
  python docs/papers_RL/figures/execution_efficiency/scripts/refresh_execution_efficiency.py --fetch
```

이미 내려받은 raw CSV만으로 파생 파일을 다시 계산하려면 `--fetch`를 생략한다.
API key와 browser cookie는 저장하지 않는다.

현재 raw CSV의 power columns는 로컬 `B_wandb.tar.zst`에서 추출한 canonical `.wandb` stats
record를 timestamp로 exact join해 고정했다. Sync 447개와 StreamWeave 1,729개의 complete
eight-GPU power row가 기존 API-exported system timeline과 정확히 일치한다. 향후 인증된
`--fetch`도 같은 `system.gpu.<i>.powerWatts` key를 직접 채운다.

## 현행 snapshot의 범위

`verified_snapshot.json`과 현재 CSV 원장은 로그인된 W&B full-history API로 수행한
2026-07-23~24 분석 결과를 고정한다. Default history는 sync 104 cycles, StreamWeave 190 cycles이며,
system stream은 각각 897, 3,475 rows다. Raw CSV를 함께 보존하므로 인증 없이도 파생 자산을 재생성할
수 있다. 원시 행을 집계값에서 역으로 추정하지 않는다.

본문의 end-to-end 결과는 full-history aggregate 하나를 사용한다. Figure 3의 active-GPU 분포는
같은 run에서 관측된 실행 기전 증거이며 별도의 speedup으로 세지 않는다. 20% 기준에서 아무 GPU도
문턱을 넘지 않은 interval은 `27.9% → 4.7%`, 평균 active GPU 수는 `5.40 → 6.92`다. 이 방향은
10–50% 문턱 전체에서 유지된다. `zero active`는 정확히 이 threshold event만 뜻하며 idle, stall,
0% utilization로 바꾸어 부르지 않는다. 첫 training cycle을 양쪽에서 제외해도 `24.9% 대 4.7%`,
평균 `5.63 대 6.93`으로 방향이 유지된다.

같은 non-validation population에서 cycle mean total GPU power를 duration과 prompt-group 수로
정규화하면 estimated energy/group은 `1.504 → 1.066 kJ`다. Pooled sample mean, cycle-edge
trimming, 첫 cycle 제외와 validation-shifted full coverage를 함께 적용한 감소 범위는
`23.5–29.1%`다. 공개 문구는 **약 24–29%**로 반올림한다. 이는 8-GPU device power의
sample-based estimate이며 node 전체, 냉각, token-normalized energy나 외부 power meter를 뜻하지
않는다.

Equal-work, block stability, startup/transient sensitivity, queue, parameter synchronization,
source mixture, per-GPU heatmap, memory-allocation churn과 threshold sensitivity는 Appendix
방어 자산이다.
