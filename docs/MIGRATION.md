# 새 서버 마이그레이션

이 문서는 git clone으로 복원되지 않는 평가 자산의 이전 범위를 고정한다. 먼저 현재 작업을 commit/push하고 새 서버에서 같은 origin HEAD를 checkout한다. 생성 결과의 파일 수와 크기가 이 문서와 다르면 각 결과 디렉터리의 generated manifest를 최종 근거(source of truth)로 사용하며 추정값으로 보정하지 않는다.

## 이전 범위

### Eval-only HF 체크포인트

병합 결과는 `checkpoints/migration_hf_eval/` 아래의 다음 11개 디렉터리다. optimizer state 없이 validation과 추론에 필요한 HF 파일만 둔다.

| arm | 결과 경로 |
| --- | --- |
| nocispo main | `qwen25_math_1_5b_openr1_async_hpt_M5abl_nocispo/global_step_{140,160,170,190}` |
| M5 | `qwen25_math_1_5b_openr1_async_hpt_M5_cleanasync/global_step_{45,65,95}` |
| RL-only | `qwen25_math_1_5b_openr1_async_RLonly_grpo/global_step_{50,110,140}` |
| sync paper-HPT | `paper_hpt_sync_qwen25_math_1_5b_beta03/global_step_50` |

정확한 파일 목록, 크기, checksum은 `checkpoints/migration_hf_eval/`에 생성된 manifest를 따른다. `models/Qwen2.5-Math-1.5B`, `models/LUFFY-Qwen-Math-1.5B-Zero`를 포함한 `models/` 전체는 이 bundle로 이전하지 않는다. 새 서버에서 원 출처로부터 다시 받는다.

### Tier-1 실험 자산

위치: `.cache/migration/tier1_assets/`

| 자산 | archive bytes | 새 서버 복원 기준 디렉터리 |
| --- | ---: | --- |
| `B_wandb.tar.zst` | 52,084,861 | `VERL_ROOT` |
| `C_val_dumps_selected.tar.zst` | 661,328,383 | `VERL_ROOT` |
| `D_sync_rollout.tar.zst` | 115,361,732 | `VERL_ROOT` |
| `F_datas_hpt.tar.zst` | 447,922,991 | `VERL_ROOT` |
| `G_entropy_math.tar.zst` | 35,957 | `Unify-Post-Training/hpt/verl/verl/mix_src` |
| `H_train_dumps_selected.tar.zst` | 79,193,064 | `VERL_ROOT` |
| `I_driver_logs_selected.tar.zst` | 22,720,225 | `VERL_ROOT` |

총 archive 크기는 1,378,647,213 bytes다. 전체 원본 경로, 파일 수, 원본/압축 크기, SHA-256, 링크 처리 방식은 `.cache/migration/tier1_assets/asset_manifest.json`에 기록되어 있다. W&B의 외부 절대 심볼릭 링크는 새 서버에서 끊어지지 않도록 archive에 실제 내용으로 저장했다. H는 M5/nocispo/RL-only learner-side train dump만, I는 세 async run과 sync paper-HPT의 driver log만 포함한다.

다음 네 archive는 `VERL_ROOT`에서 복원한다.

```bash
for archive in \
  B_wandb.tar.zst \
  C_val_dumps_selected.tar.zst \
  D_sync_rollout.tar.zst \
  F_datas_hpt.tar.zst \
  H_train_dumps_selected.tar.zst \
  I_driver_logs_selected.tar.zst; do
  tar --zstd -xf ".cache/migration/tier1_assets/${archive}" -C "${VERL_ROOT}"
done
```

채점기는 별도 checkout의 `mix_src`를 기준으로 복원하거나 launcher의 `ENTROPY_MATH`를 새 위치로 지정한다.

```bash
mkdir -p "${UPT_ROOT}/hpt/verl/verl/mix_src"
tar --zstd -xf \
  "${VERL_ROOT}/.cache/migration/tier1_assets/G_entropy_math.tar.zst" \
  -C "${UPT_ROOT}/hpt/verl/verl/mix_src"
```

### Async rollout 선별본

원본 async rollout 전체는 이전하지 않는다. M5, nocispo, RL-only 세 run의 complete group을 전수 조사한 뒤 합계 20,000 group을 결정론적으로 선별하고, 선택 group마다 8 attempts 전부를 유지한다. `k`는 raw `acc`가 아니라 실제 post-fix gate 입력인 `rm_scores`로 계산한다. incomplete group은 census에는 남기되 selection에서는 제외한다.

표본은 다음 분석 범주를 함께 보존한다.

- run × 5-policy-version bin × `k=0`/`1≤k≤7`/`k=8` strata
- M5–nocispo의 stable/pre-storm/storm/recovery 동일-prompt pair
- nocispo–RL-only의 early/late 동일-prompt panel과 run 내부 route-transition panel
- partial, cross-version span, truncation, long-response, latency의 matched case/control
- source subtype과 prompt/teacher-trajectory 길이 tail
- 모집단 비율 추정 전용 run-size 비례 1,000-group deterministic random holdout

중복 prompt는 서로 다른 version/arm의 trajectory를 분석하기 위해 여러 group으로 들어올 수 있으며, 각 strata/event 안에서만 중복을 제거한다. 희귀사건 보강을 포함한 전체 20,000개로 모집단 비율을 추정하면 편향되므로 `is_random_holdout=true` 행만 사용한다. 결과는 `.cache/migration/rollout_selection/` 아래에 둔다.

정확한 run, group 수, incomplete 수, shard/archive 목록, checksum은 그 디렉터리의 generated manifest와 `selection_summary.json`이 유일한 근거다. 전송 도구는 원본 `.cache/rollout_dump/`가 아니라 이 결과 디렉터리만 허용한다. 이 dump에는 learner version/current log-probability/advantage가 없으므로 per-token staleness·ratio·clip 분석은 함께 보존한 W&B history를 사용한다. 또한 현재 train parquet는 모든 prompt에 `tau_messages`가 있어 missing-teacher fallback은 이 표본으로 실증할 수 없다.

## 환경 재구축

conda 환경 자체와 `.rt/`는 복사하지 않는다. 새 B200 서버에서 `RL` 환경을 다시 만들고 프로젝트가 검증된 CUDA 스택(torch 2.11, CUDA 13, sglang 0.5.12, flash-attn)을 설치한다. 이후 `${CONDA_PREFIX}/etc/conda/activate.d/verl_cuda_stack.sh`를 새 환경에 다시 배치하고 activation 후 CUDA, compiler, library path가 새 서버를 가리키는지 확인한다.

## 제외 항목

- `models/` 전체와 base Qwen/LUFFY 모델
- 원본 `.cache/rollout_dump/`의 async `gen_batch.dp` 트리
- superseded `.tar.zst`, M2/M3/M4/M5R/M7 체크포인트와 미선정 step
- 모든 optimizer state
- `.rt/`와 그 아래 runtime cache/`__pycache__`, conda 환경
- 전체 `logs/`와 `logs/nohup/`는 자동 bundle 밖이다. 논문 근거용 핵심 4개 driver log만 `I_driver_logs_selected.tar.zst`로 보존한다.

## 전송

대상 루트는 미리 생성한다. 명령은 기본적으로 dry-run이며 `--execute`가 있을 때만 데이터를 쓴다.

```bash
bash scripts/migration/transfer_bundle.sh "$VERL_ROOT" user@new-host:/srv/AAAI_RL
bash scripts/migration/transfer_bundle.sh --execute "$VERL_ROOT" user@new-host:/srv/AAAI_RL
```

스크립트가 허용하는 경로는 다음 세 개뿐이다.

- `checkpoints/migration_hf_eval`
- `.cache/migration/tier1_assets`
- `.cache/migration/rollout_selection`

각 경로가 존재하고 generated manifest가 있으며 `.part` 파일이 하나도 없을 때만 rsync를 시작한다.

## 새 서버 검증 순서

1. 저장소를 clone하고 `git rev-parse HEAD`가 이전 서버의 pushed HEAD와 같은지 확인한다.
2. 전송 스크립트의 dry-run 결과에 허용된 세 경로만 표시되는지 확인한 뒤 `--execute`한다.
3. Tier-1 archive와 manifest를 검증한다.

   ```bash
   cd "$VERL_ROOT/.cache/migration/tier1_assets"
   sha256sum -c *.tar.zst.sha256
   sha256sum -c asset_manifest.json.sha256
   jq . asset_manifest.json >/dev/null
   for archive in *.tar.zst; do
     zstd -t "$archive"
     tar --zstd -tf "$archive" >/dev/null
   done
   ```

4. HF와 rollout-selection의 generated manifest/checksum을 검증한다. archive만 확인할 때는 명시적으로 `--archives-only`를 사용한다. 추출 검증은 세 run root를 모두 넘겨야 성공한다.

   ```bash
   python scripts/migration/verify_rollout_selection.py \
     --selection-dir "$VERL_ROOT/.cache/migration/rollout_selection" \
     --archives-only

   python scripts/migration/verify_rollout_selection.py \
     --selection-dir "$VERL_ROOT/.cache/migration/rollout_selection" \
     --run-root M5=/restored/M5 \
     --run-root nocispo=/restored/nocispo \
     --run-root RLonly=/restored/RLonly
   ```
5. archive를 위 복원 루트에 풀고 `models/`의 base Qwen/LUFFY를 다시 받는다.
6. `RL` 환경과 activation hook을 재구축한 뒤 `torch.cuda.is_available()`, GPU capability, sglang/flash-attn import를 확인한다.
7. 세 parquet 데이터셋을 열고 `tau_messages` 계약을 확인하며 `entropy_math`의 최소 채점 smoke test를 실행한다.
8. 각 HF 결과에서 config/tokenizer/safetensors를 읽고 validation-only smoke run을 수행한다.
