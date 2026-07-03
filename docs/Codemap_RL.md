# RL Code Map

Navigation and debugging map for this async-HPT `verl` fork. Use it to find
where code lives, how a rollout sample becomes a learner update, and — when a
run breaks — which file/symbol to open first.

Companion docs (single source of truth, do not duplicate them here):

- `../AGENTS.md` — durable rules and contracts (what you may/may not change).
- `Readme_RL.md` — how to launch and size async runs; queue/staleness budgets.
- This file — where things live and where they break.

Line numbers rot; this map anchors on **file + symbol name**. `grep`/`rg` the
symbol to jump. Update it when a symbol is renamed or a subsystem moves.

## What This Fork Is

Upstream `verl` (RL for LLMs) plus one research line: **HPT (Hybrid Policy
Training)** as a first-class objective on the **fully-asynchronous** RL runtime.
Everything custom lives under `verl/experimental/fully_async_policy/`. Baseline
before the fork work: git tag/commit "Document clean verl export baseline".

## First 15 Minutes

1. Read `../AGENTS.md` (contracts) and `Readme_RL.md` (run budgets).
2. Skim this map's *Runtime Architecture* and *Data Flow* sections.
3. Run the CPU contract tests: `pytest tests/special_RL/ -v` (no GPU/Ray needed).
4. Read the HPT loss: `verl/workers/utils/losses.py::_compute_hpt_prompt_equal_policy_loss`.

## Repo Layout (RL-relevant)

```
verl/experimental/fully_async_policy/   # THE fork's home
  fully_async_main.py        # driver: wires 3 Ray actors, runs rollouter+trainer
  fully_async_rollouter.py   # sample PRODUCER (rollout+reward, no training WG)
  fully_async_trainer.py     # sample CONSUMER + actor update ("learner")
  message_queue.py           # Ray-actor deque between them
  detach_utils.py            # RolloutSample transport + non-HPT assembly + metrics
  hpt_config.py              # async_hpt.* parse + enable-time invariants
  hpt_gate.py                # route each prompt group: SFT (tau) vs RL
  hpt_payload.py             # load/tokenize tau expert trajectories
  hpt_assembler.py           # queue-sample -> learner rows + HPT loss tensors
  hpt_rollout_accumulator.py # regroup per-trajectory attempts (scheduler mode)
  hpt_training.py            # rollout-logprob anchor + stale-row filter
  config/fully_async_ppo_trainer.yaml   # async + async_hpt config block
verl/workers/utils/losses.py            # HPT prompt-equal policy loss
verl/trainer/ppo/rollout_corr_helper.py # off-policy IS/RS correction (RL rows only)
verl/protocol.py                        # DataProto: the cross-process data contract
verl/single_controller/                 # @register / Dispatch worker RPC
verl/checkpoint_engine/                 # CheckpointEngineManager (weight sync)
examples/data_preprocess/openr1_hpt.py  # build train/tau/eval parquets
tests/special_RL/                       # CPU contract tests (see below)
main_scripts/*.sh                       # our real main-run launchers
tests/special_e2e/run_fully_async_policy_*.sh  # smoke / contract launchers
datas/openr1_hpt_{smoke,main}/          # generated datasets
models/Qwen2.5-Math-7B/                 # main-run base model
docs/Readme_RL.md                       # run checklist + budget formulas
```

Launcher split: genuine main runs live in `main_scripts/`; smoke and other
contract-coverage launchers stay in `tests/special_e2e/`. Main launchers are
value-validated only (they must compose to a valid async-HPT config); their
concrete values are the source of truth, not asserted line-by-line.

## Runtime Architecture (3 Ray actors, not single-controller)

```
 [FullyAsyncRollouter] --put_sample--> [MessageQueue] --get_sample--> [FullyAsyncTrainer]
   produces samples       (cloudpickle)    (deque)        (blocking)     consumes + trains
   rollout+reward WGs                                                     actor(+critic/ref) WG
          ^                                                                      |
          +----- reset_staleness / update_weights  (control-plane RPC handle) ---+
```

- Driver `fully_async_main.py::FullyAsyncTaskRunner` builds all three, then runs
  `rollouter.fit()` and `trainer.fit()` as concurrent Ray futures.
- Data plane = the queue (`put_sample`/`get_sample`). Control plane = trainer's
  direct rollouter handle (param sync, staleness reset, validation).
- Both rollouter and trainer subclass `SeparateRayPPOTrainer`; the rollouter owns
  no training worker group, the trainer owns the actor WG.

### Rollouter loop (`FullyAsyncRollouter`)
`fit` -> `_streaming_generation_main` spawns `_feed_samples` (dataloader ->
in-proc `pending_queue`, each prompt repeated `rollout.n`x as a `RolloutSample`)
and `_processor_worker` (-> `_process_single_sample_streaming` -> generate ->
`_queue_completed_rollout_sample` -> `put_sample`). `_async_monitor_loop` logs
stats and clears backpressure. HPT gate is invoked inside
`_queue_completed_rollout_sample`; trajectory-scheduler mode instead fans out via
`_submit_hpt_trajectory_attempts` / `_process_hpt_trajectory_attempt_streaming`.

### Trainer loop (`FullyAsyncTrainer`)
`fit` -> `while True: fit_step()`. `fit_step` is the PPO template: `_fit_generate`
(-> `_get_samples_from_queue`) -> reward -> log_prob -> ref -> critic -> advantage
-> `_fit_update_actor` -> `_fit_update_local_step` -> `_fit_update_weights` ->
dump. A `None` from the queue raises `TrainingStopException`.

## Data Flow: rollout sample -> learner row

```
config validated        hpt_config.validate_async_hpt_config
  -> gate routes         hpt_gate.HptRolloutGate.route_rollout_sample / .route
     (SFT swaps full_batch to tau payload; RL keeps generated DataProto)
  -> [scheduler] regroup hpt_rollout_accumulator.HptPromptGroupAccumulator.pop_ready
  -> queue (cloudpickle) message_queue.put_sample / get_sample
  -> assemble            hpt_assembler.HptBatchAssembler.assemble_rollout_samples
     (writes hpt_is_sft, hpt_seq_weight, hpt_length_divisor, hpt_loss_denominator)
  -> anchor + filter     hpt_training.apply_hpt_rollout_logprob_anchor / filter_hpt_stale_rollout_samples
  -> loss                losses.ppo_loss -> _compute_hpt_prompt_equal_policy_loss
```

**Key asymmetry (memorize this):** an **SFT** route collapses a prompt group to
**1 learner row**; an **RL** route expands to **`rollout.n` rows**. So
*queue-sample count != learner-row count*. The trainer collects `required_samples`
queue samples, then keeps pulling until learner rows are divisible by
`_hpt_required_training_multiple` (= `lcm(dp_size, ppo_mini_batch_size*rollout_n)`),
bounded by `max_completed_prompt_groups`. This produces the log line:

```
[FullyAsyncTrainer] Loop collection completed: <q>/<req> samples, learner_rows=<r>, required_multiple=<m>
```

## HPT Objective (the research core)

- **Route** (`hpt_gate.py`): `route_to_sft = success_prob <= gamma and not missing_tau`,
  where `success_prob` = fraction of a group's rollouts scoring `> success_threshold`.
  Failing prompts with an expert answer -> supervised; the rest stay on-policy RL.
- **tau** = expert trajectory keyed by `prompt_uid`, column `tau_messages`
  (`hpt_payload.HptTauStore` / `HptSftPayload` / `HptTauToAgentLoopOutputAdapter`).
- **Rollout-logprob anchor** (`hpt_training.py`): RL rows use the rollout engine's
  own `rollout_log_probs` as `old_log_probs` (needs `rollout.calculate_log_probs=True`).
- **prompt-equal loss** (`losses.py::_compute_hpt_prompt_equal_policy_loss`):
  batch is HPT iff it carries all four `hpt_*` tensors (`_has_hpt_loss_fields`).
  SFT rows pin ratio=1 (`old = log_prob.detach()`) -> advantage-weighted NLL;
  RL rows use clipped vanilla PPO. Both length-normalized, weighted so each prompt
  contributes equally over shared denominator `b_eff`.
- **Off-policy correction** (`rollout_corr_helper.py`): IS/RS applied to RL rows
  only; `_compute_hpt_rollout_correction_and_add_to_batch` masks SFT tokens out
  (SFT rows are not drawn from the rollout policy).

### `async_hpt.*` config knobs (`hpt_config.py`)
| Key | Meaning |
| --- | --- |
| `enabled` | master switch; off = untouched base async RL |
| `gamma` | success-prob threshold at/below which -> SFT |
| `beta` | SFT reward magnitude at last supervised token |
| `alpha` | RL row weight scale (`alpha/total_count`) |
| `k_max` | RL staleness drop bound (SFT rows exempt) |
| `success_threshold` / `success_score_key` | what counts as a successful rollout |
| `tau_dataset_path` / `tau_messages_key` | tau lookup parquet + column |
| `fail_on_missing_tau` | SFT-routed prompt without tau -> raise vs fall back to RL |
| `trajectory_scheduler.enabled` | per-attempt scheduling (needs async, `rollout.n>1`) |

Enable forces: `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False`,
`actor.policy_loss.loss_mode=vanilla`, `rollout.calculate_log_probs=True`.

## Async Budget (from `Readme_RL.md`, enforced in the rollouter)

```
required_samples     = ppo_mini_batch_size * require_batches           # trainer collection target
max_required_samples = required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step
```
Rollouter pauses (`_should_pause_generation`) when the queue is full
(`max_completed_prompt_groups`), staleness hits `max_required_samples`, or the
scheduler's completed-attempt store overflows. Weight sync fires in
`_fit_update_weights` only when `local_trigger_step == 1` (a fresh param version),
via `CheckpointEngineManager.update_weights`, then RPCs rollouter `reset_staleness`.

## "Where Did It Break?" Index

| Symptom | Look first |
| --- | --- |
| Run stalls: `active_tasks=0, mq_queue=0, pending>0` | Circular wait, not generation. Async budget too small — `_should_pause_generation` vs trainer `_get_samples_from_queue`. Fix via `Readme_RL.md` budget sizing. |
| `ValueError: could not form a trainable batch` | `_get_samples_from_queue` row-aware loop exhausted `max_completed_prompt_groups` window; check `_hpt_required_training_multiple` and assembler row counts. |
| `ValueError` mentioning `min_global_steps` | `_add_hpt_async_sample_meta` requires param-version metadata after assembly. |
| `libcudart.so.13: cannot open shared object` | SGLang subprocess without `conda activate RL`. See `Readme_RL.md` step 2, `scripts/install_vllm_sglang_mcore.sh`, `verl/utils/cuda_env.py`. |
| Wrong SFT/RL route; `hpt/missing_tau_count>0` unexpectedly | `hpt_gate.route` (`gamma`/`success_threshold`/`success_score_key`); tau `HptTauStore`; data prep `tau_messages` column; `fail_on_missing_tau`. |
| SFT rows learning with wrong probs / ratio != 1 | `_compute_hpt_prompt_equal_policy_loss` (`effective_old_log_prob`); `apply_hpt_rollout_logprob_anchor`. |
| Learner-row count unexpected | SFT collapses group->1 row, RL expands->`rollout.n`; `HptBatchAssembler.assemble_rollout_samples` / `normalize_mixed_schema`. |
| Rollout anchor missing / old_logprobs wrong | `should_use_hpt_rollout_logprob_anchor`; ensure `rollout.calculate_log_probs=True`. |
| Correction touching SFT rows | `_compute_hpt_rollout_correction_and_add_to_batch` must mask SFT out. |
| No parameter sync happening | `_fit_update_weights` gated on `local_trigger_step==1`; `_fit_update_local_step`; `CheckpointEngineManager.update_weights`. |
| Trajectory attempts out of order / wrong group | `HptPromptGroupAccumulator.pop_ready`; `_submit_hpt_trajectory_attempts`. |
| Batch shape/mask/logprob misalignment | `DataProto.check_consistency` (`verl/protocol.py`); the four `hpt_*` fields must be row-aligned with `response_mask`. |

Useful log/metric keys: `hpt/num_sft_routed`, `hpt/num_rl_routed`,
`hpt/missing_tau_count`, `hpt/old_logprob_from_rollout`, `hpt/b_eff`,
`hpt/sft_loss_component`, `hpt/rl_loss_component`. For live runs, inspect Ray
worker logs directly (see `Readme_RL.md` "Suggested Main-Run Log Checks").

## Tests

CPU-only, no Ray cluster / GPU / model (they instantiate undecorated actor
classes and (de)serialize queue payloads). One `test_GPU_*` self-skips without CUDA.

```
conda activate RL && cd <repo> && pytest tests/special_RL/ -v
```

| File | Protects |
| --- | --- |
| `test_hpt_trainer_queue_contract.py` | learner-row divisibility, fail-closed on unformable batch, row-aware window = `max_completed_prompt_groups`, param-version metadata preserved, rollout-logprob anchor, mixed SFT/RL loss reaches backprop with aligned `hpt_*` fields |
| `test_hpt_trajectory_scheduler_contract.py` | all `n` attempts share one group uid, distinct ordered `hpt_rollout_index`, `prompt_uid` propagated |
| `test_openr1_hpt_smoke_contract.py` | data-prep schema (`prompt_uid`+`tau_messages`), launcher text/Hydra-override contracts |

## Run

Data prep (`examples/data_preprocess/openr1_hpt.py`) builds `train.parquet` (carries
`prompt_uid` + `tau_messages`), `test.parquet`, and eval sets. Smoke = 12 train
rows / Qwen2.5-0.5B / `gamma=1.0`; main = 45.7k rows / `models/Qwen2.5-Math-7B` /
`gamma=0.0` / AIME24+AMC23+MATH-500 eval.

```
# smoke (8 GPU: 4 rollout + 4 train); auto-generates data if missing
bash tests/special_e2e/run_fully_async_policy_sglang_smoke.sh   # after: conda activate RL
# main (standalone launcher; resolves the repo root from its own location)
bash main_scripts/run_fully_async_policy_openr1_hpt_main.sh
```

## Reference Baseline & Sync↔Async Parity

`Unify-Post-Training/` is a **separate, read-only reference repository** (a
sibling of this repo, outside `verl/`). It is **not part of this codebase and we
do not edit it.** Its synchronous HPT launcher
`Unify-Post-Training/exp_scripts/train.sh` is the baseline our async-HPT main run
targets for a 1-to-1 comparison. `main_scripts/run_fully_async_policy_openr1_hpt_main.sh`
mirrors it on the settings that must match, and makes our own choices for the
async-only knobs:

| Matched to baseline (`train.sh`) | Our async-only decision |
| --- | --- |
| `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False` (Dr.GRPO) | `staleness_threshold`, `trigger_parameter_sync_step` |
| `ppo_mini_batch_size=64`, `rollout.n=8` | `require_batches`, `partial_rollout` |
| `lr=5e-6`, `clip_grad=80.0`, `entropy_coeff=0.001`, `use_kl_loss=False` | `max_inflight/completed_prompt_groups` |
| `max_response_length=8192`, `rope_theta=40000`, `max_position_embeddings=16384` | `data.train_batch_size=0` (forced by fully-async) |

**Why `require_batches=16`, not a naive `2`:** the baseline's learner scale is
`train_batch_size(128) × rollout.n(8) = 1024` sequences/step. Fully-async forces
`data.train_batch_size=0`, so that same 1024 scale is expressed as the trainer's
initial queue request `ppo_mini_batch_size(64) × require_batches(16) = 1024` (see
`Readme_RL.md`). A value of `2` maps only the 128 *prompt* count, not the 1024
*learner* scale, and would force HPT row-aware extra collection before the first
update. This is design intent, not a code invariant — the launcher test
deliberately does NOT assert it (it only checks the launcher composes to a valid
async-HPT config).

## Maintaining This File

Project documentation lives in `docs/` (see
`docs/contributing/editing-agent-instructions.md`). Keep this a *map*, not a
tutorial: anchor on symbols, avoid line numbers and pasted config values, and
prune stale drift. Keep under 300 lines.
