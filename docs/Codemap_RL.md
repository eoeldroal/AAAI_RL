# RL Code Map

_Last updated: 2026-07-07_

Navigation and debugging map for this async-HPT `verl` fork. Use it to find
where code lives, how a rollout sample becomes a learner update, and — when a
run breaks — which file/symbol to open first.

Companion docs (single source of truth, do not duplicate them here):

- `../AGENTS.md` — durable rules and contracts (what you may/may not change).
- `Overview_RL.md` — what this fork is and why (identity, guarantees).
- `Readme_RL.md` — environment setup, launching, and log triage.
- `AsyncBudget_RL.md` — queue/staleness/HPT-routing budget sizing.
- `Debug_RL.md` — lint/format, live profiling, and perf triangulation.
- `Ablation_RL.md` — ablation study design and analysis procedure.
- `Improvement_RL.md` — run pathology case studies and improvement decisions.
- Design records `DR-001` to `DR-005` — the decisions' rationale and theory.
- This file — where things live and where they break.

Line numbers rot; this map anchors on **file + symbol name**. `grep`/`rg` the
symbol to jump. Update it when a symbol is renamed or a subsystem moves.

## What This Fork Is

See `Overview_RL.md` for what this fork is and why. Wayfinding fact: everything
custom lives under `verl/experimental/fully_async_policy/`. Baseline before the
fork work: git commit "Document clean verl export baseline".

## First 15 Minutes

1. Read `Overview_RL.md` (what/why), `../AGENTS.md` (contracts), and
   `AsyncBudget_RL.md` (run budgets).
2. Skim this map's *Runtime Architecture* and *Data Flow* sections.
3. Run the CPU contract tests: `pytest tests/special_RL/ -v` (no GPU/Ray needed).
4. Read the HPT loss: `verl/workers/utils/losses.py::ppo_loss` HPT branch.

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
verl/workers/utils/losses.py            # HPT branch-blind policy loss wrapper
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
docs/Readme_RL.md                       # environment setup, launch, log triage
docs/AsyncBudget_RL.md                  # queue/staleness/HPT budget formulas
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
in-proc `pending_queue`, each prompt repeated `rollout.n`x as a `RolloutSample`
via `prepare_single_generation_data`) and `_processor_worker`.
`_async_monitor_loop` logs stats and clears backpressure.

```
_processor_worker
  -> _process_single_sample_streaming              (scheduler disabled)
     -> generate -> _queue_completed_rollout_sample
  -> _submit_hpt_trajectory_attempts                (scheduler enabled)
     -> _process_hpt_trajectory_attempt_streaming
        -> async_rollout_manager.generate_sequences_single
        -> HptTrajectoryAttemptResult -> HptPromptGroupAccumulator
        -> _queue_completed_rollout_sample

_queue_completed_rollout_sample
  -> HptRolloutGate.route_rollout_sample   (SFT route swaps in the tau payload)
  -> MessageQueueClient.put_sample
```

**Failure is fail-closed at the group grain**: an `infra_abort` marker on any
attempt (`_has_infra_abort_marker`) drops the *entire* prompt group before the
HPT gate and before the queue put (`_drop_hpt_scheduler_group`) — no partial
prompt-group sample is ever emitted.

### Trainer loop (`FullyAsyncTrainer`)
`fit` -> `while True: fit_step()`. `fit_step` is the PPO template: `_fit_generate`
(-> `_get_samples_from_queue`) -> reward -> log_prob -> ref -> critic -> advantage
-> `_fit_update_actor` -> `_fit_update_local_step` -> `_fit_update_weights` ->
dump. A `None` from the queue raises `TrainingStopException`.

The HPT branch of `_get_samples_from_queue`:

```
_get_samples_from_queue
  -> ray.cloudpickle.loads(...)
  -> HptBatchAssembler.assemble_rollout_samples
     -> materialize_rollout_sample     (per row: RL DataProto or SFT payload)
     -> normalize_hpt_training_batch   (per row: writes hpt_is_sft + HPT route metadata)
     -> normalize_mixed_schema         (reconciles RL/SFT tensor+non-tensor schema)
     -> DataProto.concat

_fit_compute_log_prob -> apply_hpt_rollout_logprob_anchor   (HPT path)
_fit_update_actor -> ppo_loss -> standard vanilla PPO with SFT self-detach
```

## Data Flow: rollout sample -> learner row

```
config validated        hpt_config.validate_async_hpt_config
  -> gate routes         hpt_gate.HptRolloutGate.route_rollout_sample / .route
     (SFT swaps full_batch to tau payload; RL keeps generated DataProto)
  -> [scheduler] regroup hpt_rollout_accumulator.HptPromptGroupAccumulator.pop_ready
  -> queue (cloudpickle) message_queue.put_sample / get_sample
  -> assemble            hpt_assembler.HptBatchAssembler.assemble_rollout_samples
     (writes hpt_is_sft and row-aligned HPT route metadata)
  -> anchor + filter     hpt_training.apply_hpt_rollout_logprob_anchor / filter_hpt_stale_rollout_samples
  -> loss                losses.ppo_loss -> branch-blind vanilla PPO + SFT self-detach
```

**Key asymmetry (memorize this):** an **SFT** route collapses a prompt group to
**1 learner row**; an **RL** route expands to **`rollout.n` rows**. So
*queue-sample count != learner-row count*. The trainer collects `required_samples`
queue samples, grows only if the batch is under one `_hpt_required_training_multiple`
(= `lcm(dp_size, ppo_mini_batch_size*rollout_n)`), then **trims to the largest
aligned batch and defers the residue (< one multiple) to the next step via
`_hpt_carryover_samples`** (`_plan_row_alignment_deferral`; see `Improvement_RL.md`
§5.8.6 and `AsyncBudget_RL.md` Principle 6). The batch stays bounded near
`required_samples`. Log line:

```
[FullyAsyncTrainer] Loop collection completed: <retained>/<req> samples, learner_rows=<r>,
  required_multiple=<m>, carryover_in=<n>, carried_forward=<n>, discarded=<n>, deferred_rows=<n>
```

## HPT Objective (the research core)

- **Route** (`hpt_gate.py`): `route_to_sft = success_prob <= gamma and not missing_tau`,
  where `success_prob` = fraction of a group's rollouts scoring `> success_threshold`.
  Failing prompts with an expert answer -> supervised; the rest stay on-policy RL.
- **tau** = expert trajectory keyed by `prompt_uid`, column `tau_messages`
  (`hpt_payload.HptTauStore` / `HptSftPayload` / `HptTauToAgentLoopOutputAdapter`).
- **Deferred materialization**: RL and SFT routes converge through the same
  `materialize_rollout_sample` entry point only when the trainer consumes them,
  not at generation time — RL wraps the already-generated rollout into a
  `DataProto`; SFT tokenizes the tau transcript into the same row shape. This
  symmetry is why `HptBatchAssembler` exists as a single entry point.
- **Rollout-logprob anchor** (`hpt_training.py`): RL rows use the rollout engine's
  own `rollout_log_probs` as `old_log_probs` (needs `rollout.calculate_log_probs=True`).
- **branch-blind policy loss** (`losses.py::ppo_loss` HPT branch):
  batch is HPT iff it carries `hpt_is_sft`; obsolete B_eff fields are rejected.
  SFT rows pin ratio=1 (`old = log_prob.detach()`) -> advantage-weighted NLL;
  RL rows use the standard clipped vanilla PPO path unchanged. HPT does not
  replace the base RL advantage estimator — RL rows still use the existing GRPO
  advantage path; HPT only adds the route decision, SFT self-detach, and
  auxiliary masking. Truncated-RL rows flagged `hpt_is_truncated_rl`
  (advantage zeroed) are likewise excluded from the entropy and §11
  diagnostic masks.
- **Off-policy correction** (`rollout_corr_helper.py`): IS/RS applied to RL rows
  only; `_compute_hpt_rollout_correction_and_add_to_batch` masks SFT tokens out
  (SFT rows are not drawn from the rollout policy).

### `async_hpt.*` config knobs (`hpt_config.py`)
| Key | Meaning |
| --- | --- |
| `enabled` | master switch; off = untouched base async RL |
| `gamma` | success-prob threshold at/below which -> SFT |
| `beta` | SFT terminal pseudo reward magnitude |
| `alpha` | deprecated compatibility field; must remain `1.0` |
| `loss_aggregation` | `branch_blind`; old `prompt_equal` is rejected |
| `sft_beta_mode` | `constant` or `length_inverse` terminal pseudo reward |
| `sft_entropy_enabled` / `sft_kl_enabled` | default false; include SFT rows in auxiliary masks only when explicitly true |
| `k_max` | RL staleness drop bound (SFT rows exempt) |
| `success_threshold` / `success_score_key` | what counts as a successful rollout |
| `tau_dataset_path` / `tau_messages_key` | tau lookup parquet + column |
| `fail_on_missing_tau` | SFT-routed prompt without tau -> raise vs fall back to RL |
| `trajectory_scheduler.enabled` | per-attempt scheduling (needs async, `rollout.n>1`) |

Truncation handling (`reward.reward_kwargs.*`, default off; see
`Improvement_RL.md` §5.6): `zero_reward_if_truncated` scores budget-exhausted
rollouts 0 (fixing the routing success count and the GRPO baseline);
`zero_truncated_rl_advantage` zeros those RL rows' advantage after advantage
computation. Zeroed rows are flagged `hpt_is_truncated_rl` and dropped from the
entropy loss and §11 diagnostics, mirroring the SFT exclusion.

Enable forces: `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False`,
`actor.policy_loss.loss_mode=vanilla`, `rollout.calculate_log_probs=True`.

## Async Budget (from `AsyncBudget_RL.md`, enforced in the rollouter)

```
required_samples     = ppo_mini_batch_size * require_batches           # trainer collection target
max_required_samples = required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step
```
Rollouter pauses (`_should_pause_generation`) when the queue is full
(`max_completed_prompt_groups`), the live outstanding buffer — open groups +
queued samples (`_outstanding_sample_count`) — hits `max_required_samples`, or the
scheduler's completed-attempt store overflows. The monitor loop re-checks this
live and resumes generation as the trainer drains the queue, so the pause is not
locked to the sync cadence. Weight sync fires in
`_fit_update_weights` only when `local_trigger_step == 1` (a fresh param version),
via `CheckpointEngineManager.update_weights`, then RPCs rollouter `reset_staleness`.

## "Where Did It Break?" Index

| Symptom | Look first |
| --- | --- |
| Run stalls: `active_tasks=0, mq_queue=0, pending>0` | Circular wait, not generation. Async budget too small — `_should_pause_generation` vs trainer `_get_samples_from_queue`. Fix via `AsyncBudget_RL.md` sizing. |
| `ValueError: could not trim to an aligned batch` / `could not reach one trainable multiple` | Near-impossible with trim+carryover (only if a batch has almost no SFT groups to move the residue mod `rollout.n`, or is smaller than one multiple with an exhausted queue); check `_plan_row_alignment_deferral`, `_hpt_required_training_multiple`, and assembler row counts. The old grow-until-aligned crash (`could not form a trainable batch` at large `learner_rows`) is fixed — `Improvement_RL.md` §5.8.6. |
| `ValueError` mentioning `min_global_steps` | `_add_hpt_async_sample_meta` requires param-version metadata after assembly. |
| `libcudart.so.13: cannot open shared object` | SGLang subprocess without `conda activate RL`. See `Readme_RL.md` step 2, `scripts/install_vllm_sglang_mcore.sh`, `verl/utils/cuda_env.py`. |
| Prompt groups missing / lower throughput than expected | Whole-group fail-closed drop on any attempt's `infra_abort` marker — `_has_infra_abort_marker` / `_drop_hpt_scheduler_group`. Check rollout-side exceptions before assuming a queue/staleness issue. |
| Wrong SFT/RL route; `hpt/missing_tau_count>0` unexpectedly | `hpt_gate.route` (`gamma`/`success_threshold`/`success_score_key`); tau `HptTauStore`; data prep `tau_messages` column; `fail_on_missing_tau`. |
| Routing success stuck at 0 (`hpt/onpolicy_success_rate`, `critic/score/mean`) → batch goes all-SFT | `hpt_gate.extract_score_values` must read the terminal reward via `sum(-1)`, not `rm_scores[-1]` (post-reward padding). See `Improvement_RL.md` §5.7. |
| SFT rows learning with wrong probs / ratio != 1 | `losses.py::ppo_loss` HPT self-detach branch; `apply_hpt_rollout_logprob_anchor`. |
| Learner-row count unexpected | SFT collapses group->1 row, RL expands->`rollout.n`; `HptBatchAssembler.assemble_rollout_samples` / `normalize_mixed_schema`. |
| Rollout anchor missing / old_logprobs wrong | `should_use_hpt_rollout_logprob_anchor`; ensure `rollout.calculate_log_probs=True`. |
| Correction touching SFT rows | `_compute_hpt_rollout_correction_and_add_to_batch` must mask SFT out. |
| No parameter sync happening | `_fit_update_weights` gated on `local_trigger_step==1`; `_fit_update_local_step`; `CheckpointEngineManager.update_weights`. |
| Trajectory attempts out of order / wrong group | `HptPromptGroupAccumulator.pop_ready`; `_submit_hpt_trajectory_attempts`. |
| Batch shape/mask/logprob misalignment | `DataProto.check_consistency` (`verl/protocol.py`); `hpt_is_sft`, `rollout_log_probs`, masks, rewards, and route metadata must stay row-aligned. |

Useful log/metric keys: `hpt/num_sft_routed`, `hpt/num_rl_routed`,
`hpt/missing_tau_count`, `hpt/old_logprob_from_rollout`,
`hpt/sft_response_token_count`, `hpt/sft_nll`. For live runs, inspect Ray worker
logs directly (see `Readme_RL.md` "Suggested Main-Run Log Checks").

## Tests

CPU-only, no Ray cluster / GPU / model (they instantiate undecorated actor
classes and (de)serialize queue payloads). One `test_GPU_*` self-skips without CUDA.

```
conda activate RL && cd <repo> && pytest tests/special_RL/ -v
```

| File | Protects |
| --- | --- |
| `test_hpt_trainer_queue_contract.py` | learner-row divisibility, fail-closed on unformable batch, row-aware window = `max_completed_prompt_groups`, param-version metadata preserved, rollout-logprob anchor, branch-blind HPT loss, SFT self-detach, auxiliary masks, obsolete B_eff fields rejected |
| `test_hpt_trajectory_scheduler_contract.py` | all `n` attempts share one group uid, distinct ordered `hpt_rollout_index`, `prompt_uid` propagated |
| `test_openr1_hpt_smoke_contract.py` | data-prep schema (`prompt_uid`+`tau_messages`), main-launcher Hydra-validity contract |

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
sibling of this repo, outside `verl/`, not edited here). Its synchronous
launcher `train.sh` is the baseline `main_scripts/run_fully_async_policy_openr1_hpt_main.sh`
targets for a 1-to-1 comparison — matching where it must, and choosing its own
values for the async-only knobs:

| Matched to baseline (`train.sh`) | Our async-only decision |
| --- | --- |
| `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False` (Dr.GRPO) | `staleness_threshold`, `trigger_parameter_sync_step` |
| `rollout.n=8`; same 128-prompt fit_step scale | `require_batches`, finer `ppo_mini_batch_size`, `partial_rollout` |
| `lr=5e-6`, `clip_grad=80.0`, `entropy_coeff=0.001`, `use_kl_loss=False` | `max_inflight/completed_prompt_groups` |
| `max_response_length=8192`, `rope_theta=40000`, `max_position_embeddings=16384` | `data.train_batch_size=0` (forced by fully-async) |

**`require_batches` is a queue-sample count, not a row count:** one queue
sample already holds `rollout.n` rows (the rollouter repeats one prompt
`rollout.n` times before generation — see the rollouter call graph above).
Matching the baseline's `train_batch_size(128) × rollout.n(8) = 1024`
sequences/step means `ppo_mini_batch_size * require_batches = 128`. The strict
Unify mini-batch grain is `64 * 2`; the current main launcher intentionally
uses the finer async/HPT grain `32 * 4`. Both preserve the large fit_step batch
scale. An earlier version used `require_batches=16` on the mistaken assumption
that `64 × 16 = 1024` already matched the baseline's 1024-sequence scale — it
compared a queue-sample count against a row count and missed the `× rollout.n`,
so it actually collected `1024 × 8 = 8192` rows per fit_step, 8x the baseline's
per-step volume. See `AsyncBudget_RL.md` for the full derivation and the
corrected budget.

`max_completed_prompt_groups` is not part of this parity equation. It is an
async-only completed-backlog cap; the main launcher keeps it large enough to
avoid dropping rollout samples while preserving the 128-prompt fit_step scale.
This is design intent, not a code invariant — the launcher test deliberately
does NOT assert a specific value (it only checks the launcher composes to a
valid async-HPT config).

## Maintaining This File

Project documentation lives in `docs/` (see
`docs/contributing/editing-agent-instructions.md`). Keep this a *map*, not a
tutorial: anchor on symbols, avoid line numbers and pasted config values, and
prune stale drift. Keep under 300 lines.
