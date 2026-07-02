# Async RL + HPT Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a clean async RL + HPT framework on top of the latest upstream `verl`, while preserving the current HPT learning contract and carrying over only the structural async improvements that generalize beyond CUA.

**Architecture:** Treat the export as a selective transplant onto upstream fully async RL. Export the HPT semantic layer plus the structural async layer (`E1~E3`), and leave behind current WebOS/CUA adapters and current deployment-specific tuning.

**Tech Stack:** Python 3.12, Ray, Hydra/OmegaConf, PyTorch, TensorDict/DataProto, verl fully async policy, pytest.

---

## Current Document Status

This document is the source of truth for the **export problem framing and port plan**.

Use this document to answer:

```text
What exactly should be exported?
What is the reusable async HPT core?
What stays behind as WebOS/CUA or environment-local baggage?
Which files must be added, patched, or ignored?
```

Do not use the following documents as the primary export framing when they
conflict with this file:

- [Plan.md](/NHNHOME/OSWorld/WebOSWorld/document/Plan.md)
- [Util_improvement.md](/NHNHOME/OSWorld/WebOSWorld/document/Util_improvement.md)
- [HPT_env.md](/NHNHOME/OSWorld/WebOSWorld/document/HPT_env.md)
- [async_hpt_gpu_util_root_cause.md](/NHNHOME/OSWorld/WebOSWorld/document/async_hpt_gpu_util_root_cause.md)

Those documents are still useful, but for narrower purposes:

```text
Plan.md:
  current-repo HPT contract and implementation notes

Util_improvement.md:
  why E1~E3 were introduced, plus WebOS-specific efficiency history

HPT_env.md:
  current environment/backend facts for Qwen3.5/SGLang/B200

async_hpt_gpu_util_root_cause.md:
  historical low-util debugging notes for the WebOS stack
```

## Implementation Progress

This section tracks the clean upstream port. Mark only work that is present in
this repository, not work that existed in the source OSWorld fork.

- [x] Establish the clean upstream-oriented baseline, repository principles,
      and export plan.
- [x] Add the initial HPT semantic contracts:
      `hpt_config.py`, `hpt_payload.py`, and `hpt_gate.py`.
- [x] Expose default-off HPT config in both fully-async recipe surfaces.
- [x] Wire fail-closed HPT config validation into the fully-async bootstrap.
- [x] Extend `RolloutSample` with optional HPT route metadata.
- [x] Add nullable in-flight/completed prompt-group budget knobs while
      preserving the existing `max_required_samples` behavior when unset.
- [x] Implement the initial DataProto-focused `hpt_assembler.py`.
- [x] Wire `hpt_assembler.py` into trainer queue consumption behind
      `async_hpt.enabled`.
- [ ] Implement `hpt_training.py` and HPT-aware old-logprob handling.
- [ ] Implement the prompt-equal HPT loss path.
- [ ] Exclude SFT rows from rollout correction / rejection / IS weighting.
- [ ] Export E1 trajectory scheduling and prompt-group accumulation.
- [ ] Export E3 partial rollout recovery only after HPT core is coherent.
- [ ] Add contract tests after the target environment is ready.

## 1. Export Objective

This export is **not**:

```text
a copy of WebOSWorld
a copy of the current CUA stack
a copy of SurfGym/WebOSGym integration
a copy of current Qwen3.5/SGLang/B200 operating assumptions
```

This export **is**:

```text
latest upstream verl fully async RL
  + HPT semantic layer
  + structural async improvements E1~E3
  + the minimal tests needed to keep that contract honest
```

The target is a cleaner verl environment where async RL + HPT can later be
applied to math, coding, or other non-CUA workloads.

## 2. Upstream Baseline

Checked on 2026-07-02.

```text
upstream repo:
  https://github.com/verl-project/verl

upstream HEAD:
  91666d9964282b890c75dd0b2d330edaee201c2f
  91666d9 [rollout] fix: support SGLang FP8 ignored layers for Qwen3.x GatedDeltaNet in rollout (#6906)

local repo:
  18043a099e805ec72027098524994130713dbded
```

Most important baseline facts:

```text
upstream already has fully async RL
upstream does not have async_hpt / Hpt* modules
upstream does not have local-only:
  verl/experimental/fully_async_policy/agent_loop/agent_loop.py
```

Therefore:

```text
this is not feature un-hiding
this is a real semantic port onto a moving fully async base
```

## 3. Layer Model

The current repository mixes several different concerns. Export decisions must
be made by layer, not by folder name only.

### Layer A. Base async RL engine

```text
rollouter
trainer
message queue
parameter sync
async sampling / async update
```

Representative upstream/local files:

- `verl/experimental/fully_async_policy/fully_async_main.py`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py`
- `verl/experimental/fully_async_policy/fully_async_trainer.py`
- `verl/experimental/fully_async_policy/message_queue.py`
- `verl/experimental/separation/ray_trainer.py`

Use the upstream version of Layer A as the base.

### Layer B. HPT semantic layer

```text
RL vs SFT route decision
tau payload lookup and validation
mixed RL/SFT batch assembly
prompt-equal loss
rollout old-logprob anchor
```

This is the true export core.

### Layer C. Async throughput structural layer

```text
trajectory scheduler
prompt-group accumulator
inflight/completed split
partial rollout recovery
```

This is also export scope. It is structural async logic, not CUA-local logic.

### Layer D. Workload adapter layer

```text
WebOSGym / SurfGym / browser / external-app integration
tool-loop specifics
task metadata and reward plumbing
current multimodal task conventions
```

This is not phase-1 export scope.

### Layer E. Deployment / environment layer

```text
SGLang serving
Qwen3.5 specifics
B200 / FA2 / FA4 / SP decisions
launcher knobs
current ops workarounds
```

This is operational, not semantic. It must not define the export core.

## 4. Target Model

There are two plausible export targets. They must not be conflated.

### T1. Clean async HPT core

```text
general async RL + HPT
math / coding / non-CUA workloads
no browser/session/task-server dependency
```

This is the current target.

### T2. Agentic / multimodal async HPT

```text
tool use
multimodal environment interaction
browser/app adapter layer
serving/backend-sensitive workload
```

This is a later target. Do not silently drift into it during phase 1.

## 5. Phase Split

### Phase 1

Export only:

```text
Layer B: HPT semantic layer
Layer C: async throughput structural layer
```

onto upstream Layer A.

### Phase 2

Only after T1 is stable, optionally reintroduce:

```text
Layer D: workload adapters
Layer E: deployment/environment optimizations
```

This phase split is the main guardrail against exporting the current repo's
local complexity instead of its reusable async HPT framework.

## 6. Upstream Comparison Summary

### 6.1 Upstream already moved significantly in Layer A

The latest upstream differs materially in:

```text
verl/experimental/fully_async_policy/fully_async_main.py
verl/experimental/fully_async_policy/fully_async_rollouter.py
verl/experimental/fully_async_policy/fully_async_trainer.py
verl/experimental/fully_async_policy/detach_utils.py
verl/experimental/agent_loop/agent_loop.py
verl/experimental/separation/ray_trainer.py
verl/workers/utils/losses.py
verl/trainer/ppo/rollout_corr_helper.py
```

Recent upstream history shows continued movement in:

```text
fully_async_main.py:
  hybrid validation plumbing, OPD support

fully_async_rollouter.py:
  trainer abstraction, profiling, reward support, drain behavior, LLM-server refactors

fully_async_trainer.py:
  metrics, profiling, hybrid validation reuse, engine/worker refactors

detach_utils.py:
  long-lived multimodal / sglang / partial-resume related evolution

agent_loop.py:
  heavily evolving generic rollout/tool layer

separation/ray_trainer.py:
  shared trainer logic continues to move

rollout_corr_helper.py:
  generic rollout-correction logic continues to move
```

Practical implication:

```text
Do not export by replaying old local whole-file versions.
Patch onto upstream behavior deliberately.
```

### 6.2 Upstream has no HPT semantic layer

Upstream `fully_async_policy` currently contains:

```text
fully_async_main.py
fully_async_rollouter.py
fully_async_trainer.py
detach_utils.py
message_queue.py
config/*
```

Upstream does not contain:

```text
hpt_config.py
hpt_gate.py
hpt_payload.py
hpt_assembler.py
hpt_rollout_accumulator.py
hpt_training.py
fully_async_policy/agent_loop/agent_loop.py   # local-only subtree
```

Practical implication:

```text
HPT is not a hidden upstream feature.
It must be introduced as a new semantic layer.
```

## 7. Export Scope

### 7.1 Must export

#### HPT core

```text
hpt_config
hpt_gate
hpt_payload
hpt_assembler
hpt_training
HPT-aware policy loss path
HPT monitoring metrics
```

#### Structural async improvements

```text
E1. trajectory scheduler
E2. inflight/completed split
E3. partial rollout recovery
```

#### Config surface

```text
fully_async_ppo_trainer.yaml
fully_async_ppo_megatron_trainer.yaml   # if the target export keeps the megatron recipe surface
```

Without these config patch points, the exported code has no stable way to
expose:

```text
async_hpt.*
max_inflight_prompt_groups
max_completed_prompt_groups
```

### 7.2 Export only if the target still needs it

```text
deferred_logprob_scoring
rollout tracing / request-phase tracing
GPU util sampling helper
```

These are bring-up tools or environment-dependent optimizations, not phase-1
semantic requirements.

### 7.3 Do not export in phase 1

```text
WebOSGym / SurfGym protocol code
WebOS-specific retries and timeout classification
spreadsheet task plumbing
Qwen3.5/B200-specific attention/backend tuning
SGLang serving knobs:
  mixed chunk
  prefill size
  mamba scheduler strategy
  decode step tuning
deferred scoring as the default path
```

## 8. Contracts That Must Survive Export

### 8.1 Prompt-group learning contract

`rollout.n` remains the learner-visible prompt-group grain.

Even with trajectory scheduling:

```text
one prompt
  -> repeated rollout attempts
  -> one HPT route decision
  -> one prompt-level learner contract
```

### 8.2 Queue payload contract

The queue stores pickled `RolloutSample`, not just trainer-ready `DataProto`.

That contract currently carries:

```text
RolloutSample.full_batch
  = generated RL payload OR routed tau* payload

RolloutSample.hpt_route
  = route metadata required later by trainer assembly
```

If the target codebase uses a different queue payload abstraction, this must be
replaced consciously everywhere, not accidentally.

### 8.3 Deferred materialization contract

The exported system must preserve the current convergence point:

```text
RL route:
  DeferredAgentLoopOutputs -> materialize in trainer

SFT route:
  HptSftPayload -> AgentLoopOutput -> materialize in trainer
```

This symmetry is a core reason `HptBatchAssembler` exists.

### 8.4 DataProto contract

HPT rows still have to satisfy the existing actor/trainer batch contract:

```text
- batch tensors share dim0
- non_tensor_batch arrays align with batch size
- RL and SFT rows concat under a common effective schema
- loss-visible HPT fields live in batch tensors
- padding rows are neutral for HPT loss
```

### 8.5 Old-logprob contract

Exported HPT v1 still assumes:

```text
async_hpt.rl_old_logprob_source = rollout
rollout.calculate_log_probs = True
old_log_probs := rollout_log_probs
```

This is an HPT-specific path, not just generic rollout-correction bypass.

### 8.6 Rollout-correction contract

`rollout_corr_helper.py` already needs an HPT-aware branch:

```text
exclude SFT rows from rollout correction / rejection / IS weighting
```

So HPT is not isolated to experimental files only.

### 8.7 Failure and drop contract

The current structural HPT path has a strict prompt-group failure rule:

```text
if any trajectory attempt in a prompt group fails
  -> drop the entire prompt group before HPT gate and before queue put
```

Concretely:

```text
read timeout / infra failure / infra abort in one attempt
  -> FailedHptPromptGroup
  -> no partial prompt-group learner sample is emitted
```

This is part of the exported semantics, not incidental logging behavior.

### 8.8 Completed-budget contract

`E2` is not just a new knob pair. The local code implements a specific storage
interpretation:

```text
max_inflight_prompt_groups:
  cap on prompt groups currently being executed

max_completed_prompt_groups:
  cap on learner-visible completed prompt groups waiting for trainer

accumulator storage:
  partial completed attempts count against completed storage budget
  through prompt-group-to-attempt conversion
```

This is why the accumulator exposes prompt-group-aware storage accounting rather
than only queue length.

### 8.9 HPT does not replace the base RL advantage family

The exported HPT core keeps the current split:

```text
RL rows:
  existing GRPO-style advantage path remains intact

SFT rows:
  supervised branch with beta-weighted / prompt-equal HPT semantics
```

So HPT here is a routing-and-aggregation layer over async RL, not a new base
advantage estimator.

### 8.10 Optional RL-only staleness extension

The local code also contains an optional branch-aware RL staleness filter:

```text
async_hpt.k_max
  -> stale RL rows can be dropped by generation-step lag
  -> SFT rows remain exempt
```

This is not required to define phase-1 HPT semantics, but it is a real local
extension in `hpt_training.py` and should be described as an optional policy,
not forgotten as a hidden implementation detail.

## 9. Dataset Contract To Preserve

The export should preserve the prompt-level HPT dataset shape even if the new
workload is math or coding rather than CUA.

Minimum row concept:

```text
RL row
  + prompt_uid
  + optional tau_messages
```

Key semantics:

```text
prompt_uid:
  prompt-level source identity, distinct from runtime uid

tau_messages:
  optional supervised trajectory transcript
  missing tau is allowed unless fail_on_missing_tau is explicitly enabled
```

This matters because the HPT route decision is prompt-level and the SFT branch
needs a stable prompt-level supervision lookup key.

## 10. Source Code Map

### 10.1 New modules to introduce in the target repo

- `verl/experimental/fully_async_policy/hpt_config.py`
- `verl/experimental/fully_async_policy/hpt_gate.py`
- `verl/experimental/fully_async_policy/hpt_payload.py`
- `verl/experimental/fully_async_policy/hpt_assembler.py`
- `verl/experimental/fully_async_policy/hpt_rollout_accumulator.py`
- `verl/experimental/fully_async_policy/hpt_training.py`

### 10.2 Existing files that require patching

- `verl/experimental/fully_async_policy/fully_async_main.py`
- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
- `verl/experimental/fully_async_policy/config/fully_async_ppo_megatron_trainer.yaml`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py`
- `verl/experimental/fully_async_policy/fully_async_trainer.py`
- `verl/experimental/fully_async_policy/detach_utils.py`
- `verl/experimental/separation/ray_trainer.py`
- `verl/workers/utils/losses.py`
- `verl/trainer/ppo/rollout_corr_helper.py`

### 10.3 Conditional patch point

- `verl/experimental/agent_loop/agent_loop.py`

This file is not a phase-1 HPT destination by default. Touch it only if the
target upstream path still needs our `E3` partial rollout recovery semantics
and that behavior cannot be expressed elsewhere.

### 10.4 File-by-File Export Matrix

| Local file | Upstream file exists? | Export action | Why |
|---|---:|---|---|
| `verl/experimental/fully_async_policy/hpt_config.py` | No | `new file` | HPT config contract does not exist upstream. |
| `verl/experimental/fully_async_policy/hpt_gate.py` | No | `new file` | Prompt-level RL/SFT routing is new HPT semantics. |
| `verl/experimental/fully_async_policy/hpt_payload.py` | No | `new file` | `prompt_uid`/`tau_messages` payload contract is new. |
| `verl/experimental/fully_async_policy/hpt_assembler.py` | No | `new file` | Trainer-side RL/SFT materialization and mixed-batch normalization are new. |
| `verl/experimental/fully_async_policy/hpt_rollout_accumulator.py` | No | `new file` | Prompt-group accumulator for E1/E2 does not exist upstream. |
| `verl/experimental/fully_async_policy/hpt_training.py` | No | `new file` | HPT rollout-anchor, monitoring, and optional RL-only staleness policy are new. |
| `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml` | Yes | `patch existing` | Must expose `async_hpt.*`, `max_inflight_prompt_groups`, `max_completed_prompt_groups`. |
| `verl/experimental/fully_async_policy/config/fully_async_ppo_megatron_trainer.yaml` | Yes | `patch existing` | Same config surface if megatron recipe remains supported. |
| `verl/experimental/fully_async_policy/fully_async_main.py` | Yes | `patch existing` | Needs HPT config validation and HPT-aware bootstrap wiring. |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | Yes | `patch existing` | Needs gate, trajectory scheduler, accumulator, and completed-budget semantics. |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | Yes | `patch existing` | Needs HPT-aware queue-consume assembly and HPT monitoring collection. |
| `verl/experimental/fully_async_policy/detach_utils.py` | Yes | `patch existing` | Carries `RolloutSample` queue contract, deferred materialization, schema checks, padding neutrality. |
| `verl/experimental/separation/ray_trainer.py` | Yes | `patch existing` | Needs HPT rollout-anchor branch in trainer old-logprob path and HPT-aware metrics path. |
| `verl/workers/utils/losses.py` | Yes | `patch existing` | Needs prompt-equal HPT policy loss and loss-visible HPT fields. |
| `verl/trainer/ppo/rollout_corr_helper.py` | Yes | `patch existing` | Needs HPT-specific exclusion of SFT rows from correction/rejection/IS weighting. |
| `verl/experimental/agent_loop/agent_loop.py` | Yes | `conditional patch` | Touch only if upstream still needs our `E3` partial rollout recovery semantics. |
| `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | No | `do not carry as-is` | Local-only subtree; export the required semantics, not this path wholesale. |

Practical reading:

```text
new file:
  HPT semantic layer and accumulator

patch existing:
  shared async framework and PPO/trainer boundaries

conditional patch:
  generic upstream agent_loop only if E3 cannot be realized elsewhere

do not carry as-is:
  local-only fully_async_policy/agent_loop subtree
```

## 11. Runtime Call Graph To Preserve

### 11.1 Rollouter side

```text
FullyAsyncTaskRunner._initialize_components
  -> FullyAsyncRollouter.init_workers
  -> FullyAsyncRollouter.set_max_required_samples
  -> FullyAsyncRollouter.fit
     -> _feed_samples
        -> prepare_single_generation_data
        -> RolloutSample(full_batch=repeat(prompt, rollout.n))
     -> _processor_worker
        -> _submit_hpt_trajectory_attempts        (if E1 enabled)
           -> _process_hpt_trajectory_attempt_streaming
              -> async_rollout_manager.generate_sequences_single
              -> HptTrajectoryAttemptResult
              -> HptPromptGroupAccumulator
              -> _queue_completed_rollout_sample
                 -> HptRolloutGate.route_rollout_sample
                 -> MessageQueueClient.put_sample
```

### 11.2 Trainer side

```text
FullyAsyncTrainer.fit
  -> fit_step
     -> _fit_generate
        -> _get_samples_from_queue
           -> ray.cloudpickle.loads(...)
           -> assemble_batch_from_rollout_samples
              -> HPT staleness filter
              -> HptBatchAssembler.materialize_rollout_sample
              -> HptBatchAssembler.normalize_hpt_training_batch
              -> DataProto.concat
              -> HptBatchAssembler.finalize_loss_denominator
     -> _fit_compute_log_prob
        -> apply_hpt_rollout_logprob_anchor      (HPT path)
     -> _fit_compute_advantage
     -> _fit_update_actor
        -> ppo_loss
           -> _compute_hpt_prompt_equal_policy_loss
```

Practical implication:

```text
HPT is not only a rollouter add-on.
It changes queue semantics, trainer assembly, old-logprob handling, and actor loss.
```

## 12. Paper-Reusable Framing

If this export is later written up in a paper or report, the following points
should be stated explicitly. They are present in the code and tests, but easy
to lose if the document only lists files.

### 12.1 Problem statement

```text
Base fully async RL already overlaps rollout and training.
HPT adds a prompt-level RL/SFT hybrid objective.
Naively combining prompt-level HPT with async rollout creates a mismatch:
  the learner wants prompt-group semantics,
  while the executor wants trajectory-attempt-level concurrency.
```

The exportable contribution is the reconciliation of those two grains without
changing the learner-visible HPT contract.

### 12.2 Main design contribution

The current local design can be summarized as:

```text
prompt-group semantics at the learner boundary
trajectory-attempt scheduling inside the rollouter
trainer-side deferred materialization
prompt-equal mixed RL/SFT loss
```

That is a cleaner research statement than “port WebOSWorld HPT code.”

### 12.3 Explicit guarantees worth naming

```text
G1. No partial prompt-group learner sample is emitted.
G2. RL and SFT rows share one DataProto training contract.
G3. Old-logprob semantics stay rollout-anchored for RL rows.
G4. Partial rollout recovery preserves token/logprob alignment.
G5. SFT rows are excluded from rollout correction/rejection semantics.
```

These are not formal theorems, but they are strong implementation guarantees
and good report structure.

### 12.4 Evaluation axes worth preserving

#### Contract correctness

```text
route correctness
mixed batch assembly correctness
loss correctness
old-logprob anchor correctness
partial recovery correctness
```

#### System efficiency

```text
completed prompt groups per unit time
drop rate before queue put
completed-budget pressure
trainer-visible sample throughput
```

#### Learning composition

```text
hpt/offline_data_ratio
hpt/p_success_zero_ratio
hpt/num_sft
hpt/num_rl_groups
hpt/missing_tau_count
```

These monitoring metrics already exist in local code and should not be treated
as incidental debug counters.

## 13. Test Migration Policy

Export the tests that protect contracts, not historical accidents.

### 13.1 Must migrate

- `test_async_hpt_config.py`
- `test_async_hpt_gate.py`
- `test_async_hpt_tau_adapter.py`
- `test_async_hpt_assembler.py`
- `test_async_hpt_loss.py`
- `test_async_hpt_training.py`
- `test_hpt_rollout_accumulator.py`
- `test_hpt_trajectory_scheduler_on_cpu.py`
- `test_partial_rollout_resume.py`
- `test_async_hpt_integration.py`

### 13.2 Migrate only if the target still uses the same mechanism

- `test_deferred_logprob_scoring.py`
- rollout trace tests
- request-phase trace tests

### 13.3 Do not treat as phase-1 blockers

- live WebOSGym tests
- SurfGym integration tests
- current GPU Qwen3.5/B200/SP4 live-path tests
- `gpu_use_hpt_actor_update_sp4_live_test.py`

Those are useful for this repo, but they are not the right phase-1 gate for a
clean async RL + HPT export.

## 14. Export Sequence

### Step 1. Build the upstream diff table

For each file in Sections 9.1 and 9.2, classify:

```text
new file
patch existing file
do not carry
```

Do not start by copying local whole files.

### Step 2. Export HPT core first

Bring up:

```text
hpt_config
hpt_gate
hpt_payload
hpt_assembler
hpt_training
loss path
```

before touching structural async changes.

### Step 3. Export E1~E3

Bring up:

```text
trajectory scheduler
prompt-group accumulator
inflight/completed split
partial rollout recovery
```

only after HPT core is already coherent on upstream.

### Step 4. Add observability only if needed

Tracing and measurement helpers remain default-off and should be treated as
bring-up tools, not as framework-defining behavior.

## 15. Final Recommendation

Phase 1 should be intentionally narrow:

```text
export now:
  HPT core
  E1~E3
  contract tests

do not export yet:
  WebOS/CUA adapters
  deferred scoring as default
  current SGLang/Qwen/B200 tuning
  current live multimodal GPU smokes
```

The correct export strategy is:

```text
reuse upstream Layer A
add Layer B
add Layer C
leave Layer D/E behind
```

If this discipline is not kept, the target codebase will inherit the local
WebOSWorld complexity instead of a reusable async RL + HPT framework.
