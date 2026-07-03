# RL Run Notes

This document records operational rules for the clean RL-focused `verl` fork.
It is not a replacement for the upstream docs. Use it as a run checklist for
async RL + HPT experiments, especially before launching expensive long runs.
Project-wide coding and review principles remain in `../AGENTS.md`; this file
only covers RL run and configuration practice.

## Current Direction

The project goal is to keep a clean upstream-oriented `verl` codebase while
adding HPT as a first-class objective path for fully asynchronous RL. The core
work is no longer WebOSGym/CUA-specific. CUA lessons remain useful only when
they describe general async scheduling, trainer queue, or environment-contract
issues.

Keep these boundaries clear:

- HPT objective semantics belong in HPT routing, assembly, and loss paths.
- Async scheduling belongs in rollouter/trainer queue and parameter-sync paths.
- Serving-engine tuning belongs in launch profiles or rollout engine config.
- Environment setup belongs in reproducible install scripts, not ad hoc shell
  state.

## Main Run Checklist

Before a main run:

1. Check the repo state.

   ```bash
   git status --short
   git log --oneline -5
   ```

2. Confirm the environment was built from the checked-in install path and not by
   untracked manual package edits.

   ```bash
   python - <<'PY'
   import torch, sglang, sgl_kernel, flash_attn
   print("torch", torch.__version__)
   print("torch cuda", torch.version.cuda)
   print("sglang", sglang.__version__)
   print("sglang-kernel", sgl_kernel.__version__)
   print("flash-attn", flash_attn.__version__)
   print("cuda available", torch.cuda.is_available())
   PY
   ```

   The RL environment exposes CUDA 13, cuDNN, and NCCL wheel libraries through
   its conda activation hook. A run that only prepends the RL environment's
   `bin` directory to `PATH` is not equivalent to `conda activate RL`: SGLang
   subprocesses can fail during scheduler initialization with
   `libcudart.so.13: cannot open shared object file`. Main launchers should
   activate the RL conda environment themselves or be run from a freshly
   activated RL shell.

3. Confirm the dataset and tau payloads are aligned.

   - `data.prompt_key` must exist in the train parquet.
   - The prompt-level HPT parquet should carry both `prompt_uid` and the
     `async_hpt.tau_messages_key` column.
   - `async_hpt.tau_dataset_path` may point at the same train parquet when the
     train parquet is the prompt-level HPT source of truth.
   - For strict HPT smoke and main runs, use `async_hpt.fail_on_missing_tau=True`.
     Set it to `False` only for an explicit missing-tau fallback ablation.

4. Confirm the objective normalization is intentional.

   For HPT/GRPO comparisons, keep:

   ```text
   algorithm.norm_adv_by_std_in_grpo=False
   ```

   unless the experiment explicitly studies GRPO normalization.

5. Confirm the run exercises the intended path.

   Look for logs such as:

   ```text
   async_hpt.enabled=True
   async_hpt.trajectory_scheduler.enabled=True
   async_training.partial_rollout=True
   Loop collection completed: ... learner_rows=... required_multiple=...
   ```

## HPT Queue And Staleness Settings

HPT changes the trainer queue contract. A trainer update no longer necessarily
uses exactly `required_samples` queue samples. The trainer first collects queue
samples, assembles HPT learner rows, and may keep consuming queue samples until
the learner row count satisfies the distributed training multiple.

The relevant trainer log is:

```text
[FullyAsyncTrainer] Loop collection completed:
  <queue_samples>/<required_samples> samples,
  learner_rows=<rows>,
  required_multiple=<multiple>
```

This means:

- `queue_samples` can be larger than `required_samples`.
- `learner_rows` must be divisible by `required_multiple`.
- A single trainer update can consume many queue samples.
- `async_training.max_completed_prompt_groups` also bounds how many completed
  queue samples the HPT trainer may read while trying to form one trainable
  learner batch.

Because of that, async run budgets must be chosen together:

```text
required_samples
trigger_parameter_sync_step
staleness_threshold
max_inflight_prompt_groups
max_completed_prompt_groups
actor_rollout_ref.rollout.n
```

The current rollouter pause threshold is:

```text
max_required_samples =
  required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step
```

If `max_required_samples` is too small, the rollouter can pause before the
trainer reaches `trigger_parameter_sync_step`. Then the trainer waits for more
queue samples, while the rollouter waits for sync/reset. This is a circular
wait, not a generation bottleneck.

For HPT, distinguish the trainer's initial queue request from the actor's
learner-row mini-batch multiple:

```text
initial queue request = actor.ppo_mini_batch_size * async_training.require_batches
actor learner multiple = actor.ppo_mini_batch_size * actor_rollout_ref.rollout.n
```

In an all-SFT HPT window, one queue sample usually materializes to one learner
row. With `ppo_mini_batch_size=64` and `rollout.n=8`, the first actor update
therefore needs 512 learner rows, even if the trainer initially requests only
128 queue samples with `require_batches=2`. If the staleness budget cannot
cover this learner-row demand through the next parameter-sync point, the
system can appear to be "not training" while the trainer is actually waiting
for enough HPT rows to form a legal actor update.

For the OpenR1 HPT main launcher, keep `staleness_threshold` modest and make
the initial trainer queue request match the Unify HPT learner batch scale:

```text
actor.ppo_mini_batch_size=64
rollout.n=8
async_training.require_batches=16
async_training.trigger_parameter_sync_step=4
async_training.staleness_threshold=2.0
async_training.max_completed_prompt_groups=2048
```

This gives:

```text
initial queue request = 64 * 16 = 1024
actor learner multiple = 64 * 8 = 512
Unify HPT global learner scale = data.train_batch_size * rollout.n = 128 * 8 = 1024
max_required_samples = 1024 * (2 + 1) * 4 = 12288
all-SFT rows needed for 4 updates = 1024 * 4 = 4096
```

This avoids forcing the trainer through HPT row-aware extra collection before
the first actor update while keeping the update batch scale comparable to the
reference HPT run. The row-aware fallback remains bounded by
`max_completed_prompt_groups=2048` for mixed SFT/RL routing, where queue samples
can materialize to different learner-row counts.

### Smoke Setting

For the OpenR1 HPT smoke launcher, use a generous budget:

```text
N_RESP_PER_PROMPT=4
TRAIN_PROMPT_MINI_BSZ=4
TRIGGER_PARAMETER_SYNC_STEP=4
STALENESS_THRESHOLD=19.0
async_training.max_inflight_prompt_groups=32
async_training.max_completed_prompt_groups=256
```

With `required_samples=4`, this gives:

```text
max_required_samples = 4 * (19 + 1) * 4 = 320
```

This is intentionally large for smoke coverage. The completed queue remains
bounded at 256, while staleness pauses only after 320 completed samples. This
keeps the queue budget as the first backpressure point and avoids the circular
wait where the rollouter pauses on staleness before the trainer has enough
HPT learner rows to reach parameter sync.

The inflight prompt-group setting should match the actual rollout replica cap.
The FSDP2 SGLang smoke uses 4 rollout GPUs and tensor parallel size 2, so it
has 2 SGLang replicas. The rollouter caps active prompt groups at 16 per
replica:

```text
effective_inflight_prompt_groups =
  min(max_inflight_prompt_groups, num_sglang_replicas * 16)
  = min(32, 2 * 16)
  = 32
```

Do not raise `max_inflight_prompt_groups` above this unless the number of
rollout replicas also changes. A larger value only makes the launcher look more
aggressive; it does not increase actual concurrency.

The smoke dataset currently has 12 train rows. Fully async rollouter caps the
requested rollout steps by:

```text
min(rollout.total_rollout_steps, len(train_dataloader) * trainer.total_epochs)
```

Therefore the smoke launcher uses:

```text
rollout.total_rollout_steps=7680
trainer.total_epochs=640
```

This keeps the explicit 7680-step cap reachable. With `required_samples=4` and
`trigger_parameter_sync_step=4`, the rollouter reports the nominal base async
progress:

```text
total_train_steps = 7680 / (4 * 4) = 480
```

For HPT, this nominal value is not the number of completed trainer updates.
The trainer may consume more than 4 queue samples per update while searching
for a learner-row count divisible by the distributed training multiple. In the
initial OpenR1 HPT smoke, 480 rollout samples produced 28 trainer updates, or
7 parameter-sync cycles. Scaling to 7680 rollout samples gives roughly 16 times
that coverage, i.e. about 100+ parameter-sync cycles for the strict HPT smoke
route if rows are not dropped.

### Main Run Rule

For a main run, do not copy smoke values blindly. Size the budgets from the
observed HPT learner-row consumption:

```text
completed budget >=
  trigger_parameter_sync_step
  * p95(queue_samples consumed per trainer update)
  * safety_margin
```

Use a safety margin of at least `1.5` for early experiments. If the run mixes
RL-only rows and HPT rows, use the p95 from the mixed workload, not from a toy
all-HPT or all-RL sample.

## Skip/Cache Caveat

The special E2E smoke uses rollout skip/cache to avoid paying generation cost
for every first step:

```text
skip.async_rollout.enable=True
skip.async_rollout.steps=[1]
skip.async_rollout.action=cache
```

This is useful for testing trainer/update contracts quickly, but it can produce
a burst of completed samples. With too-small staleness settings, that burst can
hit the pause threshold before the trainer reaches sync.

Interpretation rule:

- If `active_tasks_size=0`, `mq_queue_size=0`, and `pending_queue_size` is large,
  the issue is not SGLang generation. It is queue/staleness control flow.
- If `staleness_samples >= max_required_samples`, raise the completed/staleness
  budget or lower the sync interval for smoke.
- Do not diagnose GPU utilization from a skip/cache-heavy smoke as if it were a
  real generation workload.

## Partial Rollout And Parameter Sync

`async_training.partial_rollout=True` is important for realistic async behavior.
It allows interrupted rollout work to be resumed across parameter sync. Use it
in smoke tests that are meant to cover real async behavior.

Watch for:

```text
_fit_update_weights
trigger_parameter_sync_step
partial/total_partial_num
partial/partial_ratio
partial/max_partial_span
```

Do not treat partial rollout as a free correctness relaxation. Generated tokens,
response masks, old logprobs, route metadata, and parameter-version metadata
must remain aligned at the `DataProto` boundary.

## HPT Data Routing

HPT rows and RL rows may coexist only when the dataset and config explicitly
test missing-tau fallback. The default OpenR1 HPT smoke uses the stricter shape:
the train parquet itself carries `prompt_uid` and `tau_messages`, and the tau
lookup points at the same train parquet.

The route must be explicit. Useful rollouter metrics include:

```text
hpt/num_sft_routed
hpt/num_rl_routed
hpt/missing_tau_count
```

Interpretation:

- For the default strict HPT smoke, `missing_tau_count` should stay zero.
- `missing_tau_count > 0` is acceptable only if `fail_on_missing_tau=False` and
  the run intentionally tests missing-tau fallback.
- For strict HPT runs, missing tau fails closed.
- Do not infer route from missing tensors or dummy values. Route metadata must be
  explicit.

## Trainer Update Contract

HPT trainer updates must satisfy all of the following before backprop:

- Learner rows are divisible by the required distributed training multiple.
- `response_ids`, `response_mask`, `old_log_probs`, rewards, advantages, and
  non-tensor route metadata remain row-aligned.
- SFT-style supervised rows do not accidentally train with dummy rollout
  probabilities.
- Dropped or aborted samples are visible in metadata/metrics and do not
  contribute silently to loss denominators.

The update contract is stronger than a shape-only check. A batch can have valid
tensor shapes while still being semantically wrong if route metadata or old
logprob anchors are wrong.

## Suggested Main-Run Log Checks

During a run, inspect Ray worker logs directly, not only the nohup file. The
nohup file can miss actor-local details.

Find the active Ray session:

```bash
ls -td /tmp/ray/session_* | head -n 1
```

Useful checks:

```bash
SESSION=$(ls -td /tmp/ray/session_* | head -n 1)

rg -n "Loop collection completed|learner_rows|required_multiple|global_steps|_fit_update_weights|Traceback|ERROR|Exception" \
  "$SESSION/logs"/*.out "$SESSION/logs"/*.err

rg -n "MonitorLoop|staleness_samples|pending_queue_size|mq_queue_size|active_tasks_size|hpt/" \
  "$SESSION/logs"/*.out "$SESSION/logs"/*.err
```

Healthy HPT async smoke should show:

```text
Loop collection completed ... learner_rows=<multiple of required_multiple>
global_steps increasing
_fit_update_weights after the configured sync interval
no Traceback / ValueError / AssertionError
```

Potential deadlock pattern:

```text
trainer:
  Requesting 4 samples from queue

rollouter:
  staleness_samples >= max_required_samples
  active_tasks_size = 0
  mq_queue_size = 0
  pending_queue_size > 0
```

This means the trainer is waiting for more completed queue samples while the
rollouter has paused before processing pending samples.

## What Not To Change As A Utilization Fix

Do not change the learning problem just to increase utilization. By default,
avoid changing:

- reward definitions
- route labels
- tau content
- train/validation split
- sample selection
- termination semantics
- old-logprob reference policy
- response masking or loss denominators

Efficiency work should first target scheduling, queue sizing, serving-engine
configuration, or environment reproducibility.

## Smoke Versus Main Run

Smoke runs are for contract coverage:

- HPT route and assembly
- queue sample to learner-row conversion
- partial rollout and sync boundaries
- actor update entry
- failure on invalid batch contracts

Main runs are for training quality and throughput. Do not overfit main-run
settings to skip/cache smoke behavior. Conversely, do not launch a main run
until the smoke has reached the same control-flow boundaries that the main run
depends on.
