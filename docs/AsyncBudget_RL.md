# Async Queue & HPT Budget Sizing

How to size the fully-async queue, staleness, and HPT learner-row budgets.
Environment/log commands: `Readme_RL.md`. Code enforcement: `Codemap_RL.md`. Rules: `../AGENTS.md`.

## Run Correctness Checklist

Before a main run, in addition to `Readme_RL.md`'s environment checklist:

1. Confirm the dataset and tau payloads are aligned.

   - `data.prompt_key` must exist in the train parquet.
   - The prompt-level HPT parquet should carry both `prompt_uid` and the
     `async_hpt.tau_messages_key` column.
   - `async_hpt.tau_dataset_path` may point at the same train parquet when the
     train parquet is the prompt-level HPT source of truth.
   - For strict HPT smoke and main runs, use `async_hpt.fail_on_missing_tau=True`.
     Set it to `False` only for an explicit missing-tau fallback ablation.

2. Confirm the objective normalization is intentional. For HPT/GRPO
   comparisons, keep `algorithm.norm_adv_by_std_in_grpo=False` unless the
   experiment explicitly studies GRPO normalization.

3. Confirm the run exercises the intended path.

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

**Known pitfall — queue samples are not rows.** `require_batches` counts
**queue samples** (`initial queue request = ppo_mini_batch_size *
require_batches`), and one queue sample already holds `rollout.n` rows (the
rollouter repeats one prompt `rollout.n` times before generation and queues
the whole group as one item — see `Codemap_RL.md`'s rollouter call graph). An
earlier `require_batches=16` assumed `64 * 16 = 1024` already matched the
baseline's `train_batch_size(128) * rollout.n(8) = 1024`-sequence scale — a
queue-sample count compared to a row count, missing `* rollout.n`. The actual
rows collected per fit_step were `1024 * 8 = 8192`, 8x intended, running 16 SGD
steps per fit_step instead of 2 (the actor's own SGD mini-batch size is
`ppo_mini_batch_size * rollout.n`, the "actor learner multiple" below).
Reward/log-prob/ref/advantage computation holds the *entire* fit_step batch
before that chunking applies, so every fit_step was 8x larger and 8x less
frequent than intended.

The value that actually matches the baseline is `require_batches=2`. For the
OpenR1 HPT main launcher, pair it with a modest `staleness_threshold` and a
`max_completed_prompt_groups` sized to the same smaller scale:

```text
actor.ppo_mini_batch_size=64
rollout.n=8
async_training.require_batches=2
async_training.trigger_parameter_sync_step=4
async_training.staleness_threshold=2.0
async_training.max_completed_prompt_groups=256
```

This gives:

```text
initial queue request = 64 * 2 = 128 prompt groups -> 128 * 8 = 1024 rows/fit_step
actor learner multiple = 64 * 8 = 512 rows (2 SGD steps/fit_step, matching the baseline)
max_required_samples = 128 * (2 + 1) * 4 = 1536 queue samples
all-SFT rows needed for 4 updates = 1024 * 4 = 4096
```

In an all-SFT HPT window, one queue sample materializes to one row instead of
`rollout.n`, so the trainer may need more than the nominal queue-sample count
to reach a row count divisible by the actor learner multiple — if the
staleness budget can't cover that through the next parameter-sync point, the
system can look like it's "not training" while actually waiting for enough
HPT rows.

`max_completed_prompt_groups` scales down with `require_batches` (2048/8=256)
to keep the completed-queue cap, not staleness, first: `max_required_samples`
shrinks with `required_samples`, so leaving it at 2048 would flip which cap
binds first. Raise `require_batches` only when fixed per-fit_step overhead
(Ray RPC, queue deserialization) starts to dominate at the smaller scale.

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

This is intentionally large for smoke coverage: the completed queue stays
bounded at 256 while staleness pauses only after 320 samples, keeping the
queue budget — not staleness — the first backpressure point.

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
rollout replicas also changes — a larger value looks more aggressive but does
not increase actual concurrency.

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

For HPT, this nominal value is not the number of completed trainer updates —
the trainer may consume more than 4 queue samples per update while searching
for a divisible learner-row count. The initial OpenR1 HPT smoke saw 480
rollout samples produce 28 trainer updates (7 parameter-sync cycles); scaling
to 7680 gives roughly 100+ cycles for the strict HPT route if rows aren't dropped.

### Main Run Rule

For a main run, do not copy smoke values blindly. Size the budgets from the
observed HPT learner-row consumption:

```text
completed budget >=
  trigger_parameter_sync_step
  * p95(queue_samples consumed per trainer update)
  * safety_margin
```

Use a safety margin of at least `1.5` for early experiments, taking the p95
from the actual mixed RL/HPT workload rather than a toy all-HPT or all-RL sample.

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

For rollout-dump directory hygiene (keeping dumps unique per run so a later run
never replays a stale generation), see `Readme_RL.md`.

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
