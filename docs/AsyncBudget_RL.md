# Async Queue & HPT Budget Sizing

_Last updated: 2026-07-07_

How to size the fully-async queue, staleness, and HPT learner-row budgets.
Environment/log commands: `Readme_RL.md`. Code enforcement: `Codemap_RL.md`. Rules: `../AGENTS.md`.

## Operating Principles

Distilled from the async M-run experiments (case record: `Improvement_RL.md` §5.8).
These are the headline rules; where the detailed sizing below predates them, they win.

1. **Bounded-fresh beats large-stale.** The completed queue is a backlog of aging
   rollouts. Training on a large *stale* batch does not just cost compute — it
   measurably degrades the model (a leg resumed onto stale batches lost ~3 val
   points). Prefer a small, fresh, bounded batch over a large, stale one.

2. **Bound the queue to stop the batch-explosion loop.** A slow learner lets the
   queue fill; a fuller queue makes the next batch bigger; a bigger batch makes
   the learner slower. Left unbounded this self-amplifies (batch ran 22M→90M
   tokens, steps to ~30 min). Capping the completed queue at a small multiple of
   the per-step batch keeps the loop from starting.

3. **Drops are backpressure, not failure.** With a bounded queue and the learner
   as the bottleneck, rollouters overproduce and the excess pauses or drops.
   `Queue full, dropped sample` is the cap working. Rollout is not the bottleneck,
   so its wasted work is cheap; freshness is the payoff.

4. **Step time ∝ batch size; throughput is ~constant.** A slow step from a big
   batch is more-work, not slower-work (effective learner tok/s stayed constant
   through the explosion). The lever for step time is the batch/queue bound, not
   more train GPUs (and note the checkpoint is FSDP-sharded to a fixed world size,
   so changing the train GPU count is not a drop-in resume anyway).

5. **Fix plumbing, not the learning problem.** When utilization or step time is
   bad, do not reach for learning-facing knobs (mini-batch size, sync cadence) —
   that changes the experiment. Reconcile the stream-to-batch mismatch in the
   collection layer instead.

6. **Variable HPT group sizes → trim + carryover, not grow-to-align.** RL groups
   contribute `rollout.n` rows, SFT groups 1, but the learner needs a fixed
   row-multiple. Do not grow the batch until it happens to land on a multiple
   (over-collects 2-3x and can fail to converge → crash). Collect the intended
   `required_samples`, trim down to the largest aligned batch, and carry the small
   residue (< one multiple) to the next step where it trains first. Bounded,
   zero-waste, crash-free.

**Sizing that follows from these:** bound `max_completed_prompt_groups` to ~2-3×
the per-step prompt-group batch (e.g. ~384 for a 128-group batch), not thousands;
expect and accept drops; keep the batch-scale knobs (`ppo_mini_batch_size ×
require_batches`) at baseline parity — never retune them to chase throughput; let
the queue cap, not a large `staleness_threshold`, do the freshness work.

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

HPT changes the trainer queue contract. RL groups materialize to `rollout.n`
learner rows and SFT groups to 1, so `required_samples` prompt groups rarely land
exactly on the `required_multiple` the learner needs. The trainer collects the
intended `required_samples` groups (growing only if the batch is under one
multiple), then **trims down to the largest aligned batch and carries the small
residue (< one multiple) to the next step**, where it is trained first (Principle
6; `Improvement_RL.md` §5.8.6). The batch is therefore bounded near
`required_samples` — it does not balloon to whatever the queue holds.

The relevant trainer log is:

```text
[FullyAsyncTrainer] Loop collection completed:
  <retained>/<required_samples> samples, learner_rows=<rows>, required_multiple=<multiple>,
  carryover_in=<n>, carried_forward=<n>, discarded=<n>, deferred_rows=<n>
```

This means:

- `learner_rows` is a multiple of `required_multiple`, bounded near
  `required_samples * rollout.n` (not the whole queue).
- `carryover_in`/`carried_forward` are the residue groups deferred across steps;
  they stay small (< one multiple in rows) and are consumed with priority next
  step. `discarded` is a rare bounded drop when a carried group cannot be placed.
- `async_training.max_completed_prompt_groups` bounds the *completed queue depth*
  (freshness + backpressure), no longer a large grow-until-aligned read window.

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

The pause gate is evaluated live against the *outstanding buffer* — in-flight
open groups plus completed queued samples (`_outstanding_sample_count`) — not a
monotonic produced-since-sync counter. So a paused rollouter resumes (via the
monitor loop) as soon as the trainer drains the queue back below
`max_required_samples`; it does not wait for the next sync. Sizing
`max_required_samples` too small still throttles throughput (the rollouter idles
at the ceiling more often), but that is a budget-sizing choice, not a
sync-locked circular wait.

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

The prompt-batch parity rule is:

```text
ppo_mini_batch_size * require_batches = baseline train_batch_size
```

For the OpenR1 HPT main launcher, the baseline train batch is 128 prompt
groups. Two settings are therefore valid from a batch-scale perspective:

```text
strict Unify mini-batch grain:
  actor.ppo_mini_batch_size=64
  async_training.require_batches=2

finer async/HPT mini-batch grain:
  actor.ppo_mini_batch_size=32
  async_training.require_batches=4
```

Both collect 128 prompt groups per fit_step. With `rollout.n=8`, both collect
1024 generated rows per fit_step. The current main launcher uses the finer
`32 * 4` form because it keeps the large batch scale comparable while making
HPT learner-row divisibility and trainer scheduling less brittle.

Current main-run sizing:

```text
actor.ppo_mini_batch_size=32
rollout.n=8
async_training.require_batches=4
async_training.trigger_parameter_sync_step=4
async_training.staleness_threshold=2.0
async_training.max_completed_prompt_groups=384   # bounded: ~3x the 128-group batch
```

This gives:

```text
initial queue request = 32 * 4 = 128 prompt groups -> 128 * 8 = 1024 rows/fit_step
actor learner multiple = 32 * 8 = 256 rows (4 SGD mini-batches/fit_step)
max_required_samples = 128 * (2 + 1) * 4 = 1536 queue samples
all-SFT rows needed for 4 updates = 256 * 4 = 1024
```

In an all-SFT HPT window, one queue sample materializes to one row instead of
`rollout.n`, so the trainer may need more than the nominal queue-sample count
to reach a row count divisible by the actor learner multiple — if the
staleness budget can't cover that through the next parameter-sync point, the
system can look like it's "not training" while actually waiting for enough
HPT rows.

`max_completed_prompt_groups` is not a batch-size parity knob. It is the
completed-queue/backlog cap. **Bound it** to a small multiple of the per-step
batch (~384 for the 128-group main launcher). It does not change the learner
batch scale — the collection now bounds the batch by trim+carryover regardless
(Principle 6) — so a small cap simply keeps the queue fresh and the batch from
riding a stale backlog. Do not size it in the thousands: the old `2048` was what
let the batch-explosion loop run (Principle 2), and once bounded, `Queue full,
dropped sample` is the expected, safe backpressure signal (Principle 3), not a
symptom to size away.

Raise `require_batches` only when fixed per-fit_step overhead (Ray RPC, queue
deserialization) starts to dominate at the smaller scale. Do not raise it to
16 for parity; that is the unit error described above.

Validation frequency is also not directly step-identical to the synchronous
baseline. Unify validates on `global_steps % trainer.test_freq == 0`. The
fully-async trainer validates when the current parameter version reaches
`trainer.test_freq`, and parameter versions advance after
`trigger_parameter_sync_step` local trainer updates. Therefore
`trainer.test_freq=10` is intentionally kept for naming/config parity, but in
async runs it means roughly every `10 * trigger_parameter_sync_step` local
updates. Keep `trainer.val_before_train=False` for main async runs unless the
run explicitly needs an expensive initial validation baseline.

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

For a main run, do not copy smoke values blindly, and do not size the completed
queue to the p95 of grow-until-aligned consumption — that was the pre-trim regime
that let the batch explode. With trim+carryover the batch is bounded near
`required_samples` on its own, so size the completed queue for **freshness and
backpressure**, not to hold a large grow window:

```text
max_completed_prompt_groups ≈ 2-3 * (ppo_mini_batch_size * require_batches)   # prompt groups
```

That is ~2-3 fit_steps of headroom: enough that the learner never starves, small
enough that consumed data is at most a few parameter versions stale. Expect
`Queue full, dropped sample` under this cap — it is the bound working
(Principle 3), and watch `fully_async/trainer/idle_ratio` (≈0 healthy; if it
rises, the cap is too tight, raise it one step).

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
- The `count/staleness_samples` metric reports the live outstanding buffer
  (`_outstanding_sample_count`). If it sits at `max_required_samples` while the
  trainer cannot keep up, raise the completed/staleness budget or lower the sync
  interval for smoke.
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
