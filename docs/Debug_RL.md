# RL Debugging & Profiling Notes

_Last updated: 2026-07-10_

How we keep this fork's code clean and diagnose performance in the async
RL + HPT stack. This is not a replacement for upstream tool docs (`ruff`,
`py-spy`, `line_profiler`, `pyinstrument`) — it records the project- and
cluster-specific gotchas and the method that repeatedly works. For run/log
checks see `Readme_RL.md`; for queue/staleness/HPT budgets see
`AsyncBudget_RL.md`; for coding/review principles see `../AGENTS.md`.

## Lint & format (ruff, pinned)

- The authoritative ruff version is pinned by `.pre-commit-config.yaml`, and
  the rule/line-length config lives in `pyproject.toml [tool.ruff]`. Read them,
  don't assume:

  ```bash
  grep -A3 ruff-pre-commit .pre-commit-config.yaml
  grep -nA20 '^\[tool.ruff\]' pyproject.toml
  ```

- Run the **pinned** toolchain, not whatever ruff happens to be installed. A
  locally-installed ruff of a different version formats differently and will
  churn code away from CI style. Use pre-commit, which fetches the pinned
  version into an isolated env and matches CI exactly:

  ```bash
  pre-commit run ruff        --files <changed files>
  pre-commit run ruff-format --files <changed files>
  ```

  If you must call `ruff` directly, first match the pinned version.
- Scope lint fixes to the files your change already touches. Pre-existing
  violations elsewhere are a **separate** cleanup commit — bundling lint churn
  with a logic change hides both and crosses concern boundaries (see
  `../AGENTS.md` "keep changes scoped").
- Auto-fixable: `ruff check --fix` (import order `I001`, quoted annotations
  `UP037`, unused imports `F401`) and `ruff format` (line wrapping/`E501`).
  Manual: `B904` (add `from e` / `from None`), `UP022`, etc. Re-run the affected
  tests after auto-fixes — import reorders and `from None` are behavior-neutral,
  but confirm rather than assume.

## Static type-checking (mypy, pinned + scoped)

`.pre-commit-config.yaml` pins mypy (mirrors-mypy `v1.17.0`); config is
`pyproject.toml [tool.mypy]`. The default is a **blanket `ignore_errors = true`** —
a module is only actually checked when listed in the `ignore_errors = false`
override. The fork's HPT-core modules are opted in there; the orchestration files
(`fully_async_trainer`, `fully_async_rollouter`, `separation.ray_trainer`) are
not yet — they still carry `Optional`-attribute narrowing and Ray `.remote`
`attr-defined` noise, and are enabled one module at a time as they are cleaned.

- Run the pinned checker (isolated env, matches CI): `pre-commit run mypy
  --files <changed files>`. If mypy 1.17.0 is in the RL env (`--dry-run` first,
  see below), point it at a package: `python -m mypy
  verl/experimental/fully_async_policy/`.
- To measure a not-yet-enabled module before opting it in, run with a temp
  config that drops the blanket silence (`[mypy]` with
  `ignore_missing_imports=true`, `follow_imports=skip`) so you see its real error
  count, fix them, then add the module to the override.
- Prefer making a config-guaranteed non-None invariant explicit with
  `assert x is not None` (documents the contract and fails fast at runtime too)
  over `# type: ignore`. Reserve `# type: ignore[code]` for genuine tool gaps
  (e.g. the Ray `@remote`-injected `.remote`).
- mypy earns its keep on exactly the bug class this fork hits: opting the modules
  in immediately surfaced a wrong return annotation (`dict[str, dict[...]]` where
  the values are `list[str]`) and unguarded `str | None` paths.

## Installing dev tools into the RL env safely

The RL env is a tightly-pinned ML stack; a careless install can silently
upgrade `torch`/`transformers` and break it. Before installing anything,
**dry-run** and confirm the plan is purely additive:

```bash
uv pip install --python "$(command -v python)" --dry-run <pkg> ...
```

Proceed only if every change line is a new `+ <pkg>` with **no removals and no
version changes** to existing packages. Pure-Python profilers such as
`pyinstrument` and `line_profiler` resolve as additive; verify anyway.

## Profiling a live training run (py-spy)

The trainer and rollouter are long-lived Ray actors — attach a sampling
profiler instead of restarting. Find the target process:

```bash
pgrep -f "ray::FullyAsyncTrainer.fit"   # single-controller trainer / fit loop
pgrep -f "ray::FullyAsyncRollouter"     # rollouter
```

Cluster/environment gotchas that will otherwise waste a session:

- **ptrace is restricted** (`cat /proc/sys/kernel/yama/ptrace_scope` returns a
  non-zero value), so py-spy cannot attach as your user. Run it through the
  cluster's root-auth wrapper (`gcsudo`; resolve the real binary with
  `type gcsudo`), using the RL env's `py-spy`.
- **Root-auth wrapper argv splitting**: the wrapper re-joins its argv on spaces
  and re-parses through a shell, so a multi-statement `bash -c '...'` is
  fragmented at whitespace (e.g. `VAR=...` assignments are silently dropped and
  you get `command not found`). Write the loop to a script file and invoke it as
  a two-token command: `gcsudo bash /path/to/script.sh`.
- **`record` vs bounded dumps**: `py-spy record --duration N` for large `N`
  exceeds the agent shell's command timeout and gets killed. For a distribution
  of where a thread spends time, run a bounded loop of `py-spy dump` (e.g. ~40
  dumps at ~1.6s each), classify each snapshot by its **leaf** frame (py-spy
  prints the innermost frame first), and aggregate:

  ```bash
  # inside the elevated script: dump N times, keep the innermost known frame
  py-spy dump --pid "$PID" | grep -oE '<frame1>|<frame2>|...' | head -1
  # then: ... | sort | uniq -c | sort -rn
  ```

  A single `py-spy dump` also shows the full async call stack of the
  `"AsyncIO Thread: default"` — enough to identify a hot path immediately.

## Confirm the cause before acting

A profiler tells you **where** a thread is, not **why** it is expensive or **how
many times** the code runs. Always confirm magnitude with an independent method:

- Micro-benchmark the *real* code path — import the actual function, feed real
  inputs (rows from the parquet, the real tokenizer), time it, and use
  `line_profiler` (`from line_profiler import LineProfiler`) for per-line cost.
- Reconcile the numbers. If a profiler says "100% in `X`" but one call to `X` is
  cheap, the cost is **repetition** (a loop, an `O(n²)` re-run, or a high call
  count) — not per-call cost. Read the caller to find the multiplier.
- Triangulate with independent angles — live profile + micro-benchmark +
  code/data-flow read — and act only when at least two agree. Fanning these out
  as parallel subagents works well and catches wrong single-tool conclusions.

## Worked example — `O(n²)` re-tokenization at the collection boundary

- **Symptom**: ~150s/step with trainer GPUs mostly idle; py-spy showed ~40/40
  samples inside `apply_chat_template` under
  `_get_samples_from_queue → assemble_rollout_samples → materialize_sft_payload`.
- **Micro-bench**: one `apply_chat_template` on real tau ≈ 4ms; a single pass
  over a step's rows ≈ 1s — ~100× too small to explain the step time.
- **Root cause**: the learner-row-multiple top-up loop in
  `_get_samples_from_queue` re-ran `assemble_rollout_samples` over the whole
  growing sample list on every added sample → `O(n²)` re-tokenization. It only
  bites in **all-SFT** windows (one queue sample → one learner row, so it tops
  up one-by-one to reach the `ppo_mini_batch_size * rollout.n` multiple); RL
  windows yield `rollout.n` rows per sample and skip the loop entirely.
- **Fix shape**: per-row materialization is independent and padded to fixed
  config lengths, so materialize each sample **once**, accumulate, and
  `normalize_mixed_schema` + `concat` **once** — byte-identical batch, `O(n)`.
- **Lesson**: async-HPT perf bugs concentrate at the queue→learner-batch
  assembly boundary and are regime-dependent (SFT vs RL). Whenever an assembly
  step is called inside a collection loop, check whether it re-processes a
  growing list.

## Property-based testing at the collection boundary (hypothesis)

The async-HPT bugs concentrate at the queue→learner-batch assembly boundary and
are composition-dependent (SFT vs RL group sizes) — exactly where example-based
tests miss the pathological case. The grow-to-align crash (`Improvement_RL`
§5.8.4) was a *universal* claim — "for every group composition, collection
terminates and aligns" — that a few hand-picked examples could not falsify.

`tests/special_RL/test_row_alignment_properties_on_cpu.py` encodes the invariants
of `_plan_row_alignment_deferral` as hypothesis properties over random
`(row_counts, required_multiple, protected_prefix)`:

- a returned deferral only touches eligible groups, its rows sum to exactly the
  residue, and the retained batch is an exact multiple (crash-free alignment);
- `None` is returned **iff** the residue is unreachable — checked against an
  independent brute-force subset-sum oracle (two methods must agree, per "Confirm
  the cause before acting");
- aligned / trivial-multiple inputs defer nothing.

Reach for hypothesis when the correctness statement is a `for all` over a
combinatorial input (group compositions, budget arithmetic, staleness spans), and
pair the implementation with a slow-but-obvious oracle so the property tests a
second, independent computation rather than restating the code. hypothesis is in
`requirements-test.txt`; the test `importorskip`s it so it skips cleanly if
absent. It is a CPU-only `*_on_cpu.py` test.

## Known landmine — non-HPT old-logprob path is fsdp2-incompatible (2026-07-10)

**Symptom:** launching the fully-async stack with `async_hpt.enabled=False`
(and `algorithm.rollout_correction.bypass_mode=False`) dies on the first
fit-step with:

```
AssertionError: No DTensor-type parameters found in the model. FSDP2 sharding may not be enabled.
  at WorkerDict.actor_save_model_to_cpu()
```

**Mechanism:** with HPT off, `FullyAsyncTrainer._compute_old_log_prob` takes the
stock MIS branch — it snapshots/restores *version-1-of-cycle* weights via
`actor_rollout_wg.save_model_to_cpu(...)` — and that helper is incompatible with
this fork's fsdp2 build. Every HPT run bypasses the branch (entry-anchor
shortcut, or the rollout-anchor path in `separation/ray_trainer._fit_compute_log_prob`),
so the bug stayed latent through the entire M-series and only fired on the first
non-HPT launch (RLonly v1, `m3hdp3jm`, dead at startup).

**Workarounds (choose per experiment intent):**
- *Teacher-channel-off ablation while keeping everything else identical to an
  HPT main*: keep `async_hpt.enabled=True` and seal the routing with the
  sentinel `async_hpt.success_threshold=-1.0` — every score counts as success,
  `p_success≡1 > gamma`, SFT never fires, k=0 groups stay as advantage-0 RL rows
  (pure-GRPO semantics), and the entry anchor keeps the crashing branch
  unreachable. Used by `main_scripts/run_..._RLonly.sh` (v2). Caveat: the
  sentinel distorts `hpt/onpolicy_success_rate≡1.0` (aggregation only —
  read the real success rate from `critic/score/mean`).
- *Genuinely non-HPT run*: either set `rollout_correction.bypass_mode=True`
  (old_log_probs = rollout logprobs; skips the branch) or fix
  `save_model_to_cpu` for fsdp2/DTensor before launching.

**Lesson:** "gated code equals dead code" is false — a config flip can route into
paths no prior run ever exercised. When flipping a top-level mode flag
(`async_hpt.enabled`, `bypass_mode`, strategy), trace the *first fit-step's*
control flow end-to-end before burning a launch.
