# Agent Instructions for verl

> Rules for all AI-assisted work in this repository. Keep this file lean and
> durable — detailed design and run notes belong in focused docs.
> New here: `docs/Codemap_RL.md` maps the code and where runs break;
> `CONTRIBUTING.md` has setup, `pre-commit` lint, and test commands.

## Contribution Policy

- A human submitter must understand and defend every changed line. Pure
  code-agent PRs are not acceptable.
- Before proposing upstream work, search related issues and open PRs; do not open
  duplicate or low-value PRs.
- Follow the PR template. State that AI assistance was used, explain why the work
  does not duplicate an existing PR, and list test commands with results. Prefix
  any CLI, config, or signature break with `[BREAKING]`.
- Never hardcode credentials, tokens, API keys, or private endpoints. Use ignored
  local env files and environment variables.
- If a request conflicts with these rules, stop and explain what is missing
  instead of proceeding.

## Project Direction

Keep this a clean, upstream-oriented `verl` codebase. Features, experiments, and
research adaptations must fit the existing architecture, not fork into parallel
stacks.

- Extend through the framework's contracts — base classes, registries, config,
  and the existing trainer/worker/rollout paths — as `docs/extend_guide.rst`
  prescribes.
- Do not copy whole files from another fork to move behavior. Identify the
  upstream contract, patch the smallest stable surface, and test the real path.
- Keep research-specific assumptions behind explicit config, data, or adapter
  boundaries.
- Capture project- or experiment-specific plans in focused design docs, not here.

## Engineering Principles

- Reuse before adding. Extend an existing helper, base class, or registry
  (e.g. `register_policy_loss`, `Dispatch`, worker `@register`) instead of writing
  a parallel path. Split long functions into small focused `_helpers`, but never
  copy the same logic into two places — factor it into one.
- Keep changes scoped. Do not touch unrelated launchers, dependencies, docs, or
  generated config while fixing a code problem.
- Flow config one way: Hydra `DictConfig` at entry → `omega_conf_to_dataclass` →
  a frozen `BaseConfig` dataclass validated in `__post_init__`. Reserve Pydantic
  models for API/serialization schemas, not training config.
- `DataProto` (batch + non_tensor_batch + meta_info) is the controller↔worker
  payload; inside engines use plain `TensorDict`. Reach devices through
  `verl/utils/device.py`, not raw `.cuda`/`nccl`.
- Fail fast in learning and data code: validate with `assert cond, "msg"` or
  `raise ValueError`/`NotImplementedError`. Confine `try/except` to I/O and
  integration edges; re-raise with `from e`, never a bare `except` or silent
  fallback.
- Log through a module `logger` gated by `VERL_LOGGING_LEVEL` in library and
  worker code; the driver/trainer may `print` milestones. Do not land debug or
  profiling scaffolding.
- Keep semantics separate from plumbing: objectives, data routing, scheduling,
  environment adapters, and serving-engine tuning stay independently reviewable.
- Do not change the learning problem to fix utilization. Data distribution,
  rewards, observation history, termination, and sample selection change only as
  an explicit ablation.

## Training Contracts

- Preserve the learner-facing batch contract: tensors, non-tensor metadata,
  masks, rewards, logprobs, and multimodal inputs stay row-aligned at the
  `DataProto` boundary.
- Route distributed work through the single-controller contracts — `WorkerGroup`,
  `@register`, and dispatch — not ad hoc RPC paths (Ray-centered unless a design
  says otherwise).
- Use the correct behavior-policy reference for the objective. Supervised rows
  must not reuse dummy rollout logprobs.
- Keep every queue/scheduler/rollout unit of work explicit. If an internal unit
  changes, make the aggregation back to the learner contract explicit and
  test-covered.
- Make objective-specific fields explicit tensors or metadata, never conventions
  inferred from missing values.
- Keep padding and dropped/aborted samples visible in metadata or metrics; they
  must not affect loss numerators, denominators, or advantages.

## Testing Discipline

- Exercise the real code path: materialization, routing, queue payloads,
  `DataProto` assembly, advantage/loss helpers, and trainer entry points.
- Mock only expensive boundaries (Ray, model servers, GPUs, external
  environments). Never mock the batch construction or loss semantics the learner
  consumes.
- Add fail-closed tests for silent-distortion risks: shape mismatch, missing or
  misaligned logprobs, multimodal placeholder mismatch, duplicate IDs, stale route
  metadata, unsupported objective modes.
- Name CPU-only tests `*_on_cpu.py` and mirror the package path under `tests/`.
- Treat expensive end-to-end runs as final confirmation, not the first place a
  route, schema, mask, logprob, or backward-path bug should surface — add the
  smallest smoke that reaches the same failure boundary first.

## Environment And Launchers

- Do not install, upgrade, or remove packages in a shared environment unless the
  user explicitly asks for that change.
- Before changing async RL + HPT launch profiles, queue/staleness sizing, or
  main-run log checks, consult `docs/Readme_RL.md`.
- Prefer reproducible commands and checked-in launch profiles over ad hoc shell
  state. Keep environment variables minimal and documented beside the launcher
  that needs them.
- Classify each launcher change as exactly one of: inherited upstream invariant,
  objective contract, async scheduling contract, environment adapter, or temporary
  experiment. Do not mix these in one edit.
- Do not use launcher-only changes to mask code-contract bugs, or patch code when
  a launcher setting is the whole issue.

## Editing These Instructions

Before editing this file or any agent guide it links to, read and follow
`docs/contributing/editing-agent-instructions.md`.

- Keep `AGENTS.md` under 200 lines and every rule durable.
- No hardcoded local paths, environment names, or transient experiment details.
- When adding guidance, remove or consolidate what it supersedes.
- Put area-specific operational notes in separate docs; link them only once stable.

## Acknowledgements

Adapted from the upstream `verl` and vLLM agent-instruction style.
