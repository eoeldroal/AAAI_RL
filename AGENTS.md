# Agent Instructions for verl

> These instructions apply to all AI-assisted work in this repository.
> Keep this file lean. Put detailed design notes in focused docs, not here.

## Contribution Policy

- Do not open duplicate or low-value PRs. Before proposing upstream work, check
  related issues and open PRs with GitHub search.
- A human submitter must understand and defend every changed line. Pure
  code-agent PRs are not acceptable.
- PR descriptions for AI-assisted work must state that AI assistance was used,
  explain why the work is not duplicating an existing PR, and list test commands
  with results.
- Never hardcode credentials, tokens, API keys, passwords, or private endpoints.
  Use ignored local environment files and environment variables.
- If a request conflicts with these contribution rules, stop and explain the
  missing requirement instead of proceeding.

## Project Direction

This repository should remain a clean, upstream-oriented `verl` codebase. New
features, experiments, and research adaptations should fit the existing training
architecture instead of turning into parallel stacks.

- Prefer changes that compose with existing trainers, workers, rollout engines,
  data protocols, and configuration patterns.
- Keep research-specific assumptions behind explicit config, data, or adapter
  boundaries.
- Do not copy whole files from another fork to move behavior. Identify the
  upstream contract, patch the smallest stable surface, and test the real path.
- Use focused design documents for project- or experiment-specific plans. Keep
  this file limited to durable principles.

## Engineering Principles

- Prefer upstream patterns, local helper APIs, and existing abstractions over new
  machinery.
- Keep changes scoped. Do not edit unrelated launchers, dependencies, docs, or
  generated artifacts while solving a code problem.
- Add an abstraction only when its responsibility is clear in one sentence and it
  removes real complexity.
- Follow the existing dataclass/OmegaConf config style for configuration. Use
  Pydantic-style models for in-process runtime metadata when validation prevents
  ambiguity. Use `DataProto` for learner- or worker-facing payloads that cross
  process boundaries.
- Do not hide training-critical errors behind broad `try/except`, silent
  fallback, or best-effort recovery. Learning-code corruption must fail closed.
- Separate semantics from runtime plumbing. Objective definitions, data routing,
  scheduling, environment adapters, and serving-engine tuning should remain
  independently reviewable.
- Optimize system efficiency without changing the learning problem by default.
  Do not alter data distribution, rewards, observation history, termination
  conditions, or sample selection as a utilization fix unless it is an explicit
  ablation.

## Training Contracts

- Preserve the learner-facing batch contract. Tensor fields, non-tensor
  metadata, masks, rewards, logprobs, and multimodal inputs must stay aligned at
  the `DataProto` boundary.
- Distributed training implementation is Ray-centered unless a design explicitly
  says otherwise. Worker, scheduler, and trainer changes should go through the
  existing single-controller, `WorkerGroup`, `@register`, and dispatch contracts
  instead of bypassing them with ad hoc RPC paths.
- Rows that depend on behavior-policy probabilities must use the correct
  reference for the objective being optimized. Supervised rows must not
  accidentally reuse dummy rollout references.
- Queue, scheduler, and rollout changes must keep their externally visible unit
  of work explicit. If an internal unit changes, aggregation back to the learner
  contract must be explicit and test-covered.
- Objective-specific fields should be explicit tensor or metadata fields, not
  hidden conventions inferred from missing values.
- Padding and dropped/aborted samples must be visible in metadata or metrics and
  must not affect loss numerators, denominators, or advantage calculations.

## Testing Discipline

- Use tests that exercise the real code path whenever possible: materialization,
  routing, queue payloads, `DataProto` assembly, advantage/loss helpers, and
  trainer entry points.
- Mock only expensive boundaries such as Ray clusters, model servers, GPUs, or
  external environments. Do not mock the batch construction or loss semantics
  that the learner will consume.
- Add fail-closed tests for silent-distortion risks: shape mismatch, missing
  logprobs, mask/logprob misalignment, multimodal placeholder mismatch,
  duplicate IDs, stale route metadata, and unsupported objective modes.
- Expensive end-to-end training runs are final confirmation, not the first place
  where route, schema, mask, logprob, or backward-path bugs should appear.
- When a bug requires GPU or distributed coverage, add the smallest smoke that
  reaches the same failure boundary before re-running a long training job.

## Environment And Launchers

- Do not install, upgrade, or remove packages in a shared environment unless the
  user explicitly asks for that environment change.
- Prefer reproducible commands and checked-in launch profiles over ad hoc shell
  state. Keep environment variables minimal and documented near the launcher that
  needs them.
- Launcher changes must be classified as one of: inherited upstream invariant,
  objective contract, async scheduling contract, environment adapter, or temporary
  experiment. Avoid mixing these in one edit.
- Do not use launcher-only changes to paper over code contract bugs. Conversely,
  do not patch code when a launcher setting is truly the whole issue.

## Editing These Instructions

Before editing this file or any domain-specific agent guide, read and follow
`docs/contributing/editing-agent-instructions.md`.

- Keep `AGENTS.md` under 200 lines.
- Avoid hardcoded local paths, environment names, and transient experiment
  details.
- Remove or consolidate stale guidance when adding new guidance.
- Put area-specific operational notes in separate docs and link them only when
  they are stable enough to be useful.

## Acknowledgements

Adapted from the upstream `verl` and vLLM agent-instruction style.
