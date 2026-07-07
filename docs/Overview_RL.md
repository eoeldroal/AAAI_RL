# Async-HPT: What This Fork Is

Upstream `verl` (RL for LLMs) plus one research line: **HPT (Hybrid
Post-Training)** as a first-class objective on the **fully-asynchronous** RL runtime
— a prompt-level RL/SFT hybrid objective, decided per prompt group, running
under overlapped rollout and training.

For where the code lives and how it runs, see `Codemap_RL.md` and
`Readme_RL.md`. This document explains why it is shaped the way it is.

## Problem Statement

Fully-async RL already overlaps rollout generation and training. HPT adds a
prompt-level hybrid objective on top: each prompt is routed, per prompt group,
to on-policy RL or supervised imitation of an expert trajectory (`tau`).

Combining prompt-level HPT with async rollout naively creates a grain mismatch:

- the learner wants **prompt-group semantics** — one route decision per prompt
- the async executor wants **trajectory-attempt-level concurrency** — many
  in-flight generations completing in arbitrary order

This fork's contribution is reconciling those two grains without changing the
learner-visible HPT contract.

## Design Contribution

- Prompt-group semantics at the learner boundary — the trainer always sees one
  route decision (RL or SFT) per prompt.
- Trajectory-attempt scheduling inside the rollouter — attempts for one prompt
  run concurrently and can complete in any order.
- Trainer-side deferred materialization — both routes converge to a `DataProto`
  row only at consumption time, not at generation time.
- A branch-blind mixed RL/SFT loss over the reconciled batch — each row
  counts once; RL-vs-SFT relative strength is set by the supervised
  pseudo-reward `β_r`, not by row counts (`DR-001`).

## Correctness Guarantees

These are implementation guarantees enforced by the contract tests in
`tests/special_RL/`, not formal theorems:

- **G1.** No partial prompt-group learner sample is emitted.
- **G2.** RL and SFT rows share one `DataProto` training contract.
- **G3.** Old-logprob semantics stay rollout-anchored for RL rows by default (the entry-anchor decoupled path is a flag-off option; see `DR-004`).
- **G4.** Partial rollout recovery preserves token/logprob alignment.
- **G5.** SFT rows are excluded from rollout correction/rejection semantics.

## Evaluation Axes

**Contract correctness** — route correctness, mixed batch assembly correctness,
loss correctness, old-logprob anchor correctness, partial recovery correctness.

**System efficiency** — completed prompt groups per unit time, drop rate before
queue put, completed-budget pressure, trainer-visible sample throughput.

**Learning composition** — `hpt/offline_data_ratio`, `hpt/p_success_zero_ratio`,
`hpt/num_sft`, `hpt/num_rl_groups`, `hpt/missing_tau_count`.

## Where To Go Next

- Rules for working in this repo: `../AGENTS.md`.
- Code layout, control flow, and where a run breaks: `Codemap_RL.md`.
- How to launch and size a run: `Readme_RL.md`.
- Queue/staleness/HPT budget sizing: `AsyncBudget_RL.md`.
- Lint, profiling, and perf triage: `Debug_RL.md`.
- Ablation design and analysis procedure: `Ablation_RL.md`.
- Run pathology case studies and improvements: `Improvement_RL.md`.
- Decisions' rationale and theory: design records `DR-001` to `DR-005`.
- Paper draft and slide generator: `papers_RL/Draft.tex`, `papers_RL/make_asynchpt_slides.py`.
