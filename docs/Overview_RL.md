# Async-HPT: What This Fork Is

_Last updated: 2026-07-23_

> **Document scope.** This is a code-facing implementation overview, not the source of
> paper positioning or public terminology. For paper writing or review, start at
> `papers_RL/README.md`; the canonical manuscript is
> `papers_RL/Full_Paper_Draft_ko.md`. Do not infer the paper's novelty from the internal
> `Async-HPT`, `HPT`, or historical `branch-blind` labels used in this fork.
>
> **Paper-facing status (2026-07-23).** `Async-HPT` is the implementation and run-lineage
> name used inside this fork. The public paper method is **StreamWeave**, and its current
> thesis, claims, terminology, and evidence are owned by
> `papers_RL/Full_Paper_Draft_ko.md`. This document remains the codebase overview.

Upstream `verl` (RL for LLMs) plus one research implementation: a group-conditioned
policy/expert learner on the **fully-asynchronous** RL runtime. Internally, the code
calls this line **HPT (Hybrid Post-Training)** or `async-HPT`: a complete prompt group
selects policy rollouts or a matched expert trajectory, while rollout generation and
model training remain overlapped.

For where the code lives and how it runs, see `Codemap_RL.md` and
`Readme_RL.md`. This document explains why it is shaped the way it is.

## Problem Statement

Fully-async RL already overlaps rollout generation and training. Group-conditioned
policy/expert learning adds two dependencies that the homogeneous rollout stream does
not consume: a complete group is needed to select the source, and the selected source
determines the signal, reference, and correction conditions used to construct learner
inputs.

Combining these dependencies with trajectory-level async rollout creates two
implementation boundaries:

- the executor should retain **trajectory-attempt-level concurrency**, while the
  source decision consumes **complete-prompt-group context**
- source identity must survive transport until learner-input construction, but it
  should not split the primary loss, reducer, trainer, or optimizer into parallel paths

The code therefore localizes complete-group context at source selection and source
identity at learner-input construction. After those dependencies are consumed, the
existing asynchronous flow and one shared primary update resume. This is the
implementation counterpart of the StreamWeave decomposition; the exact public claim
boundary remains in the paper draft.

## Implementation Design

- **Independent attempts, local reconstruction.** Attempts for one prompt run
  independently and the complete group is reconstructed only before source selection.
- **Source before transport.** The route is fixed before a prompt-group record enters
  the queue, so queue arrival order cannot make the source decision.
- **Learner-side materialization.** The trainer converts the source-resolved record
  into source-conditioned advantages, references, masks, and correction conditions.
- **Shared primary update.** Policy and expert samples pass through the same
  `ppo_loss` entry and source-independent reduction. Their effective mixture is
  determined by routing, row and token volume, `β_r`, and the declared reducer rather
  than a source-specific optimizer path (`DR-001`).

## Correctness Guarantees

These are implementation guarantees enforced by the contract tests in
`tests/special_RL/`, not formal theorems:

- **G1.** No partial prompt-group learner sample is emitted.
- **G2.** RL and SFT rows share one `DataProto` training contract.
- **G3.** The current paper main uses a learner-entry proximal anchor for RL
  rows and separately applies rollout-to-entry token-level IS. Rollout anchoring
  remains a supported comparison setting (`DR-004`).
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
- Current paper thesis, evidence, and editing constitution:
  `papers_RL/Full_Paper_Draft_ko.md`.
- Remaining paper tasks and asset plan: `../../paper-plan.md`.
- Code layout, control flow, and where a run breaks: `Codemap_RL.md`.
- How to launch and size a run: `Readme_RL.md`.
- Queue/staleness/HPT budget sizing: `AsyncBudget_RL.md`.
- Lint, profiling, and perf triage: `Debug_RL.md`.
- Ablation design and analysis procedure: `Ablation_RL.md`.
- Run pathology case studies and improvements: `Improvement_RL.md`.
- Decisions' rationale and theory: design records `DR-001` to `DR-005`.
- Legacy implementation Appendix and slide generator: `papers_RL/Draft.tex`,
  `papers_RL/make_asynchpt_slides.py`.
