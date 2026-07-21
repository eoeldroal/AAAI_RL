# StreamWeave Diagram Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline and review each rendered figure before continuing.

**Goal:** Produce a self-explanatory Figure 1 and an editable end-to-end StreamWeave training-pipeline Figure 2.

**Architecture:** Extend the existing code-native SVG generator with two narrative diagrams that share one visual grammar. Preserve data charts as parked assets, export every active diagram to SVG/PDF/PNG, and use rendered PDF inspection rather than software tests.

**Tech Stack:** CommonJS, deterministic SVG, Playwright/Chrome export, Poppler PDF rendering.

## Global Constraints

- Do not run the repository test suite.
- Keep visible prose out of figures; use short component labels only.
- Do not include framework-specific queue sizes, Ray names, trim-and-carryover, or tensor fields.
- Preserve blue rollout, orange expert, teal context, gray control, and dark update semantics.
- Keep all source files editable and regeneration deterministic.

---

### Task 1: Replace Figure 1

**Files:**
- Modify: `docs/papers_RL/figures/src/generate_paper_figures.cjs`
- Regenerate: `docs/papers_RL/figures/figure1_streamweave_overview.svg`
- Export: `docs/papers_RL/figures/figure1_streamweave_overview.pdf`
- Export: `docs/papers_RL/figures/figure1_streamweave_overview.png`

- [ ] Replace the abstract failure symbols with explicit `Wait for group` and `Use partial group` paths.
- [ ] Add recognizable prompt-group, rollout, learner-idle, selector, rollout-stack, and expert-document forms.
- [ ] Show a previously completed group learning while another group is generating in panel (b).
- [ ] Render SVG/PNG and inspect component identity without consulting the caption.
- [ ] Render the PDF at paper scale and revise clipping, density, and label size.

### Task 2: Add the End-to-End Pipeline

**Files:**
- Modify: `docs/papers_RL/figures/src/generate_paper_figures.cjs`
- Modify: `docs/papers_RL/figures/src/export_paper_figures.cjs`
- Create: `docs/papers_RL/figures/figure2_training_pipeline.svg`
- Create: `docs/papers_RL/figures/figure2_training_pipeline.pdf`
- Create: `docs/papers_RL/figures/figure2_training_pipeline.png`

- [ ] Draw rollout, composition, and learning planes with distinct backgrounds.
- [ ] Connect prompt groups, attempt scheduling, rollouter workers, scoring, and group reconstruction.
- [ ] Connect complete-group source selection to policy-rollout and expert-trajectory payloads.
- [ ] Carry group/source/policy-version tags through a bounded mixed stream.
- [ ] Connect batch assembly, source-aware operator paths, declared composition, optimizer, and model update.
- [ ] Add a separate parameter-refresh control loop and visible generation-learning overlap.
- [ ] Render SVG/PNG and inspect the data path from left to right.
- [ ] Render the PDF at paper scale and revise clipping, density, and label size.

### Task 3: Curate the Figure Set

**Files:**
- Modify: `docs/papers_RL/figures/README.md`

- [ ] Mark Figure 1 and the training pipeline as the current narrative figures.
- [ ] Mark learning-effect and efficiency charts as parked empirical assets.
- [ ] Record recommended `figure*` placement and caption responsibilities.
- [ ] List the exact regeneration command and final output dimensions.

### Task 4: Final Visual Review

- [ ] Inspect both active PNGs together for a shared visual language.
- [ ] Inspect both PDFs after Poppler rendering for font substitution and crop safety.
- [ ] Confirm that every visible noun maps to a real stage in `Codemap_RL.md` and the implementation.
- [ ] Confirm that inherited async mechanisms are not visually presented as standalone novelty.
