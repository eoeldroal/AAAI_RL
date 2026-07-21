# StreamWeave Figure 1 Draft Implementation Plan

> **For agentic workers:** Build this draft inline in the current session. Do not run repository tests; the user requested visual drafting only.

**Goal:** Produce a clean, editable academic Figure 1 that explains the naive-composition double bind and StreamWeave's attempt-level execution, complete-context decision, and source-native learning composition.

**Architecture:** Use image generation only to explore the overall silhouette. Rebuild the selected composition as a deterministic SVG so terminology, arrows, and scientific relationships remain exact, then export matching PDF and PNG derivatives.

**Tech Stack:** Built-in ImageGen, hand-authored SVG, bundled browser/image conversion tools.

## Global Constraints

- Keep `HPT`, framework names, queue internals, objective formulas, and measured results out of Figure 1.
- Use English audience-facing labels and a white, flat, vector-first academic style.
- Show at least two interleaved prompt groups so nonblocking execution is visible.
- Show one policy-generated branch and one expert-supervision branch entering one learner while retaining distinct update semantics.
- Do not run codebase or figure tests; perform visual inspection only.

### Task 1: Explore The Visual Silhouette

**Files:**
- Generate reference only; no repository file is authoritative at this stage.

- [x] Generate one text-light academic infographic reference with a compact naive-composition panel and a larger two-layer StreamWeave panel.
- [x] Inspect whether the reference makes the local group boundary and continuous learner progress visually obvious.

### Task 2: Author The Editable Figure

**Files:**
- Create: `docs/papers_RL/figures/figure1_streamweave_draft.svg`

- [x] Draw a compact left panel with the two naive choices: `wait for groups` and `reuse the stream unchanged`.
- [x] Draw a large right panel with `EXECUTE independently`, `DECIDE with complete context`, and `LEARN by source` as the primary reading path.
- [x] Encode policy signals in blue, expert supervision in orange, provenance/context in teal, and failures in coral.
- [x] Add one bottom-line takeaway: `Preserve learning boundaries without restoring execution barriers.`

### Task 3: Export And Inspect

**Files:**
- Create: `docs/papers_RL/figures/figure1_streamweave_draft.pdf`
- Create: `docs/papers_RL/figures/figure1_streamweave_draft.png`

- [x] Export the SVG without rasterizing text in the PDF.
- [x] Render a high-resolution PNG for review.
- [x] Inspect the PNG at full size and at approximate two-column paper width; correct clipping, crowded labels, weak contrast, or ambiguous arrows.
