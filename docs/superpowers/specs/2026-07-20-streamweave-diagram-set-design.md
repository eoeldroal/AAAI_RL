# StreamWeave Diagram Set Design

## Goal

Replace the symbol-heavy overview with a self-explanatory paper figure and add an end-to-end
training-pipeline figure. The public figure set should explain the research problem and the
implemented algorithm-system architecture without relying on charts or implementation trivia.

## Figure 1: Composition Gap and StreamWeave

Figure 1 uses two panels with the same visual vocabulary.

- **(a) Naive composition** begins with a labeled prompt group whose rollout attempts finish at
  different times. It exposes the two naive choices as two explicit paths: waiting for the complete
  group restores a generation barrier and leaves the learner idle; consuming arrivals before group
  completion lets a partial group drive source selection. Every consequence has a noun label and an
  icon; no paragraph or unexplained `R/E` symbol is allowed.
- **(b) StreamWeave** shows independent rollout attempts entering a per-group reconstruction tray.
  Only a complete group reaches source selection. The selected policy-rollout stack or expert
  trajectory enters a source-aware learner while other groups continue generating. This makes the
  central judgment visible: complete context remains local and does not become a global barrier.

Visible text is limited to component identity and state: `Prompt group`, `Independent rollouts`,
`Complete group`, `Source selection`, `Policy rollouts`, `Expert trajectory`, `Learner`, `Waiting`,
`Partial group`, `Idle`, and `Wrong source`.

## Figure 2: End-to-End Training Pipeline

Figure 2 is a three-plane architecture diagram.

- **Rollout plane:** prompt groups are expanded into attempts, scheduled across a rollouter pool,
  scored, and reconstructed into complete groups.
- **Composition plane:** a complete-group selector chooses policy rollouts or retrieves a matching
  expert trajectory. The selected payload carries group, source, and rollout-policy context into a
  bounded mixed stream.
- **Learning plane:** batches are assembled, source-specific operator domains are applied, declared
  composition produces one optimizer update, and the refreshed model version returns to the
  rollouter through a control-plane loop.

The figure distinguishes the data plane from the control plane. Blue means policy-generated data,
orange means expert-provided data, teal means group/provenance context, dark ink means model update,
and gray dashed arrows mean parameter refresh or backpressure control.

## Scope Discipline

- Include nonblocking group reconstruction, complete-group selection, provenance-carrying mixed
  transport, source-aware learning, bounded flow, generation-learning overlap, and parameter refresh.
- Do not elevate accumulator data structures, exact queue sizes, subset-sum alignment,
  trim-and-carryover, Ray actors, or tensor field names into the public figure.
- Partial rollout and checkpoint-engine details remain Appendix material; the main figure may show a
  generic parameter-refresh loop without claiming that mechanism as StreamWeave novelty.
- Existing learning-effect and efficiency charts remain in the repository but are marked as parked
  empirical assets rather than members of the current narrative figure set.

## Output and Review

- Editable source: grouped SVG plus the existing JavaScript generator.
- Publication output: vector PDF.
- Review output: 2x PNG.
- Figure 1 and Figure 2 are designed for AAAI two-column `figure*` placement.
- Each iteration is reviewed at full resolution and after PDF rendering at paper scale.
