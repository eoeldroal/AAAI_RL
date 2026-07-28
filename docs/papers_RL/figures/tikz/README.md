# StreamWeave figures -- TikZ sources

> **Submission status (2026-07-28).** The timeline in `figure1.tex` is
> retired and is not included in the paper. It duplicated source routing in
> the data-state figure, while its remaining content restated inherited
> asynchronous scheduling. The data-state source `figure2.tex` is paper
> Figure 1; do not restore the timeline without a new, non-overlapping claim.

Files
  streamweave-figures.sty   colours (CMYK), 9/9.5/10 pt size macros, tikz styles, drawn glyphs
  figure1.tex               retired timeline source (not included)
  figure2.tex               current paper Figure 1 (no preamble)
  build-figure1.tex         standalone wrapper -> figure1.pdf
  build-figure2.tex         standalone wrapper -> figure2.pdf
  FIGURE3_4_HANDOFF.md      locked data, claim, scope, style, and QA contract for Figure 3--4

Figure 3--4 are being migrated from the frozen Matplotlib assets to standalone
TikZ PDFs. Read `FIGURE3_4_HANDOFF.md` before adding their source files. Their
data and comparison scopes are locked; the migration changes rendering, not
analysis.

For an agent without repository access, use `figure3_4_tikz_handoff.zip`. It
contains plot-ready CSVs, a numeric manifest, the shared Figure 1--2 style,
reference images, and the calculation sources. Regenerate the package with:

    python3 scripts/export_figure3_4_handoff.py
    zip -r -q -FS figure3_4_tikz_handoff.zip figure3_4_handoff_bundle

Build cropped PDFs

    pdflatex build-figure1.tex
    pdflatex build-figure2.tex

These commands produce `build-figure1.pdf` and `build-figure2.pdf`. Rename the
submission copies to `figure1.pdf` and `figure2.pdf`.

The standalone class with `border=0pt` makes MediaBox = CropBox = the artwork
box, so no trim/clip is needed at inclusion time. The generated figures are
approximately 6.98 inches wide and fit the AAAI 7.0-inch text width at natural
size.

Do not scale them: at 1:1 the smallest type is 9 pt.

    \begin{figure*}[t]
      \centering
      \includegraphics{figure2}
      \caption{...}
      \label{fig:data-state-boundaries}
    \end{figure*}

Keep the TikZ files as the editable source of record. The AAAI submission must
use the pre-generated PDFs through `\includegraphics`; do not `\input` the
TikZ source directly from the paper.

Compliance notes
  Fonts      no glyph outside the text font: the marks (cross, check, arrows)
             are drawn paths, so nothing falls back to a symbol font and no
             Type 3 / Identity-H / CID font can appear.
  Colours    every colour is declared in the cmyk model -> DeviceCMYK output.
  Line width 0.5 pt minimum, 1 pt maximum, three steps only (0.5 / 0.75 / 1).
  Type       9 pt minimum, 9.5 pt key labels, 10 pt stage titles.
  Greyscale  source is encoded by fill (solid = policy, hatched = expert) and
             role by line pattern, so nothing depends on colour alone.

Colour roles
  figowned   deep teal   the three boundaries this paper owns
  figdecide  deep amber  the source decision, and only that
  figsub     grey        inherited substrate labels
  figink     near-black  data glyphs
