# StreamWeave figures -- TikZ sources

Files
  streamweave-figures.sty   colours (CMYK), 9/9.5/10 pt size macros, tikz styles, drawn glyphs
  figure1.tex               retired timeline source (not included)
  figure2.tex               current paper Figure 1 (no preamble)
  build-figure1.tex         standalone wrapper -> figure1.pdf
  build-figure2.tex         standalone wrapper -> figure2.pdf

The timeline source is preserved for provenance but was removed from the
paper on 2026-07-28: it duplicated source routing in the data-state figure,
and its remaining timeline restated inherited asynchronous scheduling. Do
not restore it without a new, non-overlapping claim.

Build cropped PDFs

    pdflatex build-figure1.tex
    pdflatex build-figure2.tex

The standalone class with border=0pt makes MediaBox = CropBox = the artwork
box, so no trim/clip is needed at inclusion time:

    figure1.pdf   501.98 x 168.32 pt   =  6.972 x 2.338 in
    figure2.pdf   501.98 x 216.63 pt   =  6.972 x 3.009 in

AAAI full text width is 3.3 + 0.375 + 3.3 = 6.975 in, so the current
data-state asset (`figure2.pdf`) drops in at natural size. The retired
timeline (`figure1.pdf`) is retained only for provenance.

    \begin{figure*}[t]
      \centering
      \includegraphics{figure2}
      \caption{...}
      \label{fig:data-state-boundaries}
    \end{figure*}

Alternatively \input{figure2} directly after \usepackage{streamweave-figures};
the figure then picks up the paper's own Times and no external PDF is needed.

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


--------------------------------------------------------------------------
# Figures 3 and 4 -- added

Files
  figure3.tex               Learning dynamics (single panel)
  figure4.tex               Execution efficiency (three panels)
  build-figure3.tex         standalone wrapper -> figure3.pdf
  build-figure4.tex         standalone wrapper -> figure4.pdf
  streamweave-figures.sty   extended below \endinput's old position with an
                            ADDITIVE block only; Figure 1-2 output is unchanged.

Geometry
  Both use the same 1 unit = 1 px = 0.498 pt grid as Figures 1-2. Figure 3
  carries a 477 x 296 px one-column bounding box; Figure 4 retains the
  1008 x 320 px full-width bounding box:

    figure3.pdf   nominal canvas: 237.55 x 147.41 pt
                  rendered CropBox: 239.15 x 146.86 pt
    figure4.pdf   501.98 x 159.36 pt  =  6.972 x 2.213 in

  Use a one-column `figure` and `width=\columnwidth` for Figure 3. The
  resulting sub-percent fit absorbs text and stroke extents beyond the nominal
  canvas while preserving the intended 9 pt appearance. Use a full-width
  `figure*` at natural size for Figure 4.

Type      9 pt tick/label, 9.5 pt axis titles and key labels, 10 pt panel
          titles -- the same \figsm / \figmd / \figlg macros.
Line      0.5 pt grid and raw observations, 0.75 pt axis spines and ticks,
          1 pt smoothed trends and the trainer/rollouter rule.
Colour    figowned teal = StreamWeave (solid, filled marker)
          figsub grey   = synchronous / expert-off control (dashed, open marker)
          figdecide amber = expert routing, and its own right-hand axis
          figownedlt / figdecidelt / fign400 = raw observations only
          Every series is separated by line pattern AND marker fill, so the
          plates survive greyscale conversion.

## Data manifest

Figure 3  -- data/figure3_quality.csv, data/figure3_routing.csv
  x            progress_percent                      0-100, = 100*cycle/160
  left y       streamweave_raw / expert_off_raw      thin observation
               streamweave_smooth_w3 /
               expert_off_smooth_w3                  thick trend, centered
                                                     mean width 3, raw endpoints
  right y      expert_routing_raw_percent            thin observation
               expert_routing_weighted_smooth_w7_percent
                                                     thick trend, centered
                                                     groups-weighted mean width 7
  annotated    nothing; every reported scalar (early / late window means and
               the late routing rate) is owned by the caption and body text.
  no transformation is applied in the .tex beyond the affine map to px.

Figure 4 (a) -- data/figure4_heatmap_sync.csv (287 rows),
                data/figure4_heatmap_streamweave.csv (974 rows)
  one \heatcell per (row, GPU) at gpu_<i>_sm_active_percent, colour
  figowned!p!figbg with p = round(value); cells at 0 are left as paper.
  Columns are equal width in x_left_percent order -- no binning, no smoothing.
  Each run is normalised to its own 0-100% history; the two rows of the panel
  are NOT a shared clock.

Figure 4 (b) -- data/figure4_active_gpu_coverage.csv
  x  at_least_k_active_gpus 1..8
  y  synchronous_coverage_percent / streamweave_coverage_percent, axis 60-100

Figure 4 (c) -- data/figure4_cumulative_sync.csv,
                data/figure4_cumulative_streamweave.csv,
                data/figure4_cumulative_fill_grid.csv (fill only)
  x  wall_clock_min, 0 to the common horizon 79.6698 min
  y  relative_cumulative_work, normalised to the synchronous endpoint
  endpoints labelled 1.00x and 1.67x and nothing else.

Withheld by construction: absolute cycles, total training groups, the full
StreamWeave run length, and the 1.64x full-history throughput (Table 2 owns it).

## QA still to run locally

  pdflatex -interaction=nonstopmode -halt-on-error build-figure3.tex
  pdflatex -interaction=nonstopmode -halt-on-error build-figure4.tex

  [x] Figure 3 MediaBox = CropBox = 239.15 x 146.86 pt; include at column width
  [ ] Figure 4 MediaBox = CropBox = 501.98 x 159.36 pt
  [ ] pdffonts: Times embedded, no Type 3, no CID
  [ ] all colour operators DeviceCMYK
  [ ] smallest visible type measures 9 pt at natural size
  [ ] rotated axis titles read bottom-to-top (the picture uses y=-0.498pt;
      if a \node[rotate=90] comes out mirrored, switch it to rotate=-90)
  [ ] greyscale print: teal/grey/amber still separable by dash and marker
  [ ] figure4.tex is ~8.7k \heatcell calls -- expect a slow first run
