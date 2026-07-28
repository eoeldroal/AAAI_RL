# StreamWeave figures

`figure2.pdf` through `figure4.pdf` are the rendered assets included by
`../streamweave_v2.tex`; they appear in the paper as Figures 1 through 3.
`figure1.pdf` is the retired timeline asset and is retained only as source
history.

Editable TikZ sources, the shared style, and standalone build wrappers live in
`tikz/`. To regenerate a figure, run its `build-figureN.tex` from that
directory and copy the resulting PDF here as `figureN.pdf`.

The retired timeline duplicated source routing in the current data-state
figure, while its remaining content restated inherited asynchronous
scheduling. Do not restore it without a new, non-overlapping claim.

Source assets `figure2.pdf` and `figure4.pdf` are designed for AAAI full-width
placement and are included at their natural size. Source asset `figure3.pdf`
is designed for one-column placement
and is included at `width=\columnwidth`; this absorbs a sub-point difference
between its nominal canvas and rendered CropBox without materially changing
its 9 pt type.
