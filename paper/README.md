# Paper

`main.tex` (IEEE journal format, `IEEEtran`) + `references.bib`. Every number in it comes from
`project/experiments/runs/report_20261118_194054/tables.md` (5-seed results) or from the single-seed rounds
recorded in `PROJECT_STATUS.md` §4 (modality ablation, routing / uncertainty analysis, sensitivity, the MUAD
protocol table).

## Figures

The figures are the PNGs written by `experiments/make_report.py`. Copy them in before compiling (on the
server, from `project/`):

```bash
mkdir -p ../paper/figures
cp "$(ls -td experiments/runs/report_*/ | head -1)"*.png ../paper/figures/
```

Used: `robustness_masking_sn.png`, `robustness_noise_sn.png`, `robustness_masking_tt.png`, `routing_sn.png`,
`sensitivity_sn.png`. A missing figure compiles as a labelled box.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload the `paper/` folder (with `figures/`) to Overleaf. Packages: IEEEtran, amsmath, bm, graphicx,
booktabs, multirow, algorithm, algpseudocode, tikz, hyperref, cite (all in TeX Live / Overleaf).

Author names, affiliations and the code-availability statement are placeholders.
