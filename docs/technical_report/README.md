# UE5M3 FP4 Pretraining Report

This archive is a self-contained Overleaf project for the technical report.

The manuscript sources and assets were imported from the Overleaf export
downloaded on 2026-09-14. The compiled report is available in `main.pdf`.

## Local compilation

From this directory, with PDFLaTeX, BibTeX, and `latexmk` installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape -outdir=build main.tex
```

This writes `build/main.pdf` and keeps auxiliary files out of the source tree.
To refresh the checked-in PDF after a successful build, copy `build/main.pdf`
to `main.pdf`.

## Import

1. In Overleaf, select **New Project > Upload Project**.
2. Upload `ue5m3_fp4_training_overleaf.zip`.
3. Set the main document to `main.tex` if Overleaf does not select it
   automatically.
4. Use the **pdfLaTeX** compiler. Overleaf runs BibTeX automatically through
   its normal `latexmk` build.

The archive includes the bibliography, local report style, Graphcore symbol,
and all vector figures referenced by `main.tex`. It does not require shell
escape, Python, W&B access, or any files outside the Overleaf project.
