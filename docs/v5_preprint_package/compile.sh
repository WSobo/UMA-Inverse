#!/bin/bash
# Compile preprint.tex to PDF (run from this directory)
set -e
pdflatex -interaction=nonstopmode preprint.tex
bibtex preprint
pdflatex -interaction=nonstopmode preprint.tex
pdflatex -interaction=nonstopmode preprint.tex
echo "Done: preprint.pdf"
