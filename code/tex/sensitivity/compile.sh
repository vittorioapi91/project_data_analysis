#!/usr/bin/env bash
# Build sensitivity_analysis.pdf (must run from this directory so \input paths resolve).
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${PDFLATEX:-}" ]]; then
  if command -v pdflatex >/dev/null 2>&1; then
    PDFLATEX=pdflatex
  elif [[ -x /usr/local/texlive/2026/bin/universal-darwin/pdflatex ]]; then
    PDFLATEX=/usr/local/texlive/2026/bin/universal-darwin/pdflatex
  else
    echo "pdflatex not found; set PDFLATEX to the binary path." >&2
    exit 1
  fi
fi
"$PDFLATEX" -interaction=nonstopmode sensitivity_analysis.tex
"$PDFLATEX" -interaction=nonstopmode sensitivity_analysis.tex
echo "OK: $(pwd)/sensitivity_analysis.pdf (run twice for stable TOC.)"
