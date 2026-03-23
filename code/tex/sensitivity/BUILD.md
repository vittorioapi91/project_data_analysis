# Building `sensitivity_analysis.pdf`

1. **Generate figures and metrics** (from repo root or `code/python/`):

   ```bash
   python code/python/analyze_icma_dataset.py
   ```

   This writes `output/figures/CDX_IG_regime_rv_vix.png` and `generated/regime_switch_metrics.tex`.

2. **Compile LaTeX** — **always from this folder** (`code/tex/sensitivity/`), or use:

   ```bash
   ./compile.sh
   ```

   Run **twice** so the table of contents and cross-references settle (the script does two passes).

If you open an old `report/sensitivity_analysis.pdf`, the TOC will not list *Regime-Switching Fair Value* until you rebuild after pulling the latest `.tex` sources.
