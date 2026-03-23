# How to Reproduce the Current LaTeX PDFs

This repo produces two compiled PDFs that correspond to the current LaTeX sources in `code/tex/`:

- `report/icma_analysis_report.pdf` (main report; `code/tex/main/main.tex`)
- `report/sensitivity_analysis.pdf` (sensitivity report; `code/tex/sensitivity/sensitivity_analysis.tex`)

Both PDFs depend on Python-generated figures/CSVs under `output/` and (for the sensitivity section) a generated TeX fragment under `code/tex/sensitivity/generated/`.

---

## 0) Install dependencies

From the repo root (`project_data_analysis/`):

```bash
pip install -r code/requirements.txt
```

---

## 1) Regenerate all Python outputs used by the TeX

From the repo root:

```bash
python code/python/analyze_icma_dataset.py
```

This regenerates:

- `output/figures/*.png`
- `output/*.csv`
- `code/tex/sensitivity/generated/regime_switch_metrics.tex` (used by the regime-switching sensitivity section)

---

## 2) Build the sensitivity report PDF

From the repo root:

```bash
cd code/tex/sensitivity
./compile.sh
```

The script runs `pdflatex` twice and writes:

- `code/tex/sensitivity/sensitivity_analysis.pdf`

To copy it into `report/`:

```bash
cp sensitivity_analysis.pdf ../../report/sensitivity_analysis.pdf
```

---

## 3) Build the main report PDF

From the repo root:

```bash
cd code/tex/main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

This writes:

- `code/tex/main/main.pdf`

To copy it into `report/`:

```bash
cp main.pdf ../../report/icma_analysis_report.pdf
```

---

## Notes

- If LaTeX sources change (anything under `code/tex/`), you can skip step 1 and re-run only the relevant compile steps.
- If data/config/model logic changes, re-run step 1, then re-compile both PDFs.
