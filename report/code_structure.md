# Project Code Structure Documentation

This document describes how the repository is structured and how the analysis pipeline flows through the main Python modules.

## High-level layout

The analysis pipeline is driven from `code/python/analyze_icma_dataset.py` and relies on supporting modules under `code/python/`. Configuration is provided via YAML under the same folder, and generated artifacts are written to `output/`.

```text
project_data_analysis/
  input/                       # XLSX input dataset(s)
  output/                      # CSV outputs + generated figures
  report/                      # Compiled PDFs + this doc
  code/
    requirements.txt           # pip install -r code/requirements.txt (from repo root)
    python/
      analyze_icma_dataset.py  # full orchestrator (entry point)
      run_fair_value_pipeline.py # optional: FairValuePipeline only (debug / VS Code launch)
      utils.py                   # YAML, IO, plotting primitives
      descriptive.py             # descriptive stats, correlations, lag analysis
      fair_value_pipeline.py     # FairValuePipeline + impact_and_fair_value
      plotting.py                # fair-value + regime sensitivity figures
      models.py                    # Static / Rolling / RegimeSwitch fair-value classes
      base_model_class.py          # FairValueConfig, OLS helpers, static CSV export
      validate_config.py           # models.yaml vs dataset columns
      models.yaml
    tex/
      main/
      sensitivity/
```

## Execution flow (runtime)

### Entry point

`code/python/analyze_icma_dataset.py` follows a compute → plot → compute → plot pattern:

```python
ensure_dirs()
levels  = load_data(DATA_FILE)
validate_all(set(levels.columns) - {"Date"})
changes = compute_changes(levels)

# Compute
missing_value_statistics(levels)

# Plot
save_main_observations_plot(levels)

# Compute + Plot
corr = save_correlation_heatmap(changes)
corr.to_csv(OUTPUT_DIR / "correlation_changes.csv")

# Compute + Plot
save_rolling_correlations(changes)

# Fair-value + rolling (CSVs + plot kwargs)
plot_payload = impact_and_fair_value(changes, levels)

# Sensitivity: regime RV/VIX + static vs regime fair levels + generated TeX metrics
plot_regime_switch_sensitivity(levels)

# Compute + Plot (inside descriptive.py)
lag_analysis(changes)
```

## Fair-value pipeline

`code/python/fair_value_pipeline.py` runs static + rolling fair-value estimation via `RollingFairValueModel` and returns plot kwargs for `analyze_icma_dataset.py`.

- **`impact_and_fair_value(changes, levels)`** — `FairValuePipeline().run(...)`.
  - Returns `fair_best_model`, `rolling_fair_value`, `rolling_fair_comparison` for `plotting.py`.
  - CSV exports and diagnostics are written inside `RollingFairValueModel.evaluate_model` / `FairValueCandidateModel` (`base_model_class.py`, `models.py`).

## Modules and responsibilities

### `code/python/utils.py`

- YAML: `_load_yaml`, `get_model_param`
- IO: `load_data`, `compute_changes`, paths (`DATA_FILE`, `OUTPUT_DIR`, `FIG_DIR`)
- Plotting: `make_figure`, `make_subplots`, `style_axis`, `save_figure`, `safe_slug`
- `ensure_dirs`

### `code/python/validate_config.py`

Validates `models.yaml` against the input dataset’s columns; raises `ConfigValidationError` with a bullet list of issues. Runnable as `python validate_config.py`.

### `code/python/descriptive.py`

- `save_main_observations_plot`, `save_correlation_heatmap`, `save_rolling_correlations`
- `missing_value_statistics`, `lag_analysis`
- Internal: `rolling_corr_series`

### `code/python/plotting.py`

Fair-value and sensitivity figures (best model, rolling, comparison overlay, regime RV/VIX, regime vs static fair value); writes `code/tex/sensitivity/generated/regime_switch_metrics.tex`.

### `code/python/models.py`

`StaticFairValueModel`, `RollingFairValueModel`, `StaticFairValueModelRegimeSwitch`.

## Configuration: `code/python/models.yaml`

- `target`, `train_ratio`, `min_obs`, `rolling_window`
- `candidate_models`, optional `plot_styles`

## Output artifacts

Pipeline writes CSVs under `output/` and figures under `output/figures/` (correlation matrix, missing-value stats, fair-value series and diagnostics, etc.).

## How to extend

- Change targets/features: `models.yaml`.
- New plots: add functions in `plotting.py` (and call them from `analyze_icma_dataset.py`), or extend `FairValuePipeline` / models if new computed series are required.
