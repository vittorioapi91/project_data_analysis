from __future__ import annotations

from utils import DATA_FILE, OUTPUT_DIR, ensure_dirs, get_model_param, load_data, compute_changes

from descriptive import (
    missing_value_statistics,
    save_main_observations_plot,
    save_correlation_heatmap,
    save_rolling_correlations,
    lag_analysis,
)
from fair_value_pipeline import impact_and_fair_value
from validate_config import validate_all
from plotting import (
    plot_fair_value_best_model,
    plot_regime_switch_sensitivity,
    plot_rolling_fair_value,
    plot_rolling_comparison,
)


def main() -> None:
    """Run the full analysis pipeline and write outputs/figures."""
    
    ensure_dirs()
    levels = load_data(DATA_FILE)
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
    # Fair-value + rolling plots (CSVs written inside pipeline).
    plot_payload = impact_and_fair_value(changes, levels)

    plot_fair_value_best_model(**plot_payload["fair_best_model"])  # type: ignore[arg-type]
    plot_rolling_fair_value(
        valid=plot_payload["rolling_fair_value"]["valid"],
        target=plot_payload["rolling_fair_value"]["target"],
        slug=plot_payload["rolling_fair_value"]["slug"],
        window=plot_payload["rolling_fair_value"]["window"],
    )
    plot_rolling_comparison(
        **plot_payload["rolling_fair_comparison"],
        plot_styles=get_model_param("plot_styles"),
    )
    plot_regime_switch_sensitivity(levels)
    lag_analysis(changes)

    print("Analysis complete.")
    print(f"Outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
