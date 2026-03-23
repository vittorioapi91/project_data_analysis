from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from statsmodels.tsa.stattools import acf, ccf, pacf

from utils import OUTPUT_DIR, get_model_param, make_figure, make_subplots, save_figure, style_axis


def _integer_lag_axis(ax) -> None:
    """Lag axes are discrete days; avoid float tick labels (e.g. 2.5, 10.0)."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{int(round(x))}"))


def save_main_observations_plot(df: pd.DataFrame) -> None:
    """Save a normalized levels plot for key series."""
    cols = ["S&P", "ESTX600", "CDX IG", "CDX HY", "UST10", "VIX", "CO"]
    clean = df[["Date"] + cols].dropna()
    norm = clean.copy()
    for c in cols:
        base = norm[c].iloc[0]
        norm[c] = 100.0 * norm[c] / base

    fig, ax = make_figure((11, 6))
    for c in cols:
        ax.plot(norm["Date"], norm[c], label=c, linewidth=1.5)
    style_axis(ax, title="Normalized level series (base=100)",
               xlabel="Date", ylabel="Index (base 100)")
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, "normalized_series.png", dpi=160)


def save_correlation_heatmap(change_df: pd.DataFrame) -> pd.DataFrame:
    """Save and return a correlation heatmap of daily changes."""
    cols = [c for c in change_df.columns if c != "Date"]
    corr = change_df[cols].corr()

    fig, ax = make_figure((12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    style_axis(ax, title="Correlation matrix of daily changes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    save_figure(fig, "corr_heatmap_changes.png", dpi=180)
    return corr


def rolling_corr_series(
    change_df: pd.DataFrame, x_col: str, y_col: str, window: int = 60
) -> pd.Series:
    """Compute rolling correlation between two change series."""
    d = change_df[[x_col, y_col]].dropna()
    r = d[x_col].rolling(window).corr(d[y_col])
    r.index = d.index
    out = pd.Series(index=change_df.index, dtype=float)
    out.loc[r.index] = r
    return out


def save_rolling_correlations(change_df: pd.DataFrame) -> None:
    """Save rolling 60-day correlation plots for predefined driver pairs."""
    breakdown_pairs = [
        ("S&P", "UST10"),
        ("CDX IG", "UST10"),
        ("CDX IG", "S&P"),
    ]

    fig, axes = make_subplots(
        nrows=len(breakdown_pairs), ncols=1, figsize=(11, 8), sharex=True
    )

    for ax, (a, b) in zip(axes, breakdown_pairs):
        r = rolling_corr_series(change_df, a, b, window=60)
        ax.plot(change_df["Date"], r, linewidth=1.3)
        ax.fill_between(
            change_df["Date"], r, 0,
            where=r > 0, alpha=0.25, color="steelblue", interpolate=True,
        )
        ax.fill_between(
            change_df["Date"], r, 0,
            where=r < 0, alpha=0.25, color="coral", interpolate=True,
        )
        style_axis(ax, title=f"{a} vs {b}", ylabel="Correlation", zero_line=True)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Rolling 60-day correlations — breakdown pairs", fontsize=13, y=0.99)
    save_figure(fig, "rolling_correlations.png", dpi=160)


def missing_value_statistics(level_df: pd.DataFrame) -> None:
    """Export a CSV with missing-value counts and percentages per series."""
    total = len(level_df)
    rows: list[dict[str, object]] = []
    for col in level_df.columns:
        if col == "Date":
            continue
        n_miss = int(level_df[col].isna().sum())
        rows.append({
            "series": col,
            "n_total": total,
            "n_missing": n_miss,
            "n_available": total - n_miss,
            "pct_missing": round(n_miss / total * 100, 1),
        })
    df_out = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    df_out.to_csv(OUTPUT_DIR / "missing_value_statistics.csv", index=False)


def lag_analysis(change_df: pd.DataFrame, max_lag: int = 20) -> None:
    """Compute and plot cross-ACF and PACF of the configured target variable's
    daily changes against other variables. Uses statsmodels for all computations."""
    target = get_model_param("target")
    slug = target.replace(" ", "_").replace("&", "")
    all_cols = [c for c in change_df.columns if c not in ("Date", target)]
    drivers = [c for c in all_cols if change_df[c].notna().sum() > max_lag + 10]
    n_obs = len(change_df[target].dropna())
    conf_bound = 1.96 / np.sqrt(n_obs)

    cdx_clean = change_df[target].dropna().to_numpy()
    acf_vals = acf(cdx_clean, nlags=max_lag, fft=True)
    pacf_vals = pacf(cdx_clean, nlags=max_lag, method="ywm")

    fig_own, axes_own = make_subplots(1, 2, figsize=(13, 4.5))
    lags = np.arange(max_lag + 1)

    axes_own[0].bar(lags, acf_vals, width=0.4, color="#1f77b4", zorder=3)
    axes_own[0].axhline(conf_bound, ls="--", lw=0.8, color="grey")
    axes_own[0].axhline(-conf_bound, ls="--", lw=0.8, color="grey")
    axes_own[0].fill_between(lags, -conf_bound, conf_bound,
                             color="lightblue", alpha=0.3, zorder=2)
    style_axis(axes_own[0], title=f"ACF of $\\Delta$ {target}",
               xlabel="Lag (days)", ylabel="Autocorrelation")
    _integer_lag_axis(axes_own[0])

    axes_own[1].bar(lags, pacf_vals, width=0.4, color="#ff7f0e", zorder=3)
    axes_own[1].axhline(conf_bound, ls="--", lw=0.8, color="grey")
    axes_own[1].axhline(-conf_bound, ls="--", lw=0.8, color="grey")
    axes_own[1].fill_between(lags, -conf_bound, conf_bound,
                             color="navajowhite", alpha=0.3, zorder=2)
    style_axis(axes_own[1], title=f"PACF of $\\Delta$ {target}",
               xlabel="Lag (days)", ylabel="Partial autocorrelation")
    _integer_lag_axis(axes_own[1])

    fig_own.suptitle(f"ACF and PACF of {target} daily changes", fontsize=12, y=1.02)
    save_figure(fig_own, f"{slug}_acf_pacf.png", dpi=170)

    n_drivers = len(drivers)
    n_cols = 3
    n_rows = (n_drivers + n_cols - 1) // n_cols
    fig_ccf, axes_ccf = make_subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes_flat = axes_ccf.flatten() if hasattr(axes_ccf, "flatten") else [axes_ccf]

    for idx, drv in enumerate(drivers):
        pair = change_df[[target, drv]].dropna()
        if len(pair) < max_lag + 10:
            continue
        y = pair[target].to_numpy()
        x = pair[drv].to_numpy()
        n_pair = len(pair)
        pair_conf = 1.96 / np.sqrt(n_pair)

        ccf_pos = ccf(x, y, adjusted=False)[:max_lag + 1]
        ccf_neg = ccf(y, x, adjusted=False)[1:max_lag + 1][::-1]
        ccf_full = np.concatenate([ccf_neg, ccf_pos])
        lag_range = np.arange(-max_lag, max_lag + 1)

        ax = axes_flat[idx]
        ax.bar(lag_range, ccf_full, width=0.5, color="#2ca02c", zorder=3)
        ax.axhline(pair_conf, ls="--", lw=0.8, color="grey")
        ax.axhline(-pair_conf, ls="--", lw=0.8, color="grey")
        ax.fill_between(lag_range, -pair_conf, pair_conf,
                        color="lightgreen", alpha=0.3, zorder=2)
        ax.axvline(0, ls=":", lw=0.6, color="black")
        style_axis(ax, title=f"{drv} → {target}",
                   xlabel="Lag (days)", ylabel="CCF")
        _integer_lag_axis(ax)

    for idx in range(n_drivers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig_ccf.suptitle(
        f"Cross-correlation: driver leads (+lag) / lags (−lag) {target} daily changes",
        fontsize=12, y=1.01,
    )
    save_figure(fig_ccf, f"{slug}_cross_correlations.png", dpi=170)
