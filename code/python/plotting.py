from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from utils import PROJECT_ROOT, get_model_param, make_figure, make_subplots, safe_slug, save_figure, style_axis


def plot_fair_value_best_model(
    best_df: pd.DataFrame,
    model_table: pd.DataFrame,
    target: str,
    slug: str,
    best_name: str,
) -> None:
    """Render and save fair-value best-model figures."""

    fig, ax = make_figure((11, 5.7))
    ax.plot(
        best_df["Date"],
        best_df["actual_level"],
        label=f"Observed {target}",
        linewidth=1.3,
    )
    ax.plot(
        best_df["Date"],
        best_df["fair_level"],
        label=f"Fair value ({best_name})",
        linewidth=1.3,
    )
    style_axis(
        ax,
        title=f"{target} observed vs fair value ({best_name})",
        xlabel="Date",
        ylabel="Spread",
    )
    ax.legend()
    save_figure(fig, f"{slug}_fair_value_best_model.png", dpi=170)

    fig2, ax2 = make_figure((10, 4.8))
    sorted_tbl = model_table.sort_values(by="r2_out_sample", ascending=False)
    ax2.bar(sorted_tbl["model"], sorted_tbl["r2_out_sample"])
    style_axis(
        ax2,
        title=f"Out-of-sample R² by {target} fair value model",
        xlabel="Model",
        ylabel="Out-of-sample R²",
    )
    ax2.tick_params(axis="x", rotation=20)
    save_figure(fig2, f"{slug}_fair_value_model_comparison.png", dpi=170)


def plot_rolling_fair_value(
    valid: pd.DataFrame, target: str, slug: str, window: int
) -> None:
    """Render and save the rolling fair-value figure."""

    fig, ax = make_figure((11, 5.7))
    ax.plot(
        valid["Date"],
        valid["actual_level"],
        label=f"Observed {target}",
        linewidth=1.3,
    )
    ax.plot(
        valid["Date"],
        valid["fair_level"],
        label=f"Rolling fair value ({window}d window)",
        linewidth=1.3,
    )
    style_axis(
        ax,
        title=f"{target} observed vs rolling fair value ({window}-day window)",
        xlabel="Date",
        ylabel="Spread",
    )
    ax.legend()
    save_figure(fig, f"{slug}_rolling_fair_value.png", dpi=170)


def plot_rolling_comparison(
    base: pd.DataFrame,
    target: str,
    slug: str,
    window: int,
    all_rolling: dict[str, list[str]],
    plot_styles: dict[str, dict],
) -> None:
    """Render and save the rolling comparison figure."""
    
    fig_cmp, ax_cmp = make_figure((13, 6.5))
    ax_cmp.plot(
        base["Date"],
        base["actual_level"],
        label=f"Observed {target}",
        linewidth=1.6,
        color="#1f77b4",
    )
    ax_cmp.plot(
        base["Date"],
        base["static_fair_level"],
        label="Static (best model, global coeff.)",
        linewidth=1.3,
        color="#2ca02c",
        linestyle="--",
    )

    for model_name in all_rolling:
        col = f"rolling_{model_name}"
        if col not in base.columns:
            continue
        st = plot_styles.get(
            model_name, {"color": "grey", "linestyle": "-", "linewidth": 1.0}
        )
        ax_cmp.plot(
            base["Date"],
            base[col],
            label=f"Rolling {model_name} ({window}d)",
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
        )

    style_axis(
        ax_cmp,
        title=f"{target}: observed vs static and rolling fair-value models",
        xlabel="Date",
        ylabel="Spread (bps)",
    )
    ax_cmp.legend(fontsize=8, loc="upper left")
    save_figure(fig_cmp, f"{slug}_fair_value_comparison.png", dpi=170)


def plot_regime_switch_sensitivity(level_df: pd.DataFrame) -> None:
    """Plot RV vs VIX regime signal; fair-value levels (realised vs static vs regime); TeX metrics.

    Saves ``{slug}_regime_rv_vix.png`` and ``{slug}_regime_vs_static_fair_value.png`` (same
    best specification as ``StaticFairValueModel``: pooled OLS vs regime-switching OLS).
    Writes ``code/tex/sensitivity/generated/regime_switch_metrics.tex``.
    """
    
    from base_model_class import FairValueConfig
    from models import StaticFairValueModel, StaticFairValueModelRegimeSwitch

    target = get_model_param("target")
    slug = safe_slug(target)
    cfg = FairValueConfig(
        target=target,
        candidate_models=get_model_param("candidate_models"),
        train_ratio=float(get_model_param("train_ratio")),
        min_obs=int(get_model_param("min_obs")),
        rolling_window=int(get_model_param("rolling_window")),
    )
    regime_model = StaticFairValueModelRegimeSwitch(cfg)
    rt = regime_model._regime_table(level_df)
    rv_name = regime_model.regime_rv_column

    fig, axes = make_subplots(2, 1, figsize=(11, 6.4), sharex=True)
    ax0, ax1 = axes

    ax0.plot(
        rt["Date"],
        rt["rv_ann_pct"],
        label=f"{rv_name} realized vol (ann., %)",
        color="#1f77b4",
        linewidth=1.0,
    )
    ax0.plot(rt["Date"], rt["vix_level"], label="VIX", color="#d62728", linewidth=1.0)
    style_axis(
        ax0,
        title=(
            f"{rv_name} realized volatility ({regime_model.regime_vol_window}d, ann.) vs VIX "
            f"(regimes for {target} fair-value OLS)"
        ),
        ylabel="Percent",
    )
    ax0.legend(loc="upper left", fontsize=9)

    dates = rt["Date"].reset_index(drop=True)
    reg = rt["regime"].to_numpy()
    n = len(rt)
    i = 0
    thr = regime_model.regime_signal_threshold
    while i < n:
        j = i + 1
        while j < n and reg[j] == reg[i]:
            j += 1
        if reg[i] == 1:
            face, alpha = (0.98, 0.88, 0.55), 0.28
        else:
            face, alpha = (0.88, 0.9, 0.95), 0.22
        ax1.axvspan(dates.iloc[i], dates.iloc[j - 1], facecolor=face, alpha=alpha, linewidth=0, zorder=0)
        i = j

    for k in range(1, n):
        if reg[k] != reg[k - 1]:
            ax1.axvline(
                dates.iloc[k],
                color="0.35",
                alpha=0.45,
                linewidth=0.7,
                linestyle="-",
                zorder=1,
            )

    ax1.plot(
        rt["Date"],
        rt["rv_minus_vix"],
        color="black",
        linewidth=1.0,
        label=r"$RV_{\mathrm{ann}} - \mathrm{VIX}$",
        zorder=2,
    )
    ax1.axhline(
        thr,
        color="0.45",
        linestyle="--",
        linewidth=0.9,
        label=f"Threshold ({thr:g})",
        zorder=2,
    )
    style_axis(
        ax1,
        title="Regime signal (regime 1 when spread > threshold); shading and vertical lines mark switches",
        xlabel="Date",
        ylabel="Percentage points",
    )
    ax1.legend(loc="upper left", fontsize=9)
    fig.align_ylabels([ax0, ax1])
    plt.subplots_adjust(hspace=0.18)
    save_figure(fig, f"{slug}_regime_rv_vix.png", dpi=170)

    base_ev = StaticFairValueModel(cfg)._evaluate_candidates(level_df)
    reg_ev = regime_model._evaluate_candidates(level_df)

    if base_ev.best_one is None:
        raise ValueError(
            "Static fair-value evaluation produced no best model; cannot build "
            "regime vs static fair-value level plot."
        )
    static_best = base_ev.best_one
    best_spec_name = base_ev.best_name
    best_spec_features = base_ev.best_features
    regime_same_spec = regime_model.compute(
        level_df,
        target=target,
        model_name=best_spec_name,
        features=best_spec_features,
    )
    if regime_same_spec is None:
        raise ValueError(
            f"Regime-switching fit failed for best static specification {best_spec_name!r}."
        )
    df_static = static_best["full"][["Date", "actual_level", "fair_level"]].copy()
    df_static = df_static.rename(columns={"fair_level": "fair_static_global"})
    df_regime = regime_same_spec["full"][["Date", "fair_level"]].copy()
    df_regime = df_regime.rename(columns={"fair_level": "fair_regime_switch"})
    plot_levels = df_static.merge(df_regime, on="Date", how="inner")
    if len(plot_levels) != len(df_static) or len(plot_levels) != len(df_regime):
        raise ValueError(
            "Date alignment failed between static-global and regime-switching fair-value series."
        )

    fig_lv, ax_lv = make_figure((11, 5.7))
    ax_lv.plot(
        plot_levels["Date"],
        plot_levels["actual_level"],
        label=f"Realized {target}",
        linewidth=1.5,
        color="#1f77b4",
    )
    ax_lv.plot(
        plot_levels["Date"],
        plot_levels["fair_static_global"],
        label=f"Fair value, static pooled OLS ({best_spec_name})",
        linewidth=1.25,
        color="#2ca02c",
        linestyle="--",
    )
    ax_lv.plot(
        plot_levels["Date"],
        plot_levels["fair_regime_switch"],
        label=f"Fair value, regime-switching OLS ({best_spec_name})",
        linewidth=1.25,
        color="#ff7f0e",
    )
    style_axis(
        ax_lv,
        title=(
            f"{target}: realized vs fair value (same specification: static global vs "
            "regime-switching coefficients)"
        ),
        xlabel="Date",
        ylabel="Spread (bps)",
    )
    ax_lv.legend(loc="upper left", fontsize=8)
    save_figure(fig_lv, f"{slug}_regime_vs_static_fair_value.png", dpi=170)
    lm_base = base_ev.best_name.replace("_", r"\_")
    lm_reg = reg_ev.best_name.replace("_", r"\_")
    rv_col_tex = regime_model.regime_rv_column.replace("&", r"\&")

    gen_dir = PROJECT_ROOT / "code" / "tex" / "sensitivity" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    tex_path = gen_dir / "regime_switch_metrics.tex"
    tex_path.write_text(
        "% Generated by analyze_icma_dataset (plot_regime_switch_sensitivity).\n"
        "\\providecommand{\\RegimeSwitchBestModelBase}{"
        + lm_base
        + "}\n"
        "\\providecommand{\\RegimeSwitchBestModelReg}{"
        + lm_reg
        + "}\n"
        "\\providecommand{\\RegimeRVColumn}{"
        + rv_col_tex
        + "}\n"
        f"\\providecommand{{\\RegimeSwitchRsqOut}}{{{reg_ev.best_oos_r2:.4f}}}\n"
        f"\\providecommand{{\\BaselineStaticRsqOut}}{{{base_ev.best_oos_r2:.4f}}}\n"
        f"\\providecommand{{\\RegimeSwitchDeltaRsq}}{{{reg_ev.best_oos_r2 - base_ev.best_oos_r2:+.4f}}}\n"
        f"\\providecommand{{\\RegimeSwitchRmseOut}}{{{reg_ev.best_oos_rmse:.4f}}}\n"
        f"\\providecommand{{\\BaselineStaticRmseOut}}{{{base_ev.best_oos_rmse:.4f}}}\n"
        f"\\providecommand{{\\RegimeVolWindow}}{{{regime_model.regime_vol_window}}}\n"
        f"\\providecommand{{\\RegimeSignalThreshold}}{{{regime_model.regime_signal_threshold:g}}}\n",
        encoding="utf-8",
    )

