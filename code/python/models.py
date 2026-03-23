"""Concrete fair-value model classes (static + rolling)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base_model_class import FairValueCandidateModel, FairValueConfig, OLSHelper
from utils import aligned_actual_levels, anchored_cumsum, model_change_frame


class StaticFairValueModel(FairValueCandidateModel):
    """Static fair-value pipeline only (inherits ``evaluate_model`` / per-candidate ``compute`` unchanged)."""


class RollingFairValueModel(StaticFairValueModel):
    """Static path via ``super().evaluate_model``; rolling uses ``OLSHelper`` + ``_write_fair_value_csvs``."""

    def evaluate_model(self, level_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate static best model and compute rolling series for comparisons."""
        
        static_result = super().evaluate_model(level_df)
        target = static_result["target"]
        slug = static_result["slug"]
        best_name = static_result["best_name"]
        best_df = static_result["best_df"]
        if not isinstance(best_df, pd.DataFrame):
            raise TypeError("Expected best_df DataFrame from static compute.")
        window = self.cfg.rolling_window

        all_rolling = {
            name.removesuffix("_d"): feats
            for name, feats in self.cfg.candidate_models.items()
        }
        rolling_series: dict[str, pd.DataFrame] = {}
        for model_name, feats in all_rolling.items():
            vm = self._rolling_series(level_df, target=target, features=feats, window=window)
            if not vm.empty:
                rolling_series[model_name] = vm
        if not rolling_series:
            raise ValueError("No rolling comparison series could be computed.")

        best_key = best_name.removesuffix("_d")
        vm_best = rolling_series.get(best_key)
        if vm_best is None:
            raise ValueError(
                f"No rolling series for best static model {best_name!r} (key {best_key!r})."
            )

        valid = vm_best.copy()
        valid["predicted_change"] = valid["rolling_change"]
        valid["fair_level"] = valid["rolling_level"]
        valid["residual"] = valid["actual_level"] - valid["fair_level"]

        self._write_fair_value_csvs(slug=slug, rolling_series_df=valid)

        static_fair = best_df[["Date", "fair_level"]].rename(
            columns={"fair_level": "static_fair_level"}
        )
        base = vm_best[["Date", "actual_level"]].merge(static_fair, on="Date", how="inner")
        for model_name, vm in rolling_series.items():
            base = base.merge(
                vm[["Date", "rolling_level"]].rename(
                    columns={"rolling_level": f"rolling_{model_name}"}
                ),
                on="Date",
                how="inner",
            )

        return {
            **static_result,
            "plot_payload": {
                "rolling": {
                    "valid": valid,
                    "target": target,
                    "slug": slug,
                    "window": window,
                },
                "rolling_comparison": {
                    "base": base,
                    "target": target,
                    "slug": slug,
                    "window": window,
                    "all_rolling": all_rolling,
                },
            },
        }

    def _rolling_series(
        self,
        level_df: pd.DataFrame,
        *,
        target: str,
        features: list[str],
        window: int,
    ) -> pd.DataFrame:
        """Rolling-window OLS on first differences → ``rolling_change`` / ``rolling_level`` / ``actual_level``."""

        d = model_change_frame(level_df, target=target, features=features).reset_index(drop=True)

        pred = np.full(len(d), np.nan)
        for i in range(window, len(d)):
            sl = d.iloc[i - window : i]
            x_train = sl[features].to_numpy()
            y_train = sl[target].to_numpy()
            fit = OLSHelper.run_ols(y_train, x_train)
            beta = fit["beta"]
            if not isinstance(beta, np.ndarray):
                raise TypeError("Expected ndarray beta from rolling OLS.")
            x_cur = d.iloc[i : i + 1][features].to_numpy()
            pred[i] = float(OLSHelper.predict_ols(x_cur, beta)[0])

        d["rolling_change"] = pred
        valid = d.dropna(subset=["rolling_change"]).copy()
        if valid.empty:
            return valid

        valid["rolling_level"] = anchored_cumsum(
            level_df, valid["Date"], valid["rolling_change"], target=target
        )
        valid["actual_level"] = aligned_actual_levels(level_df, valid["Date"], target=target)
        return valid


class StaticFairValueModelRegimeSwitch(StaticFairValueModel):
    """Static fair value with two regimes from **S&P realised vol (1M) − VIX** (see ``_regime_table``).

    Realised volatility is computed from **daily percentage changes of the S&P level**
    (not the fair-value target). Subclasses ``StaticFairValueModel`` and overrides only
    ``compute`` (single-candidate fit); the rest of the static pipeline
    (``evaluate_model``, CSV export, candidate ranking) is unchanged from
    ``FairValueCandidateModel`` / ``StaticFairValueModel``.
    """

    REGIME_VOL_WINDOW: int = 21
    VIX_COLUMN: str = "VIX"
    REGIME_RV_COLUMN: str = "S&P"
    #: Regime 1 when ``RV_ann - VIX`` exceeds this (annualised vol points, same scale as VIX).
    REGIME_SIGNAL_THRESHOLD: float = 3.0
    MIN_TRAIN_ROWS_PER_REGIME: int = 30

    def __init__(
        self,
        config: FairValueConfig,
        *,
        regime_vol_window: int | None = None,
        min_train_rows_per_regime: int | None = None,
        regime_signal_threshold: float | None = None,
        regime_rv_column: str | None = None,
    ) -> None:
        """Configure regime-switching realized-vol window and threshold."""

        super().__init__(config)
        self.regime_vol_window = int(
            regime_vol_window if regime_vol_window is not None else self.REGIME_VOL_WINDOW
        )
        self.min_train_rows_per_regime = int(
            min_train_rows_per_regime
            if min_train_rows_per_regime is not None
            else self.MIN_TRAIN_ROWS_PER_REGIME
        )
        self.regime_signal_threshold = (
            float(regime_signal_threshold)
            if regime_signal_threshold is not None
            else float(self.REGIME_SIGNAL_THRESHOLD)
        )
        self.regime_rv_column = (
            regime_rv_column if regime_rv_column is not None else self.REGIME_RV_COLUMN
        )

    def _regime_table(self, level_df: pd.DataFrame) -> pd.DataFrame:
        """Build the regime indicator and RV/VIX spread time series."""

        if self.VIX_COLUMN not in level_df.columns:
            raise KeyError(
                f"{type(self).__name__} requires column {self.VIX_COLUMN!r} in level_df."
            )
        if self.regime_rv_column not in level_df.columns:
            raise KeyError(
                f"{type(self).__name__} requires column {self.regime_rv_column!r} in level_df."
            )
        lv = level_df.sort_values("Date", kind="mergesort").reset_index(drop=True)
        r = lv[self.regime_rv_column].pct_change(fill_method=None)
        r = r.replace([np.inf, -np.inf], np.nan)
        w = self.regime_vol_window
        min_p = max(5, w // 2)
        rv_ann = (
            r.rolling(w, min_periods=min_p)
            .std()
            * np.sqrt(252.0)
            * 100.0
        )
        vix = lv[self.VIX_COLUMN].astype(float)
        spread = rv_ann - vix
        spread = spread.fillna(0.0)
        regime = (spread > self.regime_signal_threshold).astype(np.int8)
        return pd.DataFrame(
            {
                "Date": lv["Date"],
                "rv_ann_pct": rv_ann,
                "vix_level": vix,
                "rv_minus_vix": spread,
                "regime": regime,
            }
        )

    def _betas_by_regime(
        self,
        train: pd.DataFrame,
        features: list[str],
        target: str,
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Return per-regime OLS coefficient vectors and the pooled fallback.

        For each regime :math:`r \\in \\{0,1\\}`, if the training subsample with
        that regime has at least ``min_train_rows_per_regime`` rows, we estimate
        a **separate** OLS on that subsample: the **entire** parameter vector
        (intercept plus **all** driver coefficients) is regime-specific. If there
        are too few points in regime :math:`r`, **all** parameters for that
        regime are set to the **pooled** OLS vector fitted on the full training
        window (same vector for both regimes only when both subsamples are
        sparse).
        """

        y_all = train[target].to_numpy()
        x_all = train[features].to_numpy()
        pooled = OLSHelper.run_ols(y_all, x_all)
        beta_pooled = pooled["beta"]
        if not isinstance(beta_pooled, np.ndarray):
            raise TypeError("Expected ndarray beta from pooled OLS.")

        betas: dict[int, np.ndarray] = {}
        for r in (0, 1):
            sub = train[train["regime"] == r]
            if len(sub) >= self.min_train_rows_per_regime:
                fit_r = OLSHelper.run_ols(
                    sub[target].to_numpy(),
                    sub[features].to_numpy(),
                )
                b = fit_r["beta"]
                if not isinstance(b, np.ndarray):
                    raise TypeError("Expected ndarray beta from regime OLS.")
                betas[r] = b
        if 0 not in betas:
            betas[0] = beta_pooled
        if 1 not in betas:
            betas[1] = beta_pooled
        return betas, beta_pooled

    def _predict_predicted_change(
        self, frame: pd.DataFrame, features: list[str], betas: dict[int, np.ndarray]
    ) -> np.ndarray:
        """Predict target changes using regime-specific coefficient vectors."""
        x = frame[features].to_numpy()
        r = frame["regime"].to_numpy().astype(int)
        pred = np.empty(len(frame), dtype=float)
        for i in range(len(frame)):
            beta = betas[int(r[i])]
            pred[i] = float(OLSHelper.predict_ols(x[i : i + 1], beta)[0])
        return pred

    def compute(
        self,
        level_df: pd.DataFrame,
        *,
        target: str,
        model_name: str,
        features: list[str],
    ) -> dict[str, Any] | None:
        """Fit a single candidate with a realized-vol/VIX regime and return metrics."""
        d = model_change_frame(level_df, target=target, features=features)
        if len(d) < self.cfg.min_obs:
            return None

        reg = self._regime_table(level_df)[["Date", "regime"]]
        d = d.merge(reg, on="Date", how="left")
        d["regime"] = d["regime"].fillna(0).astype(np.int8)

        n_train = int(self.cfg.train_ratio * len(d))
        train, test = train_test_split(d, train_size=n_train, shuffle=False, random_state=42)
        train, test = train.copy(), test.copy()

        betas, _beta_pooled = self._betas_by_regime(train, features, target)

        train["predicted_change"] = self._predict_predicted_change(train, features, betas)
        test["predicted_change"] = self._predict_predicted_change(test, features, betas)

        full = pd.concat([train, test], ignore_index=True)

        full["fair_level"] = anchored_cumsum(
            level_df, full["Date"], full["predicted_change"], target=target
        )
        full["actual_level"] = aligned_actual_levels(level_df, full["Date"], target=target)
        full["residual"] = full["actual_level"] - full["fair_level"]

        r2_in = OLSHelper.r2_score(
            train[target].to_numpy(),
            train["predicted_change"].to_numpy(),
        )
        r2_out = OLSHelper.r2_score(
            test[target].to_numpy(),
            test["predicted_change"].to_numpy(),
        )
        rmse_out = OLSHelper.rmse(
            test[target].to_numpy(),
            test["predicted_change"].to_numpy(),
        )

        pooled_fit = OLSHelper.run_ols(
            train[target].to_numpy(),
            train[features].to_numpy(),
        )
        beta = pooled_fit["beta"]
        se = pooled_fit["se"]
        t_stats = pooled_fit["t_stats"]
        if not isinstance(beta, np.ndarray) or not isinstance(se, np.ndarray) or not isinstance(t_stats, np.ndarray):
            raise TypeError("Expected ndarray beta, se, and t_stats from OLS fit.")

        var_names = ["Intercept"] + features
        diagnostics_coef_rows = [
            {
                "model": model_name,
                "variable": name,
                "coefficient": float(beta[i]),
                "std_error": float(se[i]),
                "t_statistic": float(t_stats[i]),
                "significant_5pct": "Yes" if abs(float(t_stats[i])) > 1.96 else "No",
            }
            for i, name in enumerate(var_names)
        ]
        diagnostics_model_row: dict[str, object] = {
            "model": model_name,
            "n_train": int(pooled_fit["n"]),
            "k_params": int(pooled_fit["k"]),
            "r2": float(pooled_fit["r2"]),
            "adj_r2": float(pooled_fit["adj_r2"]),
            "aic": float(pooled_fit["aic"]),
            "bic": float(pooled_fit["bic"]),
            "spec": "regime_switch_rv_minus_vix",
        }

        comparison_row: dict[str, object] = {
            "model": model_name,
            "features_changes": ", ".join(features),
            "n_obs": int(len(d)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "r2_in_sample": float(r2_in),
            "r2_out_sample": float(r2_out),
            "rmse_out_sample": float(rmse_out),
            "spec": "regime_switch_rv_minus_vix",
        }

        return {
            "comparison_row": comparison_row,
            "full": full,
            "beta": beta,
            "r2_out": float(r2_out),
            "rmse_out": float(rmse_out),
            "diagnostics_model_row": diagnostics_model_row,
            "diagnostics_coef_rows": diagnostics_coef_rows,
        }


__all__ = [
    "StaticFairValueModel",
    "StaticFairValueModelRegimeSwitch",
    "RollingFairValueModel",
]
