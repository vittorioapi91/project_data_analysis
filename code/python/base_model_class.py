from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from utils import OUTPUT_DIR, safe_slug

# Columns for main fair-value series CSVs (static + rolling best model).
FAIR_VALUE_SERIES_COLS: list[str] = [
    "Date",
    "actual_level",
    "fair_level",
    "predicted_change",
    "residual",
]
FAIR_VALUE_DISLOCATION_COLS: list[str] = [
    "Date",
    "actual_level",
    "fair_level",
    "residual",
]


@dataclass(frozen=True)
class FairValueConfig:
    target: str
    candidate_models: dict[str, list[str]]
    train_ratio: float
    min_obs: int
    rolling_window: int


class FairValueCandidateEval(NamedTuple):
    """Result of evaluating all static fair-value candidates on ``level_df``."""

    comparison_rows: list[dict[str, object]]
    best_name: str
    best_features: list[str]
    best_oos_r2: float
    best_oos_rmse: float
    best_one: dict[str, Any] | None
    diagnostics_model_rows: list[dict[str, object]]
    diagnostics_coef_rows: list[dict[str, object]]


class OLSHelper:
    @staticmethod
    def run_ols(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
        """Fit an OLS with an intercept and return fitted statistics."""
        
        x_const = sm.add_constant(x)
        result = sm.OLS(y, x_const).fit()
        return {
            "beta": np.asarray(result.params),
            "y_hat": np.asarray(result.fittedvalues),
            "r2": float(result.rsquared),
            "adj_r2": float(result.rsquared_adj),
            "se": np.asarray(result.bse),
            "t_stats": np.asarray(result.tvalues),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "n": int(result.nobs),
            "k": len(result.params),
            "sigma2": float(result.mse_resid),
        }

    @staticmethod
    def predict_ols(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute OLS predictions given x and a beta vector (with intercept)."""
        # beta is [Intercept, b1, ..., bk] and x is (n, k).
        x_const = np.column_stack([np.ones(x.shape[0]), x])
        return x_const @ beta

    @staticmethod
    def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute R^2 from y and predicted y_hat."""
        sst = np.sum((y - y.mean()) ** 2)
        sse = np.sum((y - y_hat) ** 2)
        return float(1.0 - sse / sst) if sst > 0 else float("nan")

    @staticmethod
    def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute RMSE between y and predicted y_hat."""
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))


class FairValueCandidateModel:
    """Static fair-value candidates: train/test OLS on diffs, best model, CSV export (see ``evaluate_model``)."""

    def __init__(self, config: FairValueConfig) -> None:
        """Store the shared fair-value configuration for this model."""
        self.cfg = config

    def _evaluate_candidates(self, level_df: pd.DataFrame) -> FairValueCandidateEval:
        """Fit every candidate via ``compute``; pick best by OOS R², then lower OOS RMSE on ties."""
        target = self.cfg.target
        comparison_rows: list[dict[str, object]] = []
        best_name = ""
        best_features: list[str] = []
        best_oos_r2 = -np.inf
        best_oos_rmse = np.inf
        best_one: dict[str, Any] | None = None
        diagnostics_model_rows: list[dict[str, object]] = []
        diagnostics_coef_rows: list[dict[str, object]] = []

        for model_name, features in self.cfg.candidate_models.items():
            one = self.compute(
                level_df, target=target, model_name=model_name, features=features
            )
            if one is None:
                continue
            comparison_rows.append(one["comparison_row"])
            diagnostics_model_rows.append(one["diagnostics_model_row"])
            diagnostics_coef_rows.extend(one["diagnostics_coef_rows"])
            r2_out = float(one["r2_out"])
            rmse_out = float(one["rmse_out"])
            if (r2_out > best_oos_r2) or (
                np.isclose(r2_out, best_oos_r2) and rmse_out < best_oos_rmse
            ):
                best_oos_r2 = r2_out
                best_oos_rmse = rmse_out
                best_name = model_name
                best_features = features
                best_one = one

        return FairValueCandidateEval(
            comparison_rows=comparison_rows,
            best_name=best_name,
            best_features=best_features,
            best_oos_r2=float(best_oos_r2),
            best_oos_rmse=float(best_oos_rmse),
            best_one=best_one,
            diagnostics_model_rows=diagnostics_model_rows,
            diagnostics_coef_rows=diagnostics_coef_rows,
        )

    def _write_fair_value_csvs(
        self,
        *,
        slug: str,
        comparison_rows: list[dict[str, object]] | None = None,
        best_df: pd.DataFrame | None = None,
        rolling_series_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, float, pd.DataFrame] | None:
        """Write fair-value CSVs for static and/or rolling (best-window) outputs.

        Static bundle (``comparison_rows`` + ``best_df``): model comparison, dislocations,
        and ``{slug}_fair_value_series.csv``.

        Rolling best model (``rolling_series_df``): ``{slug}_rolling_fair_value_series.csv``
        using the same column names as the static series (``FAIR_VALUE_SERIES_COLS``).

        Returns the static tuple ``(model_table, q95, disloc)`` when static inputs are
        given; otherwise ``None``.
        """

        out_static: tuple[pd.DataFrame, float, pd.DataFrame] | None = None
        if comparison_rows is not None or best_df is not None:
            if comparison_rows is None or best_df is None:
                raise ValueError(
                    "comparison_rows and best_df must both be set for static fair-value CSVs."
                )
            model_table = pd.DataFrame(comparison_rows).sort_values(
                by=["r2_out_sample", "r2_in_sample"], ascending=False
            )
            model_table.to_csv(
                OUTPUT_DIR / f"{slug}_fair_value_model_comparison.csv", index=False
            )
            q95 = float(np.quantile(np.abs(best_df["residual"]), 0.95))
            disloc = best_df[np.abs(best_df["residual"]) >= q95]
            disloc[FAIR_VALUE_DISLOCATION_COLS].to_csv(
                OUTPUT_DIR / f"{slug}_dislocations.csv", index=False
            )
            best_df[FAIR_VALUE_SERIES_COLS].to_csv(
                OUTPUT_DIR / f"{slug}_fair_value_series.csv", index=False
            )
            out_static = (model_table, q95, disloc)

        if rolling_series_df is not None:
            missing = [c for c in FAIR_VALUE_SERIES_COLS if c not in rolling_series_df.columns]
            if missing:
                raise ValueError(
                    f"rolling_series_df missing columns required for export: {missing}"
                )
            rolling_series_df[FAIR_VALUE_SERIES_COLS].to_csv(
                OUTPUT_DIR / f"{slug}_rolling_fair_value_series.csv", index=False
            )

        return out_static

    def evaluate_model(self, level_df: pd.DataFrame) -> dict[str, Any]:
        """Run the full static pipeline: evaluate all candidates, pick best, write CSVs."""

        target = self.cfg.target
        slug = safe_slug(target)
        ev = self._evaluate_candidates(level_df)
        if not ev.comparison_rows:
            raise ValueError(f"No fair value models could be estimated for {target}.")
        if ev.best_one is None:
            raise ValueError("Best fair value model selection failed unexpectedly.")

        best_df = ev.best_one["full"]
        best_beta = ev.best_one["beta"]
        if not isinstance(best_df, pd.DataFrame) or not isinstance(best_beta, np.ndarray):
            raise TypeError("Expected DataFrame and ndarray from best candidate fit.")

        static_bundle = self._write_fair_value_csvs(
            slug=slug,
            comparison_rows=ev.comparison_rows,
            best_df=best_df,
        )
        if static_bundle is None:
            raise RuntimeError("Static fair-value CSV write returned no bundle.")
        model_table, q95, disloc = static_bundle
        pd.DataFrame(ev.diagnostics_model_rows).to_csv(
            OUTPUT_DIR / f"{slug}_model_diagnostics_summary.csv", index=False
        )
        pd.DataFrame(ev.diagnostics_coef_rows).to_csv(
            OUTPUT_DIR / f"{slug}_model_coefficients_detail.csv", index=False
        )
        coef_map = {
            f"coef_{safe_slug(name)}": float(best_beta[i + 1])
            for i, name in enumerate(ev.best_features)
        }
        out: dict[str, object] = {
            "fair_value_best_model": ev.best_name,
            "fair_value_best_model_features": ", ".join(ev.best_features),
            "fair_value_best_r2_out_sample": float(ev.best_oos_r2),
            "fair_value_best_rmse_out_sample": float(ev.best_oos_rmse),
            "fair_value_best_intercept": float(best_beta[0]),
            "fair_value_residual_abs_95pct": float(q95),
            "fair_value_n_dislocations": int(len(disloc)),
        }
        out.update(coef_map)

        return {
            "out": out,
            "best_df": best_df,
            "model_table": model_table,
            "target": target,
            "slug": slug,
            "best_name": ev.best_name,
            "best_features": ev.best_features,
        }

    def compute(
        self,
        level_df: pd.DataFrame,
        *,
        target: str,
        model_name: str,
        features: list[str],
    ) -> dict[str, Any] | None:
        """Fit **one** candidate specification: train/test OLS on diffs, fair level, residuals.

        Called once per entry in ``cfg.candidate_models`` from ``_evaluate_candidates``.
        Override this in subclasses to change how a single candidate is estimated;
        override ``evaluate_model`` only if you need to change the full pipeline.

        Args:
            level_df: Panel with ``Date`` and level columns for ``target`` and ``features``.
            target: Dependent column (``cfg.target``).
            model_name: Key from ``cfg.candidate_models`` (for comparison table).
            features: Regressor column names for this candidate.

        Returns:
            Per-model metrics and ``full`` frame, or ``None`` if too few observations.
        """

        d = level_df[["Date", target] + features].copy()
        d[[target] + features] = d[[target] + features].diff(); d = d.dropna().copy()
        if len(d) < self.cfg.min_obs:
            return None

        n_train = int(self.cfg.train_ratio * len(d))
        train, test = train_test_split(d, train_size=n_train, shuffle=False, random_state=42)
        train, test = train.copy(), test.copy()

        x_train = train[features].to_numpy()
        y_train = train[target].to_numpy()
        fit = OLSHelper.run_ols(y_train, x_train)
        beta = fit["beta"]
        se = fit["se"]
        t_stats = fit["t_stats"]
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
            "n_train": int(fit["n"]),
            "k_params": int(fit["k"]),
            "r2": float(fit["r2"]),
            "adj_r2": float(fit["adj_r2"]),
            "aic": float(fit["aic"]),
            "bic": float(fit["bic"]),
        }

        train["predicted_change"] = OLSHelper.predict_ols(x_train, beta)
        x_test = test[features].to_numpy()
        test["predicted_change"] = OLSHelper.predict_ols(x_test, beta)

        full = pd.concat([train, test], ignore_index=True)

        first_date = full.loc[full.index[0], "Date"]
        anchor = float(level_df.loc[level_df["Date"] == first_date, target].iloc[0])
        full["fair_level"] = anchor + full["predicted_change"].cumsum()

        actual_level = level_df[["Date", target]].merge(full[["Date"]], on="Date", how="inner").reset_index(drop=True)[target].to_numpy()
        full["actual_level"] = actual_level
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

        comparison_row: dict[str, object] = {
            "model": model_name,
            "features_changes": ", ".join(features),
            "n_obs": int(len(d)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "r2_in_sample": float(r2_in),
            "r2_out_sample": float(r2_out),
            "rmse_out_sample": float(rmse_out),
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
