"""Orchestrates fair-value estimation (static + rolling) and plot payload for the analysis script."""

from __future__ import annotations

from typing import Any

import pandas as pd

from base_model_class import FairValueConfig
from models import RollingFairValueModel
from utils import get_model_param


class FairValuePipeline:
    def __init__(self) -> None:
        """Initialize fair-value pipeline configuration from YAML."""

        self.cfg = FairValueConfig(
            target=get_model_param("target"),
            candidate_models=get_model_param("candidate_models"),
            train_ratio=float(get_model_param("train_ratio")),
            min_obs=int(get_model_param("min_obs")),
            rolling_window=int(get_model_param("rolling_window")),
        )

        self.rolling_model = RollingFairValueModel(self.cfg)

    def run(self, change_df: pd.DataFrame, level_df: pd.DataFrame) -> dict[str, Any]:
        """Run static+rolling fair-value estimation and return plot payload."""
        _ = change_df  # kept for ``impact_and_fair_value`` / analyze script call signature.
        fair_and_rolling = self.rolling_model.evaluate_model(level_df)

        return {
            "fair_best_model": {
                "best_df": fair_and_rolling["best_df"],
                "model_table": fair_and_rolling["model_table"],
                "target": fair_and_rolling["target"],
                "slug": fair_and_rolling["slug"],
                "best_name": fair_and_rolling["best_name"],
            },
            "rolling_fair_value": fair_and_rolling["plot_payload"]["rolling"],
            "rolling_fair_comparison": fair_and_rolling["plot_payload"][
                "rolling_comparison"
            ],
        }


def impact_and_fair_value(change_df: pd.DataFrame, level_df: pd.DataFrame) -> dict[str, Any]:
    """Convenience wrapper that runs :class:`FairValuePipeline`."""
    
    return FairValuePipeline().run(change_df=change_df, level_df=level_df)


__all__ = ["FairValuePipeline", "impact_and_fair_value"]
