"""Validate models.yaml against the input dataset.

Run standalone (``python validate_config.py``) or call
``validate_all(dataset_columns)`` from the orchestrator before analysis.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_CFG_DIR = Path(__file__).resolve().parent
MODELS_FILE = _CFG_DIR / "models.yaml"

class ConfigValidationError(Exception):
    """Raised when YAML configuration is invalid."""

    def __init__(self, errors: list[str]) -> None:
        """Build an actionable exception message from validation errors."""
        self.errors = errors
        bullet_list = "\n  - ".join(errors)
        super().__init__(f"Configuration validation failed:\n  - {bullet_list}")


def _load(path: Path) -> dict:
    """Load and parse ``models.yaml`` from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# models.yaml
# ---------------------------------------------------------------------------

def validate_models(dataset_columns: set[str]) -> list[str]:
    """Return a list of error messages (empty if valid)."""
    cfg = _load(MODELS_FILE)
    errors: list[str] = []

    required_keys = {
        "target", "train_ratio", "min_obs", "rolling_window",
        "candidate_models",
    }
    missing = required_keys - set(cfg)
    if missing:
        errors.append(f"models.yaml missing required keys: {sorted(missing)}")
        return errors

    target = cfg["target"]
    if not isinstance(target, str) or not target:
        errors.append("'target' must be a non-empty string.")
    elif target not in dataset_columns:
        errors.append(
            f"target '{target}' not found in dataset. "
            f"Available columns: {sorted(dataset_columns)}"
        )

    train_ratio = cfg["train_ratio"]
    if not isinstance(train_ratio, (int, float)) or not (0 < train_ratio < 1):
        errors.append(f"'train_ratio' must be a number in (0, 1), got {train_ratio!r}.")

    min_obs = cfg["min_obs"]
    if not isinstance(min_obs, int) or min_obs < 1:
        errors.append(f"'min_obs' must be a positive integer, got {min_obs!r}.")

    rolling_window = cfg["rolling_window"]
    if not isinstance(rolling_window, int) or rolling_window < 1:
        errors.append(f"'rolling_window' must be a positive integer, got {rolling_window!r}.")

    models = cfg["candidate_models"]
    if not isinstance(models, dict) or len(models) == 0:
        errors.append("'candidate_models' must be a non-empty mapping.")
    else:
        all_features: set[str] = set()
        for name, features in models.items():
            if not isinstance(features, list) or len(features) == 0:
                errors.append(f"candidate_models.{name}: must be a non-empty list.")
                continue
            if isinstance(target, str) and target in features:
                errors.append(f"candidate_models.{name}: target '{target}' must not be a feature.")
            all_features.update(features)
        unknown_feats = all_features - dataset_columns
        if unknown_feats:
            errors.append(
                f"candidate_models reference columns not in dataset: {sorted(unknown_feats)}. "
                f"Available columns: {sorted(dataset_columns)}"
            )

    styles = cfg.get("plot_styles")
    if styles is not None:
        if not isinstance(styles, dict):
            errors.append("'plot_styles' must be a mapping if present.")
        else:
            model_stems = {n.removesuffix("_d") for n in models} if isinstance(models, dict) else set()
            extra = set(styles) - model_stems
            if extra:
                errors.append(
                    f"plot_styles contains keys that don't match any candidate model: {sorted(extra)}"
                )
            for sname, props in styles.items():
                if not isinstance(props, dict):
                    errors.append(f"plot_styles.{sname}: expected a mapping.")
                    continue
                for req in ("color", "linestyle", "linewidth"):
                    if req not in props:
                        errors.append(f"plot_styles.{sname}: missing required key '{req}'.")

    return errors


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def validate_all(dataset_columns: set[str]) -> None:
    """Validate both config files and raise on any error."""
    errors = validate_models(dataset_columns)
    if errors:
        raise ConfigValidationError(errors)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_CFG_DIR))
    from utils import DATA_FILE, load_data

    df = load_data(DATA_FILE)
    cols = set(df.columns) - {"Date"}
    try:
        validate_all(cols)
        print("All configuration files are valid.")
    except ConfigValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
