from __future__ import annotations

import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "input" / "ICMA Data Assignment_data set_March 2026.xlsx"
MODELS_FILE = Path(__file__).resolve().parent / "models.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"


# ---------------------------------------------------------------------------
# YAML config loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML config file and return it as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_MODEL_CFG: dict | None = None


def _models_cfg() -> dict:
    """Return the cached parsed contents of ``models.yaml``."""
    global _MODEL_CFG
    if _MODEL_CFG is None:
        _MODEL_CFG = _load_yaml(MODELS_FILE)
    return _MODEL_CFG


_MISSING = object()


def get_model_param(key: str, default=_MISSING):
    """Return ``models.yaml`` value for ``key``. If ``default`` is given, use it when the key is absent."""
    cfg = _models_cfg()
    if default is _MISSING:
        return cfg[key]
    return cfg.get(key, default)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create output directories used for generated artifacts."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load the input Excel dataset, parse/sort ``Date``, and validate it."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    df = pd.read_excel(path)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first differences for all non-``Date`` columns."""
    out = pd.DataFrame(index=df.index)
    out["Date"] = df["Date"]
    for col in df.columns:
        if col == "Date":
            continue
        out[col] = df[col].diff()
    return out


def model_change_frame(level_df: pd.DataFrame, *, target: str, features: list[str]) -> pd.DataFrame:
    """Prepare Date + first differences for target/features with NA rows dropped."""
    cols = [target] + features
    d = level_df[["Date"] + cols].copy()
    d[cols] = d[cols].diff()
    return d.dropna(subset=cols).copy()


def anchored_cumsum(
    level_df: pd.DataFrame, date_series: pd.Series, change_series: pd.Series, *, target: str
) -> pd.Series:
    """Convert predicted changes to level by anchoring at first date."""
    if date_series.empty:
        return pd.Series(dtype=float)
    first_date = date_series.iloc[0]
    anchor_rows = level_df.loc[level_df["Date"] == first_date, target]
    if anchor_rows.empty:
        raise ValueError(f"Anchor date {first_date!r} not found in level_df for target {target!r}.")
    anchor = float(anchor_rows.iloc[0])
    return anchor + change_series.cumsum()


def aligned_actual_levels(level_df: pd.DataFrame, date_series: pd.Series, *, target: str) -> np.ndarray:
    """Return target levels aligned on Date to the provided date sequence."""
    actual = (
        level_df[["Date", target]]
        .merge(pd.DataFrame({"Date": date_series}), on="Date", how="inner")
        .reset_index(drop=True)[target]
        .to_numpy()
    )
    if len(actual) != len(date_series):
        raise ValueError(
            f"Date alignment mismatch for target {target!r}: expected {len(date_series)}, got {len(actual)}."
        )
    return actual


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """Create a single matplotlib figure with one axes."""
    return plt.subplots(figsize=figsize)


def make_subplots(
    nrows: int, ncols: int, figsize: tuple[float, float], sharex: bool = False
) -> tuple[plt.Figure, np.ndarray]:
    """Create a matplotlib subplots grid and return ``(fig, axes)``."""
    return plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)


def style_axis(
    ax: plt.Axes,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zero_line: bool = False,
) -> None:
    """Set axis title/labels and optionally draw a horizontal zero line."""
    ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)


def save_figure(fig: plt.Figure, filename: str, dpi: int) -> None:
    """Save ``fig`` into ``FIG_DIR`` and close it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=dpi)
    plt.close(fig)


def safe_slug(name: str) -> str:
    """Convert a column name into a stable slug for filenames/column keys."""
    return name.replace(" ", "_").replace("&", "")
