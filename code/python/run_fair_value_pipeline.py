"""Run ``FairValuePipeline`` only (for debugging; see ``.vscode/launch.json``)."""

from __future__ import annotations

from fair_value_pipeline import FairValuePipeline
from utils import DATA_FILE, load_data, compute_changes
from validate_config import validate_all


def main() -> None:
    """Run :class:`FairValuePipeline` only (debug entry point)."""
    
    levels = load_data(DATA_FILE)
    validate_all(set(levels.columns) - {"Date"})
    changes = compute_changes(levels)

    result = FairValuePipeline().run(change_df=changes, level_df=levels)
    print(f"FairValuePipeline.run OK — {len(result)} top-level keys in result dict.")


if __name__ == "__main__":
    main()
