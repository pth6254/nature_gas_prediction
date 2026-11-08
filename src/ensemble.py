"""Evaluate a fixed H3 Ridge/naive blend on identical saved validation origins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import summarize_predictions


def evaluate_ensemble(backtests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare prices using a predeclared 50:50 weight, without fitting to labels."""
    part = backtests.loc[
        (backtests["stage"] == "model_comparison") & (backtests["horizon"] == 3)
    ].copy()
    part["date"] = pd.to_datetime(part["date"])
    ridge = part.loc[part["model"] == "ridge_log_return"].sort_values("date")
    naive = part.loc[part["model"] == "last_value_naive"].sort_values("date")
    keys = ["date", "horizon", "feature_group"]
    paired = ridge.merge(naive, on=keys, suffixes=("_ridge", "_naive"),
                         how="outer", validate="one_to_one", indicator=True)
    if paired.empty or not paired["_merge"].eq("both").all():
        raise ValueError("Ridge and naive must have identical nonempty validation origins")
    for column in ("actual", "current_price", "previous_price"):
        if not np.array_equal(paired[f"{column}_ridge"], paired[f"{column}_naive"]):
            raise ValueError(f"Paired forecasts disagree on {column}")
    numeric = paired.filter(regex="^(actual|predicted|current_price|previous_price)_")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Forecasts and observations must be finite")
    if not np.array_equal(paired["predicted_naive"], paired["current_price_naive"]):
        raise ValueError("Last-value naive must equal the current price")

    blend = ridge.copy()
    blend["predicted"] = 0.5 * ridge["predicted"] + 0.5 * ridge["current_price"]
    blend["model"] = "ridge_naive_equal_blend"
    blend["target_type"] = "price"
    predictions = pd.concat([ridge, naive, blend], ignore_index=True)
    predictions["stage"] = "ensemble_diagnostic"
    rows = []
    for model, forecasts in predictions.groupby("model", sort=False):
        periods = [("all", forecasts), *forecasts.groupby(forecasts["date"].dt.year)]
        for year, subset in periods:
            rows.append({"horizon": 3, "feature_group": subset["feature_group"].iloc[0],
                         "model": model, "validation_year": year,
                         **summarize_predictions(subset)})
    return predictions, pd.DataFrame(rows)


def save_ensemble_diagnostics(backtests: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    predictions, metrics = evaluate_ensemble(backtests)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "ensemble_predictions.csv", index=False, float_format="%.8f")
    metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False, float_format="%.6f")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    backtests = pd.read_csv(args.output_dir / "backtest_predictions.csv")
    metrics = save_ensemble_diagnostics(backtests, args.output_dir)
    print(metrics.loc[metrics["validation_year"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
