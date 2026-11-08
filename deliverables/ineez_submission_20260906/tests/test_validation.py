import numpy as np
import pandas as pd

from src.evaluate import summarize_predictions
from src.validation import rolling_origin_splits


def test_training_labels_are_known_at_origin() -> None:
    horizon = 3
    split = next(rolling_origin_splits(30, horizon, first_origin=12))
    assert split.train_indices.max() + horizon == split.origin
    assert split.validation_indices.tolist() == [12]


def test_summary_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [2.0, 4.0],
            "predicted": [2.0, 3.0],
            "current_price": [1.0, 3.0],
            "previous_price": [0.5, 4.0],
        }
    )
    metrics = summarize_predictions(predictions)
    assert metrics["mape_pct"] == 12.5
    assert metrics["mae"] == 0.5
    assert metrics["direction_accuracy_pct"] == 50.0


def test_flat_forecast_is_not_a_turning_point() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "predicted": [2.0],
            "current_price": [2.0],
            "previous_price": [1.0],
        }
    )
    metrics = summarize_predictions(predictions)
    assert metrics["actual_turning_points"] == 1
    assert metrics["predicted_turning_points"] == 0
