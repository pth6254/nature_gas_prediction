"""Metrics and turning-point diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def direction_accuracy(
    actual: np.ndarray, predicted: np.ndarray, current: np.ndarray
) -> float:
    actual_direction = np.sign(np.asarray(actual) - np.asarray(current))
    predicted_direction = np.sign(np.asarray(predicted) - np.asarray(current))
    return float(np.mean(actual_direction == predicted_direction) * 100)


def turning_point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    current: np.ndarray,
    previous: np.ndarray,
) -> dict[str, float]:
    previous_direction = np.sign(np.asarray(current) - np.asarray(previous))
    actual_direction = np.sign(np.asarray(actual) - np.asarray(current))
    predicted_direction = np.sign(np.asarray(predicted) - np.asarray(current))
    # A turning point is a genuine sign reversal. A flat forecast (including
    # last-value naive) is not counted as a predicted reversal.
    actual_turn = actual_direction * previous_direction < 0
    predicted_turn = predicted_direction * previous_direction < 0
    true_positive = int(np.sum(actual_turn & predicted_turn))
    false_positive = int(np.sum(~actual_turn & predicted_turn))
    false_negative = int(np.sum(actual_turn & ~predicted_turn))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "turning_precision_pct": precision * 100,
        "turning_recall_pct": recall * 100,
        "turning_f1_pct": f1 * 100,
        "actual_turning_points": int(actual_turn.sum()),
        "predicted_turning_points": int(predicted_turn.sum()),
    }


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "mape_pct": mape(predictions["actual"], predictions["predicted"]),
        "mae": mae(predictions["actual"], predictions["predicted"]),
        "direction_accuracy_pct": direction_accuracy(
            predictions["actual"], predictions["predicted"], predictions["current_price"]
        ),
        "n_origins": len(predictions),
    }
    result.update(
        turning_point_metrics(
            predictions["actual"],
            predictions["predicted"],
            predictions["current_price"],
            predictions["previous_price"],
        )
    )
    return result
