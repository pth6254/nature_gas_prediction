import numpy as np
import pandas as pd
import pytest

from src.ensemble import evaluate_ensemble


def forecasts():
    ridge = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-01", "2020-01-01"]),
        "horizon": 3, "feature_group": "E2_fundamentals",
        "stage": "model_comparison", "model": "ridge_log_return",
        "target_type": "log_return", "actual": [2., 4.],
        "predicted": [4., 2.], "current_price": [2., 4.],
        "previous_price": [1., 5.],
    })
    naive = ridge.assign(model="last_value_naive", predicted=ridge["current_price"])
    return pd.concat([ridge, naive.iloc[::-1]], ignore_index=True)


def test_pairing_and_yearly_metrics():
    source = forecasts()
    predictions, metrics = evaluate_ensemble(source)
    blend = predictions.loc[predictions["model"] == "ridge_naive_equal_blend"]
    np.testing.assert_allclose(blend["predicted"], [3., 3.])
    summary = metrics.loc[(metrics["model"] == "ridge_naive_equal_blend")
                          & (metrics["validation_year"] == "all")].iloc[0]
    assert summary["mape_pct"] == 37.5
    assert set(metrics["validation_year"]) == {"all", 2019, 2020}
    changed = source.assign(actual=source["actual"] * 2)
    updated, _ = evaluate_ensemble(changed)
    np.testing.assert_array_equal(predictions["predicted"], updated["predicted"])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "actual", "nan"])
def test_rejects_invalid_comparisons(problem):
    source = forecasts()
    if problem == "missing":
        source = source.iloc[:-1]
    elif problem == "duplicate":
        source = pd.concat([source, source.iloc[[0]]])
    elif problem == "actual":
        source.loc[2, "actual"] = 999.
    else:
        source.loc[0, "predicted"] = np.nan
    with pytest.raises(ValueError):
        evaluate_ensemble(source)
