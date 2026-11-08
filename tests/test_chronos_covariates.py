import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.chronos_covariates import GROUPS, task_at, predict_one
from src.hypotheses import build_hypotheses
from src.features import TARGET, FUNDAMENTAL_COLUMNS, ENERGY_COLUMNS


def sample():
    t = np.arange(140, dtype=float)
    data = pd.DataFrame({"date": pd.date_range("2010-01-31", periods=140, freq="ME"), TARGET: 3+t/100})
    for i, raw in enumerate([*FUNDAMENTAL_COLUMNS.values(), *ENERGY_COLUMNS, "high_temperature", "low_temperature"]):
        data[raw] = 100+i+t+10*np.sin(t*np.pi/6)
    return data


def test_chronos_covariates_share_schema_and_past_only():
    data = sample()
    features, _, _ = build_hypotheses(data)
    for group, cols in GROUPS.items():
        task = task_at(data, features, 120, group)
        assert "future_covariates" not in task
        assert set(task.get("past_covariates", {})) == set(cols)
        np.testing.assert_array_equal(task["target"], data.loc[97:120, TARGET].to_numpy(np.float32))
        for col in cols:
            np.testing.assert_array_equal(task["past_covariates"][col], features.loc[97:120, col].to_numpy(np.float32))


def test_chronos_covariates_ignore_future_raw_data():
    data = sample()
    features, _, _ = build_hypotheses(data)
    before = task_at(data, features, 120, "market")
    data.loc[121:, data.columns != "date"] *= 100
    changed, _, _ = build_hypotheses(data)
    after = task_at(data, changed, 120, "market")
    np.testing.assert_array_equal(before["target"], after["target"])
    for col in GROUPS["market"]:
        np.testing.assert_array_equal(before["past_covariates"][col], after["past_covariates"][col])


def test_chronos2_isolates_origins_and_uses_median():
    class Pipeline:
        quantiles = [.1, .5, .9]
        def predict(self, tasks, **kwargs):
            assert len(tasks) == 1
            assert kwargs["cross_learning"] is False
            assert kwargs["context_length"] == 24 and kwargs["prediction_length"] == 3
            return [torch.tensor([[[1., 1., 1.], [2., 3., 4.], [99., 99., 99.]]])]
    np.testing.assert_array_equal(predict_one(Pipeline(), {"target": np.ones(24)}), [2, 3, 4])
