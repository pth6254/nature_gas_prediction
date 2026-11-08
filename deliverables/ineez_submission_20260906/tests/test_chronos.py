import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.chronos_experiment import context_at, median_forecast


def test_chronos_context_excludes_future_and_older_history():
    values = np.arange(1., 205.)
    expected = context_at(values, 120)
    values[121:] = 9999
    values[:97] = 8888
    torch.testing.assert_close(expected, context_at(values, 120))
    assert len(expected) == 24 and expected[-1] == 121


def test_chronos_uses_median_not_quantile_mean():
    class Pipeline:
        quantiles = [.1, .5, .9]
        def predict(self, contexts, prediction_length, limit_prediction_length):
            assert prediction_length == 3 and limit_prediction_length
            return torch.tensor([[[1., 1., 1.], [2., 3., 4.], [99., 99., 99.]]])
    np.testing.assert_array_equal(median_forecast(Pipeline(), torch.ones(1, 24)), [[2., 3., 4.]])


def test_chronos_rejects_missing_context():
    with pytest.raises(ValueError):
        context_at(np.ones(20), 19)
