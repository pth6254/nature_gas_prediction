import numpy as np
import pytest

from src.statistical_experiment import forecast


def test_naive_and_seasonal_target_alignment():
    prices = np.arange(1.,37.)
    np.testing.assert_array_equal(forecast(prices,"naive")[0],[36,36,36])
    np.testing.assert_array_equal(forecast(prices,"seasonal_naive")[0],[25,26,27])


def test_level_smoothing_on_constant_price():
    predicted,status,_ = forecast(np.repeat(3.,48),"ets_level")
    assert status == "ok"
    np.testing.assert_allclose(predicted,3.,atol=1e-6)


def test_invalid_history_is_not_silently_scored():
    with pytest.raises(ValueError):
        forecast(np.zeros(48),"ets_level")


def test_failed_optimizer_is_disclosed(monkeypatch):
    import src.statistical_experiment as module
    def fail(*args,**kwargs):
        raise RuntimeError("test failure")
    monkeypatch.setattr(module,"SARIMAX",fail)
    predicted,status,message = forecast(np.repeat(3.,48),"sarima_110_000")
    assert status == "fallback_naive"
    assert message == "test failure"
    np.testing.assert_array_equal(predicted,[3.,3.,3.])
