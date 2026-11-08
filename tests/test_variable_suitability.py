from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import TARGET, validate_monthly_frame
from src.variable_suitability import assess, backtest, block_interval, build_features, redundancy


@pytest.fixture(scope="module")
def inputs():
    data = validate_monthly_frame(pd.read_csv(next(Path('.').glob('*data_train.csv'))))
    metadata = pd.read_excel(next(Path('.').glob('*metadata.xlsx')))
    return data, metadata


def test_all_inputs_explicitly_mapped(inputs):
    data, meta = inputs
    a = assess(data, meta)
    assert len(a) == 112 and a.variable.is_unique
    assert set(a.variable) == set(data)-{"date", TARGET}
    assert a.metadata_matched.all()
    assert not a.historical_release_verified.any()
    assert a.set_index("variable").loc["Brent_Crude_Oil", "metadata_name"] == "Brent_Crude_Oil_Daily"
    with pytest.raises(ValueError, match="Unmapped metadata"):
        assess(data, meta.loc[meta.name != "high_temperature"])


def test_gate_is_independent_of_future_values_and_targets(inputs):
    data, meta = inputs
    before = assess(data, meta)
    changed = data.copy()
    changed.loc[120:, changed.columns != "date"] = np.nan
    changed[TARGET] = 10000
    after = assess(changed, meta)
    for col in ("predictive_eligible", "gate", "gate_reason", "initial120_coverage_pct", "lag_months_assumed"):
        pd.testing.assert_series_equal(before[col], after[col])


def test_held_variables_do_not_enter_any_candidate(inputs):
    data, meta = inputs
    a = assess(data, meta)
    _, specs = build_features(data, a)
    assert a.predictive_eligible.sum() == 78
    assert len(specs) == 106
    for row in a.loc[~a.predictive_eligible].itertuples():
        assert "single_"+row.variable_id not in specs
        assert not any(c.startswith(row.variable_id+"_") for cols in specs.values() for c in cols)
    for cols in specs.values():
        assert len(cols) == len(set(cols))


def test_all_new_features_are_prefix_safe(inputs):
    data, meta = inputs
    a = assess(data, meta)
    x, _ = build_features(data, a)
    changed = data.copy()
    changed.loc[160:, changed.columns != "date"] *= 100
    other, _ = build_features(changed, a)
    prefix, _ = build_features(data.iloc[:160], a)
    pd.testing.assert_frame_equal(x.iloc[:160], other.iloc[:160])
    pd.testing.assert_frame_equal(x.iloc[:160], prefix)


def test_lag_sensitivity_changes_only_the_intended_columns(inputs):
    data, meta = inputs
    a = assess(data, meta)
    x, specs = build_features(data, a)
    for row in a.loc[a.predictive_eligible].itertuples():
        expected = data[row.variable].ffill(limit=1).shift(row.lag_months_assumed)
        pd.testing.assert_series_equal(x[row.variable_id+"_level"], expected, check_names=False)
    for name, cols in specs.items():
        if name.startswith("delay_"):
            for c in cols[8:]:
                pd.testing.assert_series_equal(x[c], x[c.removesuffix("_delay1")].shift(1), check_names=False)
    pd.testing.assert_series_equal(x.storage_seasonal_z_lag2, x.storage_seasonal_z.shift(1), check_names=False)
    left, right = map(set, (specs["processed_reference"], specs["processed_storage_lag2"]))
    assert left-right == {"storage_seasonal_z"} and right-left == {"storage_seasonal_z_lag2"}


def test_redundancy_uses_only_initial_prefix(inputs):
    data, meta = inputs
    a = assess(data, meta)
    changed = data.copy()
    changed.loc[120:, changed.columns != "date"] *= -999
    pd.testing.assert_frame_equal(redundancy(data, a), redundancy(changed, a))


def test_block_intervals_are_reproducible_and_paired():
    assert block_interval(np.full(81, -2)) == (-2, -2)
    assert block_interval(np.arange(83)) == block_interval(np.arange(83))


def test_backtest_purges_unavailable_training_labels(inputs, monkeypatch):
    data, _ = inputs
    calls = []
    class Recorder:
        def fit(self, x, y):
            calls.append((list(x.index), list(y.index)))
            return self
        def predict(self, x):
            assert list(x.index) == [120]
            return [0.0]
    monkeypatch.setattr("src.variable_suitability._pipeline", lambda _: Recorder())
    part, notes = backtest(data.iloc[:124], pd.DataFrame({"x": data[TARGET]}), ["x"], "ridge", "test", 3)
    assert calls == [(list(range(118)), list(range(118)))]
    assert len(part) == 1 and not notes
    assert part.predicted.iloc[0] == data[TARGET].iloc[120]
    assert part.actual.iloc[0] == data[TARGET].iloc[123]
