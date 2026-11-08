import numpy as np
import pandas as pd

from src.feature_audit import build_design
from src.features import TARGET, FUNDAMENTAL_COLUMNS, ENERGY_COLUMNS


def sample():
    t = np.arange(84, dtype=float)
    frame = pd.DataFrame({"date": pd.date_range("2010-01-31", periods=84, freq="ME"), TARGET: 3+t/100})
    for i, raw in enumerate([*FUNDAMENTAL_COLUMNS.values(), *ENERGY_COLUMNS, "high_temperature", "low_temperature"]):
        frame[raw] = 100+i+t+10*np.sin(t*np.pi/6)
    return frame


def test_audit_features_are_prefix_safe():
    data = sample()
    original, _, specs = build_design(data)
    changed = data.copy()
    changed.loc[60:, changed.columns != "date"] *= 100
    other, _, _ = build_design(changed)
    prefix, _, _ = build_design(data.iloc[:60])
    pd.testing.assert_frame_equal(original.iloc[:60], other.iloc[:60])
    pd.testing.assert_frame_equal(original.iloc[:60], prefix)
    assert len(specs) == 13
    for cols in specs.values():
        assert len(set(cols)) == len(cols) and set(cols).issubset(original.columns)


def test_audit_delays_and_compression_are_controlled():
    data = sample()
    x, _, specs = build_design(data)
    for lag in (1, 3, 6):
        for col in ("production_level", "storage_seasonal_z", "consumption_z"):
            pd.testing.assert_series_equal(x[f"{col}_extra_lag{lag}"], x[col].shift(lag), check_names=False)
        assert "BCOMNG_INDX_return" in specs[f"lag_{lag}"]
    assert x.production_level.iloc[60] == data[FUNDAMENTAL_COLUMNS["production"]].iloc[58]
    np.testing.assert_allclose(x.balance_equal, x.production_z+x.storage_z-x.consumption_z, equal_nan=True)
    left = set(specs["individual_z"])-{"production_z", "storage_z", "consumption_z"}
    assert left == set(specs["equal_index"])-{"balance_equal"}


def test_audit_ablation_removes_exactly_one_component():
    _, _, specs = build_design(sample())
    for role, col in [("production", "production_level"), ("storage", "storage_seasonal_z"),
                      ("consumption", "consumption_z"), ("market", "BCOMNG_INDX_return")]:
        assert set(specs["current"])-set(specs[f"drop_{role}"]) == {col}
        assert set(specs[f"drop_{role}"]).issubset(specs["current"])
