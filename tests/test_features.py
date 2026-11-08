import numpy as np
import pandas as pd

from src.features import TARGET, make_feature_matrix


def sample_data(n: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    base = np.arange(n, dtype=float) + 10
    return pd.DataFrame(
        {
            "date": dates,
            TARGET: 2 + base / 100,
            "Total_Natural_Gas_Marketed_Production_USA": 1000 + base,
            "Natural_Gas_Imports_USA": 100 + base,
            "Natural_Gas_Imports_From_Canada_USA": 80 + base,
            "Natural_Gas_Rotary_Rig_Count_USA": 50 + base,
            "Natural_Gas_Total_Consumption_USA": 900 + base,
            "Total_Natural_Gas_Underground_Storage_Volume_USA": 2000 + base,
        }
    )


def test_features_are_causal_when_future_changes() -> None:
    original = sample_data()
    changed = original.copy()
    changed.loc[25:, TARGET] = 99
    features_original, _ = make_feature_matrix(original)
    features_changed, _ = make_feature_matrix(changed)
    pd.testing.assert_frame_equal(features_original.iloc[:25], features_changed.iloc[:25])


def test_conservative_release_lag_is_applied() -> None:
    features, _ = make_feature_matrix(sample_data(), availability_mode="conservative")
    assert np.isnan(features.loc[0, "production__level"])
    assert np.isnan(features.loc[1, "production__level"])
    assert features.loc[2, "production__level"] == 1010

