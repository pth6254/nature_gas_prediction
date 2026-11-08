import numpy as np
import pandas as pd

from src.features import TARGET, FUNDAMENTAL_COLUMNS, ENERGY_COLUMNS
from src.hypotheses import build_hypotheses
from src.run_hypotheses import fit_predict


def sample():
    t = np.arange(84,dtype=float)
    data = pd.DataFrame({"date":pd.date_range("2010-01-31",periods=len(t),freq="ME"),
                         TARGET:3+t/100})
    for i,raw in enumerate([*FUNDAMENTAL_COLUMNS.values(),*ENERGY_COLUMNS,"high_temperature","low_temperature"]):
        data[raw] = 100+i+t+10*np.sin(t*np.pi/6)
    return data


def test_all_features_ignore_future_values_and_match_prefix():
    data = sample()
    original,_,specs = build_hypotheses(data)
    changed = data.copy()
    changed.loc[60:,changed.columns!="date"] *= 10
    other,_,_ = build_hypotheses(changed)
    prefix,_,_ = build_hypotheses(data.iloc[:60])
    pd.testing.assert_frame_equal(original.iloc[:60],other.iloc[:60])
    pd.testing.assert_frame_equal(original.iloc[:60],prefix)
    assert len({s.name for s in specs}) == len(specs)
    for spec in specs:
        assert set(spec.columns).issubset(original.columns)
        assert spec.parent in {s.name for s in specs}|{"naive"}


def test_seasonal_reference_month_and_equal_index():
    data = sample()
    features,_,_ = build_hypotheses(data)
    s = data[FUNDAMENTAL_COLUMNS["production"]]
    historical = s.iloc[[0,12,24]]
    expected = (s.iloc[36]-historical.mean())/historical.std()
    assert np.isclose(features.loc[38,"production_seasonal_z"],expected)
    np.testing.assert_allclose(features.balance_equal,
                               features.production_z+features.storage_z-features.consumption_z,
                               equal_nan=True)


def test_constraints_apply_to_core_only_and_preprocessor_uses_train():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(80,4)),columns=["production_z","storage_z","consumption_z","history"])
    # Deliberately contrary to the economic sign restrictions.
    y = x.production_z+x.storage_z-x.consumption_z-2*x.history
    _,names,coef,_ = fit_predict(x,y,x.iloc[[0]],True)
    values = dict(zip(names,coef))
    assert values["production_z"] <= 1e-8
    assert values["storage_z"] <= 1e-8
    assert values["consumption_z"] >= -1e-8
    assert values["history"] < 0
    _,_,other,_ = fit_predict(x,y,x.iloc[[0]]*10000,True)
    np.testing.assert_allclose(coef,other)
