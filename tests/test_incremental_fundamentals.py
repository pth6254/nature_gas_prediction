from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import validate_monthly_frame
from src.incremental_fundamentals import design
from src.storage_source_check import parse_series


@pytest.fixture(scope="module")
def data():
    return validate_monthly_frame(pd.read_csv(next(Path('.').glob('*data_train.csv'))))


def test_small_equal_search_budgets(data):
    for h in (1, 2, 3):
        x, specs = design(data, h)
        assert len(specs) == (16 if h == 1 else 4)
        for lag in ((0, 1, 3, 6) if h == 1 else (0,)):
            base = set(specs[f"base_lag{lag}"])
            assert len(base) == (22 if h == 1 else 12)
            assert set(specs[f"imports_lag{lag}"])-base == {"add_imports_level", "add_imports_diff1"}
            assert set(specs[f"rigs_lag{lag}"])-base == {"add_rigs_level", "add_rigs_diff1"}
            assert len(set(specs[f"both_lag{lag}"])-base) == 4
        assert all(len(c) == len(set(c)) and set(c) <= set(x) for c in specs.values())


def test_all_features_and_new_release_lags_are_causal(data):
    x, _ = design(data, 1)
    other = data.copy()
    other.loc[160:, other.columns != "date"] *= 999
    y, _ = design(other, 1)
    prefix, _ = design(data.iloc[:160], 1)
    pd.testing.assert_frame_equal(x.iloc[:160], y.iloc[:160])
    pd.testing.assert_frame_equal(x.iloc[:160], prefix)
    pd.testing.assert_series_equal(x.storage_seasonal_z_lag2, x.storage_seasonal_z.shift(1), check_names=False)
    pd.testing.assert_series_equal(x.add_imports_level, data.Natural_Gas_Imports_USA.shift(2), check_names=False)
    for lag in (1, 3, 6):
        pd.testing.assert_series_equal(x[f"storage_seasonal_z_lag2_reaction{lag}"], x.storage_seasonal_z_lag2.shift(lag), check_names=False)
    assert np.isfinite(x.add_rigs_level).all()


def test_official_parser_maps_calendar_months_and_skips_blanks():
    row = "<tr><td>&nbsp;2003</td>"+"".join(f"<td>{v}</td>" for v in ["1,234", *["NA"]*10, "5,678"])+"</tr>"
    result = parse_series(row)
    assert result.to_dict("records") == [{"month": "2003-01", "eia_value": 1234}, {"month": "2003-12", "eia_value": 5678}]
    with pytest.raises(ValueError):
        parse_series(row+row)


def test_cached_official_match_is_calendar_aligned():
    path = Path("outputs_storage_source/n5030us2m.html")
    if not path.exists():
        pytest.skip("Optional downloaded official snapshot")
    parsed = parse_series(path.read_text(encoding="utf-8"))
    assert parsed.month.is_unique
    matches = pd.read_csv("outputs_storage_source/matched_training_months.csv")
    assert len(matches) == 204 and matches.matches_nearest10.all()
    assert matches.difference_mmcf.abs().max() <= 5
