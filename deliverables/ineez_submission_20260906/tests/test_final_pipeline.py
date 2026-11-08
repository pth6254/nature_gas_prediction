from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import TARGET, validate_monthly_frame
from src.final_pipeline import RAW, features_and_specs, forecast, validate_submission
from src.incremental_fundamentals import design


@pytest.fixture(scope="module")
def data():
    return validate_monthly_frame(pd.read_csv(next(Path('.').glob('*data_train.csv'))))


def test_final_features_exactly_match_selected_experiment(data):
    x, specs = features_and_specs(data)
    for h in (1, 2, 3):
        old, old_specs = design(data, h)
        for name, cols in specs[h].items():
            assert cols == old_specs[name]
            pd.testing.assert_frame_equal(x[cols], old[cols])


def test_only_final_raw_inputs_are_required(data):
    minimal = data[["date", TARGET, *RAW.values()]]
    x, specs = features_and_specs(minimal)
    original, original_specs = features_and_specs(data)
    pd.testing.assert_frame_equal(x, original)
    assert specs == original_specs


def test_future_mutations_do_not_change_forecasts_or_past_selection(data):
    minimal = data[["date", TARGET, *RAW.values()]].copy()
    before, _, scores, _, _ = forecast(minimal, first_output=120, last_origin=122, explain=False)
    minimal.loc[123:, minimal.columns != "date"] *= 100
    after, _, after_scores, _, _ = forecast(minimal, first_output=120, last_origin=122, explain=False)
    columns = [c for c in before if c != "actual"]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    pd.testing.assert_frame_equal(scores, after_scores)
    assert (scores.inner_last_target <= scores.outer_origin).all()
    assert (before.train_last_target == before.origin).all()
    assert (before.train_last_feature+before.horizon == before.origin).all()


def test_unobserved_future_targets_can_be_forecast_and_explained(data):
    p, _, _, c, warnings = forecast(data.iloc[:121], first_output=120, explain=True)
    assert len(p) == 3 and p.actual.isna().all() and not warnings
    for row in p.itertuples():
        sub = c.loc[(c.origin == row.origin) & (c.horizon == row.horizon)]
        assert len(sub) >= row.n_features
        np.testing.assert_allclose(sub.base_log_return.iloc[0]+sub.centered_log_return_contribution.sum(), row.predicted_log_return, atol=1e-10)


def test_submission_rejects_bad_values_and_date_order():
    t = pd.DataFrame({"date": ["2020-09-30", "2020-10-31"]})
    s = t.assign(pred_1=3., pred_2=4., pred_3=5.)
    validate_submission(s, t)
    for bad in (s.iloc[::-1], s.assign(pred_1=np.inf), s.assign(pred_2=0), s.drop(columns="pred_3")):
        with pytest.raises(ValueError):
            validate_submission(bad, t)
