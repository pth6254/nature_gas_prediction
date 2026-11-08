import numpy as np
import pandas as pd

from src.candidate_stability import ALPHAS, CENTERS, model_for, names_for, pools, slices
from src.combination_experiment import select_from_past
from src.train import _pipeline


def test_stability_changes_only_alpha():
    for family, alphas in ALPHAS.items():
        baseline = _pipeline(family+"_log_return").named_steps["model"].get_params()
        for alpha in alphas:
            params = model_for(family, alpha).named_steps["model"].get_params()
            assert params == dict(baseline, alpha=alpha)


def test_stability_pools_separate_questions():
    part = pd.DataFrame([dict(combination=c, alpha=a) for c in names_for(1) for a in ALPHAS["ridge"]])
    groups = pools(part, 1, "ridge")
    assert len(groups["current_alpha"]) == 3
    assert len(groups["lag_alpha"]) == 12
    assert groups["lag_fixed_alpha"].alpha.eq(CENTERS["ridge"]).all()
    assert names_for(2) == names_for(3) == ["current", "basic_processed"]


def test_stability_selection_cannot_see_unknown_targets():
    frame = pd.DataFrame([dict(origin=t, horizon=1, experiment=f"a{a}", actual=2.,
                               predicted=2.+a, current_price=2., previous_price=1.9)
                          for t in range(72, 125) for a in [0., .1]])
    chosen, before = select_from_past(frame, 120, 1, window=36)
    frame.loc[frame.origin >= 120, "actual"] = 999.
    frame.loc[frame.origin >= 120, "predicted"] = -999.
    after_name, after = select_from_past(frame, 120, 1, window=36)
    assert chosen == after_name
    pd.testing.assert_frame_equal(before, after)
    assert before.inner_last_target.eq(120).all()


def test_stability_period_and_shock_masks():
    part = pd.DataFrame(dict(date=pd.to_datetime(["2016-12-31", "2017-01-31", "2018-01-31"]),
                             actual=[1., 1.2, .8], current_price=[1., 1., 1.]))
    groups = dict(slices(part))
    assert len(groups["early_2013_2016"]) == 1
    assert len(groups["late_2017_2020"]) == 2
    assert len(groups["shock_abs_return_ge15pct"]) == 2
    assert len(groups["nonshock"]) == 1
