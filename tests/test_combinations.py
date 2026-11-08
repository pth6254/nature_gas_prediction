import pandas as pd
import pytest

from src.combination_experiment import combinations, select_from_past


def cache():
    rows = []
    for origin in range(30):
        for name, predicted in [("A",2.1),("naive",2.0)]:
            rows.append({"origin":origin,"horizon":3,"experiment":name,
                         "actual":2.1,"predicted":predicted,"current_price":2.,"previous_price":1.9})
    return pd.DataFrame(rows)


def test_selection_ignores_not_yet_observed_targets():
    predictions = cache()
    chosen, scores = select_from_past(predictions,27,3)
    assert chosen == "A"
    assert scores.inner_last_origin.eq(24).all()
    assert scores.inner_last_target.eq(27).all()
    changed = predictions.copy()
    changed.loc[changed.origin>24,"actual"] = 1000
    changed.loc[changed.origin>24,"predicted"] = -1000
    other, other_scores = select_from_past(changed,27,3)
    assert other == chosen
    pd.testing.assert_frame_equal(scores,other_scores)


def test_selection_requires_paired_dates_and_minimum_history():
    with pytest.raises(ValueError,match="Not enough"):
        select_from_past(cache(),10,3)
    with pytest.raises(ValueError,match="identical"):
        select_from_past(cache().iloc[1:],27,3)


def test_naive_wins_tie_and_window_is_bounded():
    frame = cache().assign(actual=2.,predicted=2.)
    selected,scores = select_from_past(frame,29,3,window=24)
    assert selected == "naive"
    assert scores.inner_n.eq(24).all()
    assert scores.inner_first_origin.eq(3).all()


def test_market_changes_only_one_input():
    specs = combinations(["price_now"])
    assert len(specs)==8
    for name in "ABCD":
        assert specs[name+"_market"] == specs[name]+["BCOMNG_INDX_return"]
    assert set(specs["B"])-set(specs["A"]) == {"production_level"}
    assert set(specs["C"])-set(specs["A"]) == {"storage_seasonal_z"}
