"""Past-only regularization/lag selection and period stability checks."""

import hashlib
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.combination_experiment import select_from_past
from src.evaluate import summarize_predictions
from src.feature_audit import build_design
from src.features import TARGET, validate_monthly_frame
from src.train import _pipeline, _to_price
from src.validation import make_horizon_targets, rolling_origin_splits

OUTPUT = Path("outputs_stability")
ALPHAS = {"ridge": (1., 10., 100.), "elastic_net": (.001, .01, .1)}
CENTERS = {"ridge": 10., "elastic_net": .01}
WINDOWS = (24, 36, 48)


def model_for(family, alpha):
    model = _pipeline(family + "_log_return")
    model.set_params(model__alpha=alpha)
    return model


def names_for(horizon):
    return ["current", "lag_1", "lag_3", "lag_6"] if horizon == 1 else ["current", "basic_processed"]


def pools(part, horizon, family):
    result = {"current_alpha": part.loc[part.combination == "current"]}
    if horizon == 1:
        result["lag_fixed_alpha"] = part.loc[part.alpha == CENTERS[family]]
        result["lag_alpha"] = part
    else:
        result["basic_alpha"] = part.loc[part.combination == "basic_processed"]
    return result


def select_policies(cache):
    rows, scores = [], []
    for (family, h), part in cache.groupby(["family", "horizon"]):
        for policy, pool in pools(part, h, family).items():
            for window in WINDOWS:
                for origin in range(120, 204-h):
                    name, table = select_from_past(pool, origin, h, window=window, min_origins=24)
                    row = pool.loc[(pool.origin == origin) & (pool.experiment == name)].copy()
                    assert len(row) == 1
                    row["selected_candidate"] = name
                    row["experiment"] = f"{family}__{policy}__w{window}"
                    row["policy"] = policy
                    row["window"] = window
                    row["role"] = "past_only_policy"
                    rows.append(row)
                    scores.append(table.assign(family=family, horizon=h, policy=policy, window=window,
                                               outer_origin=origin, selected=table.experiment.eq(name)))
    return pd.concat(rows, ignore_index=True), pd.concat(scores, ignore_index=True)


def slices(part):
    yield "all", part
    for year, sub in part.groupby(part.date.dt.year):
        yield str(year), sub
    yield "early_2013_2016", part.loc[part.date.dt.year <= 2016]
    yield "late_2017_2020", part.loc[part.date.dt.year >= 2017]
    shock = (part.actual/part.current_price-1).abs() >= .15
    yield "shock_abs_return_ge15pct", part.loc[shock]
    yield "nonshock", part.loc[~shock]


def evaluate(cache):
    selected, scores = select_policies(cache)
    assert (scores.inner_last_target <= scores.outer_origin).all()
    fixed = cache.loc[cache.origin >= 120].copy()
    fixed["policy"] = "fixed"
    fixed["window"] = 0
    fixed["role"] = "fixed_sensitivity_diagnostic"
    ref = pd.read_csv("outputs_statistics/predictions.csv", parse_dates=["date"])
    ref = ref.loc[ref.experiment == "naive", ["date", "origin", "horizon", "actual", "predicted", "current_price", "previous_price"]]
    naive = ref.assign(experiment="naive", family="baseline", combination="naive", alpha=np.nan,
                       policy="fixed", window=0, role="baseline")
    predictions = pd.concat([fixed, selected, naive], ignore_index=True)
    assert not predictions.duplicated(["experiment", "origin", "horizon"]).any()
    assert np.isfinite(predictions.predicted).all() and predictions.predicted.gt(0).all()
    paired = predictions.merge(ref, on=["origin", "horizon"], suffixes=("", "_ref"), validate="many_to_one")
    assert len(paired) == len(predictions)
    assert (paired.date == paired.date_ref).all()
    for col in ("actual", "current_price", "previous_price"):
        np.testing.assert_allclose(paired[col], paired[col+"_ref"], rtol=0, atol=1e-12)
    old = pd.read_csv("outputs_feature_audit/predictions.csv")
    centers = fixed.loc[fixed.apply(lambda r: r.alpha == CENTERS[r.family], axis=1)]
    old = old.loc[old.family.isin(ALPHAS)]
    matched = centers.merge(old, on=["family", "combination", "horizon", "origin"], suffixes=("", "_old"), validate="one_to_one")
    assert len(matched) == len(centers)
    np.testing.assert_allclose(matched.predicted, matched.predicted_old, rtol=1e-8, atol=1e-8)
    records = []
    for (name, h), part in predictions.groupby(["experiment", "horizon"]):
        assert set(part.origin) == set(range(120, 204-h))
        for period, sub in slices(part):
            if sub.empty:
                continue
            errors = (sub.predicted/sub.actual-1).abs()*100
            records.append(dict(experiment=name, horizon=h, family=part.family.iloc[0],
                                policy=part.policy.iloc[0], window=int(part.window.iloc[0]),
                                role=part.role.iloc[0], period=period, **summarize_predictions(sub),
                                p90_ape_pct=float(errors.quantile(.9)), max_ape_pct=float(errors.max()),
                                worst_origin_date=str(sub.loc[errors.idxmax(), "date"].date())))
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT / "metrics.csv", index=False)
    scores.to_csv(OUTPUT / "selection_scores.csv", index=False)
    selected.groupby(["family", "horizon", "policy", "window", "combination", "alpha"]).size().rename("count").reset_index().to_csv(OUTPUT / "selection_counts.csv", index=False)


def main():
    paths = list(Path(".").glob("*data_train.csv"))
    if len(paths) != 1:
        raise ValueError("Expected one train data file")
    data = validate_monthly_frame(pd.read_csv(paths[0]))
    if len(data) != 204:
        raise ValueError("Fixed stability design expects 204 months")
    features, _, specs = build_design(data)
    plan = {
        "train_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                        ("src/candidate_stability.py", "src/feature_audit.py", "src/train.py", "src/features.py", "src/hypotheses.py", "src/validation.py", "src/combination_experiment.py")},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
        "alphas": ALPHAS, "models": {f: model_for(f, CENTERS[f]).named_steps["model"].get_params() for f in ALPHAS},
        "candidates_by_horizon": {h: {c: specs[c] for c in names_for(h)} for h in (1, 2, 3)},
        "primary_selection_window": 36, "sensitivity_windows": WINDOWS, "minimum_inner_origins": 24,
        "first_inner_origin": 72, "first_outer_origin": 120,
        "selection": "within-family pools, known target j+h<=t, latest up to window cases, min24; H1 lag_alpha jointly selects alpha and common fundamental lag; no Naive in pools",
        "target": "separate log return H1/H2/H3, same preprocessing and conservative availability as feature audit",
        "periods": "2013-2016 / 2017-2020, annual; shock diagnostic abs(actual/current-1)>=15%, never used for selection",
        "device": "CPU single BLAS thread; fixed seed42 for ElasticNet; no GPU estimator in this experiment",
        "limitations": "retrospective already-inspected data; windows24/48 are sensitivity not a search for final winner; fixed alpha tables are posthoc diagnostics; no statistical significance claim or submission mutation",
    }
    # Normalize tuple/int JSON keys before comparing resumed manifests.
    plan = json.loads(json.dumps(plan))
    old_plan = json.loads(Path("outputs_feature_audit/manifest.json").read_text(encoding="utf-8"))
    assert old_plan["train_sha256"] == plan["train_sha256"]
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "cache").mkdir(exist_ok=True)
    manifest = OUTPUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing stability configuration differs; preserve previous results")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    parts, diagnostics = [], []
    with threadpool_limits(limits=1):
        for family, alphas in ALPHAS.items():
            for h in (1, 2, 3):
                target = make_horizon_targets(data, h)
                for combination in names_for(h):
                    for alpha in alphas:
                        name = f"{family}__{combination}__a{alpha:g}"
                        file = OUTPUT / "cache" / f"{name}_h{h}.csv"
                        diag_file = file.with_suffix(".json")
                        if file.exists() and diag_file.exists():
                            frame = pd.read_csv(file, parse_dates=["date"])
                            diag = json.loads(diag_file.read_text(encoding="utf-8"))
                        else:
                            rows, diag = [], []
                            for split in rolling_origin_splits(len(data), h, first_origin=72):
                                t = split.origin
                                model = model_for(family, alpha)
                                start = time.perf_counter()
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    model.fit(features.loc[split.train_indices, specs[combination]], target.loc[split.train_indices, "log_return"])
                                    raw = float(model.predict(features.loc[[t], specs[combination]])[0])
                                pred = _to_price(raw, float(data.loc[t, TARGET]), "log_return")
                                if not np.isfinite(pred):
                                    raise ValueError("Nonfinite prediction")
                                rows.append(dict(date=data.loc[t, "date"], origin=t, horizon=h, experiment=name,
                                                 family=family, combination=combination, alpha=alpha,
                                                 actual=target.loc[t, "future_price"], predicted=pred,
                                                 current_price=data.loc[t, TARGET], previous_price=data.loc[t-1, TARGET]))
                                diag.append(dict(experiment=name, horizon=h, origin=t, last_train_target=int(split.train_indices[-1]+h),
                                                 fit_seconds=time.perf_counter()-start, warnings=[str(w.message) for w in caught]))
                            frame = pd.DataFrame(rows)
                            frame.to_csv(file, index=False)
                            diag_file.write_text(json.dumps(diag, ensure_ascii=False), encoding="utf-8")
                        assert len(frame) == 132-h and set(frame.origin) == set(range(72, 204-h))
                        assert not frame.origin.duplicated().any()
                        assert frame.experiment.eq(name).all() and frame.horizon.eq(h).all()
                        parts.append(frame)
                        diagnostics.extend(diag)
                        print(f"DONE {name} H{h}", flush=True)
    cache = pd.concat(parts, ignore_index=True)
    cache.to_csv(OUTPUT / "inner_prediction_cache.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "fit_diagnostics.csv", index=False)
    assert all(d["last_train_target"] <= d["origin"] for d in diagnostics)
    evaluate(cache)
    from src.stability_report import main as make_report
    make_report()


if __name__ == "__main__":
    main()
