"""Locked final v1: ElasticNet H1 lag policy, Ridge H2/H3; storage lag2.

Run from project root: python -m src.final_pipeline
No feature/model/hyperparameter search using test performance. H1 updates only
its predeclared reaction-lag policy using outcomes already observed at each origin.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.combination_experiment import select_from_past
from src.evaluate import summarize_predictions
from src.features import TARGET, validate_monthly_frame

OUT = Path("outputs_final_v1")
BACKUP = OUT / "previous_submission.csv"
RAW = {"production": "Total_Natural_Gas_Marketed_Production_USA",
       "storage": "Total_Natural_Gas_Underground_Storage_Volume_USA",
       "consumption": "Natural_Gas_Total_Consumption_USA", "market": "BCOMNG_INDX"}
CONFIG = {
    "version": "final_v1", "models": {"1": "elastic_net", "2": "ridge", "3": "ridge"},
    "ridge_alpha": 10.0, "elastic_net_alpha": .01, "elastic_net_l1_ratio": .2,
    "h1_reaction_lags": [0, 1, 3, 6], "selection_window": 36, "selection_min_origins": 24,
    "first_inner_origin": 72, "first_validation_origin": 120,
    "release_lags": {"production": 2, "storage": 2, "consumption": 2, "market": 0},
    "raw_inputs": RAW, "price_target": TARGET,
    "target": "log(P[t+h]/P[t]); prediction=max(.01,P[t]*exp(model_output)); no blend",
    "deployment": "monthly expanding refit; observed current test price usable only at/after its own origin",
    "limitations": "retrospective inspected validation, not untouched holdout; historical release calendar and vintage incompletely verified; monthly forecast after releases at month end",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features_and_specs(data):
    data = validate_monthly_frame(data)
    p = data[TARGET].astype(float)
    f = {"price_now": p}
    for lag in (1, 2, 3, 6, 12):
        f[f"price_lag_{lag}"] = p.shift(lag)
    for lag in (1, 3, 6, 12):
        f[f"price_log_return_{lag}"] = np.log(p/p.shift(lag))
    for window in (3, 6, 12):
        f[f"price_roll_mean_{window}"] = p.rolling(window).mean()
        f[f"price_roll_std_{window}"] = p.rolling(window).std()
    month = data.date.dt.month
    f["month_sin"], f["month_cos"] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    history = list(f)
    basic = history[:6]+["month_sin", "month_cos"]
    production = pd.to_numeric(data[RAW["production"]], errors="raise")
    storage = pd.to_numeric(data[RAW["storage"]], errors="raise")
    consumption = pd.to_numeric(data[RAW["consumption"]], errors="raise")
    market = pd.to_numeric(data[RAW["market"]], errors="raise")
    seasonal_mean = storage.groupby(month).transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    seasonal_std = storage.groupby(month).transform(lambda s: s.shift(1).expanding(min_periods=3).std())
    f["production_level"] = production.shift(2).ffill()
    f["storage_seasonal_z_lag2"] = ((storage-seasonal_mean)/seasonal_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).shift(2).ffill()
    expanding = consumption.expanding(min_periods=12)
    f["consumption_z"] = ((consumption-expanding.mean())/expanding.std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).shift(2).ffill()
    f["BCOMNG_INDX_return"] = market.pct_change(fill_method=None).ffill()
    fundamental = ["production_level", "storage_seasonal_z_lag2", "consumption_z"]
    specs = {1: {}, 2: {"base_lag0": basic+fundamental+["BCOMNG_INDX_return"]},
             3: {"base_lag0": basic+fundamental+["BCOMNG_INDX_return"]}}
    for lag in CONFIG["h1_reaction_lags"]:
        cols = []
        for name in fundamental:
            new = name if lag == 0 else f"{name}_reaction{lag}"
            if lag:
                f[new] = f[name].shift(lag)
            cols.append(new)
        specs[1][f"base_lag{lag}"] = history+cols+["BCOMNG_INDX_return"]
    x = pd.DataFrame(f)
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Nonfinite feature: inspect input rather than silently clip")
    return x, specs


def pipeline(h):
    model = ElasticNet(alpha=.01, l1_ratio=.2, max_iter=20000, random_state=42) if h == 1 else Ridge(alpha=10.)
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("model", model)])


def forecast(data, first_output=120, last_origin=None, explain=True):
    """Causal replay including origins without yet-observed future prices."""
    x, specs = features_and_specs(data)
    n = len(data)
    last_origin = n-1 if last_origin is None else last_origin
    price = data[TARGET].astype(float)
    targets = {h: np.log(price.shift(-h)/price) for h in (1, 2, 3)}
    inner, output, score_rows, contributions, diagnostics = [], [], [], [], []
    with threadpool_limits(limits=1):
        for origin in range(CONFIG["first_inner_origin"], last_origin+1):
            for h in (1, 2, 3):
                if h != 1 and origin < first_output:
                    continue
                indices = np.arange(origin-h+1)
                assert indices[-1]+h == origin
                fitted = {}
                for name, columns in specs[h].items():
                    model = pipeline(h)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(x.loc[indices, columns], targets[h].loc[indices])
                        raw = float(model.predict(x.loc[[origin], columns])[0])
                    pred = max(.01, float(price.iloc[origin]*np.exp(raw)))
                    if not np.isfinite(pred):
                        raise ValueError("Nonfinite forecast")
                    row = dict(origin=origin, date=data.date.iloc[origin], horizon=h,
                               experiment=f"{CONFIG['models'][str(h)]}__{name}", combination=name,
                               predicted=pred, current_price=price.iloc[origin], previous_price=price.iloc[origin-1],
                               actual=price.iloc[origin+h] if origin+h < n else np.nan,
                               train_last_feature=origin-h, train_last_target=origin,
                               predicted_log_return=raw)
                    if h == 1:
                        inner.append(row)
                    fitted[name] = (row, model, columns)
                    diagnostics.extend(dict(origin=origin, horizon=h, candidate=name, warning=str(w.message)) for w in caught)
                if origin < first_output:
                    continue
                if h == 1:
                    chosen, scores = select_from_past(pd.DataFrame(inner), origin, 1,
                                                     window=CONFIG["selection_window"], min_origins=CONFIG["selection_min_origins"])
                    name = chosen.split("__", 1)[1]
                    score_rows.append(scores.assign(outer_origin=origin, selected=scores.experiment.eq(chosen)))
                else:
                    name = "base_lag0"
                row, model, columns = fitted[name]
                output.append({**row, "family": CONFIG["models"][str(h)], "reaction_lag": int(name.removeprefix("base_lag")),
                               "n_features": len(columns), "target_date": data.date.iloc[origin]+pd.offsets.MonthEnd(h)})
                if explain:
                    train_z = model[:-1].transform(x.loc[indices, columns])
                    current_z = model[:-1].transform(x.loc[[origin], columns])[0]
                    center = train_z.mean(axis=0)
                    coef = model.named_steps["model"].coef_
                    values = (current_z-center)*coef
                    base = float(model.named_steps["model"].intercept_+center@coef)
                    np.testing.assert_allclose(base+values.sum(), row["predicted_log_return"], rtol=0, atol=1e-10)
                    for feature, value in zip(model[:-1].get_feature_names_out(columns), values):
                        contributions.append(dict(origin=origin, date=row["date"], horizon=h, feature=feature,
                                                  centered_log_return_contribution=float(value), base_log_return=base))
            if origin >= first_output and origin % 24 == 0:
                print(f"Completed origin {origin}", flush=True)
    return pd.DataFrame(output), pd.DataFrame(inner), pd.concat(score_rows, ignore_index=True), pd.DataFrame(contributions), diagnostics


def validate_submission(frame, template):
    if list(frame.columns) != ["date", "pred_1", "pred_2", "pred_3"]:
        raise ValueError("Incorrect submission columns/order")
    if len(frame) != len(template) or frame.date.duplicated().any() or frame.isna().any().any():
        raise ValueError("Invalid submission shape or missing values")
    if not pd.to_datetime(frame.date).reset_index(drop=True).equals(pd.to_datetime(template.date).reset_index(drop=True)):
        raise ValueError("Submission dates/order differ from template")
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Predictions must be positive and finite")


def main():
    paths = {key: next(Path('.').glob('*'+suffix)) for key, suffix in
             (("train", "data_train.csv"), ("test", "data_test.csv"), ("template", "submission_example.csv"))}
    OUT.mkdir(exist_ok=True)
    # The configuration is frozen before test values are read.
    config_path = OUT/"model_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != CONFIG:
        raise ValueError("Final v1 already has a different configuration")
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    train = validate_monthly_frame(pd.read_csv(paths["train"]))
    test = validate_monthly_frame(pd.read_csv(paths["test"]))
    template = pd.read_csv(paths["template"], parse_dates=["date"])
    assert len(train) == 204 and len(test) == len(template) == 48
    assert train.date.max() < test.date.min()
    assert list(template.columns) == ["date", "pred_1", "pred_2", "pred_3"]
    assert set(test.date) == set(template.date)
    data = validate_monthly_frame(pd.concat([train, test], ignore_index=True))
    if not np.isfinite(data[TARGET]).all():
        raise ValueError("Invalid observed price")
    x, specs = features_and_specs(data)
    raw = [TARGET, *RAW.values()]
    quality = data[raw].isna().sum()
    print("Final raw input missing counts:", quality.to_dict(), flush=True)
    predictions, inner, scores, contributions, diagnostics = forecast(data)
    validation = predictions.loc[(predictions.origin < len(train)) & (predictions.origin+predictions.horizon < len(train))].copy()
    reference = pd.read_csv("outputs_incremental_fundamentals/predictions.csv", parse_dates=["date"])
    for h in (1, 2, 3):
        part = validation.loc[validation.horizon == h]
        ref = reference.loc[(reference.family == CONFIG["models"][str(h)]) & (reference.horizon == h) & (reference.policy == "base")]
        paired = part.merge(ref, on=["origin", "date", "horizon"], suffixes=("", "_ref"), validate="one_to_one")
        assert len(paired) == len(part) == len(ref) == 84-h
        for col in ("predicted", "actual", "current_price"):
            np.testing.assert_allclose(paired[col], paired[col+"_ref"], atol=1e-9, rtol=0)
        assert paired.experiment.eq(paired.selected_candidate).all()
    assert (scores.inner_last_target <= scores.outer_origin).all()
    rows = []
    for h, sub in validation.groupby("horizon"):
        for period, part in [("all", sub), *[(str(y), p) for y, p in sub.groupby(sub.date.dt.year)]]:
            rows.append(dict(horizon=h, family=CONFIG["models"][str(h)], period=period, **summarize_predictions(part)))
    metrics = pd.DataFrame(rows)
    test_predictions = predictions.loc[predictions.origin >= len(train)].copy()
    wide = test_predictions.pivot(index="date", columns="horizon", values="predicted").rename(columns={h: f"pred_{h}" for h in (1, 2, 3)}).reset_index()
    submission = template[["date"]].merge(wide, on="date", how="left", validate="one_to_one")
    validate_submission(submission, template)
    staged = OUT/"submission.csv"
    submission.to_csv(staged, index=False, float_format="%.6f", date_format="%Y-%m-%d")
    validate_submission(pd.read_csv(staged), template)
    validation.to_csv(OUT/"validation_predictions.csv", index=False)
    metrics.to_csv(OUT/"validation_metrics.csv", index=False)
    test_predictions.to_csv(OUT/"forecast_audit.csv", index=False)
    scores.to_csv(OUT/"h1_selection_scores.csv", index=False)
    inner.to_csv(OUT/"h1_inner_predictions.csv", index=False)
    contributions.to_csv(OUT/"linear_contributions.csv", index=False)
    x.to_csv(OUT/"feature_values.csv", index_label="origin")
    (OUT/"feature_sets.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    (OUT/"fit_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if diagnostics:
        raise ValueError("Review fit warnings before publishing final submission")
    current = Path("submission.csv")
    if not BACKUP.exists():
        shutil.copy2(current, BACKUP)
    elif digest(current) not in {digest(BACKUP), digest(staged)}:
        raise ValueError("Unrelated submission edits detected; do not overwrite")
    shutil.copy2(staged, current)
    assert digest(current) == digest(staged)
    manifest = dict(config=CONFIG, input_hashes={k: digest(p) for k, p in paths.items()},
                    code_hashes={p: digest(p) for p in ("src/final_pipeline.py", "src/features.py", "src/combination_experiment.py", "src/evaluate.py")},
                    versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
                    submission_sha256=digest(current), backup_path=str(BACKUP), backup_sha256=digest(BACKUP),
                    validation_replay_max_allowed_delta=1e-9, validation_matches_previous_experiment=True,
                    test_rows=len(submission), forecast_count=len(test_predictions), missing_raw_inputs=quality.to_dict(),
                    test_performance_used_for_final_model_selection=False,
                    explanation="centered linear contributions in log-return space; base plus contributions equals model output; not causal effects")
    (OUT/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.period == "all"].to_string(index=False))
    print("Published submission.csv; prior submission preserved at", BACKUP)


if __name__ == "__main__":
    main()
