"""Four controlled feature questions across four fixed ML families."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.combination_experiment import select_from_past
from src.evaluate import summarize_predictions
from src.features import validate_monthly_frame
from src.hypotheses import build_hypotheses
from src.ml_experiment import FAMILIES, backtest, pipeline_for

OUTPUT = Path("outputs_feature_audit")
LAGS = (0, 1, 3, 6)


def build_design(data):
    data = validate_monthly_frame(data)
    features, provenance, hypotheses = build_hypotheses(data, "conservative")
    history = list(next(h.columns for h in hypotheses if h.name == "history"))
    basic = ["price_now", "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_6", "price_lag_12", "month_sin", "month_cos"]
    processed = ["production_level", "storage_seasonal_z", "consumption_z", "BCOMNG_INDX_return"]
    raw = ["production_level", "storage_level", "consumption_level", "BCOMNG_level"]
    features["BCOMNG_level"] = pd.to_numeric(data.BCOMNG_INDX, errors="raise").ffill()
    extra = [dict(feature="BCOMNG_level", raw_columns="BCOMNG_INDX", formula="x.ffill(); release lag 0", release_lag_months=0)]
    candidates = {
        "basic_raw": basic + raw,
        "basic_processed": basic + processed,
        "engineered_raw": history + raw,
        "current": history + processed,
    }
    for role, column in zip(("production", "storage", "consumption", "market"), processed):
        candidates[f"drop_{role}"] = [c for c in candidates["current"] if c != column]
    for lag in LAGS[1:]:
        shifted = []
        for col in processed[:3]:
            name = f"{col}_extra_lag{lag}"
            features[name] = features[col].shift(lag)
            shifted.append(name)
            source = provenance.loc[provenance.feature == col].iloc[0]
            extra.append(dict(feature=name, raw_columns=source.raw_columns,
                              formula=f"{col}.shift({lag}) AFTER causal release alignment",
                              release_lag_months=source.release_lag_months, extra_lag_months=lag))
        candidates[f"lag_{lag}"] = history + shifted + [processed[-1]]
    candidates["individual_z"] = history + ["production_z", "storage_z", "consumption_z", processed[-1]]
    candidates["equal_index"] = history + ["balance_equal", processed[-1]]
    provenance = pd.concat([provenance, pd.DataFrame(extra)], ignore_index=True)
    return features, provenance, candidates


def lag_selection(cache):
    selected, scores = [], []
    names = ["current", *[f"lag_{k}" for k in LAGS[1:]]]
    for family in FAMILIES:
        pool = cache.loc[(cache.family == family) & cache.combination.isin(names)].copy()
        for h in (1, 2, 3):
            for origin in range(120, 204-h):
                chosen, table = select_from_past(pool, origin, h, window=36, min_origins=24)
                row = pool.loc[(pool.horizon == h) & (pool.origin == origin) & (pool.experiment == chosen)].copy()
                if len(row) != 1:
                    raise ValueError("Missing selected forecast")
                row["selected_candidate"] = chosen
                row["experiment"] = f"{family}__lag_selected"
                row["combination"] = "lag_selected"
                row["role"] = "past_only_lag_selection"
                selected.append(row)
                scores.append(table.assign(family=family, horizon=h, outer_origin=origin, selected=table.experiment.eq(chosen)))
    return pd.concat(selected, ignore_index=True), pd.concat(scores, ignore_index=True)


def evaluate(cache):
    outer = cache.loc[cache.origin >= 120].copy()
    outer["role"] = "fixed_candidate_diagnostic"
    selected, scores = lag_selection(cache)
    assert (scores.inner_last_target <= scores.outer_origin).all()
    predictions = pd.concat([outer, selected], ignore_index=True)
    ref = pd.read_csv("outputs_statistics/predictions.csv", parse_dates=["date"])
    ref = ref.loc[ref.experiment == "naive", ["date", "origin", "horizon", "actual", "current_price", "previous_price", "predicted"]]
    merged = predictions.merge(ref, on=["origin", "horizon"], suffixes=("", "_ref"), validate="many_to_one")
    assert len(merged) == len(predictions)
    assert (merged.date == merged.date_ref).all()
    for col in ("actual", "current_price", "previous_price"):
        np.testing.assert_allclose(merged[col], merged[col+"_ref"], rtol=0, atol=1e-12)
    assert not predictions.duplicated(["experiment", "horizon", "origin"]).any()
    assert np.isfinite(predictions.predicted).all()
    for (name, h), part in predictions.groupby(["experiment", "horizon"]):
        assert set(part.origin) == set(range(120, 204-h)), name
    naive = ref.assign(experiment="naive", family="baseline", combination="naive", role="baseline")
    predictions = pd.concat([predictions, naive], ignore_index=True)
    records = []
    for (family, combination, h), part in predictions.groupby(["family", "combination", "horizon"]):
        for year, sub in [("all", part), *part.groupby(part.date.dt.year)]:
            records.append(dict(family=family, combination=combination, horizon=h, year=year,
                                role=part.role.iloc[0], **summarize_predictions(sub)))
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT / "metrics.csv", index=False)
    scores.to_csv(OUTPUT / "lag_selection_scores.csv", index=False)


def main():
    paths = list(Path(".").glob("*data_train.csv"))
    if len(paths) != 1:
        raise ValueError("Expected one training file")
    data = validate_monthly_frame(pd.read_csv(paths[0]))
    if len(data) != 204:
        raise ValueError("Fixed audit expects 204 months")
    features, provenance, candidates = build_design(data)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "cache").mkdir(exist_ok=True)
    plan = {
        "train_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                        ("src/feature_audit.py", "src/ml_experiment.py", "src/hypotheses.py", "src/features.py", "src/train.py", "src/validation.py", "src/combination_experiment.py")},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost")},
        "models": {f: pipeline_for(f).named_steps["model"].get_params() for f in FAMILIES},
        "candidates": candidates, "first_inner_origin": 72, "first_outer_origin": 120,
        "target": "separate H1/H2/H3 log returns; current_price * exp(prediction)",
        "availability": "production/consumption 2 months, storage 1, market 0; additional lag applied afterwards",
        "lag_pool": ["current", "lag_1", "lag_3", "lag_6"], "lag_selection": "last36 known past origins, min24, lowest MAPE, no Naive in pool",
        "device": "CPU, same fixed estimators as earlier ML comparison, threadpool limit1",
        "limitations": "retrospective shared evaluation, not untouched holdout; joint fundamental lag only; no hyperparameter selection; ablation measures conditional predictive value not causality; input count changes in representation/compression comparisons",
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing audit configuration differs; do not overwrite another experiment")
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    used = {c for cols in candidates.values() for c in cols}
    provenance.loc[provenance.feature.isin(used)].to_csv(OUTPUT / "feature_definitions.csv", index=False)
    features[sorted(used)].to_csv(OUTPUT / "feature_values.csv", index_label="origin")
    parts = []
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            for h in (1, 2, 3):
                for name, columns in candidates.items():
                    file = OUTPUT / "cache" / f"{family}_{h}_{name}.csv"
                    notes = file.with_suffix(".warnings.json")
                    if file.exists() and notes.exists():
                        frame = pd.read_csv(file, parse_dates=["date"])
                    else:
                        frame, warnings = backtest(data, features, columns, family, name, h)
                        frame.to_csv(file, index=False)
                        notes.write_text(json.dumps(warnings, ensure_ascii=False, indent=2), encoding="utf-8")
                    assert set(frame.origin) == set(range(72, 204-h))
                    assert len(frame) == 132-h and not frame.origin.duplicated().any()
                    assert frame.family.eq(family).all() and frame.combination.eq(name).all() and frame.horizon.eq(h).all()
                    parts.append(frame)
                    print(f"DONE {family} H{h} {name}", flush=True)
    cache = pd.concat(parts, ignore_index=True)
    cache.to_csv(OUTPUT / "inner_prediction_cache.csv", index=False)
    evaluate(cache)
    from src.feature_audit_report import main as make_report
    make_report()


if __name__ == "__main__":
    main()
