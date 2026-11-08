"""Small, declared imports/rigs additions to existing processed fundamentals.

H1 compares equal lag-search budgets; H2/H3 use basic_processed. Storage lag2
is a publication assumption, never a selectable hyperparameter.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.combination_experiment import select_from_past
from src.evaluate import summarize_predictions
from src.feature_audit import build_design
from src.features import validate_monthly_frame
from src.ml_experiment import backtest, pipeline_for
from src.variable_suitability import block_interval

OUT = Path("outputs_incremental_fundamentals")
FAMILIES = ("ridge", "elastic_net", "lightgbm")
AUGMENTS = ("base", "imports", "rigs", "both")


def design(data, horizon):
    x, provenance, old = build_design(data)
    core = old["current"] if horizon == 1 else old["basic_processed"]
    x["storage_seasonal_z_lag2"] = x.storage_seasonal_z.shift(1)
    core = ["storage_seasonal_z_lag2" if c == "storage_seasonal_z" else c for c in core]
    # Two raw forms chosen before fitting, identical to broad audit; no tuning.
    extra = {}
    for role, raw, lag in (("imports", "Natural_Gas_Imports_USA", 2),
                           ("rigs", "Natural_Gas_Rotary_Rig_Count_USA", 0)):
        s = pd.to_numeric(data[raw], errors="raise").ffill(limit=1).shift(lag)
        extra[role] = [f"add_{role}_level", f"add_{role}_diff1"]
        x[extra[role][0]], x[extra[role][1]] = s, s.diff()
    specs = {}
    for lag in ((0, 1, 3, 6) if horizon == 1 else (0,)):
        columns = []
        for c in core:
            if lag and c in ("production_level", "storage_seasonal_z_lag2", "consumption_z"):
                name = f"{c}_reaction{lag}"
                x[name] = x[c].shift(lag)
                columns.append(name)
            else:
                columns.append(c)
        for augment in AUGMENTS:
            added = extra["imports"] if augment == "imports" else extra["rigs"] if augment == "rigs" else sum(extra.values(), []) if augment == "both" else []
            specs[f"{augment}_lag{lag}"] = columns+added
    return x, specs


def select_policies(cache):
    rows, tables = [], []
    for family in FAMILIES:
        for h in (1, 2, 3):
            part = cache.loc[(cache.family == family) & (cache.horizon == h)]
            for policy in (*AUGMENTS, "all"):
                pool = part if policy == "all" else part.loc[part.combination.str.startswith(policy+"_")]
                for origin in range(120, 204-h):
                    chosen, scores = select_from_past(pool, origin, h, window=36, min_origins=24)
                    row = pool.loc[(pool.origin == origin) & (pool.experiment == chosen)].copy()
                    assert len(row) == 1
                    row["selected_candidate"], row["policy"] = chosen, policy
                    rows.append(row)
                    tables.append(scores.assign(family=family, horizon=h, policy=policy, outer_origin=origin,
                                                selected=scores.experiment.eq(chosen)))
    scores = pd.concat(tables, ignore_index=True)
    assert (scores.inner_last_target <= scores.outer_origin).all()
    return pd.concat(rows, ignore_index=True), scores


def evaluate(predictions):
    records, comparisons = [], []
    for (family, h, policy), part in predictions.groupby(["family", "horizon", "policy"]):
        for period, sub in [("all", part), ("early", part.loc[part.origin < 161]),
                            ("late", part.loc[part.origin >= 161]),
                            *[(str(y), p) for y, p in part.groupby(part.date.dt.year)]]:
            records.append(dict(family=family, horizon=h, policy=policy, period=period, **summarize_predictions(sub)))
        if policy == "base":
            continue
        ref = predictions.loc[(predictions.family == family) & (predictions.horizon == h) & (predictions.policy == "base")]
        pair = part.merge(ref, on=["origin", "horizon", "date"], suffixes=("", "_base"), validate="one_to_one")
        assert len(pair) == 84-h
        np.testing.assert_allclose(pair.actual, pair.actual_base, atol=1e-12, rtol=0)
        delta = (abs(pair.predicted-pair.actual)-abs(pair.predicted_base-pair.actual))*100/pair.actual
        lo, hi = block_interval(delta)
        comparisons.append(dict(family=family, horizon=h, policy=policy, delta_mape_pp=delta.mean(),
                                early_delta_pp=delta[pair.origin < 161].mean(), late_delta_pp=delta[pair.origin >= 161].mean(),
                                years_better=int((delta.groupby(pair.date.dt.year).mean() < 0).sum()),
                                n_years=pair.date.dt.year.nunique(), block6_low_pp=lo, block6_high_pp=hi))
    return pd.DataFrame(records), pd.DataFrame(comparisons)


def main():
    train = next(Path('.').glob('*data_train.csv'))
    data = validate_monthly_frame(pd.read_csv(train))
    assert len(data) == 204
    OUT.mkdir(exist_ok=True)
    (OUT/"cache").mkdir(exist_ok=True)
    designs = {h: design(data, h) for h in (1, 2, 3)}
    source = json.loads(Path("outputs_storage_source/summary.json").read_text(encoding="utf-8"))
    paths = [train, Path("submission.csv"), *[Path("src")/f for f in
             ("incremental_fundamentals.py", "feature_audit.py", "hypotheses.py", "features.py", "ml_experiment.py", "train.py", "validation.py", "combination_experiment.py", "variable_suitability.py")]]
    plan = dict(hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                source_check=source, candidates={str(h): specs for h, (_, specs) in designs.items()},
                models={f: pipeline_for(f).named_steps["model"].get_params() for f in FAMILIES},
                selection="H1 each augmentation separately selects reaction lag0/1/3/6; pooled policy also selects augmentation; H2/H3 four fixed augmentations and pooled policy; last36 known inner predictions, min24",
                availability="production2 consumption2 storage2 imports2 rigs0 market0; extra reaction lags only existing three fundamentals; vintage and complete historic releases unverified",
                validation="inner starts72; outer120; training j+h<=origin; same horizon dates; no test data; 9396 fits; CPU fixed estimators",
                limitations="retrospective candidates motivated by inspected data; no untouched validation; imports/rigs level+diff only, no extra reaction-lag search for added inputs; interval not multiplicity-adjusted; no automatic model/submission change")
    assert source["train_sha256"] == plan["hashes"][str(train)]
    manifest = OUT/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("Manifest differs; cannot reuse another experiment's cache")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    frames, diagnostics = [], []
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            for h, (x, specs) in designs.items():
                for name, cols in specs.items():
                    path = OUT/"cache"/f"{family}_{h}_{name}.csv"
                    note = path.with_suffix(".json")
                    if path.exists() and note.exists():
                        p = pd.read_csv(path, parse_dates=["date"])
                        notes = json.loads(note.read_text(encoding="utf-8"))
                    else:
                        p, notes = backtest(data, x, cols, family, name, h)
                        p.to_csv(path, index=False)
                        note.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
                    assert len(p) == 132-h and set(p.origin) == set(range(72, 204-h))
                    assert p.family.eq(family).all() and p.horizon.eq(h).all() and p.combination.eq(name).all()
                    frames.append(p)
                    diagnostics.extend(notes)
                print(f"DONE {family} H{h}: {len(specs)} candidates", flush=True)
    cache = pd.concat(frames, ignore_index=True)
    cache.to_csv(OUT/"inner_predictions.csv", index=False)
    pred, scores = select_policies(cache)
    metrics, comparisons = evaluate(pred)
    pred.to_csv(OUT/"predictions.csv", index=False)
    scores.to_csv(OUT/"selection_scores.csv", index=False)
    metrics.to_csv(OUT/"metrics.csv", index=False)
    comparisons.to_csv(OUT/"comparisons.csv", index=False)
    (OUT/"fit_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    assert hashlib.sha256(Path("submission.csv").read_bytes()).hexdigest() == plan["hashes"]["submission.csv"]
    print(metrics.loc[metrics.period == "all", ["family", "horizon", "policy", "mape_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
