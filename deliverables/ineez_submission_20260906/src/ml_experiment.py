"""Four ML families x eight common feature combinations; past-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from src.combination_experiment import combinations, select_from_past
from src.evaluate import summarize_predictions
from src.features import TARGET, validate_monthly_frame
from src.hypotheses import build_hypotheses
from src.train import _pipeline, _to_price, backtest_baseline
from src.validation import make_horizon_targets, rolling_origin_splits


FAMILIES = ("ridge", "elastic_net", "lightgbm", "catboost")


def pipeline_for(family):
    if family not in FAMILIES:
        raise ValueError(family)
    if family != "catboost":
        return _pipeline(family+"_log_return")
    preprocessing = _pipeline("ridge_log_return").steps[:-1]
    return Pipeline(preprocessing+[("model",CatBoostRegressor(
        iterations=100,depth=3,learning_rate=.04,l2_leaf_reg=5.,loss_function="RMSE",
        random_seed=42,thread_count=1,verbose=False,allow_writing_files=False))])


def backtest(data, features, cols, family, combination, horizon):
    target = make_horizon_targets(data,horizon)
    rows, diagnostics = [],[]
    for split in rolling_origin_splits(len(data),horizon,first_origin=72):
        origin = split.origin
        model = pipeline_for(family)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features.loc[split.train_indices,cols],target.loc[split.train_indices,"log_return"])
            raw = float(model.predict(features.loc[[origin],cols])[0])
        predicted = _to_price(raw,float(data.loc[origin,TARGET]),"log_return")
        if not np.isfinite(predicted):
            raise ValueError("Nonfinite prediction")
        rows.append({"date":data.loc[origin,"date"],"origin":origin,"horizon":horizon,
                     "experiment":f"{family}__{combination}","family":family,"combination":combination,
                     "actual":target.loc[origin,"future_price"],"predicted":predicted,
                     "current_price":data.loc[origin,TARGET],"previous_price":data.loc[origin-1,TARGET]})
        diagnostics.extend({"experiment":f"{family}__{combination}","horizon":horizon,
                            "origin":origin,"warning":str(w.message)} for w in caught)
    return pd.DataFrame(rows),diagnostics


def evaluate(cache, stats_dir, output_dir):
    outer = cache.loc[cache.origin>=120].copy()
    outer["role"] = "fixed_candidate_diagnostic"
    parts,score_parts = [outer],[]
    # Within-family policies select combinations only. Global policy also allows Naive.
    for pool in (*FAMILIES,"all"):
        subset = cache if pool=="all" else cache.loc[cache.family==pool]
        for h in (1,2,3):
            for origin in sorted(outer.loc[outer.horizon==h,"origin"].unique()):
                chosen,scores = select_from_past(subset,origin,h)
                score_parts.append(scores.assign(pool=pool,horizon=h,outer_origin=origin,
                                                 selected=lambda x:x.experiment.eq(chosen)))
                row = subset.loc[(subset.horizon==h)&(subset.origin==origin)&(subset.experiment==chosen)].copy()
                row["selected_candidate"] = chosen
                row["experiment"] = "selected_"+pool
                row["role"] = "past_only_selection_policy"
                parts.append(row)
    stats = pd.read_csv(stats_dir/"predictions.csv",parse_dates=["date"])
    for name in ("stats_best_fixed","stats_past_selected"):
        references = {1:"naive",2:"sarima_011_011",3:"sarima_011_011"}
        for h in (1,2,3):
            ref = references[h] if name=="stats_best_fixed" else "past_selected"
            part = stats.loc[(stats.horizon==h)&(stats.experiment==ref)].copy()
            expected = outer.loc[(outer.horizon==h)&(outer.experiment=="naive")]
            paired = expected.merge(part,on=["date","origin","horizon"],how="outer",suffixes=("_ml","_stat"),indicator=True,validate="one_to_one")
            if not paired._merge.eq("both").all() or not np.allclose(paired.actual_ml,paired.actual_stat,rtol=0,atol=1e-10):
                raise ValueError("Statistical reference dates or actuals differ")
            part["experiment"] = name
            part["role"] = "post_selected_reference" if name=="stats_best_fixed" else "past_only_selection_policy"
            parts.append(part)
    predictions = pd.concat(parts,ignore_index=True)
    records=[]
    for (h,name),part in predictions.groupby(["horizon","experiment"]):
        for year,sub in [("all",part),*part.groupby(part.date.dt.year)]:
            records.append({"horizon":h,"experiment":name,"year":year,"role":part.role.iloc[0],
                            **summarize_predictions(sub)})
    metrics = pd.DataFrame(records)
    overall = metrics.loc[metrics.year=="all"]
    fixed = overall.loc[overall.experiment.str.contains("__")].copy()
    fixed["family"] = fixed.experiment.str.split("__").str[0]
    best = fixed.sort_values(["mape_pct","experiment"]).groupby(["horizon","family"]).head(1)
    best.to_csv(output_dir/"best_fixed_by_family.csv",index=False)
    fixed.sort_values(["mape_pct","experiment"]).groupby("horizon").head(1).to_csv(output_dir/"best_fixed_ml.csv",index=False)
    comparisons=[]
    for h in (1,2,3):
        aggregate=overall.loc[overall.horizon==h].set_index("experiment")
        annual=metrics.loc[(metrics.horizon==h)&(metrics.year!="all")].pivot(index="year",columns="experiment",values="mape_pct")
        for candidate in ["selected_"+f for f in (*FAMILIES,"all")]:
            for reference in ("naive","stats_best_fixed","stats_past_selected"):
                comparisons.append({"horizon":h,"candidate":candidate,"reference":reference,
                                    "delta_mape_pp":aggregate.loc[candidate,"mape_pct"]-aggregate.loc[reference,"mape_pct"],
                                    "years_better":int((annual[candidate]<annual[reference]).sum()),"n_years":len(annual)})
    pd.DataFrame(comparisons).to_csv(output_dir/"comparisons.csv",index=False)
    predictions.to_csv(output_dir/"predictions.csv",index=False)
    metrics.to_csv(output_dir/"metrics.csv",index=False)
    pd.concat(score_parts,ignore_index=True).to_csv(output_dir/"selection_scores.csv",index=False)
    print(overall.loc[~overall.experiment.str.contains("__"),["horizon","experiment","mape_pct","turning_f1_pct"]].to_string(index=False))


def run(train_path, output_dir, stats_dir):
    data=validate_monthly_frame(pd.read_csv(train_path))
    if len(data)<124:
        raise ValueError("Need at least 124 monthly observations")
    digest=hashlib.sha256(train_path.read_bytes()).hexdigest()
    stats_plan=json.loads((stats_dir/"manifest.json").read_text(encoding="utf-8"))
    if stats_plan["train_sha256"]!=digest:
        raise ValueError("Statistical benchmark uses a different train file")
    features,definitions,hypotheses=build_hypotheses(data,"conservative")
    history=next(h.columns for h in hypotheses if h.name=="history")
    combos=combinations(history)
    output_dir.mkdir(parents=True,exist_ok=True)
    plan={"train_sha256":digest,"combinations":combos,"availability":"conservative",
          "models":{f:pipeline_for(f).named_steps["model"].get_params() for f in FAMILIES},
          "target":"log return; separate H1/H2/H3 models",
          "first_inner_origin":72,"first_outer_origin":120,"inner_window":36,
          "selection":"family: 8 combinations; all: 32 combinations plus Naive; known targets only",
          "scope":"one fixed configuration per family, seed 42; NOT hyperparameter or seed search",
          "limitations":"retrospective candidates from inspected data; fixed BEST is post-selected; statistical inputs are price-only; no submission changes"}
    (output_dir/"manifest.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    definitions.loc[definitions.feature.isin({c for cs in combos.values() for c in cs})].to_csv(output_dir/"feature_definitions.csv",index=False,encoding="utf-8-sig")
    frames,diagnostics=[],[]
    for family in FAMILIES:
        for h in (1,2,3):
            for combination,cols in combos.items():
                frame,notes=backtest(data,features,cols,family,combination,h)
                frames.append(frame)
                diagnostics.extend(notes)
            # Checkpoint completed work without implicit reuse of stale caches.
            pd.concat(frames,ignore_index=True).to_csv(output_dir/"inner_prediction_cache.csv",index=False)
            print(f"{family} H{h}: 8 combinations complete",flush=True)
    for h in (1,2,3):
        frame=backtest_baseline(data,h,"last_value_naive",72)
        frame["experiment"]="naive"
        frame["family"]="baseline"
        frame["origin"]=frame.date.map(pd.Series(data.index,index=data.date)).astype(int)
        frames.append(frame)
    cache=pd.concat(frames,ignore_index=True)
    cache.to_csv(output_dir/"inner_prediction_cache.csv",index=False)
    pd.DataFrame(diagnostics,columns=["experiment","horizon","origin","warning"]).to_csv(output_dir/"fit_diagnostics.csv",index=False)
    evaluate(cache,stats_dir,output_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs_ml"))
    parser.add_argument("--stats-dir",type=Path,default=Path("outputs_statistics"))
    args=parser.parse_args()
    if args.train is None:
        paths=list(Path('.').glob('*data_train.csv'))
        if len(paths)!=1:
            raise ValueError("Specify --train")
        args.train=paths[0]
    run(args.train,args.output_dir,args.stats_dir)


if __name__=="__main__":
    main()
