"""Run declared feature hypotheses on train data only; never select a submission model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.evaluate import summarize_predictions
from src.features import TARGET, validate_monthly_frame
from src.hypotheses import build_hypotheses, manifest
from src.train import _pipeline, backtest_baseline
from src.validation import make_horizon_targets, rolling_origin_splits


def fit_predict(x_train, y_train, x_now, constrained=False):
    """Ridge alpha=10 for every candidate; only core signs differ when requested."""
    pipeline = _pipeline("ridge_log_return")
    transform = pipeline[:-1]
    x = transform.fit_transform(x_train)
    now = transform.transform(x_now)
    names = list(pipeline.named_steps["imputer"].get_feature_names_out(x_train.columns))
    if constrained:
        # Unpenalized intercept; L2 penalty on ALL slopes, matching ordinary Ridge.
        design = np.column_stack([np.ones(len(x)),x])
        penalty = np.column_stack([np.zeros(x.shape[1]),np.sqrt(10.)*np.eye(x.shape[1])])
        lower = np.full(x.shape[1]+1,-np.inf)
        upper = np.full(x.shape[1]+1,np.inf)
        for name in ("production_z","storage_z"):
            upper[names.index(name)+1] = 0
        lower[names.index("consumption_z")+1] = 0
        fit = lsq_linear(np.vstack([design,penalty]),np.r_[y_train,np.zeros(x.shape[1])],
                         bounds=(lower,upper),tol=1e-10,max_iter=1000)
        if not fit.success:
            raise RuntimeError(fit.message)
        intercept, coefficients = fit.x[0],fit.x[1:]
    else:
        model = pipeline.named_steps["model"].fit(x,y_train)
        intercept, coefficients = model.intercept_,model.coef_
    return float(intercept+now[0]@coefficients), names, coefficients, float(intercept)


def run(train_path, output_dir, first_origin=120, mode="conservative"):
    data = validate_monthly_frame(pd.read_csv(train_path))
    if first_origin < 48 or first_origin > len(data)-4:
        raise ValueError("first_origin must be >=48 and leave an observable H3 target")
    features, provenance, hypotheses = build_hypotheses(data,mode)
    output_dir.mkdir(parents=True,exist_ok=True)
    plan = {"availability_mode":mode,"train_file":str(train_path),
            "train_sha256":hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "first_origin":first_origin,"model":"Ridge log_return, alpha=10 fixed, not tuned",
            "preprocessing":"fold-local median+missing indicators, StandardScaler",
            "scope":"exploratory train-only comparison; no independent holdout or automatic adoption",
            "hypotheses":manifest(hypotheses)}
    (output_dir/"manifest.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    provenance.to_csv(output_dir/"feature_definitions.csv",index=False,encoding="utf-8-sig")
    features.assign(date=data.date).to_csv(output_dir/"feature_values.csv",index=False)
    all_predictions, coefficient_rows, metric_rows = [],[],[]
    for horizon in (1,2,3):
        targets = make_horizon_targets(data,horizon)
        baseline = backtest_baseline(data,horizon,"last_value_naive",first_origin)
        baseline["experiment"] = "naive"
        parts = [baseline]
        for hypothesis in hypotheses:
            rows = []
            cols = list(hypothesis.columns)
            for split in rolling_origin_splits(len(data),horizon,first_origin):
                origin = split.origin
                raw,names,coef,intercept = fit_predict(
                    features.loc[split.train_indices,cols],targets.loc[split.train_indices,"log_return"],
                    features.loc[[origin],cols],hypothesis.constrained)
                current = float(data.loc[origin,TARGET])
                prediction = max(current*float(np.exp(raw)),.01)
                if not np.isfinite(prediction):
                    raise ValueError("Nonfinite prediction")
                rows.append({"date":data.loc[origin,"date"],"horizon":horizon,
                             "experiment":hypothesis.name,"actual":targets.loc[origin,"future_price"],
                             "predicted":prediction,"current_price":current,
                             "previous_price":data.loc[origin-1,TARGET]})
                coefficient_rows.extend({"date":data.loc[origin,"date"],"horizon":horizon,
                                         "experiment":hypothesis.name,"feature":name,"coefficient":float(value)}
                                        for name,value in [("intercept",intercept),*zip(names,coef)])
            parts.append(pd.DataFrame(rows))
        combined = pd.concat(parts,ignore_index=True)
        all_predictions.append(combined)
        for name,part in combined.groupby("experiment"):
            periods = [("all",part),*part.groupby(part.date.dt.year)]
            for year,subset in periods:
                metric_rows.append({"experiment":name,"horizon":horizon,"year":year,
                                    **summarize_predictions(subset)})
        print(f"H{horizon}: {len(hypotheses)} combinations complete",flush=True)
    predictions = pd.concat(all_predictions,ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    comparisons = []
    for horizon in (1,2,3):
        overall = metrics[(metrics.horizon==horizon)&(metrics.year=="all")].set_index("experiment")
        annual = metrics[(metrics.horizon==horizon)&(metrics.year!="all")].pivot(
            index="year",columns="experiment",values="mape_pct")
        for h in hypotheses:
            child,parent = overall.loc[h.name],overall.loc[h.parent]
            assert child.n_origins == parent.n_origins
            comparisons.append({"horizon":horizon,"experiment":h.name,"parent":h.parent,
                                "question":h.question,"mape_pct":child.mape_pct,
                                "delta_mape_pp":child.mape_pct-parent.mape_pct,
                                "delta_turning_f1_pp":child.turning_f1_pct-parent.turning_f1_pct,
                                "years_better_than_parent":int((annual[h.name]<annual[h.parent]).sum()),
                                "n_years":len(annual),"n_origins":int(child.n_origins),
                                "status":"exploratory; not automatically adopted"})
    predictions.to_csv(output_dir/"predictions.csv",index=False)
    metrics.to_csv(output_dir/"metrics.csv",index=False)
    pd.DataFrame(comparisons).to_csv(output_dir/"comparisons.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(output_dir/"coefficients.csv",index=False)
    print(metrics[metrics.year=="all"].sort_values(["horizon","mape_pct"]).groupby("horizon").head(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs_hypotheses"))
    parser.add_argument("--first-origin",type=int,default=120)
    parser.add_argument("--availability-mode",choices=["conservative","as_provided"],default="conservative")
    args = parser.parse_args()
    if args.train is None:
        matches = list(Path(".").glob("*data_train.csv"))
        if len(matches)!=1:
            raise ValueError("Specify --train: expected one train file")
        args.train = matches[0]
    run(args.train,args.output_dir,args.first_origin,args.availability_mode)


if __name__ == "__main__":
    main()
