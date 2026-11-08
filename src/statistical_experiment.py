"""Small predeclared ETS/SARIMA benchmark, using price history only."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.combination_experiment import select_from_past
from src.evaluate import summarize_predictions, turning_point_metrics, direction_accuracy
from src.features import TARGET, validate_monthly_frame


SPECS = {
    "naive": {"family": "baseline"},
    "seasonal_naive": {"family": "baseline", "period": 12},
    "ets_level": {"family": "ETS", "trend": None, "seasonal": None, "damped_trend": False},
    "ets_damped": {"family": "ETS", "trend": "add", "seasonal": None, "damped_trend": True},
    "ets_damped_seasonal": {"family": "ETS", "trend": "add", "seasonal": "add", "damped_trend": True},
    "sarima_110_000": {"family": "SARIMA", "order": [1,1,0], "seasonal_order": [0,0,0,0]},
    "sarima_011_011": {"family": "SARIMA", "order": [0,1,1], "seasonal_order": [0,1,1,12]},
}


def forecast(history, name):
    """Fit once at origin and generate H1-H3; fitting never receives future rows."""
    price = np.asarray(history, dtype=float)
    if len(price)<24 or not np.isfinite(price).all() or (price<=0).any():
        raise ValueError("Need at least 24 finite positive monthly prices")
    if name == "naive":
        return np.repeat(price[-1],3), "ok", ""
    if name == "seasonal_naive":
        return price[[len(price)-12,len(price)-11,len(price)-10]], "ok", ""
    spec = SPECS[name]
    # Same fixed log-price transformation for every fitted statistical candidate.
    # exp(point forecast) is used without an estimated mean-bias correction.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if spec["family"] == "ETS":
                fit = ExponentialSmoothing(
                    np.log(price),trend=spec["trend"],seasonal=spec["seasonal"],
                    damped_trend=spec["damped_trend"],
                    seasonal_periods=12 if spec["seasonal"] else None,
                    initialization_method="estimated",
                ).fit(optimized=True)
                converged = bool(fit.mle_retvals.get("success",True))
            else:
                fit = SARIMAX(np.log(price),order=tuple(spec["order"]),
                              seasonal_order=tuple(spec["seasonal_order"]),trend="n",
                              enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
                converged = bool(fit.mle_retvals.get("converged",False))
            if not converged:
                raise RuntimeError("optimizer did not converge")
            prediction = np.exp(np.asarray(fit.forecast(3),dtype=float))
            if not np.isfinite(prediction).all() or (prediction<=0).any():
                raise ValueError("nonfinite or nonpositive forecast")
            return prediction,"ok"," | ".join(str(w.message) for w in caught)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # Preserve identical origins and explicitly disclose fallback instead of dropping failures.
            return np.repeat(price[-1],3),"fallback_naive",str(exc)


def run(train_path, output_dir, first_origin=120, inner_first=72):
    data = validate_monthly_frame(pd.read_csv(train_path))
    if inner_first<24 or first_origin-inner_first-3+1<24 or first_origin>len(data)-4:
        raise ValueError("Insufficient history for selection and H3 evaluation")
    output_dir.mkdir(parents=True,exist_ok=True)
    plan = {"train_sha256":hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "input":"observed Henry Hub price only; current month price assumed known",
            "specs":SPECS,"first_inner_origin":inner_first,"first_outer_origin":first_origin,
            "transformation":"log price for ETS/SARIMA, exponentiate forecast without mean correction",
            "fitting":"expanding price history through origin; forecast H1-H3 once per fit",
            "selection":"past-only MAPE over last 36 known targets, minimum 24; ties naive then name",
            "fallback":"nonconvergence or invalid forecast => naive, logged; no origin removed",
            "limitations":"retrospective evaluation on previously inspected data; fixed small grid; no submission changes"}
    (output_dir/"manifest.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    rows,diagnostics = [],[]
    for name in SPECS:
        for origin in range(inner_first,len(data)-1):
            prediction,status,message = forecast(data.loc[:origin,TARGET],name)
            diagnostics.append({"experiment":name,"origin":origin,"date":data.loc[origin,"date"],
                                "status":status,"message":message})
            for h in (1,2,3):
                if origin+h>=len(data):
                    continue
                rows.append({"date":data.loc[origin,"date"],"origin":origin,"horizon":h,
                             "experiment":name,"actual":data.loc[origin+h,TARGET],
                             "predicted":prediction[h-1],"current_price":data.loc[origin,TARGET],
                             "previous_price":data.loc[origin-1,TARGET],"fit_status":status})
        print(f"{name} complete",flush=True)
    evaluate_cache(pd.DataFrame(rows),pd.DataFrame(diagnostics),output_dir,first_origin)


def evaluate_cache(cache, diagnostics, output_dir, first_origin=120):
    """Score cached fits; allow diagnostics to be reproduced without refitting."""
    path = output_dir/"manifest.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["diagnostic_flat_relative_threshold"] = .001
    plan["best_fixed_tie_tolerance_mape_pp"] = .0001
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    outer = cache.loc[cache.origin>=first_origin].copy()
    selected, score_parts = [],[]
    for h in (1,2,3):
        for origin in sorted(outer.loc[outer.horizon==h,"origin"].unique()):
            name,scores = select_from_past(cache,origin,h)
            score_parts.append(scores.assign(horizon=h,outer_origin=origin,selected=lambda x:x.experiment.eq(name)))
            row = outer.loc[(outer.horizon==h)&(outer.origin==origin)&(outer.experiment==name)].copy()
            row["selected_candidate"] = name
            row["experiment"] = "past_selected"
            selected.append(row)
    predictions = pd.concat([outer,*selected],ignore_index=True)
    metrics = []
    for (h,name),part in predictions.groupby(["horizon","experiment"]):
        for year,subset in [("all",part),*part.groupby(part.date.dt.year)]:
            # Additional direction diagnostics treat moves <=0.1% as flat.
            # MAPE and exact-sign metrics remain unchanged. Threshold is descriptive, not tuned.
            current = subset.current_price.to_numpy()
            def flatten(values):
                values = np.asarray(values)
                return np.where(np.abs(values/current-1)<=.001,current,values)
            actual, predicted, previous = [flatten(subset[c]) for c in ("actual","predicted","previous_price")]
            threshold_turn = turning_point_metrics(actual,predicted,current,previous)
            metrics.append({"horizon":h,"experiment":name,"year":year,
                            "fallback_origins":int(subset.fit_status.ne("ok").sum()),**summarize_predictions(subset),
                            "direction_accuracy_flat_0_1pct":direction_accuracy(actual,predicted,current),
                            "turning_f1_flat_0_1pct":threshold_turn["turning_f1_pct"]})
    metrics = pd.DataFrame(metrics)
    aggregate = metrics.loc[metrics.year=="all"]
    fixed = aggregate.loc[aggregate.experiment!="past_selected"]
    best_rows = []
    for h,part in fixed.groupby("horizon"):
        tied = part.loc[part.mape_pct<=part.mape_pct.min()+.0001].copy()
        tied["priority"] = tied.experiment.ne("naive").astype(int)
        best_rows.append(tied.sort_values(["priority","mape_pct","experiment"]).iloc[0])
    best = pd.DataFrame(best_rows)
    best["tie_tolerance_mape_pp"] = .0001
    best["interpretation"] = "best fixed candidate on full inspected interval; not independent performance"
    cache.to_csv(output_dir/"inner_prediction_cache.csv",index=False)
    predictions.to_csv(output_dir/"predictions.csv",index=False)
    diagnostics.to_csv(output_dir/"fit_diagnostics.csv",index=False)
    pd.concat(score_parts,ignore_index=True).to_csv(output_dir/"selection_scores.csv",index=False)
    metrics.to_csv(output_dir/"metrics.csv",index=False)
    best.to_csv(output_dir/"best_fixed.csv",index=False)
    print(aggregate[["horizon","experiment","mape_pct","turning_f1_pct","fallback_origins"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs_statistics"))
    args = parser.parse_args()
    if args.train is None:
        paths = list(Path('.').glob('*data_train.csv'))
        if len(paths)!=1:
            raise ValueError("Specify --train")
        args.train = paths[0]
    run(args.train,args.output_dir)


if __name__ == "__main__":
    main()
