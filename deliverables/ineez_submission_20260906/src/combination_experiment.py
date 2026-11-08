"""Eight controlled combinations with past-only monthly feature selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import summarize_predictions
from src.features import make_feature_matrix, validate_monthly_frame
from src.hypotheses import build_hypotheses
from src.train import backtest_baseline, backtest_model


def combinations(history):
    """2 production forms x 2 storage forms x BCOMNG present/absent."""
    result = {}
    for name, production, storage in [
        ("A", "production_z", "storage_z"),
        ("B", "production_level", "storage_z"),
        ("C", "production_z", "storage_seasonal_z"),
        ("D", "production_level", "storage_seasonal_z"),
    ]:
        columns = list(history) + [production, storage, "consumption_z"]
        result[name] = columns
        result[name + "_market"] = columns + ["BCOMNG_INDX_return"]
    return result


def select_from_past(predictions, origin, horizon, window=36, min_origins=24):
    """Score only outcomes whose target month is known by the outer origin.

    Cached predictions are fitted at their own origin, so reusing them is equivalent
    to rerunning the inner rolling backtests. Never use outer-origin actuals here.
    """
    if window < min_origins or min_origins < 1:
        raise ValueError("window must be >= min_origins >= 1")
    part = predictions.loc[(predictions.horizon == horizon) &
                           (predictions.origin + horizon <= origin)].copy()
    if part.duplicated(["experiment", "origin"]).any():
        raise ValueError("Duplicate candidate origins")
    names = sorted(predictions.loc[predictions.horizon == horizon, "experiment"].unique())
    dates = [set(part.loc[part.experiment == name, "origin"]) for name in names]
    if not dates or any(x != dates[0] for x in dates):
        raise ValueError("Candidates must have identical inner validation origins")
    eligible = sorted(dates[0])[-window:]
    if len(eligible) < min_origins:
        raise ValueError("Not enough observed inner validation targets")
    rows = []
    for name in names:
        subset = part.loc[(part.experiment == name) & part.origin.isin(eligible)]
        score = summarize_predictions(subset)["mape_pct"]
        if not np.isfinite(score):
            raise ValueError("Nonfinite selection score")
        rows.append({"experiment": name, "inner_mape_pct": score,
                     "inner_first_origin": eligible[0], "inner_last_origin": eligible[-1],
                     "inner_last_target": eligible[-1]+horizon, "inner_n": len(eligible)})
    scores = pd.DataFrame(rows)
    # Deterministic ties prefer naive, then experiment name. No tuned threshold.
    ranked = scores.assign(priority=(scores.experiment != "naive").astype(int)).sort_values(
        ["inner_mape_pct", "priority", "experiment"])
    return str(ranked.iloc[0].experiment), scores


def run(train_path, output_dir, first_origin=120, inner_first=72, window=36):
    data = validate_monthly_frame(pd.read_csv(train_path))
    if inner_first < 48 or first_origin > len(data)-4 or first_origin-inner_first-3+1 < 24:
        raise ValueError("Need >=48 initial months, >=24 inner observations and observable H3")
    features, definitions, hypotheses = build_hypotheses(data, "conservative")
    history = next(h.columns for h in hypotheses if h.name == "history")
    candidates = combinations(history)
    legacy_features, legacy_groups = make_feature_matrix(data, "conservative")
    # Explicit historical references, NOT eligible for past-only selection:
    # these identities were previously chosen using the full validation period.
    references = {1: ("E3_energy", "elastic_net_log_return"),
                  2: ("E3_energy", "ridge_log_return"),
                  3: ("E2_fundamentals", "ridge_log_return")}
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "availability_mode": "conservative", "first_outer_origin": first_origin,
            "first_inner_origin": inner_first, "inner_window": window, "minimum_inner_origins": 24,
            "model": "Ridge alpha=10 fixed, log return; no hyperparameter or blend search",
            "candidates": candidates, "selection_pool": [*candidates, "naive"],
            "rule": "lowest past-only MAPE; inner_origin+h<=outer_origin; ties prefer naive then name",
            "references": references,
            "limitations": "Candidates motivated by previously inspected data; retrospective evaluation, not untouched holdout. Historical references are post-selected. No submission changes."}
    (output_dir/"manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    used = set(c for cols in candidates.values() for c in cols)
    definitions.loc[definitions.feature.isin(used)].to_csv(output_dir/"feature_definitions.csv", index=False, encoding="utf-8-sig")
    cache_parts, outer_parts, score_parts = [], [], []
    date_to_origin = pd.Series(data.index, index=data.date)
    for h in (1, 2, 3):
        parts = []
        for name, cols in candidates.items():
            frame = backtest_model(data, features, cols, h, "ridge_log_return", name, inner_first)
            frame["experiment"] = name
            parts.append(frame)
        naive = backtest_baseline(data, h, "last_value_naive", inner_first)
        naive["experiment"] = "naive"
        parts.append(naive)
        cache = pd.concat(parts, ignore_index=True)
        cache["origin"] = cache.date.map(date_to_origin).astype(int)
        cache_parts.append(cache)
        outer = cache.loc[cache.origin >= first_origin].copy()
        outer["evaluation_role"] = "fixed_candidate_diagnostic"
        outer_parts.append(outer)
        for origin in sorted(outer.origin.unique()):
            name, scores = select_from_past(cache, origin, h, window)
            scores = scores.assign(outer_origin=origin, date=data.loc[origin,"date"], horizon=h,
                                   selected=lambda x: x.experiment.eq(name))
            score_parts.append(scores)
            selected = cache.loc[(cache.origin == origin) & (cache.experiment == name)].copy()
            selected["selected_candidate"] = name
            selected["experiment"] = "past_selected"
            selected["evaluation_role"] = "past_only_selection_policy"
            outer_parts.append(selected)
        group, model = references[h]
        reference = backtest_model(data, legacy_features, legacy_groups[group], h, model, group, first_origin)
        reference["experiment"] = "legacy_reference"
        reference["origin"] = reference.date.map(date_to_origin).astype(int)
        reference["evaluation_role"] = "previously_post_selected_reference"
        outer_parts.append(reference)
        print(f"H{h}: eight combinations and past-only selection complete", flush=True)
    predictions = pd.concat(outer_parts, ignore_index=True)
    rows = []
    for (h, name), part in predictions.groupby(["horizon", "experiment"]):
        for year, subset in [("all", part), *part.groupby(part.date.dt.year)]:
            rows.append({"horizon": h, "experiment": name, "year": year,
                         "evaluation_role": subset.evaluation_role.iloc[0], **summarize_predictions(subset)})
    metrics = pd.DataFrame(rows)
    edges = [("B","A"),("C","A"),("D","B"),("D","C")]
    edges += [(a+"_market", b+"_market") for a,b in edges.copy()]
    edges += [(a+"_market",a) for a in "ABCD"]
    edges += [("past_selected","naive"),("past_selected","legacy_reference")]
    contrasts = []
    for h in (1,2,3):
        aggregate = metrics.loc[(metrics.horizon==h)&(metrics.year=="all")].set_index("experiment")
        annual = metrics.loc[(metrics.horizon==h)&(metrics.year!="all")].pivot(index="year",columns="experiment",values="mape_pct")
        for child,parent in edges:
            contrasts.append({"horizon":h,"candidate":child,"reference":parent,
                              "delta_mape_pp":aggregate.loc[child,"mape_pct"]-aggregate.loc[parent,"mape_pct"],
                              "delta_turning_f1_pp":aggregate.loc[child,"turning_f1_pct"]-aggregate.loc[parent,"turning_f1_pct"],
                              "years_better":int((annual[child]<annual[parent]).sum()),"n_years":len(annual)})
    pd.concat(cache_parts,ignore_index=True).to_csv(output_dir/"inner_prediction_cache.csv",index=False)
    predictions.to_csv(output_dir/"predictions.csv",index=False)
    pd.concat(score_parts,ignore_index=True).to_csv(output_dir/"selection_scores.csv",index=False)
    metrics.to_csv(output_dir/"metrics.csv",index=False)
    pd.DataFrame(contrasts).to_csv(output_dir/"comparisons.csv",index=False)
    print(metrics.loc[(metrics.year=="all") & metrics.experiment.isin(["past_selected","legacy_reference","naive"])].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs_combinations"))
    parser.add_argument("--first-origin",type=int,default=120)
    parser.add_argument("--inner-first",type=int,default=72)
    parser.add_argument("--inner-window",type=int,default=36)
    args = parser.parse_args()
    if args.train is None:
        matches = list(Path(".").glob("*data_train.csv"))
        if len(matches)!=1:
            raise ValueError("Specify --train")
        args.train = matches[0]
    run(args.train,args.output_dir,args.first_origin,args.inner_first,args.inner_window)


if __name__ == "__main__":
    main()
