"""Train, compare, interpret, and export Henry Hub forecasts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import summarize_predictions
from src.ensemble import save_ensemble_diagnostics
from src.features import TARGET, make_feature_matrix, validate_monthly_frame
from src.validation import make_horizon_targets, rolling_origin_splits


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_type: str


MODEL_SPECS = {
    "ridge_log_return": ModelSpec("ridge_log_return", "log_return"),
    "elastic_net_log_return": ModelSpec("elastic_net_log_return", "log_return"),
    "lightgbm_log_return": ModelSpec("lightgbm_log_return", "log_return"),
    "ridge_raw_price": ModelSpec("ridge_raw_price", "future_price"),
}


def _find_file(suffix: str) -> Path:
    matches = sorted(Path(".").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one '*{suffix}', found {matches}")
    return matches[0]


def _pipeline(model_name: str) -> Pipeline:
    common: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        ),
        ("scaler", StandardScaler()),
    ]
    if model_name in {"ridge_log_return", "ridge_raw_price"}:
        estimator = Ridge(alpha=10.0)
    elif model_name == "elastic_net_log_return":
        estimator = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20_000, random_state=42)
    elif model_name == "lightgbm_log_return":
        estimator = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    else:
        raise KeyError(f"Unknown model: {model_name}")
    return Pipeline(common + [("model", estimator)])


def _to_price(prediction: float, current_price: float, target_type: str) -> float:
    if target_type == "log_return":
        prediction = current_price * float(np.exp(prediction))
    return max(float(prediction), 0.01)


def backtest_baseline(
    data: pd.DataFrame, horizon: int, model_name: str, first_origin: int
) -> pd.DataFrame:
    targets = make_horizon_targets(data, horizon)
    rows: list[dict[str, Any]] = []
    for split in rolling_origin_splits(len(data), horizon, first_origin=first_origin):
        origin = split.origin
        current = float(targets.loc[origin, "current_price"])
        if model_name == "last_value_naive":
            prediction = current
        elif model_name == "seasonal_naive":
            prediction = float(data.loc[origin + horizon - 12, TARGET])
        else:
            raise KeyError(model_name)
        rows.append(
            {
                "date": data.loc[origin, "date"],
                "horizon": horizon,
                "feature_group": "baseline",
                "model": model_name,
                "target_type": "price",
                "actual": float(targets.loc[origin, "future_price"]),
                "predicted": prediction,
                "current_price": current,
                "previous_price": float(data.loc[origin - 1, TARGET]),
            }
        )
    return pd.DataFrame(rows)


def backtest_model(
    data: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    model_name: str,
    feature_group: str,
    first_origin: int,
) -> pd.DataFrame:
    targets = make_horizon_targets(data, horizon)
    spec = MODEL_SPECS[model_name]
    rows: list[dict[str, Any]] = []
    for split in rolling_origin_splits(len(data), horizon, first_origin=first_origin):
        origin = split.origin
        train_indices = split.train_indices
        estimator = _pipeline(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(
                features.loc[train_indices, feature_columns],
                targets.loc[train_indices, spec.target_type],
            )
        raw_prediction = float(
            estimator.predict(features.loc[[origin], feature_columns])[0]
        )
        current = float(targets.loc[origin, "current_price"])
        rows.append(
            {
                "date": data.loc[origin, "date"],
                "horizon": horizon,
                "feature_group": feature_group,
                "model": model_name,
                "target_type": spec.target_type,
                "actual": float(targets.loc[origin, "future_price"]),
                "predicted": _to_price(raw_prediction, current, spec.target_type),
                "current_price": current,
                "previous_price": float(data.loc[origin - 1, TARGET]),
            }
        )
    return pd.DataFrame(rows)


def _metric_row(predictions: pd.DataFrame, stage: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "horizon": int(predictions["horizon"].iloc[0]),
        "feature_group": predictions["feature_group"].iloc[0],
        "model": predictions["model"].iloc[0],
        "target_type": predictions["target_type"].iloc[0],
    }
    row.update(summarize_predictions(predictions))
    return row


def _yearly_metrics(predictions: pd.DataFrame, stage: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for year, part in predictions.groupby(predictions["date"].dt.year):
        row = _metric_row(part, stage)
        row["validation_year"] = int(year)
        records.append(row)
    return pd.DataFrame(records)


def choose_feature_groups(
    ablation_predictions: list[pd.DataFrame], ablation_metrics: pd.DataFrame
) -> dict[int, str]:
    """Choose a group only when improvement over E1 is broad across years."""
    chosen: dict[int, str] = {}
    combined = pd.concat(ablation_predictions, ignore_index=True)
    for horizon in (1, 2, 3):
        metrics_h = ablation_metrics[ablation_metrics["horizon"] == horizon].sort_values(
            "mape_pct"
        )
        baseline = combined[
            (combined["horizon"] == horizon)
            & (combined["feature_group"] == "E1_history")
        ].copy()
        baseline["ape"] = np.abs(
            (baseline["actual"] - baseline["predicted"]) / baseline["actual"]
        )
        selected = "E1_history"
        e1_mape = float(
            metrics_h.loc[metrics_h["feature_group"] == "E1_history", "mape_pct"].iloc[0]
        )
        for candidate in metrics_h.itertuples(index=False):
            if candidate.feature_group == "E1_history" or candidate.mape_pct >= e1_mape:
                continue
            candidate_predictions = combined[
                (combined["horizon"] == horizon)
                & (combined["feature_group"] == candidate.feature_group)
            ].copy()
            candidate_predictions["ape"] = np.abs(
                (candidate_predictions["actual"] - candidate_predictions["predicted"])
                / candidate_predictions["actual"]
            )
            paired = baseline[["date", "ape"]].merge(
                candidate_predictions[["date", "ape"]], on="date", suffixes=("_e1", "_candidate")
            )
            yearly = paired.assign(year=paired["date"].dt.year).groupby("year")[
                ["ape_e1", "ape_candidate"]
            ].mean()
            if float((yearly["ape_candidate"] < yearly["ape_e1"]).mean()) >= 0.5:
                selected = candidate.feature_group
                break
        chosen[horizon] = selected
    return chosen


def choose_final_models(metrics: pd.DataFrame) -> dict[int, str]:
    """Select by MAPE, requiring learned models to beat last-value naive."""
    selected: dict[int, str] = {}
    for horizon in (1, 2, 3):
        part = metrics[metrics["horizon"] == horizon].sort_values("mape_pct")
        naive_mape = float(
            part.loc[part["model"] == "last_value_naive", "mape_pct"].iloc[0]
        )
        best = part.iloc[0]
        if best["model"] in MODEL_SPECS and float(best["mape_pct"]) > naive_mape * 0.995:
            selected[horizon] = "last_value_naive"
        else:
            selected[horizon] = str(best["model"])
    return selected


def _feature_contributions(
    estimator: Pipeline,
    current_features: pd.DataFrame,
    raw_feature_names: list[str],
    model_name: str,
) -> tuple[np.ndarray, list[str], str]:
    imputer = estimator.named_steps["imputer"]
    scaler = estimator.named_steps["scaler"]
    model = estimator.named_steps["model"]
    imputed = imputer.transform(current_features)
    transformed = scaler.transform(imputed)
    names = list(imputer.get_feature_names_out(raw_feature_names))
    if model_name == "lightgbm_log_return":
        contributions = model.predict(transformed, pred_contrib=True)[0][:-1]
        method = "tree_shap"
    else:
        contributions = transformed[0] * model.coef_
        method = "linear_shap_independence"
    return np.asarray(contributions), names, method


def walk_forward_test_predictions(
    combined_data: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    train_rows: int,
    horizon: int,
    model_name: str,
    feature_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = make_horizon_targets(combined_data, horizon)
    prediction_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for origin in range(train_rows, len(combined_data)):
        current = float(combined_data.loc[origin, TARGET])
        if model_name == "last_value_naive":
            prediction = current
        elif model_name == "seasonal_naive":
            prediction = float(combined_data.loc[origin + horizon - 12, TARGET])
        else:
            spec = MODEL_SPECS[model_name]
            train_end = origin - horizon
            train_indices = np.arange(train_end + 1, dtype=int)
            estimator = _pipeline(model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(
                    features.loc[train_indices, feature_columns],
                    targets.loc[train_indices, spec.target_type],
                )
            current_x = features.loc[[origin], feature_columns]
            raw_prediction = float(estimator.predict(current_x)[0])
            prediction = _to_price(raw_prediction, current, spec.target_type)
            contributions, names, method = _feature_contributions(
                estimator, current_x, feature_columns, model_name
            )
            contribution_rows.extend(
                {
                    "date": combined_data.loc[origin, "date"],
                    "horizon": horizon,
                    "model": model_name,
                    "feature_group": feature_group,
                    "target_space": spec.target_type,
                    "method": method,
                    "feature": name,
                    "contribution": float(value),
                }
                for name, value in zip(names, contributions, strict=True)
            )
        prediction_rows.append(
            {
                "date": combined_data.loc[origin, "date"],
                f"pred_{horizon}": prediction,
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(contribution_rows)


def _make_figures(
    data: pd.DataFrame,
    train_rows: int,
    comparison_metrics: pd.DataFrame,
    final_backtests: list[pd.DataFrame],
    contributions: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["date"], data[TARGET], color="#174A7E", linewidth=1.6)
    ax.axvline(data.loc[train_rows, "date"], color="#C44E52", linestyle="--", label="test starts")
    ax.set(title="Henry Hub monthly price", xlabel="Date", ylabel="Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "price_history.png", dpi=160)
    plt.close(fig)

    pivot = comparison_metrics.pivot(index="model", columns="horizon", values="mape_pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set(title="Rolling-origin MAPE by horizon", xlabel="Model", ylabel="MAPE (%)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_mape.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for horizon, predictions, ax in zip((1, 2, 3), final_backtests, axes, strict=True):
        ax.plot(predictions["date"], predictions["actual"], label="actual", color="#174A7E")
        ax.plot(predictions["date"], predictions["predicted"], label="predicted", color="#E45756")
        ax.set_title(f"H{horizon}: {predictions['model'].iloc[0]}")
        ax.set_ylabel("Price")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Forecast origin")
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_predictions.png", dpi=160)
    plt.close(fig)

    if not contributions.empty:
        summary = (
            contributions.assign(abs_contribution=lambda x: x["contribution"].abs())
            .groupby("feature", as_index=False)["abs_contribution"]
            .mean()
            .nlargest(15, "abs_contribution")
            .sort_values("abs_contribution")
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(summary["feature"], summary["abs_contribution"], color="#4C78A8")
        ax.set(
            title="Top mean absolute SHAP contributions (test forecasts)",
            xlabel="Mean absolute contribution in model target space",
            ylabel="Feature",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_contributions.png", dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    train = validate_monthly_frame(pd.read_csv(args.train))
    test = validate_monthly_frame(pd.read_csv(args.test))
    if train["date"].max() >= test["date"].min():
        raise ValueError("Train dates must precede test dates")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train and test schemas differ")

    combined = validate_monthly_frame(pd.concat([train, test], ignore_index=True))
    train_features, groups = make_feature_matrix(train, args.availability_mode)
    combined_features, _ = make_feature_matrix(combined, args.availability_mode)
    first_origin = min(args.first_origin, len(train) - 4)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_backtests: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    ablation_predictions: list[pd.DataFrame] = []
    for horizon in (1, 2, 3):
        for group_name, columns in groups.items():
            predictions = backtest_model(
                train,
                train_features,
                columns,
                horizon,
                "ridge_log_return",
                group_name,
                first_origin,
            )
            predictions["stage"] = "ablation"
            ablation_predictions.append(predictions)
            all_backtests.append(predictions)
            metric_rows.append(_metric_row(predictions, "ablation"))
            yearly_frames.append(_yearly_metrics(predictions, "ablation"))

    metrics = pd.DataFrame(metric_rows)
    chosen_groups = choose_feature_groups(
        ablation_predictions, metrics[metrics["stage"] == "ablation"]
    )

    comparison_predictions: list[pd.DataFrame] = []
    for horizon in (1, 2, 3):
        group_name = chosen_groups[horizon]
        columns = groups[group_name]
        for baseline in ("last_value_naive", "seasonal_naive"):
            predictions = backtest_baseline(train, horizon, baseline, first_origin)
            predictions["feature_group"] = group_name
            predictions["stage"] = "model_comparison"
            comparison_predictions.append(predictions)
        # Ridge log-return was already evaluated, but recomputing keeps this
        # stage self-contained and makes saved predictions easy to audit.
        for model_name in MODEL_SPECS:
            predictions = backtest_model(
                train,
                train_features,
                columns,
                horizon,
                model_name,
                group_name,
                first_origin,
            )
            predictions["stage"] = "model_comparison"
            comparison_predictions.append(predictions)

    for predictions in comparison_predictions:
        all_backtests.append(predictions)
        metric_rows.append(_metric_row(predictions, "model_comparison"))
        yearly_frames.append(_yearly_metrics(predictions, "model_comparison"))

    metrics = pd.DataFrame(metric_rows)
    comparison_metrics = metrics[metrics["stage"] == "model_comparison"].copy()
    selected_models = choose_final_models(comparison_metrics)

    final_test_parts: list[pd.DataFrame] = []
    contribution_parts: list[pd.DataFrame] = []
    final_backtests: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    combined_comparisons = pd.concat(comparison_predictions, ignore_index=True)
    for horizon in (1, 2, 3):
        model_name = selected_models[horizon]
        group_name = chosen_groups[horizon]
        columns = groups[group_name]
        predictions, contributions = walk_forward_test_predictions(
            combined,
            combined_features,
            columns,
            len(train),
            horizon,
            model_name,
            group_name,
        )
        final_test_parts.append(predictions)
        if not contributions.empty:
            contribution_parts.append(contributions)
        backtest = combined_comparisons[
            (combined_comparisons["horizon"] == horizon)
            & (combined_comparisons["model"] == model_name)
        ].copy()
        final_backtests.append(backtest)
        chosen_metric = comparison_metrics[
            (comparison_metrics["horizon"] == horizon)
            & (comparison_metrics["model"] == model_name)
        ].iloc[0]
        selection_rows.append(
            {
                "horizon": horizon,
                "feature_group": group_name,
                "model": model_name,
                "validation_mape_pct": float(chosen_metric["mape_pct"]),
                "validation_mae": float(chosen_metric["mae"]),
                "direction_accuracy_pct": float(chosen_metric["direction_accuracy_pct"]),
                "turning_f1_pct": float(chosen_metric["turning_f1_pct"]),
            }
        )

    template = pd.read_csv(args.submission_template, parse_dates=["date"])[["date"]]
    submission = template.copy()
    for part in final_test_parts:
        submission = submission.merge(part, on="date", how="left", validate="one_to_one")
    submission = submission[["date", "pred_1", "pred_2", "pred_3"]]
    if submission.isna().any().any() or len(submission) != len(test):
        raise RuntimeError("Submission is incomplete or has the wrong row count")

    contributions = (
        pd.concat(contribution_parts, ignore_index=True)
        if contribution_parts
        else pd.DataFrame(
            columns=[
                "date",
                "horizon",
                "model",
                "feature_group",
                "target_space",
                "method",
                "feature",
                "contribution",
            ]
        )
    )
    backtest_output = pd.concat(all_backtests, ignore_index=True)
    save_ensemble_diagnostics(backtest_output, output_dir)
    yearly_output = pd.concat(yearly_frames, ignore_index=True)
    selection = pd.DataFrame(selection_rows)

    submission.to_csv(args.submission, index=False, float_format="%.6f")
    metrics.to_csv(output_dir / "experiment_metrics.csv", index=False, float_format="%.6f")
    yearly_output.to_csv(output_dir / "yearly_metrics.csv", index=False, float_format="%.6f")
    backtest_output.to_csv(output_dir / "backtest_predictions.csv", index=False, float_format="%.6f")
    selection.to_csv(output_dir / "model_selection.csv", index=False, float_format="%.6f")
    contributions.to_csv(output_dir / "shap_contributions.csv", index=False, float_format="%.8f")

    summary = {
        "availability_mode": args.availability_mode,
        "train_rows": len(train),
        "test_rows": len(test),
        "first_validation_origin": str(train.loc[first_origin, "date"].date()),
        "feature_groups": chosen_groups,
        "models": selected_models,
        "selection": selection_rows,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _make_figures(
        combined,
        len(train),
        comparison_metrics,
        final_backtests,
        contributions,
        figures_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--submission-template", type=Path, default=None)
    parser.add_argument(
        "--availability-mode",
        choices=("conservative", "as_provided"),
        default="conservative",
    )
    parser.add_argument("--first-origin", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--submission", default="submission.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.train = args.train or _find_file("data_train.csv")
    args.test = args.test or _find_file("data_test.csv")
    args.submission_template = args.submission_template or _find_file("submission_example.csv")
    run(args)


if __name__ == "__main__":
    main()
