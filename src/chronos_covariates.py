"""Chronos-2 GPU controlled price/fundamentals/market ablation, past only."""

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from src.evaluate import summarize_predictions
from src.features import TARGET, validate_monthly_frame
from src.hypotheses import build_hypotheses

MODEL = "amazon/chronos-2"
OUTPUT = Path("outputs_chronos_covariates")
LOOKBACK = 24
GROUPS = {"price": [], "fundamentals": ["production_level", "storage_seasonal_z", "consumption_z"],
          "market": ["production_level", "storage_seasonal_z", "consumption_z", "BCOMNG_INDX_return"]}


def task_at(data, features, origin, group):
    if origin < LOOKBACK-1 or origin >= len(data):
        raise ValueError("Invalid origin")
    start = origin-LOOKBACK+1
    target = data.loc[start:origin, TARGET].to_numpy(dtype=np.float32, copy=True)
    if len(target) != LOOKBACK or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Invalid price history")
    task = {"target": target}
    if GROUPS[group]:
        covariates = {c: features.loc[start:origin, c].to_numpy(dtype=np.float32, copy=True) for c in GROUPS[group]}
        # Model supports masked NaNs; preserve causal missingness, never future interpolation.
        if any(np.isinf(v).any() or len(v) != LOOKBACK for v in covariates.values()):
            raise ValueError("Invalid covariate history")
        task["past_covariates"] = covariates
    return task


def predict_one(pipeline, task):
    # One origin per call: no possibility of borrowing later origins' observations.
    with torch.inference_mode():
        result = pipeline.predict([task], prediction_length=3, context_length=LOOKBACK,
                                  batch_size=8, cross_learning=False, limit_prediction_length=True)
    assert len(result) == 1
    index = list(pipeline.quantiles).index(.5)
    prediction = result[0][0, index, :].detach().cpu().numpy().astype(float)
    if prediction.shape != (3,) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Invalid median prices; no silent clipping or baseline fallback")
    return prediction


def main():
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi, snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback disabled")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    paths = list(Path(".").glob("*data_train.csv"))
    if len(paths) != 1:
        raise ValueError("Expected one train file")
    data = validate_monthly_frame(pd.read_csv(paths[0]))
    if len(data) != 204:
        raise ValueError("Expected 204 months")
    features, definitions, _ = build_hypotheses(data, "conservative")
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT / "manifest.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    revision = old["revision"] if old else HfApi().model_info(MODEL).sha
    snapshot = Path(snapshot_download(MODEL, revision=revision, cache_dir=".cache/huggingface",
                                     allow_patterns=["config.json", "model.safetensors", "README.md"]))
    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cuda", torch_dtype=torch.float32)
    pipeline.model.eval()
    assert next(pipeline.model.parameters()).device.type == "cuda"
    plan = {
        "model": MODEL, "revision": revision,
        "train_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                        ("src/chronos_covariates.py", "src/hypotheses.py", "src/features.py", "src/evaluate.py")},
        "weights_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "numpy", "pandas")},
        "gpu": torch.cuda.get_device_name(), "device": str(next(pipeline.model.parameters()).device),
        "dtype": "float32", "seed": 42, "lookback": LOOKBACK, "groups": GROUPS,
        "prediction_length": 3, "quantile": .5, "first_origin": 120, "last_origin": 202,
        "inference": "one origin per call, cross_learning=False, frozen zero-shot, no tuning",
        "past_covariates_only": True, "future_covariates": "not supplied",
        "availability": "production/consumption lag2; storage lag1; market lag0; causal features reused from ML/DL",
        "normalization": "model internal per-context normalization; precomputed causal Z features shared with ML/DL; no global imputer",
        "external_economic_data": "none: existing data already contains all required aligned inputs; downloaded only model weights",
        "limitations": "Chronos-2 differs from Bolt; compare input ablations WITHIN Chronos-2. Public historical evaluation may overlap pretraining; not certified uncontaminated OOS. Existing ML references differ in training, target transform, and history representation.",
    }
    if old is not None and old != plan:
        raise ValueError("Output configuration differs; preserve previous results")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    definitions.loc[definitions.feature.isin(GROUPS["market"])].to_csv(OUTPUT / "feature_definitions.csv", index=False)
    features[GROUPS["market"]].assign(date=data.date).to_csv(OUTPUT / "aligned_features.csv", index_label="origin")
    records, diagnostics, input_rows = [], [], []
    for group in GROUPS:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        repeat_delta = None
        for origin in range(120, 203):
            task = task_at(data, features, origin, group)
            prediction = predict_one(pipeline, task)
            if origin == 120:
                repeated = predict_one(pipeline, task)
                np.testing.assert_allclose(prediction, repeated, rtol=0, atol=1e-6)
                repeat_delta = float(np.max(np.abs(prediction-repeated)))
            for col, values in task.get("past_covariates", {}).items():
                input_rows.append(dict(group=group, origin=origin, feature=col, first_context_origin=origin-LOOKBACK+1,
                                       last_context_origin=origin, missing_count=int(np.isnan(values).sum())))
            for h in (1, 2, 3):
                if origin+h < len(data):
                    records.append(dict(group=group, experiment=f"chronos2_{group}", date=data.loc[origin, "date"],
                                        origin=origin, horizon=h, actual=float(data.loc[origin+h, TARGET]),
                                        predicted=prediction[h-1], current_price=float(data.loc[origin, TARGET]),
                                        previous_price=float(data.loc[origin-1, TARGET])))
        torch.cuda.synchronize()
        diagnostics.append(dict(group=group, inference_seconds=time.perf_counter()-start_time,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(), repeat_max_difference=repeat_delta))
        print(f"DONE Chronos-2 {group} on GPU", flush=True)
    predictions = pd.DataFrame(records)
    reference = pd.read_csv("outputs_statistics/predictions.csv", parse_dates=["date"])
    reference = reference.loc[reference.experiment == "naive"]
    paired = predictions.merge(reference, on=["origin", "horizon"], suffixes=("", "_ref"), validate="many_to_one")
    assert len(paired) == len(predictions) == 738
    assert not predictions.duplicated(["group", "origin", "horizon"]).any()
    assert (paired.date == paired.date_ref).all()
    for col in ("actual", "current_price", "previous_price"):
        np.testing.assert_allclose(paired[col], paired[col+"_ref"], rtol=0, atol=1e-12)
    metrics = []
    for (group, h), part in predictions.groupby(["group", "horizon"]):
        assert set(part.origin) == set(range(120, 204-h))
        for year, sub in [("all", part), *part.groupby(part.date.dt.year)]:
            metrics.append(dict(group=group, horizon=h, year=year, **summarize_predictions(sub)))
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT / "metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "runtime.csv", index=False)
    pd.DataFrame(input_rows).to_csv(OUTPUT / "input_audit.csv", index=False)
    from src.chronos_covariates_report import main as make_report
    make_report()


if __name__ == "__main__":
    main()
