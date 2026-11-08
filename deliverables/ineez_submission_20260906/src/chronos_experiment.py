"""GPU zero-shot Chronos-Bolt Small, fixed price-only 24-month context."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from src.evaluate import summarize_predictions
from src.features import TARGET, validate_monthly_frame

MODEL = "amazon/chronos-bolt-small"
LOOKBACK = 24


def context_at(price, origin):
    values = np.asarray(price[origin-LOOKBACK+1:origin+1], dtype=np.float32)
    if origin < LOOKBACK-1 or len(values) != LOOKBACK or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected 24 finite positive past prices")
    return torch.tensor(values.copy())


def median_forecast(pipeline, contexts):
    with torch.inference_mode():
        quantiles = pipeline.predict(contexts, prediction_length=3, limit_prediction_length=True)
    index = list(pipeline.quantiles).index(0.5)
    predicted = quantiles[:, index, :].detach().cpu().numpy().astype(float)
    if predicted.shape != (len(contexts), 3) or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Invalid median forecast; no silent clipping or fallback")
    return predicted


def run(args):
    from chronos import BaseChronosPipeline
    from huggingface_hub import HfApi, snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; install the project's CUDA torch build. No CPU fallback.")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = validate_monthly_frame(pd.read_csv(args.train))
    if len(data) != 204:
        raise ValueError("This fixed comparison expects 204 training months")
    output = Path("outputs_chronos")
    output.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    revision = args.revision or (old["revision"] if old else HfApi().model_info(MODEL).sha)
    snapshot = Path(snapshot_download(MODEL, revision=revision, cache_dir=".cache/huggingface",
                                     allow_patterns=["config.json", "model.safetensors", "README.md"]))
    pipeline = BaseChronosPipeline.from_pretrained(str(snapshot), device_map="cuda", torch_dtype=torch.float32)
    pipeline.model.eval()
    assert next(pipeline.model.parameters()).device.type == "cuda"
    plan = {
        "model": MODEL, "revision": revision,
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "huggingface-hub")},
        "device": str(next(pipeline.model.parameters()).device), "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda, "dtype": "float32", "batch_size": 16,
        "lookback": LOOKBACK, "origins": [120, 202], "horizons": [1, 2, 3],
        "inputs": "raw price only; internal per-context normalization; no external scaler or covariates",
        "point_forecast": "0.5 quantile (median), NOT an ensemble weight or MAPE-tuned quantile",
        "training": "none: frozen pretrained weights, no fine-tuning or validation selection",
        "seed": 42, "contamination_status": "unverified: public Henry Hub history may overlap pretraining; checkpoint released after evaluation period",
        "role": "retrospective exploratory comparison; NOT proven contamination-free historical forecast",
    }
    if old is not None and old != plan:
        raise ValueError("Existing output configuration differs; preserve old results before changing settings")
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    price = data[TARGET].to_numpy()
    rows = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for first in range(120, 203, 16):
        origins = list(range(first, min(first+16, 203)))
        contexts = torch.stack([context_at(price, t) for t in origins])
        forecasts = median_forecast(pipeline, contexts)
        if first == 120:
            repeated = median_forecast(pipeline, contexts)
            np.testing.assert_allclose(forecasts, repeated, rtol=0, atol=1e-6)
        for origin, forecast in zip(origins, forecasts):
            for h in (1, 2, 3):
                if origin+h < len(price):
                    rows.append(dict(date=str(data.loc[origin, "date"].date()), origin=origin, horizon=h,
                                     experiment="chronos_bolt_small_price24", actual=price[origin+h],
                                     predicted=forecast[h-1], current_price=price[origin], previous_price=price[origin-1]))
        print(f"GPU origins through {origins[-1]} complete", flush=True)
    torch.cuda.synchronize()
    runtime = {"inference_seconds_including_repeat": time.perf_counter()-started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "repeat_first_batch_max_atol": 1e-6}
    predictions = pd.DataFrame(rows)
    reference = pd.read_csv("outputs_statistics/predictions.csv")
    reference = reference.loc[reference.experiment == "naive"]
    merged = predictions.merge(reference, on=["origin", "horizon"], suffixes=("", "_ref"), validate="one_to_one")
    assert len(merged) == len(predictions) == 246
    assert (merged.date == merged.date_ref).all()
    for col in ("actual", "current_price", "previous_price"):
        np.testing.assert_allclose(merged[col], merged[col+"_ref"], rtol=0, atol=1e-12)
    metrics = []
    for h, part in predictions.groupby("horizon"):
        assert set(part.origin) == set(range(120, 204-h))
        for year, sub in [("all", part), *part.groupby(pd.to_datetime(part.date).dt.year)]:
            metrics.append(dict(horizon=h, year=year, **summarize_predictions(sub)))
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    report(output, plan, pd.DataFrame(metrics))


def report(output, plan, metrics):
    ml_plan = json.loads(Path("outputs_ml/manifest.json").read_text(encoding="utf-8"))
    assert ml_plan["train_sha256"] == plan["train_sha256"]
    comparisons = []
    def add(name, values):
        comparisons.append(dict(model=name, **{f"H{h}": v for h, v in zip((1, 2, 3), values)}))
    add("Chronos-Bolt Small price24", metrics.loc[metrics.year == "all"].sort_values("horizon").mape_pct)
    ml = pd.read_csv("outputs_ml/metrics.csv")
    for name in ("naive", "ridge__D_market"):
        add(name, ml.loc[(ml.year == "all") & (ml.experiment == name)].sort_values("horizon").mape_pct)
    dl = pd.read_csv("outputs_dl/seed_stability.csv")
    for group in ("price", "fundamentals"):
        add(f"NBEATSx {group} (3-seed mean)", dl.loc[(dl.model == "NBEATSx") & (dl.input_group == group)].sort_values("horizon").mape_mean)
    table = pd.DataFrame(comparisons)
    assert np.isfinite(table[["H1", "H2", "H3"]]).all().all()
    table.to_csv(output / "comparison.csv", index=False)
    lines = ["# Chronos-Bolt Small 비교 결과", "", "MAPE(%), 낮을수록 좋다. H1/H2/H3는 동일한 83/82/81개 원점이다.", "",
             "| 모델 | H1 | H2 | H3 |", "|---|---:|---:|---:|"]
    for r in table.itertuples():
        lines.append(f"| {r.model} | {r.H1:.4f} | {r.H2:.4f} | {r.H3:.4f} |")
    lines += ["", "최근 24개월 가격만 입력하고 GPU에서 사전학습 가중치를 고정한 zero-shot 예측을 수행했다.",
              "0.5 분위수(중앙값)를 사전에 고정했다. 이는 50:50 혼합 가중치가 아니며 MAPE 최적 분위수라는 뜻도 아니다.",
              "NBEATSx price는 동일 입력 길이의 가격-only 참고다. Ridge D_market은 수급·시장 변수도 사용한 사후 탐색 BEST다.",
              "추가 학습이나 시드별 모델 선택이 없으며 첫 배치 반복 추론의 일치 여부를 검사했다.", "",
              "## 검증 한계", "",
              "사전학습 데이터와 공개 Henry Hub 평가 시계열의 중복 여부는 확인하지 못했다. 체크포인트 공개 시점도 평가 기간 이후다.",
              "입력의 미래값 누수 방지는 검사했으나 가중치의 평가 데이터 노출 가능성까지 배제한 것은 아니다.",
              "따라서 이 결과는 회고적 탐색 비교이며 오염 없는 역사적 out-of-sample 성능이나 최종 제출 모델 우위로 주장하지 않는다.",
              "submission.csv는 변경하지 않았다.", "", f"고정 revision: `{plan['revision']}`", "",
              "[설계](../notebooks/CHRONOS_COMPARISON.md) · [설정](manifest.json) · [실행 시간/GPU 메모리](runtime.json) · [연도별 지표](metrics.csv)", ""]
    (output / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--revision", help="Pinned Hugging Face commit; subsequent runs reuse manifest revision")
    args = parser.parse_args()
    if args.train is None:
        paths = list(Path(".").glob("*data_train.csv"))
        if len(paths) != 1:
            raise ValueError("Specify --train")
        args.train = paths[0]
    run(args)
