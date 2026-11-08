"""Audit completed DL forecasts and write a reproducible descriptive report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path("outputs_dl")
    plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "predictions.csv")
    stability = pd.read_csv(root / "seed_stability.csv")
    fits = pd.concat([
        pd.read_csv(root / f"{m}_{g}_{s}_fits.csv")
        for m in plan["models"] for g in plan["inputs"] for s in plan["seeds"]
    ], ignore_index=True)
    keys = ["model", "input_group", "seed"]
    assert not predictions.duplicated(keys + ["horizon", "origin"]).any()
    assert not fits.duplicated(keys + ["origin"]).any()
    assert np.isfinite(predictions[["actual", "predicted"]]).all().all()
    assert (predictions[["actual", "predicted"]] > 0).all().all()
    assert (fits.train_last_target <= fits.validation_first_origin).all()
    assert (fits.scaler_last_observation <= fits.validation_first_origin).all()
    assert (fits.validation_last_target <= fits.origin).all()
    configurations = len(plan["models"]) * len(plan["inputs"]) * len(plan["seeds"])
    assert fits.groupby(keys).ngroups == configurations
    for _, group in fits.groupby(keys):
        assert set(group.origin) == set(range(120, 203))
    for (_, _, _, h), group in predictions.groupby(keys + ["horizon"]):
        assert set(group.origin) == set(range(120, 204 - h))
    assert predictions.groupby(keys + ["horizon"]).ngroups == configurations * 3
    assert (stability.seeds == len(plan["seeds"])).all()
    assert len(predictions) == configurations * (83 + 82 + 81)
    reference_plan = json.loads(Path("outputs_ml/manifest.json").read_text(encoding="utf-8"))
    assert reference_plan["train_sha256"] == plan["train_sha256"]
    reference = pd.read_csv("outputs_statistics/predictions.csv")
    reference = reference.loc[reference.experiment == "naive", ["origin", "horizon", "actual", "current_price", "date"]]
    checked = predictions.merge(reference, on=["origin", "horizon"], suffixes=("", "_ref"), validate="many_to_one")
    assert len(checked) == len(predictions)
    for column in ("actual", "current_price"):
        np.testing.assert_allclose(checked[column], checked[f"{column}_ref"], rtol=0, atol=1e-12)
    assert (checked.date == checked.date_ref).all()

    best = stability.loc[stability.groupby(["model", "horizon"]).mape_mean.idxmin()].copy()
    best.to_csv(root / "posthoc_best_inputs.csv", index=False)
    lines = ["# 소규모 딥러닝 비교 결과", "",
             "3개 시드 MAPE의 평균 ± 표본 표준편차(%). 입력 조합 BEST는 전체 평가 결과를 본 사후 선택이며 독립 최종 성능이 아니다.", "",
             "| 모델 | H1 | H2 | H3 |", "|---|---:|---:|---:|"]
    for model in plan["models"]:
        cells = []
        for h in (1, 2, 3):
            row = best.loc[(best.model == model) & (best.horizon == h)].iloc[0]
            cells.append(f"{row.mape_mean:.4f} ± {row.mape_std:.4f} ({row.input_group})")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", "## 모든 입력 조합", "",
              "| 모델 | 입력 | H1 | H2 | H3 |", "|---|---|---:|---:|---:|"]
    for (model, group), part in stability.groupby(["model", "input_group"]):
        cells = [f"{r.mape_mean:.4f} ± {r.mape_std:.4f}" for r in part.sort_values("horizon").itertuples()]
        lines.append(f"| {model} | {group} | " + " | ".join(cells) + " |")
    lines += ["", "## 기존 후보와 같은 외부 원점에서의 비교", "",
              "| 고정 후보 | H1 MAPE | H2 MAPE | H3 MAPE |", "|---|---:|---:|---:|"]
    ml = pd.read_csv("outputs_ml/metrics.csv")
    for experiment in ("naive", "ridge__D_market"):
        part = ml.loc[(ml.year == "all") & (ml.experiment == experiment)].sort_values("horizon")
        if len(part) == 3:
            lines.append(f"| {experiment} | " + " | ".join(f"{x:.4f}%" for x in part.mape_pct) + " |")
    lines += ["", "Ridge D_market도 앞선 전체 구간 탐색의 사후 고정 BEST다. 과거 성적 기반 조합 선택 절차의 성능과 혼동하지 않는다.",
              "DL은 공동 로그 가격 출력·내부 조기 종료, 기존 ML은 horizon별 로그 수익률 회귀이므로 구조만 바꾼 통제 실험은 아니다.",
              "", "이번 고정 설정에서는 NBEATSx가 DL 후보 중 가장 좋지만 세 기간 모두 Ridge D_market보다 MAPE가 높다.",
              "NBEATSx H3 fundamentals는 Naive보다 평균 0.1671%p 낮지만 일부 시드는 Naive보다 나쁘다. 안정적인 우위를 주장하지 않는다.",
              "MLP·TFT는 시장 입력으로 개선됐으나 Naive보다 오차가 크다. NBEATSx는 수급 입력 대비 시장 추가가 H2에서만 개선됐다.",
              "기존 제출 모델을 DL로 교체할 근거는 이번 실험에서 확인하지 못했다.",
              "", "## 실행 및 무결성", "",
              f"- {configurations}개 모델·입력·시드 조합, {len(fits):,}회 원점별 학습, {len(predictions):,}개 예측을 확인했다.",
              "- 조합마다 H1/H2/H3 83/82/81개이며 중복·비유한 예측이 없다. 학습/검증 타깃 경계와 스케일러 시점 검사를 통과했다.",
              f"- 개별 학습 시간 합계: {fits.fit_seconds.sum():.1f}초. 파일 저장·데이터 준비·환경 설치 시간은 제외한다.",
              f"- 최적 step 중앙값: {fits.best_step.median():.0f}; 상한 80 step이 최적으로 선택된 비율: {(fits.best_step == 80).mean() * 100:.1f}%.",
              "- CPU 단일 스레드, 작은 고정 구조, 최대 80 step의 제한된 실험이다. 더 큰 예산에서의 성능이나 DL 전체의 우열을 단정하지 않는다.",
              "- 시드 표준편차는 초기화 민감도이지 통계적 신뢰구간이 아니다. 반복 탐색한 평가 구간은 미사용 홀드아웃이 아니다.",
              "- submission.csv는 이 실험에서 변경하지 않았다.", "",
              "[실험 설계](../notebooks/DL_COMPARISON.md) · [시드별/연도별 지표](metrics.csv) · [시드 안정성](seed_stability.csv) · [예측](predictions.csv) · [실행 설정](manifest.json)", ""]
    (root / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(best.to_string(index=False))
    print(f"Audit passed: {len(fits)} fits, {len(predictions)} predictions")


if __name__ == "__main__":
    main()
