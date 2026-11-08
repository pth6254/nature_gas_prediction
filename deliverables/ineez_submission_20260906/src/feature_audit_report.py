"""Paired contrasts for the four feature-audit questions (no model selection)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_audit import OUTPUT
from src.ml_experiment import FAMILIES


def main():
    metrics = pd.read_csv(OUTPUT / "metrics.csv", dtype={"year": str})
    predictions = pd.read_csv(OUTPUT / "predictions.csv")
    plan = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    old_plan = json.loads(Path("outputs_ml/manifest.json").read_text(encoding="utf-8"))
    assert plan["train_sha256"] == old_plan["train_sha256"]
    old = pd.read_csv("outputs_ml/predictions.csv")
    old = old.loc[old.experiment.isin([f"{f}__D_market" for f in FAMILIES])]
    current = predictions.loc[predictions.combination == "current"]
    check = current.merge(old, on=["family", "origin", "horizon"], suffixes=("", "_old"), validate="one_to_one")
    assert len(check) == len(current) == 984
    np.testing.assert_allclose(check.predicted, check.predicted_old, atol=1e-8, rtol=1e-8)
    overall = metrics.loc[metrics.year == "all"].set_index(["family", "combination", "horizon"])
    edges = [
        ("representation", "current", "basic_raw"),
        ("representation", "basic_processed", "basic_raw"),
        ("representation", "current", "engineered_raw"),
        ("representation", "engineered_raw", "basic_raw"),
        ("representation", "current", "basic_processed"),
        *[("ablation", f"drop_{r}", "current") for r in ("production", "storage", "consumption", "market")],
        *[("lag", c, "current") for c in ("lag_1", "lag_3", "lag_6", "lag_selected")],
        ("compression", "equal_index", "individual_z"),
    ]
    contrasts = []
    for question, candidate, reference in edges:
        for family in FAMILIES:
            for h in (1, 2, 3):
                c = overall.loc[(family, candidate, h)]
                r = overall.loc[(family, reference, h)]
                a = predictions.loc[(predictions.family == family) & (predictions.horizon == h) & (predictions.combination == candidate)]
                b = predictions.loc[(predictions.family == family) & (predictions.horizon == h) & (predictions.combination == reference)]
                paired = a.merge(b, on="origin", suffixes=("_a", "_b"), validate="one_to_one")
                assert len(paired) == 84-h
                delta = ((paired.predicted_a/paired.actual_a-1).abs() - (paired.predicted_b/paired.actual_b-1).abs())*100
                annual = delta.groupby(paired.date_a.str[:4]).mean()
                contrasts.append(dict(question=question, family=family, horizon=h, candidate=candidate,
                                      reference=reference, candidate_mape=c.mape_pct, reference_mape=r.mape_pct,
                                      delta_mape_pp=c.mape_pct-r.mape_pct, delta_mae=c.mae-r.mae,
                                      delta_turning_f1_pp=c.turning_f1_pct-r.turning_f1_pct,
                                      origins_better=int((delta < 0).sum()), n_origins=len(delta),
                                      years_better=int((annual < 0).sum()), n_years=len(annual)))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(OUTPUT / "contrasts.csv", index=False)
    scores = pd.read_csv(OUTPUT / "lag_selection_scores.csv")
    counts = scores.loc[scores.selected].groupby(["family", "horizon", "experiment"]).size().rename("count").reset_index()
    counts.to_csv(OUTPUT / "lag_selection_counts.csv", index=False)
    warnings = [w for p in (OUTPUT / "cache").glob("*.warnings.json") for w in json.loads(p.read_text(encoding="utf-8"))]
    pd.DataFrame(warnings, columns=["experiment", "horizon", "origin", "warning"]).to_csv(OUTPUT / "fit_warnings.csv", index=False)
    lines = ["# 변수 설계 편향·제거·시차·압축 실험 결과", "",
             "4개 고정 ML 모델에 13개 공통 입력을 적용했다. 전체 수치는 회고적 탐색이며 최종 홀드아웃 성능이 아니다.",
             "각 행의 H1/H2/H3는 MAPE(%)이며 낮을수록 좋다. 동일 원점 83/82/81개로 비교한다.", ""]

    def table(names):
        lines.extend(["| 모델 | 입력 | H1 | H2 | H3 |", "|---|---|---:|---:|---:|"])
        for f in FAMILIES:
            for name in names:
                vals = [overall.loc[(f, name, h), "mape_pct"] for h in (1, 2, 3)]
                lines.append(f"| {f} | {name} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        lines.append("")

    lines.extend(["## 1. 입력 표현이 특정 모델에 유리한가?", "",
                  "2×2 비교: 가격 입력은 기본 시차/가공 특징, 외생 입력은 원시 수준/현재 가공 표현으로 나눈다.",
                  "current가 이전 D_market이다. 원시 수준도 발표 지연·학습 구간 내 결측 대체·표준화를 적용한다.", ""])
    table(["basic_raw", "basic_processed", "engineered_raw", "current"])
    rep = contrasts.loc[(contrasts.question == "representation") & (contrasts.candidate == "current") & (contrasts.reference == "basic_raw")]
    lines.append(f"현재 가공 조합은 기본 원시 조합 대비 {int((rep.delta_mape_pp < 0).sum())}/12개 모델·기간에서 개선됐다.")
    lines.extend(["가격 특징 수가 기본 8개/가공 18개로 다르므로 정보량과 표현이 함께 변한다. 이 결과만으로 선형 편향의 존재·부재를 확정하지 않는다.", "",
                  "## 2. 각 변수군의 조건부 예측 가치", "",
                  "현재 조합에서 해당 변수 하나를 제거한 뒤 매 원점에서 다시 학습했다. 값만 0으로 덮은 실험이 아니다.", ""])
    table(["current", "drop_production", "drop_storage", "drop_consumption", "drop_market"])
    for role in ("production", "storage", "consumption", "market"):
        part = contrasts.loc[(contrasts.question == "ablation") & (contrasts.candidate == f"drop_{role}")]
        lines.append(f"- {role} 제거 후 악화: {int((part.delta_mape_pp > 0).sum())}/12; 개선: {int((part.delta_mape_pp < 0).sum())}/12.")
    lines.extend(["", "제거 후 개선은 현재 다른 입력·설정에서의 중복/잡음 가능성이지 해당 경제 요소의 무관함이나 인과효과 부재를 뜻하지 않는다.", "",
                  "## 3. 발표 지연을 유지한 추가 반응 시차", "",
                  "생산·재고·소비를 함께 추가 0/1/3/6개월 이동했다. 시장·가격 이력은 유지했다. 변수별 서로 다른 시차의 전체 탐색은 아니다.",
                  "lag_selected는 원점마다 정답이 알려진 최근 36개 과거 예측으로 시차만 선택한 절차다. 같은 원점의 미래 정답은 선택에 쓰지 않았다.", ""])
    table(["current", "lag_1", "lag_3", "lag_6", "lag_selected"])
    lag = contrasts.loc[contrasts.candidate == "lag_selected"]
    lines.extend([f"과거 기반 시차 선택이 추가 시차 0보다 개선된 비교는 {int((lag.delta_mape_pp < 0).sum())}/12개다.",
                  "고정 시차의 전체 구간 최저값과 선택 절차의 성능은 구분해야 한다. Naive는 시차 선택 풀에 넣지 않았다.", "",
                  "## 4. 합성 지수의 정보 압축", "",
                  "동일한 production_z·storage_z·consumption_z 개별 입력과 그 합(+생산+재고−소비) 하나를 비교했다. 가격 이력·시장 입력은 동일하다.",
                  "현재 D의 생산 수준과 계절 재고를 그대로 더하지 않았다. 혼합 단위 문제를 피하기 위해 두 비교군 모두 같은 Z 표현으로 맞췄다.", ""])
    table(["individual_z", "equal_index"])
    comp = contrasts.loc[contrasts.question == "compression"]
    lines.extend([f"합성 지수로 압축했을 때 악화된 비교는 {int((comp.delta_mape_pp > 0).sum())}/12개다.",
                  "압축은 세 변수의 독립적 효과와 상호작용을 제한한다. 성능 차이에는 차원 및 규제 변화도 포함돼 순수한 정보 손실만 측정한 것은 아니다.", "",
                  "## 검증·한계", "",
                  "- 현재 조합의 984개 예측은 기존 네 모델 D_market 예측과 허용오차 1e-8로 일치했다.",
                  "- 모든 후보가 같은 실제값·예측 원점으로 비교됐고 선택 점수의 마지막 타깃 시점이 원점을 넘지 않았다.",
                  f"- 학습 경고 {len(warnings)}건. 상세 경고와 연도별·원점별 개선은 CSV에 보존했다.",
                  "- 고정 모델 설정·CPU 조건을 유지했다. DL/Chronos 입력 비교나 모델별 하이퍼파라미터 최적화는 포함하지 않았다.",
                  "- 원점별 오차는 시간적으로 독립이 아니며 반복 탐색의 선택 편향이 있다. 승리 횟수는 유의성 검정이 아니다.",
                  "- 이 실험만으로 새 최종 후보를 자동 채택하지 않으며 submission.csv를 유지한다.", "",
                  "[실험 설계](../notebooks/FEATURE_AUDIT.md) · [입력 명세](manifest.json) · [쌍별 차이](contrasts.csv) · [시차 선택 기록](lag_selection_scores.csv) · [연도별 지표](metrics.csv)", ""])
    (OUTPUT / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(contrasts.loc[((contrasts.question == "representation") & (contrasts.candidate == "current") & (contrasts.reference == "basic_raw")) |
                        (contrasts.question == "ablation") | (contrasts.candidate == "lag_selected") | (contrasts.question == "compression"),
                       ["question", "family", "horizon", "candidate", "delta_mape_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
