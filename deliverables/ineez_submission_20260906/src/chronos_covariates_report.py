"""Report within-Chronos-2 covariate effects separately from model changes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.chronos_covariates import OUTPUT


def main():
    m = pd.read_csv(OUTPUT / "metrics.csv", dtype={"year": str})
    overall = m.loc[m.year == "all"].set_index(["group", "horizon"])
    plan = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    differences = []
    for candidate, reference in [("fundamentals", "price"), ("market", "fundamentals"), ("market", "price")]:
        for h in (1, 2, 3):
            a, b = overall.loc[(candidate, h)], overall.loc[(reference, h)]
            yearly = m.loc[(m.year != "all") & (m.horizon == h)].pivot(index="year", columns="group", values="mape_pct")
            differences.append(dict(candidate=candidate, reference=reference, horizon=h,
                                    delta_mape_pp=a.mape_pct-b.mape_pct, delta_mae=a.mae-b.mae,
                                    delta_turning_f1_pp=a.turning_f1_pct-b.turning_f1_pct,
                                    years_better=int((yearly[candidate] < yearly[reference]).sum()), n_years=len(yearly)))
    contrasts = pd.DataFrame(differences)
    contrasts.to_csv(OUTPUT / "contrasts.csv", index=False)
    rows = []
    def add(name, vals, role):
        assert len(vals) == 3 and np.isfinite(vals).all()
        rows.append(dict(model=name, H1=vals[0], H2=vals[1], H3=vals[2], role=role))
    for group in ("price", "fundamentals", "market"):
        add(f"Chronos-2 {group}", [overall.loc[(group, h), "mape_pct"] for h in (1, 2, 3)], "within_model_control")
    bolt = pd.read_csv("outputs_chronos/metrics.csv", dtype={"year": str})
    add("Chronos-Bolt Small price", bolt.loc[bolt.year == "all"].sort_values("horizon").mape_pct.tolist(), "different_checkpoint_reference")
    ml = pd.read_csv("outputs_ml/metrics.csv", dtype={"year": str})
    for name in ("naive", "ridge__D_market"):
        add(name, ml.loc[(ml.year == "all") & (ml.experiment == name)].sort_values("horizon").mape_pct.tolist(), "different_pipeline_reference")
    audit = pd.read_csv("outputs_feature_audit/metrics.csv", dtype={"year": str})
    add("Ridge basic_processed", audit.loc[(audit.year == "all") & (audit.family == "ridge") & (audit.combination == "basic_processed")].sort_values("horizon").mape_pct.tolist(), "different_pipeline_reference")
    for path in ("outputs_ml/manifest.json", "outputs_chronos/manifest.json", "outputs_feature_audit/manifest.json"):
        assert json.loads(Path(path).read_text(encoding="utf-8"))["train_sha256"] == plan["train_sha256"]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    lines = ["# Chronos 수급·시장 정보 재평가", "",
             "Chronos-Bolt에는 과거 외생변수를 직접 입력할 수 없어 이를 지원하는 Chronos-2를 사용했다.",
             "모델 변경과 입력 추가 효과를 섞지 않도록 같은 Chronos-2 가중치로 가격-only/수급/수급+시장 세 조건을 평가했다.", "",
             "MAPE(%), 동일 H1/H2/H3 83/82/81개 원점. 앞의 세 행만 모델·이력·추론 조건을 고정한 입력 추가 실험이다.", "",
             "| 모델·입력 | H1 | H2 | H3 |", "|---|---:|---:|---:|"]
    for r in comparison.itertuples():
        lines.append(f"| {r.model} | {r.H1:.4f} | {r.H2:.4f} | {r.H3:.4f} |")
    lines += ["", "## 같은 모델에서의 정보 추가 효과", "",
              "ΔMAPE는 추가한 조건−참조 조건으로 음수이면 개선이다. 연도별 승리 수는 유의성 검정이 아니다.", "",
              "| 추가 조건 / 참조 | 기간 | ΔMAPE(%p) | 개선 연도 |", "|---|---|---:|---:|"]
    for r in contrasts.itertuples():
        lines.append(f"| {r.candidate} / {r.reference} | H{r.horizon} | {r.delta_mape_pp:+.4f} | {r.years_better}/{r.n_years} |")
    lines += ["", "## 입력·장치·검증", "",
              "- 가격은 원점까지 최근 24개월. 수급은 생산 수준·계절 재고 Z·소비 Z, 시장은 BCOMNG 수익률이다.",
              "- 기존 데이터 및 변환 코드를 그대로 재사용했다. 추가 경제 데이터 다운로드·병합은 하지 않았다. 다운로드한 것은 모델 가중치다.",
              "- 생산·소비 2개월, 재고 1개월, 시장 0개월의 기존 가용성 가정을 유지했다. 추가 경제 반응 시차는 0개월로 고정했다.",
              "- 미래 외생변수는 전달하지 않았다. 과거 covariates만 제공하고 원점별 단독 추론 및 cross_learning=False로 시점 간 정보 공유를 막았다.",
              "- GPU float32, 고정 체크포인트 zero-shot, 중앙값(0.5 분위수), 추가 학습·최고 시드·분위수 탐색 없음.",
              "- 738개 예측의 중복·원점·실제값을 검사했다. 첫 원점 반복 추론과 입력 미래값 변조 테스트를 수행했다.", "",
              "## 해석 제한", "",
              "Bolt→Chronos-2 성능 차이는 모델 변경 효과도 포함하므로 외생변수 추가 효과라고 부르지 않는다.",
              "Ridge와는 같은 수급·시장 특징의 가용성을 맞췄지만 가격 이력 표현·학습 데이터·타깃 변환·튜닝 방식은 다르다. 모델 구조만의 완전 통제 비교가 아니다.",
              "Chronos-2의 공개 사전학습 데이터에 평가 시계열이 포함됐는지 미확인이다. 입력 누수 방지와 가중치의 사전 노출 가능성은 별개다.",
              "평가 구간을 반복 사용한 탐색 결과이며 미래 성능이나 최종 모델 우위를 보장하지 않는다. submission.csv는 유지한다.", "",
              f"고정 revision: `{plan['revision']}`", "",
              "[설계](../notebooks/CHRONOS_COVARIATES.md) · [실제 정렬 특징](aligned_features.csv) · [입력 검사](input_audit.csv) · [GPU 실행](runtime.csv) · [연도별 지표](metrics.csv)", ""]
    (OUTPUT / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
