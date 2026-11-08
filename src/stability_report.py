"""Report primary-window policies separately from sensitivity diagnostics."""

import json

import numpy as np
import pandas as pd

from src.candidate_stability import OUTPUT, CENTERS


def main():
    m = pd.read_csv(OUTPUT / "metrics.csv", dtype={"period": str})
    indexed = m.set_index(["experiment", "horizon", "period"])
    policies = m.loc[(m.role == "past_only_policy") & (m.period == "all")]
    comparisons = []
    for r in policies.itertuples():
        references = ["naive", f"{r.family}__current__a{CENTERS[r.family]:g}"]
        if r.horizon != 1:
            references.append(f"{r.family}__basic_processed__a{CENTERS[r.family]:g}")
        for ref in references:
            for period in m.period.unique():
                if (r.experiment, r.horizon, period) not in indexed.index:
                    continue
                candidate = indexed.loc[(r.experiment, r.horizon, period)]
                baseline = indexed.loc[(ref, r.horizon, period)]
                assert candidate.n_origins == baseline.n_origins
                comparisons.append(dict(experiment=r.experiment, family=r.family, policy=r.policy,
                                        window=r.window, horizon=r.horizon, reference=ref, period=period,
                                        n_origins=candidate.n_origins,
                                        delta_mape_pp=candidate.mape_pct-baseline.mape_pct,
                                        delta_mae=candidate.mae-baseline.mae,
                                        delta_p90_ape_pp=candidate.p90_ape_pct-baseline.p90_ape_pct,
                                        delta_max_ape_pp=candidate.max_ape_pct-baseline.max_ape_pct,
                                        delta_turning_f1_pp=candidate.turning_f1_pct-baseline.turning_f1_pct))
    contrasts = pd.DataFrame(comparisons)
    contrasts.to_csv(OUTPUT / "comparisons.csv", index=False)
    lines = ["# 최종 후보의 규제 강도·선택 기간 안정성 검증", "",
             "주 분석은 과거 선택 기간 36개월이다. 24/48개월은 민감도 진단이며 최저값을 골라 주 성능으로 대체하지 않는다.",
             "추가 시차·규제 선택은 매 원점 이전에 정답이 알려진 예측만 사용했다. 그러나 후보 자체가 반복 탐색에서 나왔으므로 미사용 최종 홀드아웃은 아니다.", "",
             "## 1. 주 분석: 고정 기준과 과거 기반 선택", "",
             "MAPE(%), 낮을수록 좋다. H1/H2/H3 원점은 83/82/81개다.", "",
             "| 기간 | 모델 | 절차 | 전체 | 전반 2013–16 | 후반 2017–20 | 급변 | 급변 n | P90 APE |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    primary = []
    for h in (1, 2, 3):
        names = ["naive"]
        for family in CENTERS:
            names.append(f"{family}__current__a{CENTERS[family]:g}")
            if h != 1:
                names.append(f"{family}__basic_processed__a{CENTERS[family]:g}")
            names.extend(policies.loc[(policies.family == family) & (policies.horizon == h) & (policies.window == 36), "experiment"].tolist())
        for name in names:
            all_row = indexed.loc[(name, h, "all")]
            early = indexed.loc[(name, h, "early_2013_2016")]
            late = indexed.loc[(name, h, "late_2017_2020")]
            shock = indexed.loc[(name, h, "shock_abs_return_ge15pct")]
            label = name.split("__", 1)[-1]
            lines.append(f"| H{h} | {all_row.family} | {label} | {all_row.mape_pct:.4f} | {early.mape_pct:.4f} | {late.mape_pct:.4f} | {shock.mape_pct:.4f} | {shock.n_origins} | {all_row.p90_ape_pct:.4f} |")
            primary.append(dict(horizon=h, experiment=name, family=all_row.family,
                                mape=all_row.mape_pct, early_mape=early.mape_pct, late_mape=late.mape_pct,
                                shock_mape=shock.mape_pct, shock_n=shock.n_origins, p90_ape=all_row.p90_ape_pct))
    pd.DataFrame(primary).to_csv(OUTPUT / "primary_summary.csv", index=False)
    lines.extend(["", "current_alpha는 현재 조합의 규제만 선택, basic_alpha는 간소화 조합의 규제만 선택한다.",
                  "H1 lag_fixed_alpha는 기존 규제 강도를 유지하고 시차만 선택, lag_alpha는 시차와 규제 강도를 함께 선택한다.",
                  "급변은 |미래 실제 가격/현재 가격−1|≥15%인 사례다. 미래 실제값으로 나눈 사후 오차 진단이며 학습·선택 조건이 아니다.", "",
                  "## 2. 선택 기간 민감도", "", "| 기간 | 모델 | 절차 | 과거 24개월 | 과거 36개월(주) | 과거 48개월 |", "|---|---|---|---:|---:|---:|"])
    sensitivity = []
    for (h, family, policy), part in policies.groupby(["horizon", "family", "policy"]):
        part = part.set_index("window")
        vals = [part.loc[w, "mape_pct"] for w in (24, 36, 48)]
        lines.append(f"| H{h} | {family} | {policy} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        sensitivity.append(dict(horizon=h, family=family, policy=policy, mape24=vals[0], mape36=vals[1], mape48=vals[2], range_pp=max(vals)-min(vals)))
    pd.DataFrame(sensitivity).to_csv(OUTPUT / "window_sensitivity.csv", index=False)
    lines.extend(["", "48개월 창은 이용 가능한 과거 사례가 부족하면 최소 24개 이상인 가용 사례만 사용한다. 실제 선택 관측 수는 selection_scores.csv에 기록한다.", "",
                  "## 3. 고정 규제 강도 민감도", "", "전체 외부 결과를 본 고정값 비교는 진단이다. 이 표의 최저 alpha를 사전 선택 성능으로 주장하지 않는다.", "",
                  "| 기간 | 모델 | 입력 | 약한 규제 | 기존 규제 | 강한 규제 |", "|---|---|---|---:|---:|---:|"])
    fixed = m.loc[(m.role == "fixed_sensitivity_diagnostic") & (m.period == "all")].copy()
    fixed["combination"] = fixed.experiment.str.split("__").str[1]
    fixed["alpha"] = fixed.experiment.str.split("__a").str[-1].astype(float)
    for (h, family, combo), part in fixed.groupby(["horizon", "family", "combination"]):
        vals = part.sort_values("alpha").mape_pct
        lines.append(f"| H{h} | {family} | {combo} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines.extend(["", "Ridge alpha=1/10/100, ElasticNet alpha=0.001/0.01/0.1. ElasticNet l1_ratio=0.2는 고정했다.", "",
                  "## 4. 기간별 개선의 지속성", "",
                  "아래 비교의 참조는 같은 계열 기존 current 고정 규제다. 연도별 승리 횟수는 유의성 검정이 아니다.", "",
                  "| 기간 | 후보(36개월) | 전체 ΔMAPE | 전반 Δ | 후반 Δ | 급변 Δ | 개선 연도/전체 연도 |", "|---|---|---:|---:|---:|---:|---:|"])
    persistence = []
    for r in policies.loc[policies.window == 36].itertuples():
        ref = f"{r.family}__current__a{CENTERS[r.family]:g}"
        sub = contrasts.loc[(contrasts.experiment == r.experiment) & (contrasts.horizon == r.horizon) & (contrasts.reference == ref)].set_index("period")
        annual = sub.loc[sub.index.str.fullmatch(r"\d{4}")]
        vals = [sub.loc[p, "delta_mape_pp"] for p in ("all", "early_2013_2016", "late_2017_2020", "shock_abs_return_ge15pct")]
        wins = int((annual.delta_mape_pp < 0).sum())
        lines.append(f"| H{r.horizon} | {r.experiment} | " + " | ".join(f"{v:+.4f}" for v in vals) + f" | {wins}/{len(annual)} |")
        persistence.append(dict(horizon=r.horizon, experiment=r.experiment, delta_all=vals[0], delta_early=vals[1],
                                delta_late=vals[2], delta_shock=vals[3], years_better=wins, n_years=len(annual)))
    pd.DataFrame(persistence).to_csv(OUTPUT / "period_stability.csv", index=False)
    diag = pd.read_csv(OUTPUT / "fit_diagnostics.csv")
    assert (diag.last_train_target <= diag.origin).all()
    warning_fits = int(diag.warnings.ne("[]").sum())
    lines.extend(["", "## 검증 및 해석의 한계", "",
                  f"- {len(diag):,}회 고정 후보 학습. 경고가 발생한 학습 {warning_fits}회. 학습 시간 합계 {diag.fit_seconds.sum():.1f}초(선택·파일 저장 제외).",
                  "- 기존 규제 강도에서의 현재/간소화/시차 입력 예측은 앞선 변수 통제 실험과 허용오차 내에서 일치했다.",
                  "- 동일 원점·실제값·과거 학습 타깃 및 내부 선택 타깃 시점을 검사했다.",
                  "- 같은 평가 구간을 반복 사용했으므로 새 후보의 사후 선택 편향과 시계열 오차 의존성이 남는다.",
                  "- 선택 절차가 고정 후보보다 나쁘면 튜닝을 무조건 추가하지 않는다. 평균 개선만으로 급변 위험이 줄었다고 해석하지 않는다.",
                  "- 이 실행은 최종 모델 확정이나 submission.csv 변경을 수행하지 않는다.", "",
                  "[실험 설계](../notebooks/STABILITY.md) · [실행 설정](manifest.json) · [연도/급변/오차 꼬리](metrics.csv) · [비교](comparisons.csv) · [선택 기록](selection_scores.csv)", ""])
    (OUTPUT / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(primary).to_string(index=False))
    print(pd.DataFrame(persistence).to_string(index=False))


if __name__ == "__main__":
    main()
