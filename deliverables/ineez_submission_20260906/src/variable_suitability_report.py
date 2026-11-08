"""Render the completed suitability audit without fitting or selecting models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.variable_suitability import OUT, GROUP_LABELS


def table(frame):
    def value(x):
        if isinstance(x, (float, np.floating)):
            return "—" if pd.isna(x) else f"{x:.3f}"
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join("---" for _ in frame.columns) + " |",
                      *["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def main():
    a = pd.read_csv(OUT/"variable_assessment.csv")
    m = pd.read_csv(OUT/"metrics.csv")
    c = pd.read_csv(OUT/"comparisons.csv")
    r = pd.read_csv(OUT/"redundancy.csv")
    p = pd.read_csv(OUT/"predictions.csv")
    plan = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    for path, digest in plan["hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert len(a) == 112 and a.variable.is_unique and a.metadata_matched.all()
    assert not p.duplicated(["family", "candidate", "horizon", "origin"]).any()
    assert set(a.loc[a.predictive_eligible, "variable_id"]) == set(x[7:] for x in c.candidate.unique() if x.startswith("single_"))
    for _, part in p.groupby(["family", "candidate", "horizon"]):
        h = int(part.horizon.iloc[0])
        assert len(part) == 84-h and set(part.origin) == set(range(120, 204-h))
    overall = m.loc[m.period == "all"]
    group = a.groupby("group").agg(variables=("variable", "size"), tested=("predictive_eligible", "sum"))
    group["held"] = group.variables-group.tested
    group.index = group.index.map(GROUP_LABELS)
    group.index.name = "변수군"
    group.columns = ["전체", "조건부 실험", "보류"]
    blocks = ["# 전체 입력 변수 사용 적절성 평가\n",
              "평가 대상: 제공 훈련자료 204개월의 입력 112개(날짜·정답 제외). 메타데이터 113행 중 정답 1행을 제외하고 전수 대응했다. Brent/WTI의 Daily 이름 차이는 명시적 별칭으로 처리했다. test 자료를 읽거나 submission.csv를 수정하지 않았다.\n",
              "## 1. 판정 원칙\n",
              "점수를 임의 가중합하지 않는다. **정의·가용성 게이트 → 품질·중복 → 공통 조건 예측 실험 → 안정성 → 조건부 판단** 순서다. 성능 개선이 공표 시점·지역 불명을 상쇄하지 않는다.\n",
              "초기 120개월 관측률 ≥80%, 서로 다른 관측 ≥2, 대상 지역 명시를 실험 진입 조건으로 고정했다. 이는 소표본용 실무 기준이지 검증된 최적 임계값이 아니다. 전체 204개월 품질 통계는 설명용이며 진입 조건에는 사용하지 않았다.\n",
              table(group.reset_index()), "",
              f"**{int(a.predictive_eligible.sum())}개 조건부 실험, {int((~a.predictive_eligible).sum())}개 보류.** 조건부는 완전 승인과 다르다. 모든 112개에 대해 제공 자료의 정확한 원천 시리즈 ID·과거 공표 시각·개정 빈티지의 일대일 검증은 미완료다. 공식 기관의 일반 설명을 확인해도 이 CSV의 출처를 확정할 수 없다.\n",
              "## 2. 정의·품질에서 확인된 중요 문제\n",
              "- PMI 27개 및 누적 YoY 4개: 설명서에 국가가 없어 보류했다. 중국 자료처럼 보인다는 추정으로 국가를 채우지 않았다.\n- EU 재고 2개: 초기 관측률은 각각 26.7%, 16.7%. 늦게 시작한 글로벌 대리지표를 충분한 미국 재고 이력처럼 취급하지 않는다.\n- 난방유: 전 기간 매년 4~9월 결측(50%). 비난방기 값을 겨울 관측으로 채우지 않는다.\n- 보스턴 최고·최저 월 기온: 미국 전체 평균 기온이 아니며, 두 월 극값의 평균을 실제 월평균 또는 HDD/CDD라고 부를 수 없다.\n- Electricity_Price_USA: cents/kWh는 가격 단위인데 한국어 설명은 판매량이다. 생산 변수로 오분류하지 않는다.\n- marketed production은 판매량/건성가스 생산량과 구분한다. 재고는 stock이고 생산·소비는 기간 물량이므로 원시값을 더해 가격 공식으로 삼지 않는다. 총수입과 캐나다 수입의 단순 합은 부분집합을 중복 계산한다.\n- BCOM 계열은 선물 기반 시장 신호 계열이다. 물리적 수급량이나 Henry Hub 현물 자체로 간주하지 않는다. 정확한 지수 변형은 추가 확인이 필요하다.\n- CLI는 정규화/진폭조정 지수지만 다른 모든 열까지 정규화된 것은 아니다. 개정된 과거값을 사용했는지는 시차 추가만으로 해결되지 않는다.\n",
              f"초기 120개월에서 |상관|≥0.95, 유효 쌍≥60인 관계는 수준 {int((r.kind=='level').sum())}쌍, 1차 차분 {int((r.kind=='first_difference').sum())}쌍이다. 원시 수준 상관만으로 제거하지 않았다. 아래 관계는 중복 경고이며 인과관계 증거가 아니다.\n",
              table(r.head(12)), "",
              "## 3. 실제 예측 평가 프로토콜\n",
              "Ridge(alpha=10)와 LightGBM(기존 120-tree 고정 설정)에 같은 입력을 적용했다. 이번 평가는 단순 모델로 변수 적절성을 선별하는 것이며 모든 모델의 최적 설정을 다시 찾는 실험은 아니다. 기존 4개 ML의 가공/시차/합성 실험은 아래 6절에서 별도로 연결한다.\n",
              "기준선은 현재 가격·가격 lag 1/2/3/6/12·월 sin/cos의 8개 특징이다. 각 입력은 수준+1차 차분 2개 특징으로 표현했다. (1) 한 변수씩 추가, (2) 변수군 추가, (3) 전체 적격 변수군에서 한 군 제거, (4) 추가한 변수군의 공표 가정을 한 달 더 지연했다. 전체 78개를 넣는 164차원 조합은 과다 입력의 스트레스 테스트이지 권장 최종 모델이 아니다.\n",
              "추정 공표 시차: 월별 수급·재고·거시 2개월, 보스턴 1개월, 시장·리그·금리 0개월. 보류 변수는 사용하지 않았다. 원시 결측은 최대 1개월 과거값 전방채움만 허용하고 잔여 결측 중앙값·결측 표시·스케일러는 각 학습 구간에서 적합했다. 가격 now는 과제의 예측 원점에 관측된 가격이라는 가정이다.\n",
              "월말 원점 2013-09부터 H1/H2/H3 각각 83/82/81회, 매번 학습 라벨 j+h≤t를 확인했다. MAPE를 주지표, MAE·방향 적중률·변곡점 F1을 보조 지표로 저장했다. 전반은 origin<161, 후반은 ≥161이다. 원점 기준 연도 8개 중 개선 연도 수도 확인했다(2013·2020은 부분 연도).\n",
              f"총 {len(plan['candidates'])}개 조합 ×2모델 ×3 horizon, **{len(plan['candidates'])*2*(83+82+81):,}회 적합**. CPU의 기존 고정 추정기를 유지하여 앞선 비교와 구현 조건을 맞췄으며 DL/GPU 재학습은 하지 않았다.\n",
              "ΔMAPE = 후보−비교군, 음수이면 개선(%p). 안정 개선 표시는 전체·전반·후반 모두 개선하면서 4개 이상 연도 개선인 경우다. 이 기준은 보고용 진단일 뿐 채택 규칙이 아니다. 6개월 moving-block bootstrap 1,000회 구간을 저장했으나 다중 비교 보정·독립 검증이 없으므로 통계적 확정으로 읽지 않는다.\n",
              "## 4. 변수군 결과\n"]
    for family in ("ridge", "lightgbm"):
        names = ["naive", "history", "processed_reference", "processed_storage_lag2", "all_groups"]
        names += ["add_"+g for g in GROUP_LABELS if "add_"+g in set(m.candidate)]
        pivot = overall.loc[(overall.family == family) & overall.candidate.isin(names)].pivot(index="candidate", columns="horizon", values="mape_pct").reindex(names)
        pivot.columns = [f"H{h} MAPE(%)" for h in pivot.columns]
        blocks += [f"### {family}\n", table(pivot.reset_index()), ""]
    groups = c.loc[c.candidate.str.startswith(("add_", "drop_", "delay_"))]
    groups.to_csv(OUT/"group_comparisons.csv", index=False)
    blocks += ["그룹별 추가·제거·추가 지연의 전후반·연도·구간 결과는 [group_comparisons.csv](group_comparisons.csv). `drop_`은 전체 78개 입력 조합 대비이며, `add_`/`delay_`는 가격 이력 대비이므로 비교군을 혼동하지 않는다.\n",
               "### 재고 공표 가정 1개월 → 2개월\n",
               table(c.loc[c.candidate == "processed_storage_lag2", ["family", "horizon", "delta_mape_pp", "early_delta_pp", "late_delta_pp", "years_better", "block6_low_pp", "block6_high_pp"]]), "",
               "이 비교는 기존 basic_processed 특징 12개에서 재고 시차만 변경한다. 시차는 MAPE가 낮은 쪽을 고르는 것이 아니라 실제 이용 가능 시점으로 정해야 한다. 기존 storage lag1은 월별 원천 확인 전까지 잠정 가정이다.\n",
               "## 5. 112개 전수 평가표\n",
               "아래에는 게이트·품질·예측 근거를 요약했다. [전체 평가 CSV](variable_assessment.csv)에는 원문 정의, 단위, 지역, 시차 가정, 첫/마지막 관측, 6개 모델×horizon 개선폭, 구간 정보 요약, 미확인 플래그 및 주의사항이 있다. 보류는 무용함이 아니고, 단독 추가가 실패해도 다른 변환/다른 조합의 가치를 부정하지 않는다.\n"]
    display = a[["variable", "group", "initial120_coverage_pct", "gate", "predictive_evidence", "decision"]].copy()
    display.group = display.group.map(GROUP_LABELS)
    blocks += [table(display), "", "## 6. 기존 가공·절제 실험과 함께 해석\n",
               "이번의 단독 수준+차분 추가와 기존 가공 조합의 조건부 제거는 다른 질문이다. 생산·재고·소비·BCOMNG를 유지한 채 다른 후보의 추가 가치를 검증하지 않고 단독 순위를 최종 변수 세트로 옮기면 안 된다.\n"]
    old = pd.read_csv("outputs_feature_audit/metrics.csv")
    manifest = json.loads(Path("outputs_feature_audit/manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_sha256"] == next(v for k, v in plan["hashes"].items() if k.endswith("data_train.csv"))
    old = old.loc[old.year.astype(str) == "all"]
    checks = []
    for name in ("drop_production", "drop_storage", "drop_consumption", "drop_market", "equal_index"):
        paired = old.loc[old.combination == name].merge(old.loc[old.combination == "current"], on=["family", "horizon"], suffixes=("", "_current"))
        if name == "equal_index":
            paired = old.loc[old.combination == name].merge(old.loc[old.combination == "individual_z"], on=["family", "horizon"], suffixes=("", "_current"))
        checks.append(dict(comparison=name, worse_of12=int((paired.mape_pct > paired.mape_pct_current).sum())))
    blocks += [table(pd.DataFrame(checks)), "",
               "기존 4개 모델×3 horizon 절제 결과: 재고·시장 정보의 조건부 기여가 비교적 일관되며 생산·소비는 모델 의존적이다. equal_index는 individual_z 대비다. `가격=생산+재고−수요`는 확정 가격 등식이 아니라 비교 후 기각/보완할 초기 합성 지수 가설로 남긴다.\n",
               "## 7. 종합 판단과 후속 결정\n",
               "**현재 핵심 수급·시장 가공 변수를 잘못 골랐다고 단정할 근거는 없다. 반면 전체 입력의 정의·가용성을 모두 확인했다고 쓰는 것도 부정확하다.**\n",
               "- 원시 변수 확대는 모델 의존적이다. Ridge의 가격 이력 H1/H2/H3 MAPE 7.882/11.036/14.634%가 전체 적격 변수 투입에서 10.017/26.632/52.519%로 악화했다. LightGBM은 7.678/11.151/15.763%에서 7.057/10.514/14.552%로 좋아졌지만 기존 가공 조합 7.043/9.808/13.668%보다 못했다. 고차원 조합·고정 규제·수준 표현에 대한 결과이지 모든 모델에서 그 변수들이 무용하다는 결론은 아니다.\n- 미국 수급·재고·리그군과 에너지 가격군은 두 모델의 세 horizon 모두에서 가격 이력 대비 개선했다. 그러나 수급군을 한 달 더 지연하면 LightGBM H1 개선은 유지되지 않았다. 공표 가정 민감도를 무시하면 안 된다.\n- 금융/거시/CLI를 통째로 추가하는 효과는 모델·horizon에 따라 달랐다. 특히 해외 CLI 대량 투입에서 Ridge가 불안정했다. 후속 실험은 전체 묶음 확대보다 경제적 관련성이 있는 소수 후보의 조건부 추가가 적절하다.\n- 재고 lag2의 Ridge H3 MAPE는 11.723%(lag1 11.905%), 전후반 모두 개선하고 8개 중 5개 연도 개선이었다. paired block6 구간은 −0.374~−0.038%p이나 다중 비교 보정 없는 탐색 구간이다. 공표 시점 확인 없이 더 좋은 점수를 이유로 lag를 선택하지 않는다.\n",
               f"단독 추가에서 {int((a.single_stable_of6 > 0).sum())}/78개가 6개 모델×horizon 중 **적어도 하나**의 안정 진단을 만족했다. 매우 약한 탐색 조건이며 이 수만큼 최종 채택하라는 의미가 아니다. 예컨대 영국 CLI 소비자물가는 6/6 안정 표시이나 bootstrap 구간 전체가 0 아래인 경우는 1/6뿐이고 지역 관련성·빈티지는 추가 검증이 필요하다. 미국 총수입·리그·전기가격·제조업 PPI·BCOMCL도 제한된 후속 후보이지 이미 확정된 추가 입력이 아니다.\n",
               "이 보고서는 112개를 모두 평가했지만 112개의 실시간 사용 가능성을 모두 입증한 것은 아니다. 78개는 가정하에 예측 실험했고 34개는 데이터/정의 게이트에서 보류했다. 반복 열람한 훈련 검증 구간의 탐색 결과이며 신규 최종 변수 세트·최종 모델을 확정하지 않는다.\n",
               "우선 원천 ID와 월 집계 방식을 확인하고 특히 재고 공표 시차 및 CLI 빈티지를 해결해야 한다. 이후 안정 후보를 현재 수급·시장 조합에 하나씩 추가/제거하는 제한된 사전 고정 실험과 과거 내부 선택으로 재검증한다. 중요도/SHAP는 이 단계를 대체하지 않으며 인과 증거도 아니다.\n",
               "## 8. 외부 정의 교차 확인 (2026-09-06)\n",
               "원천 데이터 추가 다운로드는 하지 않았다. 아래는 정의·위험 확인용 공식 자료이며 제공 CSV와의 완전한 시리즈 매칭을 주장하지 않는다.\n",
               "- [EIA Natural Gas Monthly](https://www.eia.gov/naturalgas/monthly/): 현재 2026년 6월 자료의 공표일은 8월 31일이며 월별 생산·소비·재고를 포함한다. 이 사례만으로 2003~2020의 전 기간 지연을 확정할 수는 없다.\n- [EIA underground storage](https://www.eia.gov/dnav/ng/ng_stor_type_s1_m.htm): base gas/working gas 구분이 있어 제공 열의 정확한 저장 개념 확인이 필요하다.\n- [OECD CLI](https://www.oecd.org/en/data/datasets/oecd-composite-leading-indicators-clis.html): 경기 전환 신호와 과거 발표 스냅샷/개정 자료를 제공한다. 단순 lag로 빈티지 문제까지 해결되지 않는다.\n- [NWS degree days](https://www.weather.gov/key/climate_heat_cool): 일별 평균기온과 65°F의 차이를 사용하는 정의다. 월 최고·최저 극값만으로 실제 월별 HDD/CDD를 복원할 수 없다.\n- [Bloomberg 공식 BCOM methodology](https://assets.bbhub.io/professional/sites/27/BCOM-Methodology-APR-2024_FINAL-1.pdf): 선물 계약·롤 구조를 다룬다. 제공 BCOMNG 열의 정확한 변형은 미확인이다.\n",
               "## 재현\n", "`python -m src.variable_suitability` (프로젝트 .venv, 최초 실행은 전체 적합). 동일 manifest일 때 완료 캐시를 재사용한다. `python -m src.variable_suitability_report`는 결과 무결성을 검사하고 이 문서만 다시 만든다. 실행 정보·해시는 manifest.json, 원점별 예측은 predictions.csv, 모든 지표는 metrics.csv, 경고는 fit_diagnostics.json에 저장했다.\n"]
    (OUT/"RESULTS.md").write_text("\n".join(blocks), encoding="utf-8")
    print(f"Verified {len(a)} inputs; {int(a.predictive_eligible.sum())} tested; {len(p):,} predictions incl. Naive", flush=True)


if __name__ == "__main__":
    main()
