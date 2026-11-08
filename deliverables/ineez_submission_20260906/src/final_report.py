"""Verify published final-v1 artifacts and generate a matching concise report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import summarize_predictions
from src.final_pipeline import OUT, digest, validate_submission
from src.variable_suitability_report import table


def main():
    manifest = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert digest("submission.csv") == digest(OUT/"submission.csv") == manifest["submission_sha256"]
    assert digest(manifest["backup_path"]) == manifest["backup_sha256"]
    for path, expected in manifest["code_hashes"].items():
        assert digest(path) == expected
    template = pd.read_csv(next(Path('.').glob('*submission_example.csv')))
    submission = pd.read_csv("submission.csv")
    validate_submission(submission, template)
    v = pd.read_csv(OUT/"validation_predictions.csv")
    audit = pd.read_csv(OUT/"forecast_audit.csv")
    contributions = pd.read_csv(OUT/"linear_contributions.csv")
    scores = pd.read_csv(OUT/"h1_selection_scores.csv")
    assert (scores.inner_last_target <= scores.outer_origin).all()
    assert len(audit) == 144 and not audit.duplicated(["date", "horizon"]).any()
    published = audit.pivot(index="date", columns="horizon", values="predicted").reindex(submission.date)
    np.testing.assert_allclose(published.to_numpy(), submission.iloc[:, 1:].to_numpy(), atol=5.01e-7, rtol=0)
    all_predictions = pd.concat([v, audit], ignore_index=True)
    for row in all_predictions.itertuples():
        part = contributions.loc[(contributions.origin == row.origin) & (contributions.horizon == row.horizon)]
        np.testing.assert_allclose(part.base_log_return.iloc[0]+part.centered_log_return_contribution.sum(), row.predicted_log_return, atol=1e-10, rtol=0)
    records = []
    for h, sub in v.groupby("horizon"):
        naive = sub.copy()
        naive["predicted"] = naive.current_price
        records.append(dict(horizon=h, model=sub.family.iloc[0], MAPE=summarize_predictions(sub)["mape_pct"],
                            Naive_MAPE=summarize_predictions(naive)["mape_pct"],
                            MAE=summarize_predictions(sub)["mae"], direction_pct=summarize_predictions(sub)["direction_accuracy_pct"]))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/"final_vs_naive.csv", index=False)
    case_keys = [("2019-09-30", 1), ("2019-09-30", 3), ("2018-11-30", 1), ("2013-11-30", 3)]
    cases, details = [], []
    for date, h in case_keys:
        row = v.loc[(v.date == date) & (v.horizon == h)].iloc[0]
        actual_turn = (row.current_price-row.previous_price)*(row.actual-row.current_price) < 0
        pred_turn = (row.current_price-row.previous_price)*(row.predicted-row.current_price) < 0
        cases.append(dict(date=date, horizon=h, current_price=row.current_price, actual=row.actual,
                          predicted=row.predicted, actual_turn=actual_turn, predicted_turn=pred_turn))
        part = contributions.loc[(contributions.origin == row.origin) & (contributions.horizon == h)].copy()
        details.append(part.reindex(part.centered_log_return_contribution.abs().sort_values(ascending=False).index).head(5))
    cases = pd.DataFrame(cases)
    cases.to_csv(OUT/"report_cases.csv", index=False)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(OUT/"case_top_contributions.csv", index=False)
    counts = audit.loc[audit.horizon == 1].reaction_lag.value_counts().sort_index().to_dict()
    sections = ["# 최종 모델 및 제출 결과: final_v1\n",
                "2026-09-06 사용자 승인 후 고정한 최종 설정이다. 모델 종류·특징·규제·선택 창을 테스트 성적으로 재선택하지 않았다.\n",
                "## 고정 모델\n",
                "- H1: ElasticNet(alpha=0.01, l1_ratio=0.2), 가격 이력18+가공 수급·시장4. 생산·재고·소비의 추가 반응 시차0/1/3/6만 매월 알려진 과거36개 예측 성적(최소24)으로 선택한다.\n- H2/H3: Ridge(alpha=10), 기본 가격 이력8+생산 수준·재고 계절Z·소비 누적Z·BCOMNG 수익률의12특징. 반응 시차 추가 없음.\n- 생산·재고·소비 공표 지연2개월, BCOMNG0개월. 총수입·리그·기타 후보는 최종 입력에 포함하지 않는다.\n- 목표는 log(P[t+h]/P[t]), 가격 복원은 max(0.01,P[t]×exp(예측값)). 현재 가격과50:50 평균하거나 고정 수급 합성식을 쓰지 않는다.\n",
                "## 검증 결과\n", table(metrics), "",
                "최종 코드의 검증 예측은 기존 추가 기여 실험의 기본 조합과1e-9 이내로 일치한다. H1/H2/H3 각각83/82/81개 동일 원점이다. 반복 열람한 검증 구간이므로 새 독립 검증이나 대회 테스트 점수가 아니다.\n",
                "## 제출\n",
                "submission.csv는2020-09-30~2024-08-31의48개 예측 원점, date/pred_1/pred_2/pred_3 순서다. 날짜는 목표월이 아니라 예측 수행 월이다. 마지막2024-08-31 행은9/10/11월 가격을 예측한다. 수치는 소수6자리로 저장했다.\n",
                "각 월말까지 관측된 테스트 가격을 이용해 expanding-window 재학습한다. 학습 행j의 정답은j+h≤t인 경우만 포함하고 H1 선택도 같은 제약을 적용한다. 미래 테스트 행·아직 알려지지 않은 정답은 선택에 사용하지 않는다. 이는 테스트 가격 전체를 처음부터 쓰거나 테스트 점수로 최종 모델을 고르는 것과 다르다.\n",
                f"제출 구간 H1 반응 시차 선택 횟수: {counts}. 이것은 사후 최적 시차를 고른 분포가 아니라 사전에 고정한 과거 선택 절차의 실행 기록이다.\n",
                "원시 생산·재고·소비는 마지막2개월이 결측이다. 공표lag2 이후 실제 제출 원점에 필요한 값은 존재하며 미래 값으로 보간하지 않는다. 실제 최초 발표·개정 빈티지의 전 기간 검증은 미완료라는 한계는 유지한다.\n",
                f"제출 SHA256: `{manifest['submission_sha256']}`. 이전 파일은 `{manifest['backup_path']}`에 원본 바이트 그대로 보존했다.\n",
                "## 동일 사례의 최종 모델 재계산\n", table(cases), "",
                "기존 보고서의 네 원점 자체를 유지하고 최종 예측으로 교체했다. 반전은 sign(P[t]−P[t−1])×sign(P[t+h]−P[t])<0으로 정의한다. 2013-11의 직전 변동은 작아 반전 정의에 민감하다.\n",
                table(detail[["date", "horizon", "feature", "centered_log_return_contribution"]]), "",
                "기여도는 각 학습 구간의 변환된 특징 평균을 기준으로 한 선형 log-return 기여다. 기준값+기여도 합이 모델의 log-return 출력과 일치하는지 검증했다. 가격 달러 기여·실제 인과효과·상관특징을 조건부 처리한 SHAP라고 해석하지 않는다. 개별 특징의 상관성이 있으므로 순위만으로 변수 중요성을 확정하지 않는다.\n",
                "## 재현·감사 파일\n",
                "프로젝트 루트에서 `python -m src.final_pipeline`, 이어 `python -m src.final_report`. 예측 재실행은 같은 결과를 생성하며 다른 사용자의 제출 수정이 발견되면 덮어쓰기를 거부한다.\n",
                "[고정 설정](model_config.json), [모델별 특징](feature_sets.json), [원점·학습 종료·목표월](forecast_audit.csv), [H1 과거 선택 점수](h1_selection_scores.csv), [검증 지표](validation_metrics.csv), [선형 기여도](linear_contributions.csv), [실행 해시·환경](manifest.json).\n",
                "과거 실험 manifest의 submission 해시는 이전 파일을 가리키는 역사 기록이다. 일부 과거 실험/보고서 명령은 이 해시 보호 때문에 새 최종 제출 상태에서 재실행을 거부할 수 있다. 기존 결과를 덮어쓰지 말고 최종 산출물 재현에는 위 final 명령을 사용한다. 이전 src.train은 최종 설정이 아닌 초기 비교 파이프라인이다.\n"]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    print("Verified submission, temporal selection, centered explanations and final report")


if __name__ == "__main__":
    main()
