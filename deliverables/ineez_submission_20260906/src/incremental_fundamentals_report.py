"""Verify and report conditional imports/rigs experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import summarize_predictions
from src.incremental_fundamentals import OUT
from src.variable_suitability_report import table


def main():
    plan = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    for path, digest in plan["hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    cache = pd.read_csv(OUT/"inner_predictions.csv")
    scores = pd.read_csv(OUT/"selection_scores.csv")
    p = pd.read_csv(OUT/"predictions.csv", parse_dates=["date"])
    m = pd.read_csv(OUT/"metrics.csv")
    c = pd.read_csv(OUT/"comparisons.csv")
    assert len(cache) == 9396 and (scores.inner_last_target <= scores.outer_origin).all()
    assert not cache.duplicated(["family", "horizon", "combination", "origin"]).any()
    assert not p.duplicated(["family", "horizon", "policy", "origin"]).any()
    assert np.isfinite(p.predicted).all()
    for (_, h, _), sub in p.groupby(["family", "horizon", "policy"]):
        assert len(sub) == 84-h and set(sub.origin) == set(range(120, 204-h))
    assert p.selected_candidate.notna().all()
    baseline = []
    for (family, h), sub in p.loc[p.policy == "base"].groupby(["family", "horizon"]):
        sub = sub.copy()
        sub["predicted"] = sub.current_price
        baseline.append(dict(family=family, horizon=h, policy="naive", period="all", **summarize_predictions(sub)))
    naive = pd.DataFrame(baseline)
    naive.to_csv(OUT/"naive_metrics.csv", index=False)
    overall = pd.concat([m.loc[m.period == "all"], naive], ignore_index=True)
    pivot = overall.pivot(index=["family", "policy"], columns="horizon", values="mape_pct")
    pivot.columns = [f"H{h} MAPE(%)" for h in pivot.columns]
    key = c.loc[((c.family == "elastic_net") & (c.horizon == 1)) | ((c.family == "ridge") & c.horizon.isin([2, 3]))]
    text = ["# 현재 수급·시장 조합에 대한 총수입·리그의 추가 기여\n",
            "## 결론\n",
            "**이번 범위에서는 총수입·리그를 일괄 추가하지 않는 쪽을 권한다.** 단독 가격 이력 기준의 개선이 기존 가공 수급·시장 조합 위의 안정적 개선으로 이어지지는 않았다. 과거 성적으로 추가 여부까지 선택한 절차도 H1 ElasticNet, H2/H3 Ridge에서 모두 악화했다.\n",
            "- H1 ElasticNet: 기본6.314%, 리그6.313%의 차이는 약−0.0006%p이며 전후반 부호가 다르다. 총수입은6.325%로 악화했다.\n- H2 Ridge: 총수입 고정 추가는9.212→9.071%지만 전반−0.375%p, 후반+0.092%p다. 5/8개 연도 개선, block6 구간−0.802~+0.508%p로 안정적 우월성을 확인하지 못했다.\n- H3 Ridge: 총수입은11.723→11.721%의 미세한 차이다. 리그/둘다 추가는 악화했다.\n- 추가 여부까지 과거 선택한 all 정책: H1 ElasticNet6.422%, H2 Ridge9.281%, H3 Ridge11.795%로 모두 기본 정책보다 나쁘다.\n",
            "## 원천과 고정 실험 조건\n",
            "[재고 원천 확인](../outputs_storage_source/FINDINGS.md): 204개월 모두 EIA 월별 총저장량을10 MMcf 단위 반올림한 값과 일치했다. working gas와 구분하고2013년 알래스카 포함 범위 변경을 기록했다. 새 실험의 재고는 **lag2 고정**으로 성능을 보고 공표 지연을 고르지 않는다. 원 제공자의 수집 경로·과거 빈티지·전 기간 최초 공표 달력은 여전히 미확인이다.\n",
            "조합은 base/imports/rigs/both 4개다. 총수입·리그 각각 수준+1차 차분2특징, 전기가격·PPI·CLI로 탐색 확대하지 않았다. Ridge alpha10, ElasticNet alpha0.01/l1_ratio0.2, LightGBM 기존 설정을 유지했다. H1은 가격 이력18+핵심4, H2/H3는 가격 이력8+핵심4다. 생산/소비/재고/수입 공표lag2, 리그/시장lag0은 실험 가정이다. 학습 구간에서만 중앙값·결측 표시·스케일러를 적합한다. 추가 변수의 전방채움은 최대1개월이다.\n",
            "H1은 각 추가 조합별 동일하게 핵심 생산·재고·소비 반응 시차0/1/3/6을 최근36개 알려진 과거 정답(최소24)으로 선택한다. 추가 수입/리그 자체의 반응 시차는 따로 탐색하지 않았다. all은 조합도 함께 선택한다. H2/H3는4개 고정 조합 및 과거 조합 선택 정책을 비교한다.\n",
            "내부 원점72, 외부120부터, 모든 학습·선택 라벨j+h≤원점. H1/H2/H3 각각83/82/81개 동일 원점. 총9,396회 적합, 경고0건. 기존 CPU 추정기를 유지하고 DL/GPU 재학습은 하지 않았다. 테스트 자료는 읽지 않았으며 submission.csv를 보존했다.\n",
            "## 동일 원점 MAPE\n", table(pivot.reset_index()), "",
            "H1의 base/imports/rigs/both는 각각 동일 예산의 과거 반응 시차 선택 정책이다. H2/H3 네 조합은 고정 후보이며 all만 추가 여부의 과거 선택까지 평가한다. Naive는 선택 풀에 없는 동일 원점 참고 기준선이다.\n",
            "## 우선 후보의 안정성\n", table(key), "",
            "ΔMAPE=후보−동일 모델 기본 조합(%p), 음수 개선. 전반origin<161, 후반≥161. paired moving-block6 bootstrap1,000회 seed42. 원점 기준8개 연도이며 처음/마지막은 부분 연도다. 다중 비교 보정 없는 탐색 구간이다.\n",
            "## 후속 결정과 한계\n",
            "새 변수 추가 없이 H1 ElasticNet+과거 반응 시차 선택, H2/H3 Ridge basic_processed, 재고 공표lag2를 우선 최종화 후보로 제안한다. 실제 최종 파이프라인·제출은 아직 고정/생성하지 않았다. 반복 열람한 검증자료에서 발상한 후보이므로 과거 내부 선택이 안전해도 독립 검증을 통과한 최적 모델이라는 뜻은 아니다.\n",
            "## 재현\n",
            "`python -m src.storage_source_check` → `python -m src.incremental_fundamentals` → `python -m src.incremental_fundamentals_report`.\n",
            "[실험 선언·입력 해시](manifest.json), [과거 선택 점수](selection_scores.csv), [원점별 예측](predictions.csv), [전체/전후반/연도별 지표](metrics.csv), [기본 대비 개선폭](comparisons.csv), [Naive 지표](naive_metrics.csv). MAPE 외 MAE·방향·변곡점 지표도 저장했다.\n"]
    (OUT/"FINDINGS.md").write_text("\n".join(text), encoding="utf-8")
    assert json.loads((OUT/"fit_diagnostics.json").read_text(encoding="utf-8")) == []
    print("Verified 9396 fits, past-only selection, paired origins and unchanged submission")


if __name__ == "__main__":
    main()
