"""Generate report tables directly from the ML experiment's saved metrics."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR",str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def markdown_table(frame):
    """Small Markdown table without an optional tabulate dependency."""
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"]*len(frame.columns)) + " |",
        *["| " + " | ".join(f"{v:.4f}" if isinstance(v,float) else str(v) for v in row) + " |"
          for row in frame.itertuples(index=False,name=None)],
    ])


def run(output_dir=Path("outputs_ml")):
    metrics=pd.read_csv(output_dir/"metrics.csv")
    all_metrics=metrics.loc[metrics.year=="all"]
    best=pd.read_csv(output_dir/"best_fixed_by_family.csv")
    best_table=best[["family","horizon","experiment","mape_pct","turning_f1_pct"]].sort_values(["family","horizon"])
    selected=all_metrics.loc[~all_metrics.experiment.str.contains("__")]
    pivot=selected.pivot(index="experiment",columns="horizon",values="mape_pct")
    effects=[]
    values=all_metrics.set_index(["horizon","experiment"])
    for family in ("ridge","elastic_net","lightgbm","catboost"):
        for h in (1,2,3):
            for base in "ABCD":
                parent=values.loc[(h,f"{family}__{base}"),"mape_pct"]
                child=values.loc[(h,f"{family}__{base}_market"),"mape_pct"]
                effects.append({"family":family,"horizon":h,"base":base,
                                "without_market_mape":parent,"with_market_mape":child,"delta_mape_pp":child-parent})
    pd.DataFrame(effects).to_csv(output_dir/"market_effects.csv",index=False)
    diagnostic=pd.read_csv(output_dir/"fit_diagnostics.csv")
    text=("# 규제 선형회귀·트리 부스팅 비교 결과\n\n"
          "Ridge·ElasticNet은 통계적 회귀이면서 지도학습 모델이다. LightGBM·CatBoost는 트리 기반 부스팅이다. "
          "통계와 ML을 배타적으로 분류하지 않으며, 전통 시계열·규제 선형회귀·트리 부스팅·딥러닝의 구조별 그룹으로 설명한다.\n\n"
          "4개 모델 × 8개 조합을 같은 원점에서 비교했다. H1/H2/H3 표본수는 83/82/81이다.\n"
          "후보당 고정 설정 하나와 시드 42를 사용했으며, 하이퍼파라미터 최적화나 시드 안정성 검증은 아니다.\n\n"
          "## 모델별 고정 조합 BEST (사후 선택)\n\n"+markdown_table(best_table)+"\n\n"
          "전체 평가 구간을 본 후의 순위이며 당시 선택할 수 있었던 성능과 다르다.\n\n"
          "## 과거 기반 선택 절차와 전통 시계열 기준선 MAPE\n\n"+markdown_table(pivot.reset_index())+"\n\n"
          "selected_<family>는 해당 모델의 8개 조합 중 과거 성적으로 선택한 결과다. "
          "selected_all은 32개 후보와 Naive 중 선택했다.\n"
          "stats_best_fixed는 사후 전통 시계열 BEST이며 stats_past_selected는 전통 시계열의 과거 기반 선택 절차다.\n\n"
          "![ML past-only selection MAPE](selection_mape.png)\n\n"
          "## 해석과 검증 범위\n\n"
          "- BCOMNG 추가 효과는 market_effects.csv에서 모델·horizon·조합별로 확인한다. 음수이면 MAPE 개선이다.\n"
          "- family별 BEST를 한 번 더 사후 선택하는 경우에도 독립 평가가 되지 않는다.\n"
          "- 전통 시계열 후보는 가격만 사용하고 규제 선형회귀·트리 부스팅은 수급·시장 정보도 사용하므로 입력 정보와 모델이 함께 달라진 결과다.\n"
          "- 미리 탐색한 데이터에서 조합을 정했으므로 완전히 미관측인 검증 구간이 아니다.\n"
          "- 경고 기록 수: "+str(len(diagnostic))+". 내용은 fit_diagnostics.csv에 저장했다. 실패한 원점을 제외하거나 Naive로 조용히 대체하지 않는다.\n"
          "- 보고된 반전 F1은 기존 정확 부호 기준이며 작은 변화에도 민감할 수 있다.\n"
          "- 제출 파일과 기존 최종 모델을 변경하지 않았다. DL 비교 및 모델별 설정 검증은 후속 단계다.\n\n"
          "재현: `uv run python -m src.ml_experiment`, `uv run python -m src.ml_report`.\n")
    (output_dir/"FINDINGS.md").write_text(text,encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    ax=pivot.plot.bar(figsize=(12,6),rot=30)
    ax.set(ylabel="MAPE (%)",xlabel="Selection policy / reference",title="Regularized linear / boosted tree selection vs traditional time-series references")
    ax.legend(title="Horizon")
    plt.tight_layout()
    plt.savefig(output_dir/"selection_mape.png",dpi=140)
    plt.close()
    print(pivot.round(4).to_string())


if __name__=="__main__":
    run()
