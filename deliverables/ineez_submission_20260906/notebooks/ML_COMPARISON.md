# 규제 선형회귀·트리 부스팅과 변수 조합 비교

Ridge·ElasticNet은 통계적 회귀이면서 ML의 지도학습 알고리즘입니다. LightGBM·CatBoost는 트리 부스팅,
TFT·NBEATSx는 딥러닝 시계열 그룹으로 구분합니다. 통계와 ML을 배타적인 범주로 분류하지 않습니다.
`ml_experiment`, `outputs_ml` 등의 기존 코드·결과 경로는 두 지도학습 그룹을 포괄하는 이름으로 유지합니다.

```powershell
uv sync --extra dev
uv run python -m src.ml_experiment
```

전통 시계열 비교 결과 `outputs_statistics/`가 필요합니다. 없으면 먼저 `uv run python -m src.statistical_experiment`를 실행합니다.
입력 train 파일 해시와 통계 비교 원점·실제값이 일치하는지 확인하며 test와 submission은 변경하지 않습니다.

## 비교하는 질문

1. 생산 수준과 재고 계절편차의 효과가 Ridge에만 나타나는가?
2. BCOMNG 수익률 추가가 다른 모델에도 도움이 되는가?
3. 트리 모델을 추가하면 동일 특징 입력의 선형모델보다 개선되는가?

규제 선형회귀는 Ridge·ElasticNet, 트리 부스팅은 LightGBM·CatBoost이며 후보당 하나의 설정으로 제한했습니다.
Ridge alpha=10, ElasticNet alpha=0.01/l1_ratio=0.2, LightGBM은 기존 설정을 유지합니다.
CatBoost는 iterations=100, depth=3, learning_rate=0.04, l2_leaf_reg=5, RMSE loss입니다.
설정과 시드는 고정이며 최적화한 값이 아닙니다. 재현을 위한 실제 파라미터 전체는 manifest.json에 저장합니다.
설정 개수를 같게 둔다고 모델 간 계산 비용이나 최적화 정도까지 같아지는 것은 아닙니다.
LightGBM은 log-return의 absolute-error 목적, 다른 세 모델은 제곱오차 계열이므로 학습 손실까지 동일한 비교는 아닙니다.
선택 지표는 모두 가격 공간 MAPE입니다.

## 공통 입력

모든 모델에 같은 8개 조합을 제공합니다. 가격 이력과 소비 누적 Z는 고정합니다.

| 조합 | 생산 | 저장 |
|---|---|---|
| A | 누적 Z | 누적 Z |
| B | 수준 | 누적 Z |
| C | 누적 Z | 과거 동일월 계절 Z |
| D | 수준 | 과거 동일월 계절 Z |

각각 BCOMNG 1개월 수익률 추가 여부를 비교합니다. 원자료·변환·발표 지연은 기존 가설 코드를 재사용합니다.
모든 모델의 median imputer, missing indicator와 StandardScaler는 해당 원점의 학습 구간에서만 적합합니다.
H1/H2/H3 각각 log-return을 학습하며 label이 원점까지 알려진 행만 학습에 넣습니다.
CatBoost는 수치 특징만 사용하며 범주형 처리의 이점을 시험하는 실험이 아닙니다.

## 성능표의 세 가지 의미

- `family__combination`: 고정 모델·조합의 진단 결과. 전체 결과의 최저값은 사후 선택 성능입니다.
- `selected_ridge/elastic_net/lightgbm/catboost`: 해당 가족의 8개 조합 중 과거 성적으로 매월 선택한 절차입니다. Naive는 이 가족별 후보군에 넣지 않습니다.
- `selected_all`: 32개 ML 후보와 Naive 중 과거 성적으로 매월 선택한 절차입니다.

선택은 최근 최대 36개 내부 검증 원점의 MAPE로 합니다. 내부 원점 j의 정답은 j+h<=현재 원점인 경우만 사용합니다.
내부 예측 생성 시작은 2009-09, 외부 비교는 2013-09부터 H1 83개/H2 82개/H3 81개 원점입니다.
동점은 Naive 우선, 그다음 이름 순서입니다. 모델 실패를 조용히 제외하거나 다른 모델로 대체하지 않고 오류로 중단합니다.

전통 시계열 기준 `stats_best_fixed`는 이전 실험에서 사후 선택한 H1 Naive/H2·H3 SARIMA입니다.
`stats_past_selected`는 전통 시계열 후보를 과거 성적으로 고른 선택 절차입니다.
전통 시계열 입력은 가격만, 이번 규제 선형회귀·트리 부스팅은 수급·시장 정보를 포함하므로 비교는 정보 집합과 알고리즘이 함께 달라진 결과입니다.
가격-only ML 비교와 모델별 설정 최적화는 추가 실험 대상입니다.

이미 탐색한 자료에서 후보를 설계했으므로 과거 기반 선택을 적용해도 완전한 미관측 평가로 주장하지 않습니다.
가족별 과거 기반 선택 정책 중 사후 최고 정책을 고르는 것 역시 최종 정책의 독립 평가와 다릅니다.

## 결과 파일

- manifest.json / feature_definitions.csv: 후보·설정·계산식
- inner_prediction_cache.csv: 모든 내부 및 외부 원점의 후보 예측
- selection_scores.csv: 원점별 후보 점수와 선택 이유
- predictions.csv / metrics.csv: 전체·연도별 MAPE·MAE·방향·반전
- best_fixed_by_family.csv / best_fixed_ml.csv: 전체 구간 사후 고정 후보 순위
- comparisons.csv: 과거 기반 선택 결과의 Naive·통계 대비 MAPE 차이와 개선 연도 수
- fit_diagnostics.csv: 학습·예측 중 경고 기록

기존 전체 특징 E2/E3의 모델 비교와 이번 A~D 비교를 구분해야 합니다. 이번 결과만으로 자동 제출 모델을 바꾸지 않습니다.
