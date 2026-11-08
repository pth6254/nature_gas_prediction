# 변수 설계 편향·제거·시차·압축 실험 결과

4개 고정 ML 모델에 13개 공통 입력을 적용했다. 전체 수치는 회고적 탐색이며 최종 홀드아웃 성능이 아니다.
각 행의 H1/H2/H3는 MAPE(%)이며 낮을수록 좋다. 동일 원점 83/82/81개로 비교한다.

## 1. 입력 표현이 특정 모델에 유리한가?

2×2 비교: 가격 입력은 기본 시차/가공 특징, 외생 입력은 원시 수준/현재 가공 표현으로 나눈다.
current가 이전 D_market이다. 원시 수준도 발표 지연·학습 구간 내 결측 대체·표준화를 적용한다.

| 모델 | 입력 | H1 | H2 | H3 |
|---|---|---:|---:|---:|
| ridge | basic_raw | 7.2212 | 9.4620 | 12.0629 |
| ridge | basic_processed | 6.7521 | 9.2025 | 11.9051 |
| ridge | engineered_raw | 7.2212 | 9.6057 | 12.3897 |
| ridge | current | 6.3946 | 9.3131 | 12.0778 |
| elastic_net | basic_raw | 7.2684 | 9.5620 | 12.1424 |
| elastic_net | basic_processed | 6.7986 | 9.2496 | 11.9972 |
| elastic_net | engineered_raw | 7.2870 | 9.8461 | 12.7660 |
| elastic_net | current | 6.4102 | 9.4634 | 12.4343 |
| lightgbm | basic_raw | 7.7081 | 11.3074 | 16.0464 |
| lightgbm | basic_processed | 7.0427 | 9.8083 | 13.6676 |
| lightgbm | engineered_raw | 7.7894 | 11.7035 | 15.9414 |
| lightgbm | current | 7.1528 | 10.2973 | 14.4142 |
| catboost | basic_raw | 7.7698 | 10.7534 | 14.6970 |
| catboost | basic_processed | 7.1153 | 9.8552 | 13.2688 |
| catboost | engineered_raw | 7.8859 | 10.9818 | 15.0985 |
| catboost | current | 7.1171 | 10.3915 | 13.7271 |

현재 가공 조합은 기본 원시 조합 대비 10/12개 모델·기간에서 개선됐다.
가격 특징 수가 기본 8개/가공 18개로 다르므로 정보량과 표현이 함께 변한다. 이 결과만으로 선형 편향의 존재·부재를 확정하지 않는다.

## 2. 각 변수군의 조건부 예측 가치

현재 조합에서 해당 변수 하나를 제거한 뒤 매 원점에서 다시 학습했다. 값만 0으로 덮은 실험이 아니다.

| 모델 | 입력 | H1 | H2 | H3 |
|---|---|---:|---:|---:|
| ridge | current | 6.3946 | 9.3131 | 12.0778 |
| ridge | drop_production | 6.5099 | 10.2203 | 13.3351 |
| ridge | drop_storage | 6.5676 | 9.5013 | 12.1570 |
| ridge | drop_consumption | 6.4909 | 9.9036 | 12.5851 |
| ridge | drop_market | 7.3299 | 9.5726 | 12.3125 |
| elastic_net | current | 6.4102 | 9.4634 | 12.4343 |
| elastic_net | drop_production | 6.4042 | 10.3269 | 13.6023 |
| elastic_net | drop_storage | 6.5679 | 9.6044 | 12.3946 |
| elastic_net | drop_consumption | 6.4554 | 10.1177 | 13.0964 |
| elastic_net | drop_market | 7.4436 | 9.7148 | 12.7490 |
| lightgbm | current | 7.1528 | 10.2973 | 14.4142 |
| lightgbm | drop_production | 7.1332 | 10.2444 | 14.2466 |
| lightgbm | drop_storage | 7.2358 | 11.0258 | 15.8297 |
| lightgbm | drop_consumption | 7.0059 | 10.2507 | 14.0884 |
| lightgbm | drop_market | 7.6292 | 10.8544 | 14.4068 |
| catboost | current | 7.1171 | 10.3915 | 13.7271 |
| catboost | drop_production | 7.2337 | 10.5137 | 14.0776 |
| catboost | drop_storage | 7.2426 | 10.5876 | 14.6810 |
| catboost | drop_consumption | 7.1372 | 10.2957 | 13.6615 |
| catboost | drop_market | 7.8067 | 10.6886 | 13.8011 |

- production 제거 후 악화: 8/12; 개선: 4/12.
- storage 제거 후 악화: 11/12; 개선: 1/12.
- consumption 제거 후 악화: 7/12; 개선: 5/12.
- market 제거 후 악화: 11/12; 개선: 1/12.

제거 후 개선은 현재 다른 입력·설정에서의 중복/잡음 가능성이지 해당 경제 요소의 무관함이나 인과효과 부재를 뜻하지 않는다.

## 3. 발표 지연을 유지한 추가 반응 시차

생산·재고·소비를 함께 추가 0/1/3/6개월 이동했다. 시장·가격 이력은 유지했다. 변수별 서로 다른 시차의 전체 탐색은 아니다.
lag_selected는 원점마다 정답이 알려진 최근 36개 과거 예측으로 시차만 선택한 절차다. 같은 원점의 미래 정답은 선택에 쓰지 않았다.

| 모델 | 입력 | H1 | H2 | H3 |
|---|---|---:|---:|---:|
| ridge | current | 6.3946 | 9.3131 | 12.0778 |
| ridge | lag_1 | 6.2919 | 9.9401 | 12.4344 |
| ridge | lag_3 | 6.3584 | 9.2517 | 12.0893 |
| ridge | lag_6 | 6.3441 | 9.2586 | 12.4840 |
| ridge | lag_selected | 6.2846 | 9.4523 | 12.3154 |
| elastic_net | current | 6.4102 | 9.4634 | 12.4343 |
| elastic_net | lag_1 | 6.2469 | 10.1584 | 12.8328 |
| elastic_net | lag_3 | 6.4314 | 9.3004 | 12.3668 |
| elastic_net | lag_6 | 6.3837 | 9.4411 | 12.7486 |
| elastic_net | lag_selected | 6.2487 | 10.0176 | 12.7973 |
| lightgbm | current | 7.1528 | 10.2973 | 14.4142 |
| lightgbm | lag_1 | 7.1314 | 10.1511 | 14.0316 |
| lightgbm | lag_3 | 7.0332 | 9.7671 | 13.4574 |
| lightgbm | lag_6 | 6.9667 | 10.4420 | 14.0599 |
| lightgbm | lag_selected | 7.1352 | 10.3286 | 14.5352 |
| catboost | current | 7.1171 | 10.3915 | 13.7271 |
| catboost | lag_1 | 7.2185 | 10.2845 | 13.2260 |
| catboost | lag_3 | 7.0948 | 10.0694 | 12.9758 |
| catboost | lag_6 | 7.0226 | 10.3403 | 14.0392 |
| catboost | lag_selected | 7.1186 | 10.4838 | 13.7520 |

과거 기반 시차 선택이 추가 시차 0보다 개선된 비교는 3/12개다.
고정 시차의 전체 구간 최저값과 선택 절차의 성능은 구분해야 한다. Naive는 시차 선택 풀에 넣지 않았다.

## 4. 합성 지수의 정보 압축

동일한 production_z·storage_z·consumption_z 개별 입력과 그 합(+생산+재고−소비) 하나를 비교했다. 가격 이력·시장 입력은 동일하다.
현재 D의 생산 수준과 계절 재고를 그대로 더하지 않았다. 혼합 단위 문제를 피하기 위해 두 비교군 모두 같은 Z 표현으로 맞췄다.

| 모델 | 입력 | H1 | H2 | H3 |
|---|---|---:|---:|---:|
| ridge | individual_z | 6.6487 | 10.4939 | 14.0243 |
| ridge | equal_index | 6.9119 | 11.3358 | 15.6112 |
| elastic_net | individual_z | 6.6181 | 10.6539 | 13.9383 |
| elastic_net | equal_index | 6.8026 | 11.5056 | 16.0041 |
| lightgbm | individual_z | 7.1669 | 11.0647 | 14.9751 |
| lightgbm | equal_index | 7.2165 | 11.1214 | 15.8389 |
| catboost | individual_z | 7.2794 | 10.6396 | 14.2424 |
| catboost | equal_index | 7.2334 | 10.7825 | 14.7219 |

합성 지수로 압축했을 때 악화된 비교는 11/12개다.
압축은 세 변수의 독립적 효과와 상호작용을 제한한다. 성능 차이에는 차원 및 규제 변화도 포함돼 순수한 정보 손실만 측정한 것은 아니다.

## 검증·한계

- 현재 조합의 984개 예측은 기존 네 모델 D_market 예측과 허용오차 1e-8로 일치했다.
- 모든 후보가 같은 실제값·예측 원점으로 비교됐고 선택 점수의 마지막 타깃 시점이 원점을 넘지 않았다.
- 학습 경고 0건. 상세 경고와 연도별·원점별 개선은 CSV에 보존했다.
- 고정 모델 설정·CPU 조건을 유지했다. DL/Chronos 입력 비교나 모델별 하이퍼파라미터 최적화는 포함하지 않았다.
- 원점별 오차는 시간적으로 독립이 아니며 반복 탐색의 선택 편향이 있다. 승리 횟수는 유의성 검정이 아니다.
- 이 실험만으로 새 최종 후보를 자동 채택하지 않으며 submission.csv를 유지한다.

[실험 설계](../notebooks/FEATURE_AUDIT.md) · [입력 명세](manifest.json) · [쌍별 차이](contrasts.csv) · [시차 선택 기록](lag_selection_scores.csv) · [연도별 지표](metrics.csv)
