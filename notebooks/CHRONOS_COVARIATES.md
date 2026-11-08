# Chronos 수급·시장 입력의 공정한 재평가

## 모델과 데이터 결정

기존 Chronos-Bolt Small의 직접 추론은 과거 외생변수를 지원하지 않는다. 이를 지원하는 **Chronos-2**로 가격-only / 수급 추가 / 수급+시장 추가를 모두 다시 평가했다. 모델을 바꾸고 외생변수를 넣은 결과만 이전 Bolt와 비교하면 두 효과가 섞이므로 같은 Chronos-2 체크포인트 안에서 차이를 측정한다.

추가 경제 데이터는 필요하지 않았다. 기존 train에 생산·재고·소비·BCOMNG가 모두 있고, 이전 ML/DL에서 사용한 변환과 날짜 정렬을 그대로 재사용했다. 외부 경제 데이터 다운로드·조인·보간은 하지 않았다. 웹에서 받은 것은 공개 모델 가중치이며 데이터는 로컬 GPU에서 처리한다.

## 세 입력 조건

| 조건 | 가격 | 추가 과거 입력 |
|---|---|---|
| price | 최근 24개월 원시 가격 | 없음 |
| fundamentals | 동일 | 생산 수준·계절 재고 Z·소비 Z |
| market | 동일 | 위 수급 3개 + BCOMNG 수익률 |

조건별로 모델·revision·24개월 길이·H1/H2/H3 출력·GPU float32·0.5 분위수(중앙값)를 고정한다. 추가 학습·규제 선택·입력 길이 탐색·최고 시드 선택은 하지 않는다. 중앙값은 혼합 가중치가 아니며 MAPE 최적값이라는 뜻도 아니다.

## 발표 시점과 미래 정보 차단

기존 `build_hypotheses(..., conservative)`로 계산한다. 생산·소비는 2개월, 재고 1개월의 가용성 지연을 유지하고 BCOMNG는 당월 관측 가정을 유지한다. 추가 경제 반응 시차는 0개월이다. 실제 공표 빈티지·개정 시점까지 완전히 복원했다는 의미는 아니다.

가공 특징은 시간순으로 계산된 값이며 모델의 `past_covariates`에만 전달한다. `future_covariates`는 전달하지 않는다. 미래 실제 생산·재고·소비·시장값을 예측 원점에서 이미 안다고 가정하지 않는다.

Chronos-2는 관련 시계열 사이에 정보를 공유할 수 있으므로 **한 원점씩 단독 추론**하고 `cross_learning=False`를 명시한다. 서로 다른 원점들을 묶어 미래 시점의 정보가 과거 원점에 전달될 가능성을 차단한다.

기존 인과적 전방 채움 후 남는 결측은 NaN으로 유지하며 모델의 결측 마스킹에 맡긴다. 미래값 보간이나 전체 구간 평균 대체는 하지 않는다. `input_audit.csv`에 실제 입력 창과 결측 수, `aligned_features.csv`에 정렬된 값, `feature_definitions.csv`에 원시 열·공식·발표 지연을 보존한다.

## 비교 범위와 한계

- Chronos-2 내부 세 조건: 동일 체크포인트·입력 길이·예측 원점의 입력 추가 통제 실험이다.
- Chronos-Bolt와의 비교: 모델 변경도 포함한 참고이며 외생변수 추가의 독립 효과가 아니다.
- 기존 Ridge와의 비교: 수급·시장 변수와 가용성은 맞췄지만, 가격 이력 표현·학습 데이터·타깃 변환·최적화 방식은 다르다. 완전히 동일한 파이프라인은 아니다.
- 사전학습에 공개 Henry Hub 평가 시계열이 포함됐는지 확인하지 못했다. 모델 공개 시점도 평가 구간 이후다. 입력 누수 차단이 가중치의 데이터 노출 가능성까지 제거하지 않는다.
- 평가 구간을 반복 탐색했으므로 독립 최종 홀드아웃 성능이나 미래 우위를 주장하지 않는다.

H1/H2/H3 83/82/81개 원점에서 MAPE·MAE·방향·반전 F1·연도별 성능을 기록한다. 제출 모델 변경은 이 실험의 범위가 아니다.

## 재현

```powershell
uv sync --extra dev --extra dl --extra chronos
uv run --extra dl --extra chronos python -m src.chronos_covariates
uv run --extra dl --extra chronos python -m src.chronos_covariates_report
uv run --extra dev --extra dl --extra chronos python -m pytest -q -p no:cacheprovider
```

GPU 사용을 검사하며 CPU로 조용히 전환하지 않는다. 최초 실행에서 모델 commit과 파일 SHA256을 고정하고 이후 재사용한다. 결과는 `outputs_chronos_covariates/`에 분리한다. 기존 Bolt 결과와 `submission.csv`는 보존한다.

근거: [공식 모델 카드와 기능 비교](https://huggingface.co/amazon/chronos-2), [공식 과거/미래 외생변수 API](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py).
