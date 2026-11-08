# Chronos-Bolt Small 비교 결과

MAPE(%), 낮을수록 좋다. H1/H2/H3는 동일한 83/82/81개 원점이다.

| 모델 | H1 | H2 | H3 |
|---|---:|---:|---:|
| Chronos-Bolt Small price24 | 7.7420 | 10.7297 | 13.9191 |
| naive | 7.6663 | 10.4804 | 13.5698 |
| ridge__D_market | 6.3946 | 9.3131 | 12.0778 |
| NBEATSx price (3-seed mean) | 8.1466 | 11.4549 | 14.4129 |
| NBEATSx fundamentals (3-seed mean) | 8.0916 | 10.9180 | 13.4027 |

최근 24개월 가격만 입력하고 GPU에서 사전학습 가중치를 고정한 zero-shot 예측을 수행했다.
0.5 분위수(중앙값)를 사전에 고정했다. 이는 50:50 혼합 가중치가 아니며 MAPE 최적 분위수라는 뜻도 아니다.
NBEATSx price는 동일 입력 길이의 가격-only 참고다. Ridge D_market은 수급·시장 변수도 사용한 사후 탐색 BEST다.
추가 학습이나 시드별 모델 선택이 없으며 첫 배치 반복 추론의 일치 여부를 검사했다.

## 검증 한계

사전학습 데이터와 공개 Henry Hub 평가 시계열의 중복 여부는 확인하지 못했다. 체크포인트 공개 시점도 평가 기간 이후다.
입력의 미래값 누수 방지는 검사했으나 가중치의 평가 데이터 노출 가능성까지 배제한 것은 아니다.
따라서 이 결과는 회고적 탐색 비교이며 오염 없는 역사적 out-of-sample 성능이나 최종 제출 모델 우위로 주장하지 않는다.
submission.csv는 변경하지 않았다.

고정 revision: `772f3d25d38aec6d914c8949dab4462e2d46f5d8`

[설계](../notebooks/CHRONOS_COMPARISON.md) · [설정](manifest.json) · [실행 시간/GPU 메모리](runtime.json) · [연도별 지표](metrics.csv)
