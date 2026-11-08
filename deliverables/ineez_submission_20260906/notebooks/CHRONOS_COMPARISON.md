# Chronos-Bolt Small: GPU 사전학습 모델 비교

## 가설과 사전 고정 설정

204개월만으로 새 신경망을 학습하는 대신 외부 시계열에서 사전학습한 패턴을 사용하면 가격-only 예측이 개선되는지 확인한다. 사용자 요청에 따라 Chronos 계열 하나만 비교하며 다른 크기·입력 길이·분위수를 성능에 따라 탐색하지 않는다.

- 모델: `amazon/chronos-bolt-small` (약 48M parameters, Apache-2.0). 크기와 추론 비용을 고려해 Small을 선택했으며 성능 최적이라는 근거는 없다.
- 첫 다운로드에서 Hugging Face commit을 확인해 manifest에 고정한다. 이후 실행은 해당 revision을 재사용한다. 가중치·설정 파일 SHA256도 저장한다.
- 입력: 각 원점까지의 최근 24개월 원시 가격. 앞선 DL의 가격-only 입력 길이와 맞춘 고정값이다. 수급·시장 변수와 미래값은 입력하지 않는다.
- 정규화: 공식 모델 내부의 개별 입력 창 정규화를 사용한다. 전 데이터 스케일러나 추가 로그 변환은 적용하지 않는다.
- 출력: 세 미래 기간의 분위수를 직접 예측한다. 사전 고정한 0.5 분위수(중앙값)를 점 예측으로 사용한다. MAPE 최적 분위수나 혼합 가중치가 아니다.
- 학습: frozen zero-shot. 미세조정·조기 종료·시드별 후보 선택 없음. `eval()` 및 inference mode로 실행한다.
- 장치: CUDA GPU, float32, batch 16. CUDA가 없으면 CPU로 조용히 전환하지 않고 오류로 알린다. 첫 배치를 재추론해 수치 일치를 검사한다.
- 평가: 기존과 같은 H1 83개/H2 82개/H3 81개 원점. MAPE·MAE·방향·반전 지표 및 연도별 결과를 저장한다.

## 공정성 및 사전학습 오염

공식 모델 카드는 광범위한 시계열 사전학습을 설명하지만 이 프로젝트의 Henry Hub 월별 가격이 제외됐다는 증거는 확인하지 못했다. 체크포인트는 평가 구간(2013~2020년) 이후 공개된 모델이다. 따라서 다음을 구분한다.

1. 구현상 입력 누수: 과거 창만 전달하며 미래값 변조 테스트로 검사한다.
2. 가중치의 사전 노출: 평가 시계열이나 이후 구간이 사전학습에 포함됐는지 미확인이다. 입력 누수 테스트로 해결되지 않는다.

결과는 **사전학습 중복 여부가 확인되지 않은 회고적 탐색 비교**로 표시한다. 엄격한 과거 실시간 배포 가능 성능, 검증된 zero-shot 데이터 독립성, 최종 제출 우위를 주장하지 않는다. 여기서 zero-shot은 이 프로젝트 데이터로 추가 가중치 학습을 하지 않았다는 의미다.

NBEATSx price와는 최근 24개월 가격 입력을 맞췄지만 사전학습 자료와 타깃 처리 방식은 다르다. Ridge D_market은 수급·시장 변수도 쓰는 사후 선택 후보다. 비교표를 모델 구조만의 통제 실험으로 해석하지 않는다.

## 재현

```powershell
uv sync --extra dev --extra dl --extra chronos
uv run --extra dl --extra chronos python -m src.chronos_experiment
uv run --extra dev --extra dl --extra chronos python -m pytest -q -p no:cacheprovider
```

Windows의 torch는 pyproject에 지정한 공식 CUDA 12.8 인덱스를 사용한다. GPU 드라이버를 이 작업에서 변경하지 않는다. 모델은 `.cache/huggingface`에 다운로드한 뒤 로컬에서 추론하며 원본 데이터를 외부 추론 API에 보내지 않는다.

`outputs_chronos/`에 예측·지표·비교표·GPU 실행 정보·모델 revision을 기록한다. `submission.csv`는 변경하지 않는다. 동일 설정의 재실행은 같은 결과 파일을 재생성하며 설정이 달라지면 기존 파일 보호를 위해 중단한다.

근거: [공식 모델 카드](https://huggingface.co/amazon/chronos-bolt-small), [공식 구현](https://github.com/amazon-science/chronos-forecasting).
