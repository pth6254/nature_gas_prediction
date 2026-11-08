# Henry Hub 천연가스 가격 예측 — 제출 파일 안내

## 먼저 확인할 파일

1. **분석 보고서:** [presentation/REPORT_SLIDES.html](presentation/REPORT_SLIDES.html) — 10장, 실제 실험 수치와 그래프 포함
2. **예측 결과:** [submission.csv](submission.csv) — `date,pred_1,pred_2,pred_3`, 48행
3. **최종 코드:** [src/final_pipeline.py](src/final_pipeline.py)
4. **상세 설명:** [REPORT.md](REPORT.md)

HTML은 Chrome/Edge에서 인터넷 연결 없이 열 수 있습니다. 좌우 방향키로 이동하고, `인쇄 / PDF` 버튼으로 PDF 저장이 가능합니다. 안내문은 보고서 확장자를 지정하지 않았으므로 접수 시스템이 별도 형식을 요구하면 그 형식에 맞춰 제출해야 합니다. HTML 파일만 보내지 말고 코드와 CSV가 들어 있는 이 패키지를 함께 제출합니다.

## 최종 모델과 결과

| 예측 기간 | 모델 | 검증 MAPE |
|---|---|---:|
| 1개월 후 | ElasticNet + 과거 오차 기반 수급 시차 선택 | 6.3141% |
| 2개월 후 | Ridge | 9.2125% |
| 3개월 후 | Ridge | 11.7226% |

훈련자료 내 시간순 검증이며 대회 테스트 점수가 아닙니다. 생산·재고·소비의 발표 지연은 2개월입니다. H1은 수급의 추가 반응 시차 0/1/3/6개월 중 과거 성적으로 선택합니다. 현재 가격과 예측 가격의 50:50 혼합은 하지 않습니다.

`date`는 예측하는 시점입니다. 예를 들어 마지막 `2024-08-31` 행의 세 값은 각각 2024년 9·10·11월 가격입니다. 가격 단위는 USD/MMBtu입니다.

## 환경 설치

Python 3.11 또는 3.12와 uv를 사용합니다. **압축을 해제한 이 README가 있는 폴더에서 실행**합니다.

```powershell
uv sync --frozen --extra dev
```

`pyproject.toml`은 의존성 정의, `uv.lock`은 고정된 패키지 해석 결과입니다. 설치에는 인터넷이 필요하지만 최종 모델 재학습에는 GPU나 모델 다운로드가 필요하지 않습니다. 가상환경과 다운로드된 모델 가중치는 배포하지 않습니다.

## 최종 예측 재현

```powershell
uv run --frozen --extra dev python -m src.final_pipeline
uv run --frozen --extra dev python -m src.final_report
uv run --frozen --extra dev python -m pytest -q -p no:cacheprovider
```

첫 명령은 제공 학습·테스트 자료로 월별 예측을 다시 계산하고 `submission.csv` 및 `outputs_final_v1/`을 갱신합니다. 두 번째 명령은 저장된 제출·검증 값과 코드 해시를 확인하고 최종 요약을 만듭니다. 테스트 중 DL/Chronos 관련 항목은 선택 의존성을 설치하지 않으면 건너뛸 수 있습니다.

기존 `src.train`은 **초기 탐색용**이며 기본 실행하면 다른 모델로 `submission.csv`를 덮어쓸 수 있습니다. 최종 제출을 재현할 때는 반드시 `src.final_pipeline`을 사용합니다.

다음 두 파일은 최종 코드의 검증에 필요하므로 삭제하지 않습니다.

- `outputs_incremental_fundamentals/predictions.csv`: 앞선 실험과 최종 예측의 일치 여부 확인용
- `outputs_final_v1/previous_submission.csv`: 이전 제출의 보존·해시 검증용. **실제 제출할 예측은 루트의 submission.csv 하나**입니다.

## 코드 구성

| 코드 | 역할 |
|---|---|
| `src/final_pipeline.py` | 최종 특징 생성, 기간별 학습, 과거 시차 선택, 제출 생성 |
| `src/final_report.py` | 최종 수치·기여·제출 무결성 확인과 결과 요약 |
| `src/features.py`, `src/validation.py`, `src/evaluate.py` | 시점별 특징, 시간순 분할, 오차·방향·반전 평가 |
| `src/hypotheses.py`, `src/run_hypotheses.py` | 생산·재고·소비의 개별/합성/부호 제약 가설 |
| `src/combination_experiment.py`, `src/feature_audit.py` | 입력 조합, 공통 변수 가공·제거·시차 실험 |
| `src/statistical_experiment.py`, `src/ml_experiment.py` | 전통 시계열 및 Ridge/ElasticNet/LightGBM/CatBoost 비교 |
| `src/dl_experiment.py`, `src/chronos_experiment.py`, `src/chronos_covariates.py` | 딥러닝 및 Chronos 비교 |
| `src/variable_suitability.py`, `src/storage_source_check.py` | 112개 입력 적절성 평가 및 EIA 재고 원천 대조 |
| `src/candidate_stability.py`, `src/incremental_fundamentals.py` | 규제·시차 안정성과 수입·리그 추가 기여 검증 |
| `src/*_report.py` | 각 실험의 지표·해석 정리 |
| `tests/` | 미래 정보 사용 방지, 특징 산술, 제출 형식 등의 테스트 |
| `notebooks/02_data_analysis.ipynb` | 결과가 저장된 데이터 분석 노트북 |
| `notebooks/03_variable_suitability.ipynb` | 전체 변수 평가 노트북 |

일부 최종 모듈이 초기 실험의 공통 함수를 가져오므로 초기 코드도 포함했습니다. 파일을 임의로 옮기거나 몇 개만 삭제하면 import와 근거 파일 경로가 끊길 수 있습니다.

## 보고서의 실험을 확인하는 방법

`outputs_*/`에는 실험별 요약, 지표, 예측, 설정과 검사 기록이 있습니다. 보고서 그래프와 표는 이 파일을 근거로 합니다. 재생성 가능한 중간 캐시는 정리했습니다. 가설별 계수, 내부 검증 예측, DL 개별 학습 기록과 변수 감사 경고는 근거 확인에 필요하므로 보존했습니다. 모든 실험을 새로 수행해야 최종 CSV를 만들 수 있는 구조는 아닙니다.

실험 실행 순서와 설정은 `notebooks/`의 `HYPOTHESES.md`, `COMBINATIONS.md`, `ML_COMPARISON.md`, `DL_COMPARISON.md`, `CHRONOS_COMPARISON.md`, `CHRONOS_COVARIATES.md`, `FEATURE_AUDIT.md`, `STABILITY.md`를 참고합니다. 해당 문서의 '후속 단계'는 당시 실험 기록입니다.

일부 과거 실험은 **당시 제출 파일의 해시**를 설정에 저장했습니다. 최종 제출로 바뀐 현재 폴더에서 재실행하면 보호 검사로 중단될 수 있습니다. 이 검사를 지우거나 결과에 맞춰 해시를 수정하지 않습니다. 과거 실험 전체 재수행은 별도 작업 복사본에서 당시 입력 상태와 설정을 준비해야 하며, 여기서는 전체 재실행을 검증했다고 주장하지 않습니다. 최종 제출 재현 명령은 별도로 검증했습니다.

DL/Chronos 추가 실험 환경:

```powershell
uv sync --frozen --extra dev --extra dl --extra chronos
```

Chronos 실험은 CUDA GPU와 최초 가중치 다운로드가 필요합니다. 원래 DL 실험은 제한된 CPU 학습, Chronos는 GPU 추론이었습니다. 선택 의존성을 유지하려면 이후 uv 명령에도 같은 `--extra` 옵션을 넣습니다. 최종 Ridge/ElasticNet 예측에는 이 추가 설치가 필요하지 않습니다.

## 외부 자료와 제공 데이터

제공받은 train/test CSV, 제출 예시, 메타데이터를 포함했습니다. `outputs_storage_source/n5030us2m.html`은 EIA 월별 총저장량의 원문 스냅샷입니다. 제공 재고가 어떤 계열인지 확인하는 용도이며 새 경제 입력을 추가한 것은 아닙니다. 다운로드 없이 저장된 원문을 검사할 수 있습니다.

보고서의 초기 비교는 재고 공표 지연 1개월, 최종 실험은 2개월 조건입니다. 최초 발표 당시의 수정 전 자료 전체는 미확인입니다. 모델 기여는 예측의 설명이지 가격 변화의 실제 인과효과는 아닙니다.

## 파일 검증

`FILE_MANIFEST.json`에는 포함 파일의 상대 경로·크기·SHA256이 있습니다. 패키지 검수 결과는 `PACKAGE_CHECKS.json`을 참고합니다. 파일 목록 자체와 검수 결과 파일은 자기 참조를 피하기 위해 목록 해시 대상에서 제외합니다.

실제 제출 파일은 `submission.csv`, 보고서는 `presentation/REPORT_SLIDES.html`입니다. 코드·데이터·환경·근거 파일은 폴더 구조를 유지해 함께 전달합니다.
