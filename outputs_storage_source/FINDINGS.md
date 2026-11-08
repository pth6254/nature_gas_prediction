# 재고 원천·발표 시점 확인

확인일: 2026-09-06. 원본·제출 데이터는 수정하지 않았다. EIA 공식 시계열은 대조용으로만 다운로드했다.

## 1. 원자료 대응: 204개월 모두 반올림 수준 일치

제공 `Total_Natural_Gas_Underground_Storage_Volume_USA`와
[EIA U.S. Natural Gas Underground Storage Volume](https://www.eia.gov/dnav/ng/hist/n5030us2m.htm)을
날짜·단위(MMcf) 기준으로 대조했다. 2003-09~2020-08의 **204/204개월이 공식 수치를 10 MMcf 단위로 반올림하면 일치**한다.
원시 정수값까지 일치한 달은 24개이며 최대 차이는 5 MMcf다. 이름·단위·전 기간 수치 대응은 같은 시계열 계열이라는 강한 근거다.
다만 제공자가 어떤 경로·시점에 자료를 수집했는지까지 증명하지는 않는다.

월별 대조는 [matched_training_months.csv](matched_training_months.csv), 다운로드 시각·해시는 [summary.json](summary.json)에 저장했다.
공식 HTML 원문도 보존해 이후 개정과 관계없이 이번 대조를 재현할 수 있게 했다.

## 2. 의미: working gas가 아닌 전체 저장 가스

[EIA 전체 운영자 표](https://www.eia.gov/dnav/ng/ng_stor_sum_dcu_nus_m.htm)의 Natural Gas in Storage 이력은 위 시계열로 연결된다.
이 표는 Base Gas와 Working Gas를 따로 제공한다. 따라서 입력은 **base gas를 포함한 총저장량**이며,
즉시 인출 가능한 working gas만의 재고 또는 저장 시설 용량과 구분해야 한다.
[EIA 정의·주석](https://www.eia.gov/dnav/ng/TblDefs/ng_stor_sum_tbldef2.asp)은 base gas의 압력 유지 역할,
기간 말 저장량과 유출입의 차이 가능성, 반올림, 2013년 이전 미국 합계에 알래스카가 포함되지 않았다는 점을 설명한다.
2013년 범위 변경이 있으므로 완전히 동일한 지리 범위의 시계열이라고 표현하지 않는다.

## 3. 발표 시차: 새 실험은 2개월 가정으로 고정

[EIA 2020년 6월 30일 공지](https://www.eia.gov/about/new/index.php?r=2250)는 2020년 4월 자료를 포함한 Natural Gas Monthly 발표를 기록한다.
또 [현재 저장량 표](https://www.eia.gov/dnav/ng/ng_stor_sum_dcu_nus_m.htm)는 2026년 6월까지 자료의 발표일을 8월 31일로 표시한다.
이는 월별 자료에 2개월 지연을 적용할 근거를 보강한다. 주간 working gas의 빠른 공표를 월별 총저장량에 적용해서는 안 된다.

**이번 후속 실험에서는 재고 2개월을 고정하고, 성능으로 1/2개월 중 선택하지 않는다.**
예를 들어 6월 말 예측에는 4월 재고를 쓴다. 예측은 해당 월말의 공표가 끝난 후라는 가정이다.
이 확인은 2003~2020 전체 월별 최초 공표 달력이나 실제 당시의 수치를 복원한 것은 아니다.
발표 연기 및 이후 개정 빈티지 위험은 남아 있어 `완전한 실시간 백테스트`라고 주장할 수 없다.
기존 lag1 결과는 삭제하거나 덮어쓰지 않고 과거 잠정 가정의 참고 결과로 유지한다.

## 재현

캐시 대조: `python -m src.storage_source_check`.
처음 다운로드: `python -m src.storage_source_check --download` (네트워크 필요; 기존 원문은 덮어쓰지 않음).
