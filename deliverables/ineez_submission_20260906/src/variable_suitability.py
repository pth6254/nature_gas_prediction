"""Audit every supplied input; controlled retrospective predictive diagnostics.

Run: python -m src.variable_suitability
Does not read test outcomes or change submission/model selection.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.evaluate import summarize_predictions
from src.feature_audit import build_design
from src.features import TARGET, validate_monthly_frame
from src.train import _pipeline, _to_price
from src.validation import make_horizon_targets, rolling_origin_splits

OUT = Path("outputs_variable_suitability")
PREFIX = 120
ALIASES = {"Brent_Crude_Oil": "Brent_Crude_Oil_Daily", "WTI_Crude_Oil": "WTI_Crude_Oil_Daily"}
GROUP_LABELS = {
    "fundamentals": "미국 가스 수급·재고·리그", "energy": "에너지 가격",
    "commodities": "원자재 선물 지수", "financial": "주가·환율",
    "macro": "미국·세계 거시", "cli_us": "미국 CLI",
    "cli_foreign": "해외 CLI", "pmi_unknown": "국가 미확인 PMI",
    "activity_unknown": "국가 미확인 누적 YoY", "eu_storage": "EU 가스·LNG 재고",
    "weather": "보스턴 기온",
}


def category(name):
    if "PMI" in name:
        return "pmi_unknown"
    if "Accumulated_YoY" in name:
        return "activity_unknown"
    if name.startswith("EU_"):
        return "eu_storage"
    if name in ("high_temperature", "low_temperature"):
        return "weather"
    if name.startswith(("CLI_", "Composite_Leading_")):
        return "cli_us" if name.endswith("_USA") else "cli_foreign"
    if "Natural_Gas" in name:
        return "fundamentals"
    if name.startswith("BCOM"):
        return "commodities"
    if name.endswith(("_INDX", "_FOREX")):
        return "financial"
    if name in (*ALIASES, "Electricity_Price_USA", "Heating_Oil_Price_USA"):
        return "energy"
    return "macro"


def lag_assumption(name, group):
    # Scenario assumptions, NOT verified source-specific historical release calendars.
    if group in ("commodities", "financial") or name in (*ALIASES, "Natural_Gas_Rotary_Rig_Count_USA", "Fed_Fund_Effective_USA"):
        return 0
    if group == "weather":
        return 1
    # Monthly storage deliberately uses 2, compared with legacy 1 separately.
    return 2


def region(name, group):
    if group.endswith("unknown"):
        return "미확인"
    if group == "weather":
        return "USA/Boston (전국 아님)"
    if group == "eu_storage":
        return "EU"
    if name == "CSI300_INDX":
        return "CHN"
    if name == "USDEUR_FOREX":
        return "USD/EUR (호가 방향 확인 필요)"
    if name.startswith("DXY"):
        return "USD 통화 바스켓"
    if name in ALIASES or name.startswith("BCOM"):
        return "국제 시장 벤치마크 (Henry Hub 현물과 별개)"
    return name.rsplit("_", 1)[-1] if name.rsplit("_", 1)[-1] in {"USA", "WORLD", "CHN", "DEU", "GBR", "IND", "JPN", "KOR"} else "USA"


def longest_missing(s):
    a = s.isna()
    return int(a.groupby(a.ne(a.shift()).cumsum()).sum().max())


def assess(data, metadata):
    if metadata.name.duplicated().any():
        raise ValueError("Duplicate metadata key")
    meta = metadata.set_index("name")
    inputs = [c for c in data if c not in ("date", TARGET)]
    rows = []
    for i, name in enumerate(inputs):
        key = ALIASES.get(name, name)
        if key not in meta.index:
            raise ValueError(f"Unmapped metadata: {name}")
        raw = data[name]
        s = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        pre = s.iloc[:PREFIX]
        group = category(name)
        failures, notes = [], []
        if group.endswith("unknown"):
            failures.append("국가/대상 경제 미확인")
        if pre.notna().mean() < .8:
            failures.append("초기120개월 관측률80% 미만")
        if pre.nunique() < 2:
            failures.append("초기120개월 비상수 관측 부족")
        invalid = int((raw.notna() & s.isna()).sum())
        if invalid:
            notes.append("숫자 변환 실패/비유한 값 존재")
        if name == "Heating_Oil_Price_USA":
            notes.append("4~9월 반복 결측: 계절적 관측 공백; 여름 가격으로 전방채움 금지")
        if group == "eu_storage":
            notes.append("늦은 시작; 미국 가스 재고와 지역·단위가 다름")
        if group in ("cli_us", "cli_foreign"):
            notes.append("정규화/진폭조정 지수; 과거값 개정 및 빈티지 미확인")
        if group == "pmi_unknown":
            notes.append("메타데이터 단위 %; 확산지수를 증가율로 해석하지 않음")
        if group == "activity_unknown":
            notes.append("누적 YoY는 월별 물량 아님; 누적기간/연초 결측 확인 필요")
        if group == "weather":
            notes.append("보스턴 월 극값; 전국 평균 기온/실제 HDD·CDD 아님")
        if name in ALIASES:
            notes.append("메타데이터 Daily 별칭 명시 매핑; 월평균/월말 집계 미확인")
        if name == "Electricity_Price_USA":
            notes.append("단위 cents/kWh는 가격; 한국어 판매량 설명과 불일치")
        if name == "USDEUR_FOREX":
            notes.append("단위 Index와 환율 설명 불일치; 호가 방향 확인 필요")
        if "Marketed_Production" in name:
            notes.append("marketed production은 판매량/건성가스 생산량과 구분")
        if "Underground_Storage" in name:
            notes.append("stock; 총가스/working gas 구분 및 월말값 여부 미확인; 기존 lag1 재검증")
        if name.startswith("BCOM"):
            notes.append("선물 기반 지수 계열; 현물 물량 아님; 정확한 지수 변형/롤 규칙 미확인")
        if "Imports" in name:
            notes.append("캐나다 수입은 총수입의 부분집합: 단순 합산 시 중복")
        first = s.first_valid_index()
        last = s.last_valid_index()
        rows.append(dict(variable_id=f"v{i:03d}", variable=name, metadata_name=key,
                         unit=str(meta.loc[key, "unit"]), description=str(meta.loc[key, "description"]),
                         korean_description=str(meta.loc[key, "Korean description"]),
                         group=group, region=region(name, group), metadata_matched=True,
                         source_series_verified=False, historical_release_verified=False, vintage_verified=False,
                         lag_months_assumed=lag_assumption(name, group),
                         initial120_coverage_pct=float(pre.notna().mean()*100),
                         train204_missing_pct=float(s.isna().mean()*100),
                         first_observation="none" if first is None else str(data.loc[first, "date"].date()),
                         last_observation="none" if last is None else str(data.loc[last, "date"].date()),
                         initial120_unique=int(pre.nunique()), invalid_numeric_count=invalid,
                         longest_missing_months=longest_missing(s),
                         train_min=s.min(), train_max=s.max(),
                         predictive_eligible=not failures,
                         gate="조건부 실험" if not failures else "보류",
                         gate_reason="; ".join(failures) or "초기 관측·지역 조건 충족; 공표/빈티지 미검증",
                         caveats="; ".join(notes) or "정확한 원천 시리즈·월 집계·공표 시각 확인 필요"))
    expected = {ALIASES.get(c, c) for c in inputs} | {TARGET}
    if set(meta.index) != expected:
        raise ValueError("Unexpected orphan metadata")
    return pd.DataFrame(rows)


def redundancy(data, assessment):
    names = assessment.variable.tolist()
    prefix = data[names].iloc[:PREFIX].apply(pd.to_numeric, errors="coerce")
    records = []
    for kind, values in (("level", prefix), ("first_difference", prefix.diff())):
        corr = values.corr(min_periods=60)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                r = corr.loc[a, b]
                if np.isfinite(r) and abs(r) >= .95:
                    records.append(dict(kind=kind, variable_a=a, variable_b=b, correlation=r,
                                        paired_n=int(values[[a, b]].notna().all(axis=1).sum()),
                                        exact_duplicate=bool(prefix[a].equals(prefix[b]))))
    return pd.DataFrame(records, columns=["kind", "variable_a", "variable_b", "correlation", "paired_n", "exact_duplicate"])


def build_features(data, assessment):
    old, _, designs = build_design(data)
    history = designs["basic_processed"][:8]
    columns = {c: old[c] for c in history}
    groups, singles = {}, {}
    for row in assessment.loc[assessment.predictive_eligible].itertuples():
        # No future fill, and at most one month forward fill of ordinary gaps.
        # Seasonal/late series fail the gate rather than being filled indiscriminately.
        s = pd.to_numeric(data[row.variable], errors="raise").replace([np.inf, -np.inf], np.nan)
        s = s.ffill(limit=1).shift(row.lag_months_assumed)
        names = [row.variable_id+"_level", row.variable_id+"_diff1"]
        columns[names[0]], columns[names[1]] = s, s.diff()
        singles[row.variable_id] = history + names
        groups.setdefault(row.group, []).extend(names)
    all_columns = [c for names in groups.values() for c in names]
    candidates = {"history": history, "all_groups": history + all_columns}
    for group, names in groups.items():
        candidates["add_"+group] = history + names
        candidates["drop_"+group] = history + [c for c in all_columns if c not in names]
        delayed = []
        for c in names:
            columns[c+"_delay1"] = columns[c].shift(1)
            delayed.append(c+"_delay1")
        candidates["delay_"+group] = history + delayed
    candidates.update({"single_"+k: v for k, v in singles.items()})
    # Same processed reference, only storage publication assumption changes.
    for c in designs["basic_processed"]:
        columns[c] = old[c]
    candidates["processed_reference"] = designs["basic_processed"]
    columns["storage_seasonal_z_lag2"] = old.storage_seasonal_z.shift(1)
    candidates["processed_storage_lag2"] = ["storage_seasonal_z_lag2" if c == "storage_seasonal_z" else c for c in designs["basic_processed"]]
    return pd.DataFrame(columns), candidates


def block_interval(delta, block=6, repetitions=1000):
    """Paired moving-block bootstrap, descriptive 95% interval; no multiplicity claim."""
    a = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(a)-block+1, size=(repetitions, int(np.ceil(len(a)/block))))
    sampled = a[(starts[..., None]+np.arange(block)).reshape(repetitions, -1)[:, :len(a)]]
    return tuple(np.quantile(sampled.mean(axis=1), [.025, .975]))


def backtest(data, features, columns, family, name, h):
    target = make_horizon_targets(data, h)
    rows, diagnostics = [], []
    for split in rolling_origin_splits(len(data), h, first_origin=120):
        model = _pipeline(family+"_log_return")
        origin = split.origin
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features.loc[split.train_indices, columns], target.loc[split.train_indices, "log_return"])
            raw = float(model.predict(features.loc[[origin], columns])[0])
        current = float(data.loc[origin, TARGET])
        rows.append(dict(date=data.loc[origin, "date"], origin=origin, horizon=h, family=family,
                         candidate=name, actual=target.loc[origin, "future_price"],
                         predicted=_to_price(raw, current, "log_return"), current_price=current,
                         previous_price=data.loc[origin-1, TARGET]))
        diagnostics.extend(dict(family=family, candidate=name, horizon=h, origin=origin,
                                warning=str(w.message)) for w in caught)
    return pd.DataFrame(rows), diagnostics


def evaluate(predictions):
    metrics, comparisons = [], []
    for (family, name, h), part in predictions.groupby(["family", "candidate", "horizon"]):
        for period, sub in [("all", part), ("early", part.loc[part.origin < 161]),
                            ("late", part.loc[part.origin >= 161]),
                            *[(str(y), p) for y, p in part.groupby(part.date.dt.year)]]:
            metrics.append(dict(family=family, candidate=name, horizon=h, period=period, **summarize_predictions(sub)))
        if name in ("history", "naive"):
            continue
        ref_name = "all_groups" if name.startswith("drop_") else "history"
        if name == "processed_storage_lag2":
            ref_name = "processed_reference"
        ref = predictions.loc[(predictions.family == family) & (predictions.candidate == ref_name) & (predictions.horizon == h)]
        pair = part.merge(ref, on=["origin", "horizon", "date"], suffixes=("", "_ref"), validate="one_to_one")
        assert len(pair) == len(part) == 84-h
        np.testing.assert_allclose(pair.actual, pair.actual_ref, rtol=0, atol=1e-12)
        delta = 100*(abs(pair.predicted-pair.actual)-abs(pair.predicted_ref-pair.actual))/pair.actual
        annual = delta.groupby(pair.date.dt.year).mean()
        lo, hi = block_interval(delta)
        comparisons.append(dict(family=family, candidate=name, horizon=h, reference=ref_name,
                                delta_mape_pp=float(delta.mean()), early_delta_pp=float(delta[pair.origin < 161].mean()),
                                late_delta_pp=float(delta[pair.origin >= 161].mean()),
                                years_better=int((annual < 0).sum()), n_years=len(annual),
                                block6_low_pp=lo, block6_high_pp=hi))
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def enrich(assessment, comparisons):
    result = assessment.copy()
    for i, row in result.iterrows():
        sub = comparisons.loc[comparisons.candidate == "single_"+row.variable_id]
        if sub.empty:
            result.loc[i, "predictive_evidence"] = "미실험(게이트 보류); 무용함을 의미하지 않음"
            result.loc[i, "decision"] = "보류: " + row.gate_reason
            continue
        assert len(sub) == 6
        improving = sub.delta_mape_pp < 0
        stable = improving & (sub.early_delta_pp < 0) & (sub.late_delta_pp < 0) & (sub.years_better >= 4)
        result.loc[i, "single_improving_of6"] = int(improving.sum())
        result.loc[i, "single_stable_of6"] = int(stable.sum())
        result.loc[i, "single_ci_below_zero_of6"] = int((sub.block6_high_pp < 0).sum())
        for r in sub.itertuples():
            result.loc[i, f"{r.family}_h{r.horizon}_delta_pp"] = r.delta_mape_pp
        result.loc[i, "predictive_evidence"] = f"단독추가 개선 {improving.sum()}/6; 전후반·연도 안정 {stable.sum()}/6"
        result.loc[i, "decision"] = "조건부 재검증 후보" if stable.any() else "현 표현에서 안정 개선 근거 부족"
    return result


def main():
    train = next(Path('.').glob('*data_train.csv'))
    metadata = next(Path('.').glob('*metadata.xlsx'))
    data = validate_monthly_frame(pd.read_csv(train))
    if len(data) != 204:
        raise ValueError("This audit protocol expects the supplied 204 training months")
    assessment = assess(data, pd.read_excel(metadata))
    features, candidates = build_features(data, assessment)
    OUT.mkdir(exist_ok=True)
    (OUT/"cache").mkdir(exist_ok=True)
    paths = [train, metadata, Path("submission.csv"), *[Path("src")/n for n in
             ("variable_suitability.py", "features.py", "hypotheses.py", "feature_audit.py", "train.py", "validation.py", "evaluate.py")]]
    plan = dict(hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm")},
                candidates=candidates, models={f: _pipeline(f+"_log_return").named_steps["model"].get_params() for f in ("ridge", "lightgbm")},
                audit_scope="112 inputs, provided metadata; test file not read; submission unchanged",
                eligibility="first120 coverage>=80%, >=2 unique, known region; pragmatic fixed gate, no target ranking",
                features="same price-history8; each raw variable contributes level + first difference; ffill limit1 then assumed release shift; fold-only median+indicator+StandardScaler",
                availability="assumed lag2 monthly macro/fundamentals including storage; lag1 Boston; lag0 traded market/rigs/Fed; group +1 sensitivity; vintage UNKNOWN",
                validation="origins120..203-h, expanding, training labels j+h<=origin, H1/H2/H3 independent log returns",
                stability="early origins<161, late>=161; years with lower MAPE; paired moving-block6 bootstrap1000 seed42 descriptive intervals",
                decision="no scalar score; metadata gate independent of prediction; stable flag=lower MAPE overall+both halves+>=4 calendar years; not final selection",
                limitations="retrospective repeatedly inspected training validation; no untouched holdout; no multiplicity adjustment; raw level/difference only is not optimal transformation; conditional lag scenarios do not reconstruct original vintages; CPU fixed comparable estimators; no DL rerun")
    manifest = OUT/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing manifest differs; use a new output directory/version")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    assessment.to_csv(OUT/"variable_assessment.csv", index=False, encoding="utf-8-sig")
    redundancy(data, assessment).to_csv(OUT/"redundancy.csv", index=False)
    features.to_csv(OUT/"feature_values.csv", index_label="origin")
    parts, diagnostics = [], []
    with threadpool_limits(limits=1):
        for family in ("ridge", "lightgbm"):
            for h in (1, 2, 3):
                for name, columns in candidates.items():
                    path = OUT/"cache"/f"{family}_{h}_{name}.csv"
                    notes = path.with_suffix(".json")
                    if path.exists() and notes.exists():
                        part = pd.read_csv(path, parse_dates=["date"])
                        diag = json.loads(notes.read_text(encoding="utf-8"))
                    else:
                        part, diag = backtest(data, features, columns, family, name, h)
                        part.to_csv(path, index=False)
                        notes.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
                    assert len(part) == 84-h and set(part.origin) == set(range(120, 204-h))
                    assert part.family.eq(family).all() and part.candidate.eq(name).all() and part.horizon.eq(h).all()
                    assert np.isfinite(part.predicted).all()
                    parts.append(part)
                    diagnostics.extend(diag)
                naive = parts[-1].copy()
                naive["candidate"], naive["predicted"] = "naive", naive.current_price
                parts.append(naive)
                print(f"DONE {family} H{h}: {len(candidates)} candidates", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    metrics, comparisons = evaluate(predictions)
    predictions.to_csv(OUT/"predictions.csv", index=False)
    metrics.to_csv(OUT/"metrics.csv", index=False)
    comparisons.to_csv(OUT/"comparisons.csv", index=False)
    enrich(assessment, comparisons).to_csv(OUT/"variable_assessment.csv", index=False, encoding="utf-8-sig")
    (OUT/"fit_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    assert hashlib.sha256(Path("submission.csv").read_bytes()).hexdigest() == plan["hashes"]["submission.csv"]
    from src.variable_suitability_report import main as report
    report()


if __name__ == "__main__":
    main()
