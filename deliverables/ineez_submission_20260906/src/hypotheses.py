"""Predeclared hypotheses: raw columns -> transformations -> controlled combinations.

These are experimental candidates, not approved final features. No test data is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.features import (
    CONSERVATIVE_RELEASE_LAGS, ENERGY_COLUMNS, FUNDAMENTAL_COLUMNS,
    _expanding_zscore, make_feature_matrix, validate_monthly_frame,
)


@dataclass(frozen=True)
class Hypothesis:
    name: str
    parent: str
    question: str
    columns: tuple[str, ...]
    constrained: bool = False


def build_hypotheses(data: pd.DataFrame, mode: str = "conservative"):
    """Return feature values, provenance, and a fixed experiment manifest."""
    if mode not in {"conservative", "as_provided"}:
        raise ValueError("Invalid availability mode")
    data = validate_monthly_frame(data)
    base, groups = make_feature_matrix(data, mode)
    history = tuple(groups["E1_history"])
    features = base[list(history)].copy()
    provenance = [{"feature": c, "raw_columns": "date" if c.startswith("month_") else
                   "Natural_Gas_US_Henry_Hub_Gas", "formula": c,
                   "release_lag_months": 0, "implementation": "src/features.py:make_feature_matrix"}
                  for c in history]
    month = data.date.dt.month

    def add(name, values, raw, formula, lag):
        features[name] = values
        provenance.append({"feature": name, "raw_columns": raw, "formula": formula,
                           "release_lag_months": lag, "implementation": "src/hypotheses.py"})

    for role, raw in FUNDAMENTAL_COLUMNS.items():
        if raw not in data:
            raise ValueError(f"Missing candidate column: {raw}")
        s = pd.to_numeric(data[raw], errors="raise")
        lag = CONSERVATIVE_RELEASE_LAGS.get(raw, 0) if mode == "conservative" else 0
        # Transform on reference-month observations BEFORE release alignment.
        # Same-month seasonal history excludes the observation being evaluated.
        seasonal_mean = s.groupby(month).transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
        seasonal_std = s.groupby(month).transform(lambda x: x.shift(1).expanding(min_periods=3).std())
        transforms = {
            "level": (s, "x"),
            "change": (s.diff(), "x - x.shift(1)"),
            "yoy": (s.pct_change(12, fill_method=None), "x / x.shift(12) - 1"),
            "z": (_expanding_zscore(s), "(x - expanding_mean(x)) / expanding_std(x), min_n=12"),
            "seasonal_z": ((s-seasonal_mean)/seasonal_std.replace(0,np.nan),
                           "(x - prior_same_month_mean) / prior_same_month_std, min_n=3"),
        }
        for suffix, (values, formula) in transforms.items():
            add(f"{role}_{suffix}", values.replace([np.inf,-np.inf],np.nan).shift(lag).ffill(),
                raw, formula + "; then shift(release_lag).ffill()", lag)
    for lag in (3,6):
        add(f"rigs_z_lag{lag}", features.rigs_z.shift(lag), FUNDAMENTAL_COLUMNS["rigs"],
            f"rigs_z.shift({lag}) after release alignment", lag)
    for raw in ENERGY_COLUMNS:
        if raw not in data:
            raise ValueError(f"Missing candidate column: {raw}")
        add(f"{raw}_return", pd.to_numeric(data[raw],errors="raise").pct_change(fill_method=None).ffill(),
            raw, "x / x.shift(1) - 1; ffill", 0)
    temperature_columns = ["high_temperature", "low_temperature"]
    if not set(temperature_columns).issubset(data.columns):
        raise ValueError("Missing temperature columns")
    temp = data[temperature_columns].apply(pd.to_numeric,errors="raise").mean(axis=1,skipna=False)
    mean = temp.groupby(month).transform(lambda x:x.shift(1).expanding(min_periods=3).mean())
    std = temp.groupby(month).transform(lambda x:x.shift(1).expanding(min_periods=3).std())
    lag = 1 if mode == "conservative" else 0
    add("temperature_seasonal_z", ((temp-mean)/std.replace(0,np.nan)).shift(lag).ffill(),
        ";".join(temperature_columns), "mean(high,low); prior_same_month_z(min_n=3); release shift; ffill", lag)
    add("balance_equal", features.production_z + features.storage_z - features.consumption_z,
        ";".join(FUNDAMENTAL_COLUMNS[k] for k in ["production","storage","consumption"]),
        "production_z + storage_z - consumption_z; no imports or rigs", "component-specific")
    add("demand_storage_interaction", features.consumption_seasonal_z.clip(lower=0) *
        (-features.storage_seasonal_z).clip(lower=0),
        ";".join(FUNDAMENTAL_COLUMNS[k] for k in ["consumption","storage"]),
        "max(consumption_seasonal_z,0) * max(-storage_seasonal_z,0)", "component-specific")

    hypotheses = []
    def register(name, parent, question, cols, constrained=False):
        hypotheses.append(Hypothesis(name,parent,question,tuple(cols),constrained))
    register("history", "naive", "가격 이력만으로 Naive보다 개선되는가?", history)
    core = history + ("production_z","storage_z","consumption_z")
    register("core_z", "history", "생산·재고·소비를 개별 입력하면 추가 정보가 있는가?", core)
    for role in ("production","storage","consumption"):
        for form in ("level","change","yoy","seasonal_z"):
            cols = tuple(f"{role}_{form}" if c==f"{role}_z" else c for c in core)
            register(f"{role}_{form}", "core_z", f"다른 입력을 고정하고 {role}를 {form}로 교체하면 개선되는가?", cols)
        register(f"without_{role}", "core_z", f"{role}를 제거하고 재학습하면 성능이 악화되는가?",
                 tuple(c for c in core if c!=f"{role}_z"))
    register("equal_index", "core_z", "같은 세 Z특징을 +생산+재고-소비 지수로 압축하면 어떠한가?",
             history+("balance_equal",))
    register("signed_core", "core_z", "생산·재고 음수, 소비 양수 계수 제약이 예측에 도움이 되는가?",core,True)
    register("plus_imports", "core_z", "총수입이 세 수급 변수에 추가 정보를 제공하는가?",core+("imports_z",))
    register("plus_canada", "plus_imports", "총수입에 캐나다 수입을 더하면 중복 이상의 가치가 있는가?",
             core+("imports_z","canada_imports_z"))
    for lag in (0,3,6):
        col = "rigs_z" if lag==0 else f"rigs_z_lag{lag}"
        register(f"plus_rigs_{lag}","core_z",f"rig count의 추가 시차 {lag}개월이 도움이 되는가?",core+(col,))
    for raw in ENERGY_COLUMNS:
        register(f"plus_{raw}","core_z",f"{raw} 수익률에 수급 이상의 추가 정보가 있는가?",core+(f"{raw}_return",))
    register("plus_energy", "core_z", "시장 수익률을 함께 추가하면 개선되는가?",
             core+tuple(f"{c}_return" for c in ENERGY_COLUMNS))
    register("plus_weather", "core_z", "계절 대비 기온이 소비 이외의 정보를 제공하는가?",
             core+("temperature_seasonal_z",))
    seasonal = history+tuple(f"{r}_seasonal_z" for r in ("production","storage","consumption"))
    register("seasonal_core", "core_z", "세 요소를 모두 계절편차로 표현하면 어떠한가?",seasonal)
    register("plus_interaction", "seasonal_core", "높은 소비와 낮은 재고의 결합 효과가 있는가?",
             seasonal+("demand_storage_interaction",))
    return features.replace([np.inf,-np.inf],np.nan), pd.DataFrame(provenance), hypotheses


def manifest(hypotheses):
    return [asdict(h) for h in hypotheses]
