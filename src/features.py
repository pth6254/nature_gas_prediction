"""Causal feature engineering for monthly Henry Hub forecasts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


TARGET = "Natural_Gas_US_Henry_Hub_Gas"

FUNDAMENTAL_COLUMNS = {
    "production": "Total_Natural_Gas_Marketed_Production_USA",
    "imports": "Natural_Gas_Imports_USA",
    "canada_imports": "Natural_Gas_Imports_From_Canada_USA",
    "rigs": "Natural_Gas_Rotary_Rig_Count_USA",
    "consumption": "Natural_Gas_Total_Consumption_USA",
    "storage": "Total_Natural_Gas_Underground_Storage_Volume_USA",
}

ENERGY_COLUMNS = [
    "BCOMNG_INDX",
    "WTI_Crude_Oil",
    "Brent_Crude_Oil",
    "BCOMCL_INDX",
    "DXY_INDX",
]

MACRO_COLUMNS = [
    "Kilian_Global_Economy_Index_WORLD",
    "Fed_Fund_Effective_USA",
    "Composite_Leading_Indicator_USA",
    "Composite_Leading_Indicator_CHN",
]

EXTERNAL_COLUMNS = [
    "EU_Aggregated_Gas_Storage_Inventory",
    "EU_Aggregated_LNG_System_Inventory",
    "high_temperature",
    "low_temperature",
]

# Months by which a release is assumed to trail its reference month. This is a
# deliberately conservative default for a real-time backtest. The alternative
# ``as_provided`` mode assumes every value on a row was available at month-end.
CONSERVATIVE_RELEASE_LAGS = {
    FUNDAMENTAL_COLUMNS["production"]: 2,
    FUNDAMENTAL_COLUMNS["imports"]: 2,
    FUNDAMENTAL_COLUMNS["canada_imports"]: 2,
    FUNDAMENTAL_COLUMNS["rigs"]: 0,
    FUNDAMENTAL_COLUMNS["consumption"]: 2,
    FUNDAMENTAL_COLUMNS["storage"]: 1,
    "Kilian_Global_Economy_Index_WORLD": 2,
    "Fed_Fund_Effective_USA": 0,
    "Composite_Leading_Indicator_USA": 2,
    "Composite_Leading_Indicator_CHN": 2,
    "high_temperature": 1,
    "low_temperature": 1,
    "EU_Aggregated_Gas_Storage_Inventory": 0,
    "EU_Aggregated_LNG_System_Inventory": 0,
}


def validate_monthly_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a date-sorted copy after validating core input assumptions."""
    required = {"date", TARGET}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")
    periods = frame["date"].dt.to_period("M").astype(int).to_numpy()
    if len(periods) > 1 and not (np.diff(periods) == 1).all():
        raise ValueError("Input must have uninterrupted monthly observations")
    if (frame[TARGET] <= 0).any() or frame[TARGET].isna().any():
        raise ValueError("Henry Hub prices must be positive and complete")
    return frame


def _available_series(
    frame: pd.DataFrame, column: str, availability_mode: str
) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if availability_mode not in {"conservative", "as_provided"}:
        raise ValueError("availability_mode must be 'conservative' or 'as_provided'")
    lag = CONSERVATIVE_RELEASE_LAGS.get(column, 0) if availability_mode == "conservative" else 0
    # Forward-fill is causal. Remaining gaps are retained for fold-local imputation.
    return pd.to_numeric(frame[column], errors="coerce").shift(lag).ffill()


def _expanding_zscore(series: pd.Series, min_periods: int = 12) -> pd.Series:
    expanding = series.expanding(min_periods=min_periods)
    std = expanding.std().replace(0, np.nan)
    return (series - expanding.mean()) / std


def _add_changes(output: pd.DataFrame, name: str, series: pd.Series) -> list[str]:
    columns: list[str] = []
    output[f"{name}__level"] = series
    columns.append(f"{name}__level")
    output[f"{name}__chg1"] = series.diff(1)
    columns.append(f"{name}__chg1")
    output[f"{name}__pct1"] = series.pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan)
    columns.append(f"{name}__pct1")
    output[f"{name}__yoy"] = series.pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan)
    columns.append(f"{name}__yoy")
    return columns


def make_feature_matrix(
    data: pd.DataFrame, availability_mode: str = "conservative"
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build nested E1-E5 feature groups using trailing information only."""
    frame = validate_monthly_frame(data)
    price = frame[TARGET].astype(float)
    features = pd.DataFrame(index=frame.index)

    e1: list[str] = []
    features["price_now"] = price
    e1.append("price_now")
    for lag in (1, 2, 3, 6, 12):
        name = f"price_lag_{lag}"
        features[name] = price.shift(lag)
        e1.append(name)
    for lag in (1, 3, 6, 12):
        name = f"price_log_return_{lag}"
        features[name] = np.log(price / price.shift(lag))
        e1.append(name)
    for window in (3, 6, 12):
        mean_name = f"price_roll_mean_{window}"
        std_name = f"price_roll_std_{window}"
        features[mean_name] = price.rolling(window).mean()
        features[std_name] = price.rolling(window).std()
        e1.extend([mean_name, std_name])
    month = frame["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    e1.extend(["month_sin", "month_cos"])

    available = {
        key: _available_series(frame, column, availability_mode)
        for key, column in FUNDAMENTAL_COLUMNS.items()
    }
    fundamental: list[str] = []
    for key, series in available.items():
        fundamental.extend(_add_changes(features, key, series))

    storage = available["storage"]
    features["storage_vs_12m_mean"] = storage / storage.rolling(12).mean() - 1
    seasonal_storage = storage.groupby(month).transform(
        lambda values: values.shift(1).expanding(min_periods=2).mean()
    )
    features["storage_seasonal_deviation"] = storage / seasonal_storage - 1
    fundamental.extend(["storage_vs_12m_mean", "storage_seasonal_deviation"])

    features["rigs_lag_3"] = available["rigs"].shift(3)
    features["rigs_lag_6"] = available["rigs"].shift(6)
    fundamental.extend(["rigs_lag_3", "rigs_lag_6"])

    features["supply_demand_proxy"] = (
        available["production"] + available["imports"] - available["consumption"]
    )
    fundamental.append("supply_demand_proxy")

    supply_z = pd.concat(
        [
            _expanding_zscore(available["production"]),
            _expanding_zscore(available["imports"]),
            _expanding_zscore(available["canada_imports"]),
            _expanding_zscore(available["rigs"]),
        ],
        axis=1,
    )
    features["production_index_equal_weight"] = supply_z.mean(axis=1, skipna=False)
    features["storage_index_equal_weight"] = _expanding_zscore(storage)
    features["consumption_index_equal_weight"] = _expanding_zscore(available["consumption"])
    features["fundamental_balance_equal_weight"] = (
        features["production_index_equal_weight"]
        + features["storage_index_equal_weight"]
        - features["consumption_index_equal_weight"]
    )
    fundamental.extend(
        [
            "production_index_equal_weight",
            "storage_index_equal_weight",
            "consumption_index_equal_weight",
            "fundamental_balance_equal_weight",
        ]
    )

    energy: list[str] = []
    for column in ENERGY_COLUMNS:
        series = _available_series(frame, column, availability_mode)
        features[f"{column}__level"] = series
        features[f"{column}__return1"] = series.pct_change(fill_method=None)
        features[f"{column}__return3"] = series.pct_change(3, fill_method=None)
        energy.extend(
            [f"{column}__level", f"{column}__return1", f"{column}__return3"]
        )

    macro: list[str] = []
    for column in MACRO_COLUMNS:
        series = _available_series(frame, column, availability_mode)
        features[f"{column}__level"] = series
        features[f"{column}__chg1"] = series.diff()
        macro.extend([f"{column}__level", f"{column}__chg1"])

    external: list[str] = []
    for column in EXTERNAL_COLUMNS[:2]:
        series = _available_series(frame, column, availability_mode)
        external.extend(_add_changes(features, column, series))
    high = _available_series(frame, "high_temperature", availability_mode)
    low = _available_series(frame, "low_temperature", availability_mode)
    average_temp = (high + low) / 2
    features["temperature_average"] = average_temp
    features["temperature_deviation_12m"] = average_temp - average_temp.rolling(12).mean()
    features["temperature_demand_proxy"] = (65 - average_temp).clip(lower=0) + (
        average_temp - 65
    ).clip(lower=0)
    external.extend(
        ["temperature_average", "temperature_deviation_12m", "temperature_demand_proxy"]
    )

    groups = {
        "E1_history": e1,
        "E2_fundamentals": e1 + fundamental,
        "E3_energy": e1 + fundamental + energy,
        "E4_macro": e1 + fundamental + energy + macro,
        "E5_external": e1 + fundamental + energy + macro + external,
    }
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, groups


def available_columns(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    """Drop features that are entirely unavailable in a particular sample."""
    return [column for column in columns if column in frame and frame[column].notna().any()]
