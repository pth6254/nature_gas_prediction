"""Direct-horizon datasets and leakage-safe rolling-origin splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import TARGET


@dataclass(frozen=True)
class RollingOriginSplit:
    origin: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


def make_horizon_targets(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in {1, 2, 3}:
        raise ValueError("horizon must be 1, 2, or 3")
    current = data[TARGET].astype(float)
    future = current.shift(-horizon)
    return pd.DataFrame(
        {
            "current_price": current,
            "future_price": future,
            "log_return": np.log(future / current),
        },
        index=data.index,
    )


def rolling_origin_splits(
    n_rows: int,
    horizon: int,
    first_origin: int = 120,
    last_origin: int | None = None,
    step: int = 1,
) -> Iterator[RollingOriginSplit]:
    """Yield monthly origins; every training target is known by the origin."""
    if first_origin < horizon:
        raise ValueError("first_origin leaves no trainable labels")
    if last_origin is None:
        last_origin = n_rows - horizon - 1
    last_origin = min(last_origin, n_rows - horizon - 1)
    for origin in range(first_origin, last_origin + 1, step):
        # A target for feature row j is known at j+h. Equality is permitted
        # because the price at the current forecast origin is observed.
        train_end = origin - horizon
        yield RollingOriginSplit(
            origin=origin,
            train_indices=np.arange(train_end + 1, dtype=int),
            validation_indices=np.array([origin], dtype=int),
        )

