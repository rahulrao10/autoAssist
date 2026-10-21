"""profiler.py -- looks at the data and measures things (it never changes anything).

Contains: date-column detection and per-column statistics (missing share,
distinct values, ...). The planner then turns these numbers into decisions.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portable import parse_dates

# Looks like 2021-03-04, 2021/3/4, 04.03.2021 ... at the start of the text
DATE_PATTERN = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def find_datetime_columns(df, cfg):
    """A column is a date column if it already has a datetime dtype, or if >= 90% of its
    non-missing text values look like dates AND can be parsed as dates."""
    found = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        s = s.dropna()
        if len(s) == 0:
            continue
        text = s.astype(str)
        if text.str.match(DATE_PATTERN).mean() < cfg.datetime_min_parse_fraction:
            continue
        if parse_dates(s).notna().mean() >= cfg.datetime_min_parse_fraction:
            found.append(col)
    return found


@dataclass
class ColumnProfile:
    name: str
    kind: str               # "numeric" or "categorical"
    dtype: str
    n_rows: int
    n_missing: int
    missing_fraction: float
    n_unique: int           # distinct non-missing values
    unique_ratio: float     # n_unique / number of non-missing values
    is_integer: bool
    range_ratio: float      # (max-min+1)/n_unique for integers (close to 1 = consecutive numbers), else 0


def profile_columns(X, cfg):
    """One ColumnProfile per column of X."""
    profiles = []
    n = len(X)
    for col in X.columns:
        s = X[col]
        non_missing = s.dropna()
        n_unique = int(non_missing.nunique())
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_int = bool(is_numeric and (pd.api.types.is_integer_dtype(s) or
                      (len(non_missing) > 0 and bool((non_missing == np.round(non_missing)).all()))))
        range_ratio = 0.0
        if is_int and n_unique > 0:
            range_ratio = float((non_missing.max() - non_missing.min() + 1) / n_unique)
        profiles.append(ColumnProfile(
            name=col, kind="numeric" if is_numeric else "categorical", dtype=str(s.dtype),
            n_rows=n, n_missing=int(s.isna().sum()), missing_fraction=float(s.isna().mean()),
            n_unique=n_unique,
            unique_ratio=float(n_unique / len(non_missing)) if len(non_missing) else 0.0,
            is_integer=is_int, range_ratio=range_ratio))
    return profiles
