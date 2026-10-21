"""portable.py -- the code that is COPIED WORD FOR WORD into the generated script.

IMPORTANT: this file may only use pandas, numpy and scikit-learn.
Both the library and the generated script call these exact functions, which is
why the script reproduces the library's scores.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Critical values of Student's t distribution (two-sided 95%), indexed by degrees of freedom.
# With k CV folds we have k-1 degrees of freedom (5 folds -> 4 -> 2.776).
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_and_ci(scores):
    """Mean of the fold scores and the half-width of the 95% confidence interval.
    CI half-width = t * (standard deviation / sqrt(number of folds)).
    Read it as: mean +- half-width."""
    scores = np.asarray(scores, dtype=float)
    k = len(scores)
    if k < 2:
        return float(scores.mean()), 0.0
    std_error = scores.std(ddof=1) / np.sqrt(k)
    return float(scores.mean()), float(T_CRITICAL_95.get(k - 1, 1.96) * std_error)


def clean_target(df, target):
    """Drop rows whose target is missing; turn whole-number floats into ints (3.0 -> 3)."""
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y = df[target]
    if pd.api.types.is_float_dtype(y) and len(y) > 0 and bool((y == np.round(y)).all()):
        df[target] = y.astype("int64")
    return df


def parse_dates(series):
    """Turn a column of date strings into datetimes; unreadable values become NaT (missing)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce")


def expand_datetime_columns(df, datetime_cols):
    """Replace each date column by three plain numbers: year, month, day-of-week (Mon=0)."""
    df = df.copy()
    for col in datetime_cols:
        parsed = parse_dates(df[col])
        df[col + "_year"] = parsed.dt.year.astype("float64")
        df[col + "_month"] = parsed.dt.month.astype("float64")
        df[col + "_dayofweek"] = parsed.dt.dayofweek.astype("float64")
        df = df.drop(columns=[col])
    return df


def select_and_cast(df, num_cols, cat_cols):
    """Keep only the chosen columns; numbers stay numbers, categories become text (NaN stays missing)."""
    out = df[list(num_cols) + list(cat_cols)].copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cat_cols:
        out[c] = out[c].astype(object).map(lambda v: v if pd.isna(v) else str(v))
    return out


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category by how often it appears in the TRAINING data (share between 0 and 1).
    A category never seen in training gets 0. Used for columns with many distinct values,
    where one-hot would create too many columns."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_ = [X.iloc[:, i].value_counts(normalize=True).to_dict() for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        columns = [X.iloc[:, i].map(self.freq_[i]).fillna(0.0).to_numpy(dtype=float)
                   for i in range(X.shape[1])]
        return np.column_stack(columns)


def build_preprocessor(num_cols, low_card_cols, high_card_cols, scale_numeric):
    """Build the preprocessing step. It is fitted ONLY on the training part of each CV fold.
    numbers      -> fill missing with the median (+ standardise if the model needs it)
    few-category -> fill missing with the most frequent value, then one-hot encode
    many-category-> fill missing with the most frequent value, then frequency encode"""
    parts = []
    if num_cols:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        parts.append(("numeric", Pipeline(steps), list(num_cols)))
    if low_card_cols:
        parts.append(("onehot", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))]), list(low_card_cols)))
    if high_card_cols:
        parts.append(("frequency", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", FrequencyEncoder())]), list(high_card_cols)))
    # sparse_threshold=0 -> always return a normal dense table (some models cannot read sparse)
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)
