"""ingestion.py -- step 1: load the CSV, validate the target, drop bad rows, expand dates."""
import difflib
from pathlib import Path

import pandas as pd

from .portable import clean_target, expand_datetime_columns
from .profiler import find_datetime_columns


def load_table(data, log):
    """`data` is a CSV path or an already-loaded DataFrame. Returns (DataFrame, source_description)."""
    if isinstance(data, pd.DataFrame):
        df, source = data.copy(), "<DataFrame>"
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError("CSV file not found: %s" % path)
        df, source = pd.read_csv(path), str(path.resolve())
    log.add("load", "Loaded %d rows x %d columns from %s" % (len(df), df.shape[1], source))
    return df, source


def validate_target(df, target):
    """Stop early with a helpful message if the target column does not exist."""
    if target not in df.columns:
        close = difflib.get_close_matches(target, [str(c) for c in df.columns], n=3)
        hint = " Did you mean: %s?" % ", ".join(close) if close else ""
        raise ValueError("Target column '%s' not found. Columns are: %s.%s"
                         % (target, list(df.columns), hint))


def remove_missing_target_rows(df, target, cfg, log):
    """Rows without a target cannot be used for supervised learning -> drop them."""
    before = len(df)
    df = clean_target(df, target)
    dropped = before - len(df)
    if dropped:
        log.add("drop_missing_target", "Dropped %d row(s) with a missing target" % dropped,
                "supervised learning needs a known answer for every training row")
    else:
        log.add("drop_missing_target", "No rows with a missing target")
    if len(df) < cfg.min_rows:
        raise ValueError("Only %d usable rows (minimum is %d)." % (len(df), cfg.min_rows))
    return df


def expand_dates(features, cfg, log):
    """Find date-looking columns and replace each by year / month / day-of-week.
    Returns (expanded feature table, list of date column names)."""
    if not cfg.expand_datetimes:
        log.add("datetime", "Date expansion switched off in config")
        return features, []
    cols = find_datetime_columns(features, cfg)
    if not cols:
        log.add("datetime", "No date-like columns found")
        return features, []
    for c in cols:
        log.add("datetime", "Expanded date column '%s' into %s_year, %s_month, %s_dayofweek" % (c, c, c, c),
                "a raw date string is not a number a model can use; these three parts are")
    return expand_datetime_columns(features, cols), cols
