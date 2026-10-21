"""planner.py -- step 3: the cleaning RULES. Turns column profiles into a cleaning plan.

Rules (checked in this order for every column):
  1. constant column            -> drop   (all rows equal: carries no information)
  2. mostly missing (> 60%)     -> drop   (too little real data to learn from)
  3. ID-like                    -> drop   (an ID identifies rows, it does not explain the target)
  4. otherwise keep it:
       numeric                  -> median imputation
       categorical, <= 10 values-> most-frequent imputation + one-hot
       categorical, > 10 values -> most-frequent imputation + frequency encoding
NO new features are created. Every decision goes into the log with its reason.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Plan:
    dropped: dict = field(default_factory=dict)     # column -> reason
    numeric: list = field(default_factory=list)
    low_card: list = field(default_factory=list)    # one-hot
    high_card: list = field(default_factory=list)   # frequency encoding
    datetime_columns: list = field(default_factory=list)

    @property
    def categorical(self):
        return self.low_card + self.high_card


def _name_says_id(name):
    """True if the column name contains the word 'id' / 'uuid' / 'index' (e.g. patient_id, ID, row_index)."""
    tokens = re.split(r"[^a-z0-9]+", str(name).lower())
    return any(t in ("id", "uuid", "index") for t in tokens)


def _id_like_reason(p, cfg):
    """Return a reason string if the column looks like an ID, else ''."""
    if p.unique_ratio < cfg.id_unique_ratio:
        return ""
    if p.kind == "categorical":
        return "text column where %.0f%% of values are unique (looks like an ID or free text)" % (100 * p.unique_ratio)
    if p.is_integer and _name_says_id(p.name):
        return "integer column named like an ID with %.0f%% unique values" % (100 * p.unique_ratio)
    if p.is_integer and 0 < p.range_ratio <= cfg.id_max_range_ratio:
        return "integer column of almost-consecutive unique numbers (a row counter / ID)"
    return ""


def make_plan(profiles, cfg, log):
    plan = Plan()
    for p in profiles:
        if cfg.drop_constant and p.n_unique <= 1:
            _drop(plan, log, p.name, "constant column (%d distinct value)" % p.n_unique)
            continue
        if cfg.drop_mostly_missing and p.missing_fraction > cfg.mostly_missing_fraction:
            _drop(plan, log, p.name, "%.0f%% of values are missing (limit %.0f%%)"
                  % (100 * p.missing_fraction, 100 * cfg.mostly_missing_fraction))
            continue
        id_reason = _id_like_reason(p, cfg) if cfg.drop_id_like else ""
        if id_reason:
            _drop(plan, log, p.name, id_reason)
            continue
        if p.kind == "numeric":
            plan.numeric.append(p.name)
        elif p.n_unique <= cfg.onehot_max_categories:
            plan.low_card.append(p.name)
        else:
            plan.high_card.append(p.name)

    # Log the keep-decisions per group (one line each, with the reason).
    n_missing_num = sum(1 for p in profiles if p.name in plan.numeric and p.n_missing)
    n_missing_cat = sum(1 for p in profiles if p.name in plan.categorical and p.n_missing)
    if plan.numeric:
        log.add("impute_numeric", "Numeric columns kept (%d): missing values -> median; %d column(s) had gaps"
                % (len(plan.numeric), n_missing_num),
                "the median is robust to outliers and keeps the original value distribution")
    if plan.low_card:
        log.add("encode_onehot", "One-hot encoded (%d): %s" % (len(plan.low_card), plan.low_card),
                "<= %d distinct values, so one 0/1 column per value stays small; missing -> most frequent value"
                % cfg.onehot_max_categories)
    if plan.high_card:
        log.add("encode_frequency", "Frequency encoded (%d): %s" % (len(plan.high_card), plan.high_card),
                "> %d distinct values, one-hot would create too many columns; each value becomes its share of rows"
                % cfg.onehot_max_categories)
    if n_missing_cat:
        log.add("impute_categorical", "%d categorical column(s) had gaps -> most frequent value" % n_missing_cat,
                "simple, and keeps the column's own categories (no invented category)")
    if not plan.numeric and not plan.categorical:
        raise ValueError("Every column was dropped by the cleaning rules; nothing left to learn from.")
    log.add("no_feature_engineering", "No new features were created",
            "AutoAssist only cleans and encodes; your columns stay as original as possible")
    return plan


def _drop(plan, log, name, reason):
    plan.dropped[name] = reason
    log.add("drop_column", "Dropped column '%s'" % name, reason)
