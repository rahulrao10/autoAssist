"""config.py -- ALL thresholds and settings live here (one place to look).

Every number the library uses to make a decision is a field of `Config`.
Change a value by passing your own Config:  autoassist.run(..., config=Config(seed=1))
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- reproducibility -------------------------------------------------
    seed: int = 42                      # one fixed seed used everywhere

    # ---- data checks -----------------------------------------------------
    min_rows: int = 30                  # refuse to run on fewer rows than this
    max_classes: int = 50               # more distinct target values than this = not a sensible classification

    # ---- task detection --------------------------------------------------
    max_int_classes: int = 10           # integer target with <= this many distinct values -> classification

    # ---- validation scheme -----------------------------------------------
    test_size: float = 0.20             # holdout share, never used for any decision
    cv_folds: int = 5                   # k of k-fold cross-validation
    imbalance_threshold: float = 0.20   # smallest class < 20% of rows -> use macro-F1 instead of accuracy

    # ---- cleaning rules (thresholds) -------------------------------------
    datetime_min_parse_fraction: float = 0.90  # >=90% of values must parse as dates to call a column a date
    mostly_missing_fraction: float = 0.60      # drop a column if more than 60% of its values are missing
    id_unique_ratio: float = 0.95              # "almost every value is different" = at least 95% unique
    id_max_range_ratio: float = 1.5            # integer IDs are densely packed: (max-min+1)/n_unique <= 1.5
    onehot_max_categories: int = 10            # <=10 distinct values -> one-hot, otherwise frequency encoding

    # ---- cleaning rules (on/off switches, handy for an ablation study) ---
    expand_datetimes: bool = True
    drop_constant: bool = True
    drop_mostly_missing: bool = True
    drop_id_like: bool = True

    # ---- model settings --------------------------------------------------
    n_estimators: int = 100             # trees in random forest / extra trees

    # ---- rating rules ----------------------------------------------------
    min_gain_over_baseline: float = 0.05   # must beat the dummy baseline by at least this (in metric units)
    max_cv_holdout_gap: float = 0.05       # CV score minus holdout score above this -> "Fair" (2x -> "Poor")
    max_ci_halfwidth: float = 0.10         # 95% CI wider than +-0.10 -> scores are unstable -> "Fair"
    max_train_cv_gap: float = 0.50         # train score minus CV score above this -> memorising -> "Fair" (trees always score high on their own training data, so the limit is loose)

    def to_dict(self):
        return asdict(self)
