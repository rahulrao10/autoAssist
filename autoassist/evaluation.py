"""evaluation.py -- steps 4 and 5: split, cross-validate, score on the holdout, rate each model.

Anti-leakage design:
  * The 20% HOLDOUT is split off first and only used at the very end to REPORT a score.
  * Preprocessing sits INSIDE a scikit-learn Pipeline, so in every CV fold it is
    fitted on that fold's training rows only (never on the validation rows).
  * Models are RANKED by CV score only (never by holdout).
"""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .portable import build_preprocessor, mean_and_ci


@dataclass
class ModelResult:
    spec: object
    cv_scores: list = field(default_factory=list)
    cv_mean: float = float("nan")
    cv_ci: float = 0.0
    train_mean: float = float("nan")
    holdout: float = float("nan")
    gap: float = 0.0            # CV score - holdout score (positive = holdout is worse)
    train_cv_gap: float = 0.0   # train score - CV score (big = memorising)
    gain: float = 0.0           # CV score - dummy baseline CV score
    rating: str = ""
    rating_reason: str = ""
    seconds: float = 0.0
    rank: int = 0

    @property
    def key(self):
        return self.spec.key


# ----------------------------------------------------------------------------- split
def split_holdout(X, y, task, cfg, log):
    """Split off the holdout. Classification: keep class proportions equal (stratify)."""
    stratify = None
    if task != "regression":
        if int(y.value_counts().min()) < 2:
            raise ValueError("A class has only 1 row; remove or merge rare classes first.")
        stratify = y
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify)
    log.add("split", "Holdout split: %d rows for development, %d rows (%.0f%%) locked away as holdout%s"
            % (len(X_dev), len(X_hold), 100 * cfg.test_size, ", stratified" if stratify is not None else ""),
            "the holdout is never used for any decision, only for the final honest score")
    return X_dev, X_hold, y_dev, y_hold, stratify is not None


# ----------------------------------------------------------------------------- metric
def choose_metric(task, y_dev, cfg, log):
    """R2 for regression; accuracy for balanced classes; macro-F1 if smallest class < 20%.
    Returns (scoring_name_for_sklearn, label)."""
    if task == "regression":
        log.add("metric", "Metric: R2", "R2 = share of the target's variance the model explains (1 = perfect, 0 = as good as the mean)")
        return "r2", "R2"
    smallest = float(y_dev.value_counts(normalize=True).min())
    if smallest < cfg.imbalance_threshold:
        log.add("metric", "Metric: macro-F1",
                "smallest class is only %.1f%% of rows (< %.0f%%), so plain accuracy would hide mistakes on it; "
                "macro-F1 averages the F1 of every class equally" % (100 * smallest, 100 * cfg.imbalance_threshold))
        return "f1_macro", "macro-F1"
    log.add("metric", "Metric: accuracy", "classes are balanced (smallest class %.1f%% >= %.0f%%)"
            % (100 * smallest, 100 * cfg.imbalance_threshold))
    return "accuracy", "accuracy"


# ----------------------------------------------------------------------------- CV
def make_cv(task, y_dev, cfg, log):
    """Build the cross-validation splitter (stratified for classification)."""
    k = cfg.cv_folds
    if task == "regression":
        log.add("cv", "%d-fold cross-validation (shuffled, seed %d)" % (k, cfg.seed),
                "average over several train/validation splits instead of trusting one lucky split")
        return KFold(n_splits=k, shuffle=True, random_state=cfg.seed), k
    smallest = int(y_dev.value_counts().min())
    if smallest < 2:
        raise ValueError("A class has fewer than 2 development rows; cannot cross-validate.")
    if smallest < k:
        k = smallest
        log.add("cv_folds_reduced", "Folds reduced to %d" % k, "the smallest class has only %d development rows" % smallest)
    log.add("cv", "%d-fold STRATIFIED cross-validation (shuffled, seed %d)" % (k, cfg.seed),
            "stratified = every fold keeps the class proportions")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed), k


def make_pipeline(spec, plan):
    """Preprocessing + model in ONE pipeline (this is what prevents leakage)."""
    prep = build_preprocessor(plan.numeric, plan.low_card, plan.high_card, spec.scale_numeric)
    return Pipeline([("prep", prep), ("model", spec.estimator)])


# ----------------------------------------------------------------------------- run all models
def evaluate_models(specs, plan, X_dev, y_dev, X_hold, y_hold, scoring, cv, log):
    results = []
    scorer = get_scorer(scoring)
    for spec in specs:
        t0 = time.perf_counter()
        pipe = make_pipeline(spec, plan)
        try:
            out = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=scoring,
                                 return_train_score=True, n_jobs=1, error_score="raise")
            mean, ci = mean_and_ci(out["test_score"])
            pipe.fit(X_dev, y_dev)                       # fit on ALL development rows ...
            holdout = float(scorer(pipe, X_hold, y_hold))  # ... and score on the untouched holdout
        except Exception as exc:                          # one broken model must not stop the run
            log.add("model_failed", "Model %s failed and was skipped: %s" % (spec.display, exc))
            continue
        r = ModelResult(spec=spec, cv_scores=[float(s) for s in out["test_score"]], cv_mean=mean, cv_ci=ci,
                        train_mean=float(np.mean(out["train_score"])), holdout=holdout,
                        seconds=time.perf_counter() - t0)
        results.append(r)
        log.say("  trained %-32s CV %.4f +- %.4f   (%.1fs)" % (spec.display, mean, ci, r.seconds))
    return results


# ----------------------------------------------------------------------------- rating
def rate_models(results, cfg):
    """Fill in gap / gain / rating for every result and sort them best-first (by CV only)."""
    base = next((r for r in results if r.key == "dummy_baseline"), None)
    for r in results:
        r.gap = r.cv_mean - r.holdout
        r.train_cv_gap = r.train_mean - r.cv_mean
        r.gain = r.cv_mean - base.cv_mean if base else 0.0
        if r.key == "dummy_baseline":
            r.rating, r.rating_reason = "Baseline", "reference point: every real model must beat this"
            continue
        poor, fair = [], []
        if r.gain < cfg.min_gain_over_baseline:
            poor.append("beats the baseline by only %+.3f (needs >= %.2f)" % (r.gain, cfg.min_gain_over_baseline))
        if base and r.holdout <= base.holdout:
            poor.append("holdout score is not better than the baseline's")
        if r.gap > 2 * cfg.max_cv_holdout_gap:
            poor.append("holdout is %.3f below CV (large gap)" % r.gap)
        elif r.gap > cfg.max_cv_holdout_gap:
            fair.append("holdout is %.3f below CV (possible overfitting)" % r.gap)
        if r.cv_ci > cfg.max_ci_halfwidth:
            fair.append("unstable across folds (CI +-%.3f)" % r.cv_ci)
        if r.train_cv_gap > cfg.max_train_cv_gap:
            fair.append("train score is %.2f above CV (memorising training data)" % r.train_cv_gap)
        if poor:
            r.rating, r.rating_reason = "Poor", "; ".join(poor + fair)
        elif fair:
            r.rating, r.rating_reason = "Fair", "; ".join(fair)
        else:
            r.rating, r.rating_reason = "Good", "beats baseline by %+.3f, CV and holdout agree, stable folds" % r.gain
    results.sort(key=lambda r: r.cv_mean, reverse=True)
    for i, r in enumerate(results, 1):
        r.rank = i
    return results
