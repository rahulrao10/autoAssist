"""models.py -- the list of models AutoAssist tries (scikit-learn only)."""
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class ModelSpec:
    key: str             # short id, e.g. "random_forest"
    display: str         # nice name for tables
    estimator: object    # un-fitted scikit-learn model
    scale_numeric: bool  # does this model need standardised numbers?
    import_line: str     # import statement used by the generated script
    why: str             # one sentence for the script comments


def get_models(task, cfg):
    """Return the model list for the task. The dummy baseline is always last."""
    s, n = cfg.seed, cfg.n_estimators
    if task == "regression":
        return [
            ModelSpec("ridge", "Ridge regression", Ridge(alpha=1.0), True,
                      "from sklearn.linear_model import Ridge",
                      "a linear model with a penalty on big weights; simple and fast, needs scaled numbers"),
            ModelSpec("random_forest", "Random forest", RandomForestRegressor(n_estimators=n, random_state=s), False,
                      "from sklearn.ensemble import RandomForestRegressor",
                      "many decision trees averaged; handles non-linear patterns, needs no scaling"),
            ModelSpec("extra_trees", "Extra trees", ExtraTreesRegressor(n_estimators=n, random_state=s), False,
                      "from sklearn.ensemble import ExtraTreesRegressor",
                      "like a random forest but with random split points; often faster and smoother"),
            ModelSpec("hist_gradient_boosting", "Hist. gradient boosting",
                      HistGradientBoostingRegressor(random_state=s), False,
                      "from sklearn.ensemble import HistGradientBoostingRegressor",
                      "trees built one after another, each fixing the previous one's errors"),
            ModelSpec("dummy_baseline", "Dummy baseline (mean)", DummyRegressor(strategy="mean"), False,
                      "from sklearn.dummy import DummyRegressor",
                      "always predicts the average target; any real model must beat it"),
        ]
    return [
        ModelSpec("logistic_regression", "Logistic regression", LogisticRegression(max_iter=2000), True,
                  "from sklearn.linear_model import LogisticRegression",
                  "a linear model giving class probabilities; simple and fast, needs scaled numbers"),
        ModelSpec("random_forest", "Random forest", RandomForestClassifier(n_estimators=n, random_state=s), False,
                  "from sklearn.ensemble import RandomForestClassifier",
                  "many decision trees voting together; handles non-linear patterns, needs no scaling"),
        ModelSpec("extra_trees", "Extra trees", ExtraTreesClassifier(n_estimators=n, random_state=s), False,
                  "from sklearn.ensemble import ExtraTreesClassifier",
                  "like a random forest but with random split points; often faster and smoother"),
        ModelSpec("hist_gradient_boosting", "Hist. gradient boosting",
                  HistGradientBoostingClassifier(random_state=s), False,
                  "from sklearn.ensemble import HistGradientBoostingClassifier",
                  "trees built one after another, each fixing the previous one's errors"),
        ModelSpec("dummy_baseline", "Dummy baseline (most frequent)", DummyClassifier(strategy="most_frequent"), False,
                  "from sklearn.dummy import DummyClassifier",
                  "always predicts the most common class; any real model must beat it"),
    ]
