"""codegen.py -- step 7: write a COMPLETE, STANDALONE Python script for the chosen model.

The script needs only pandas, numpy, scikit-learn and joblib. The cleaning code inside it is
portable.py copied word for word, so it behaves exactly like the library.
Comments in the script come from the decision log.
"""
import datetime
import inspect
import textwrap
from pathlib import Path

from . import portable


def _comment(text, indent=""):
    """Turn a sentence into wrapped '# ...' comment lines."""
    return "\n".join(textwrap.wrap(text, 92, initial_indent=indent + "# ", subsequent_indent=indent + "# "))


def _portable_source():
    """Source of portable.py without its docstring (everything from the first import on)."""
    src = inspect.getsource(portable)
    return src[src.index("import warnings"):]


def generate_script(ai):
    """Build the script text for ai.chosen (an AutoAssist object that has been fitted)."""
    r, plan, log, cfg = ai.chosen, ai.plan, ai.log, ai.config
    spec = r.spec
    scaling_reason = ("this model is sensitive to the size of numbers, so numeric columns are standardised"
                      if spec.scale_numeric else "tree models do not care about the scale of numbers, so no scaling")
    dropped = "\n".join("#   - %s: %s" % (c, why) for c, why in plan.dropped.items()) or "#   (none)"
    source = ai.source if ai.source != "<DataFrame>" else "data.csv"

    script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoAssist generated pipeline: @@MODEL@@
Generated on @@DATE@@

What this script does, step by step
  1. loads the CSV and removes rows without a target
  2. cleans/preprocesses the columns (same rules AutoAssist used)
  3. holds out 20% of the rows, then cross-validates the model on the other 80%
  4. scores the model on the holdout and prints the results
  5. re-trains on ALL rows and saves the model with joblib
  6. can predict on a new CSV

Run it:     python @@SCRIPT_HINT@@
Predict:    python @@SCRIPT_HINT@@ --predict new_data.csv --out predictions.csv
Needs only: pandas, numpy, scikit-learn, joblib (no AutoAssist).
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import @@CV_IMPORT@@, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
@@MODEL_IMPORT@@

# =============================================================================
# SETTINGS
# =============================================================================
DATA_PATH = @@DATA_PATH@@          # change this to point to your CSV (or use --data)
TARGET = @@TARGET@@                # the column we want to predict
TASK = @@TASK@@
@@TASK_COMMENT@@
SCORING = @@SCORING@@              # metric: @@METRIC_LABEL@@
@@METRIC_COMMENT@@
SEED = @@SEED@@                    # fixed seed -> same results every run
TEST_SIZE = @@TEST_SIZE@@          # share of rows locked away as the holdout
CV_FOLDS = @@CV_FOLDS@@
STRATIFY = @@STRATIFY@@            # keep class proportions in every split (classification only)
MODEL_PATH = @@MODEL_PATH@@        # where the trained model is saved

# =============================================================================
# CLEANING PLAN (found by AutoAssist on the development rows only)
# =============================================================================
# Columns that were removed, and why:
@@DROPPED@@
DATETIME_COLUMNS = @@DATETIME@@    # expanded into year / month / day-of-week
@@DATETIME_COMMENT@@
NUMERIC_COLUMNS = @@NUMERIC@@
@@NUMERIC_COMMENT@@
ONEHOT_COLUMNS = @@ONEHOT@@
@@ONEHOT_COMMENT@@
FREQUENCY_COLUMNS = @@FREQ@@
@@FREQ_COMMENT@@
# Scaling: @@SCALING_REASON@@.
SCALE_NUMERIC = @@SCALE@@
# No feature engineering: no new ratio / interaction columns are created.

# =============================================================================
# CLEANING / PREPROCESSING CODE (identical to the code AutoAssist ran)
# =============================================================================
@@PORTABLE@@

# =============================================================================
# THE MODEL: @@MODEL@@
# =============================================================================
@@MODEL_COMMENT@@
def build_model():
    return @@MODEL_CODE@@


def build_pipeline():
    """Preprocessing + model in ONE pipeline. Inside cross-validation the preprocessing is fitted on
    each fold's training rows only, so nothing leaks from the validation rows."""
    prep = build_preprocessor(NUMERIC_COLUMNS, ONEHOT_COLUMNS, FREQUENCY_COLUMNS, SCALE_NUMERIC)
    return Pipeline([("prep", prep), ("model", build_model())])


def prepare_features(df):
    """Raw table (without the target) -> clean table with exactly the columns the model expects."""
    df = expand_datetime_columns(df, DATETIME_COLUMNS)
    return select_and_cast(df, NUMERIC_COLUMNS, ONEHOT_COLUMNS + FREQUENCY_COLUMNS)


# =============================================================================
# TRAIN + EVALUATE + SAVE
# =============================================================================
def train(data_path):
    df = clean_target(pd.read_csv(data_path), TARGET)        # drop rows with missing target
    y = df[TARGET]
    X = prepare_features(df.drop(columns=[TARGET]))
    print("Loaded %d rows, using %d feature columns" % (len(X), X.shape[1]))

    # (a) lock away the holdout; it is only used once, for the final score
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y if STRATIFY else None)

    # (b) k-fold cross-validation on the development rows
    if STRATIFY:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    out = cross_validate(build_pipeline(), X_dev, y_dev, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    mean, ci = mean_and_ci(out["test_score"])
    print("CV score (mean +- 95%% CI): %.6f +- %.6f" % (mean, ci))
    print("Train score (CV average): %.6f" % np.mean(out["train_score"]))

    # (c) fit on all development rows, score on the holdout
    pipe = build_pipeline().fit(X_dev, y_dev)
    holdout = get_scorer(SCORING)(pipe, X_hold, y_hold)
    print("Holdout score: %.6f" % holdout)

    # (d) final model: re-train on ALL rows (more data = better model) and save it
    final = build_pipeline().fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print("Saved model to", MODEL_PATH)


def predict(data_path, out_path):
    model = joblib.load(MODEL_PATH)
    new = pd.read_csv(data_path)
    new = new.drop(columns=[TARGET], errors="ignore")      # the target is not needed for predicting
    result = new.copy()
    result["prediction"] = model.predict(prepare_features(new))
    result.to_csv(out_path, index=False)
    print("Wrote %d predictions to %s" % (len(result), out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAssist generated pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="training CSV")
    parser.add_argument("--predict", help="CSV with new rows to predict")
    parser.add_argument("--out", default="predictions.csv", help="where to write predictions")
    args = parser.parse_args()
    if args.predict:
        predict(args.predict, args.out)
    else:
        train(args.data)
'''
    cv_import = "KFold" if ai.task == "regression" else "StratifiedKFold"
    # the template mentions both splitters; import both so the script is valid either way
    cv_import = "KFold, StratifiedKFold"
    model_path = "autoassist_%s_model.joblib" % spec.key

    fill = {
        "MODEL": spec.display, "DATE": datetime.date.today().isoformat(),
        "SCRIPT_HINT": "this_script.py", "CV_IMPORT": cv_import, "MODEL_IMPORT": spec.import_line,
        "DATA_PATH": repr(source), "TARGET": repr(ai.target), "TASK": repr(ai.task),
        "TASK_COMMENT": _comment("Why this task: " + ai.task_reason + "."),
        "SCORING": repr(ai.scoring), "METRIC_LABEL": ai.metric_label,
        "METRIC_COMMENT": _comment(log.reason("metric")),
        "SEED": repr(cfg.seed), "TEST_SIZE": repr(cfg.test_size), "CV_FOLDS": repr(ai.cv_folds),
        "STRATIFY": repr(ai.stratified), "MODEL_PATH": repr(model_path),
        "DROPPED": dropped,
        "DATETIME": repr(plan.datetime_columns), "DATETIME_COMMENT": _comment(
            "A raw date string is not a number a model can use; year, month and day-of-week are."),
        "NUMERIC": repr(plan.numeric), "NUMERIC_COMMENT": _comment("Numbers: " + log.reason("impute_numeric", "kept as they are") + "."),
        "ONEHOT": repr(plan.low_card), "ONEHOT_COMMENT": _comment("One-hot: " + log.reason("encode_onehot", "no such columns") + "."),
        "FREQ": repr(plan.high_card), "FREQ_COMMENT": _comment("Frequency: " + log.reason("encode_frequency", "no such columns") + "."),
        "SCALING_REASON": scaling_reason, "SCALE": repr(spec.scale_numeric),
        "PORTABLE": _portable_source().rstrip(),
        "MODEL_COMMENT": _comment("Why this model: " + spec.why + "."),
        "MODEL_CODE": repr(spec.estimator),
    }
    for key, value in fill.items():
        script = script.replace("@@%s@@" % key, value)
    return script


def write_script(code, path, if_exists=None, interactive=True, input_fn=input, say=print):
    """Write `code` to `path`. If the file exists: new name (default) / append / overwrite."""
    # Windows "Copy as path" wraps paths in quotes; remove them (and stray spaces) so they are not taken literally.
    path = Path(str(path).strip().strip("\"'").strip()).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = if_exists
        if mode is None:
            mode = "new"
            if interactive:
                answer = input_fn("'%s' already exists. [n]ew file / [a]ppend / [o]verwrite? [n]: " % path).strip().lower()
                mode = {"a": "append", "o": "overwrite"}.get(answer[:1], "new")
        if mode == "overwrite":
            path.write_text(code, encoding="utf-8")
            say("  Overwrote %s" % path)
        elif mode == "append":
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            section = ("\n\n# " + "=" * 76 + "\n# >>> AUTOASSIST GENERATED SECTION (added %s) >>>\n"
                       "# Note: this section is a full script with its own main block.\n# " % stamp
                       + "=" * 76 + "\n" + code + "\n# <<< END AUTOASSIST GENERATED SECTION <<<\n")
            with open(path, "a", encoding="utf-8") as f:
                f.write(section)
            say("  Appended a marked section to %s" % path)
        elif mode == "new":
            n = 1
            while path.with_name("%s_%d%s" % (path.stem, n, path.suffix)).exists():
                n += 1
            path = path.with_name("%s_%d%s" % (path.stem, n, path.suffix))
            path.write_text(code, encoding="utf-8")
            say("  File existed, saved under a new name: %s" % path)
        else:
            raise ValueError("if_exists must be 'new', 'append' or 'overwrite'")
    else:
        path.write_text(code, encoding="utf-8")
        say("  Wrote %s" % path)
    return path
