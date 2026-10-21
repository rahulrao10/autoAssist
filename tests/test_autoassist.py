"""Tests for AutoAssist. Run with:  python -m pytest -q"""
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from autoassist import AutoAssist, Config, run
from autoassist.decision_log import DecisionLog
from autoassist.planner import make_plan
from autoassist.portable import FrequencyEncoder, mean_and_ci
from autoassist.profiler import find_datetime_columns, profile_columns
from autoassist.task import detect_task

CFG = Config(n_estimators=20)          # small forests keep the tests fast


def make_df(n=200, kind="classification", seed=0):
    """Small messy table: ID, constant, mostly-missing, date, low- and high-cardinality categories."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "const": "same",
        "ghost": np.where(rng.random(n) < 0.9, np.nan, 1.0),
        "when": pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "color": rng.choice(["red", "green", "blue"], n),
        "code": ["K%02d" % k for k in rng.integers(0, 15, n)],
    })
    df["when"] = df["when"].dt.strftime("%Y-%m-%d")
    signal = df["x1"] + 0.5 * df["x2"] + (df["color"] == "red")
    df["y"] = (signal > signal.median()).map({True: "yes", False: "no"}) if kind == "classification" \
        else signal * 10 + rng.normal(size=n)
    df.loc[rng.random(n) < 0.05, "x1"] = np.nan
    return df


# ---------------------------------------------------------------- task detection
def test_task_detection():
    cfg = Config()
    assert detect_task(pd.Series(["a", "b", "a", "b"]), cfg)[0] == "binary"
    assert detect_task(pd.Series(["a", "b", "c"] * 5), cfg)[0] == "multiclass"
    assert detect_task(pd.Series([0, 1, 2, 1, 0] * 4), cfg)[0] == "multiclass"          # int, <= 10 values
    assert detect_task(pd.Series(range(50)), cfg)[0] == "regression"                      # int, > 10 values
    assert detect_task(pd.Series(np.linspace(0, 1, 30)), cfg)[0] == "regression"          # decimals


# ---------------------------------------------------------------- cleaning rules
def test_cleaning_rules_drop_the_right_columns():
    df = make_df().drop(columns=["y", "when"])
    plan = make_plan(profile_columns(df, CFG), CFG, DecisionLog(verbose=False))
    assert set(plan.dropped) == {"row_id", "const", "ghost"}
    assert plan.low_card == ["color"] and plan.high_card == ["code"]
    assert plan.numeric == ["x1", "x2"]


def test_switching_a_rule_off_keeps_the_column():
    df = make_df().drop(columns=["y", "when"])
    cfg = Config(drop_constant=False)
    plan = make_plan(profile_columns(df, cfg), cfg, DecisionLog(verbose=False))
    assert "const" not in plan.dropped


def test_date_detection():
    assert find_datetime_columns(make_df(), CFG) == ["when"]


# ---------------------------------------------------------------- preprocessing
def test_frequency_encoder_unseen_category_is_zero():
    enc = FrequencyEncoder().fit(np.array([["a"], ["a"], ["b"], ["a"]], dtype=object))
    out = enc.transform(np.array([["a"], ["b"], ["never_seen"]], dtype=object))
    assert out.ravel().tolist() == [0.75, 0.25, 0.0]


def test_confidence_interval_formula():
    mean, ci = mean_and_ci([0.8, 0.9, 0.85, 0.95, 0.9])
    assert abs(mean - 0.88) < 1e-9
    assert abs(ci - 2.776 * np.std([0.8, 0.9, 0.85, 0.95, 0.9], ddof=1) / np.sqrt(5)) < 1e-9


def test_preprocessing_lives_inside_the_pipeline():
    """Leakage guard: the preprocessor must be a step of the pipeline that cross_validate clones per fold."""
    ai = AutoAssist("y", CFG, verbose=False).fit(make_df())
    from autoassist.evaluation import make_pipeline
    pipe = make_pipeline(ai.results[0].spec, ai.plan)
    assert [name for name, _ in pipe.steps] == ["prep", "model"]


# ---------------------------------------------------------------- end to end
def test_fit_ranks_models_and_holdout_is_20_percent():
    ai = AutoAssist("y", CFG, verbose=False).fit(make_df())
    assert [r.rank for r in ai.results] == [1, 2, 3, 4, 5]
    assert ai.results[0].cv_mean >= ai.results[1].cv_mean
    assert len(ai.y_hold) == 40 and len(ai.y_dev) == 160
    assert ai.results[-1].rating == "Baseline" or any(r.rating == "Baseline" for r in ai.results)


def test_missing_target_column_gives_helpful_error():
    try:
        AutoAssist("prize", CFG, verbose=False).fit(make_df())
        raise AssertionError("should have failed")
    except ValueError as e:
        assert "not found" in str(e)


def _script_scores(script, cwd):
    out = subprocess.run([sys.executable, str(script)], cwd=cwd, capture_output=True, text=True, check=True).stdout
    cv = re.search(r"CV score.*?: ([\-\d.]+) \+\- ([\d.]+)", out)
    hold = re.search(r"Holdout score: ([\-\d.]+)", out)
    return float(cv.group(1)), float(cv.group(2)), float(hold.group(1))


def test_generated_script_reproduces_library_scores(tmp_path):
    for kind, model in [("classification", "random_forest"), ("regression", "ridge")]:
        csv = tmp_path / ("%s.csv" % kind)
        make_df(kind=kind).to_csv(csv, index=False)
        ai = run(str(csv), "y", model=model, script_path=str(tmp_path / ("%s_script.py" % kind)),
                 output_dir=str(tmp_path / ("out_" + kind)), interactive=False, config=CFG, verbose=False)
        assert "autoassist" not in "\n".join(l for l in ai.script_path.read_text().splitlines()
                                             if l.startswith(("import", "from")))
        cv, ci, hold = _script_scores(ai.script_path, tmp_path)
        r = ai.chosen
        assert abs(cv - r.cv_mean) < 1e-6 and abs(ci - r.cv_ci) < 1e-6 and abs(hold - r.holdout) < 1e-6
        assert (tmp_path / ("out_" + kind) / "decision_log.json").exists()
        assert (tmp_path / ("out_" + kind) / "report.html").exists()


def test_generated_script_can_predict(tmp_path):
    csv = tmp_path / "d.csv"
    df = make_df()
    df.to_csv(csv, index=False)
    ai = run(str(csv), "y", model=1, script_path=str(tmp_path / "s.py"), output_dir=str(tmp_path / "o"),
             interactive=False, config=CFG, verbose=False)
    _script_scores(ai.script_path, tmp_path)                       # trains + saves the model
    df.drop(columns=["y"]).head(10).to_csv(tmp_path / "new.csv", index=False)
    subprocess.run([sys.executable, str(ai.script_path), "--predict", "new.csv", "--out", "p.csv"],
                   cwd=tmp_path, check=True, capture_output=True)
    assert len(pd.read_csv(tmp_path / "p.csv")) == 10


def test_if_exists_modes(tmp_path):
    csv = tmp_path / "d.csv"
    make_df().to_csv(csv, index=False)
    target = tmp_path / "s.py"
    kw = dict(model=1, script_path=str(target), output_dir=str(tmp_path / "o"), interactive=False,
              config=CFG, verbose=False)
    run(str(csv), "y", **kw)                                                  # creates s.py
    p2 = run(str(csv), "y", **kw).script_path                                 # default -> new name
    assert p2.name == "s_1.py"
    size = target.stat().st_size
    run(str(csv), "y", if_exists="append", **kw)                              # marked section appended
    assert "AUTOASSIST GENERATED SECTION" in target.read_text() and target.stat().st_size > size
    run(str(csv), "y", if_exists="overwrite", **kw)
    assert "AUTOASSIST GENERATED SECTION (added" not in target.read_text()


def test_menu_accepts_typed_choice_and_default():
    answers = iter(["99", "2"])                                               # invalid first, then 2
    ai = AutoAssist("y", CFG, verbose=False, input_fn=lambda prompt: next(answers)).fit(make_df())
    assert ai.menu().rank == 2
    ai2 = AutoAssist("y", CFG, verbose=False, input_fn=lambda prompt: "").fit(make_df())
    assert ai2.menu().rank == 1                                               # Enter = top model


def test_quoted_script_path_is_accepted(tmp_path):
    """Windows 'Copy as path' adds quotes around the path; they must be ignored."""
    csv = tmp_path / "d.csv"
    make_df().to_csv(csv, index=False)
    quoted = '"%s"' % (tmp_path / "new_folder" / "model.py")
    ai = run(str(csv), "y", model=1, script_path=quoted, output_dir=str(tmp_path / "o"),
             interactive=False, config=CFG, verbose=False)
    assert ai.script_path == tmp_path / "new_folder" / "model.py" and ai.script_path.exists()
