# AutoAssist

A small AutoML-style **Python library** for **supervised learning on tabular data** (a CSV with one column to predict).
It loads and cleans your data, trains several scikit-learn models, rates them honestly, lets you pick one in the
terminal, and writes a **standalone Python script** for that model.

```python
import autoassist
autoassist.run("data.csv", target="price")
```

## Install and run (Windows)

Open **Command Prompt** (cmd.exe) in the project folder. (PowerShell may block the venv activation script; see the note below.)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
python demo.py
python -m pytest -q
```

**If you prefer PowerShell** and `.venv\Scripts\Activate.ps1` says "running scripts is disabled", run this first
(it only affects the current window):

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` and `python` commands.

## Ways to use it

| How | Command |
|---|---|
| One call (asks you questions) | `autoassist.run("data.csv", target="price")` |
| Terminal | `python -m autoassist data.csv price` |
| No questions (for tests/scripts) | `autoassist.run("data.csv", "price", model="random_forest", script_path="out/pipe.py", interactive=False)` |
| Step by step | `ai = autoassist.AutoAssist("price"); ai.fit("data.csv"); ai.menu(); ai.export()` |
| Terminal, no questions | `python -m autoassist data.csv price --model 1 --script out/pipe.py --if-exists overwrite` |

`model` can be the rank number (`1`) or the key (`ridge`, `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `dummy_baseline`).
`if_exists` can be `"new"` (default), `"append"` or `"overwrite"`.

Outputs: your script, `autoassist_output/decision_log.json` (every decision with its reason) and `autoassist_output/report.html`.

Run the generated script on its own (needs only pandas, numpy, scikit-learn, joblib):

```
python my_script.py                                          # trains, prints CV + holdout scores, saves the model
python my_script.py --predict new_rows.csv --out preds.csv   # predicts on new data
```

## What happens, step by step

1. **Load + validate** (`ingestion.py`): read CSV, check the target exists, drop rows with a missing target, expand date columns into year / month / day-of-week.
2. **Task detection** (`task.py`): text target -> classification; integer target with <= 10 distinct values -> classification; otherwise regression. 2 classes = binary, more = multiclass.
3. **Diagnosis + cleaning** (`profiler.py`, `planner.py`): drop constant, ID-like and mostly-missing (> 60%) columns; impute (median for numbers, most-frequent for categories); one-hot for <= 10 categories, frequency encoding above that; scale numbers only for the linear model. **No new features are created.**
4. **Training + scoring** (`models.py`, `evaluation.py`): 5 models incl. a dummy baseline. 5-fold CV, mean +- 95% CI, plus a 20% holdout. Metric: R2 (regression), accuracy (balanced classes), macro-F1 (smallest class < 20%).
5. **Rating**: per model: CV score, holdout score, gap, gain over the dummy baseline -> Good / Fair / Poor with the reason.
6. **Menu** (`menu.py`): ranked table, you type a number (Enter = top model).
7. **Code generation** (`codegen.py`): complete standalone script. If the file exists you choose new name / append (marked section) / overwrite.
8. **Reports** (`report.py`): JSON log + HTML report.

## Terms explained

- **Cross-validation (CV)**: split the training data into 5 parts; train on 4, test on 1, repeat 5 times, average the 5 scores.
- **95% confidence interval (CI)**: the +- after the mean. A narrow CI means the 5 fold scores agree.
- **Holdout**: 20% of rows locked away at the very start. Used once at the end for an honest score. It is never used for ranking or cleaning decisions.
- **Data leakage**: information from the test rows sneaking into training (e.g. computing a median on all rows). Here preprocessing is part of the scikit-learn `Pipeline`, so it is re-fitted inside every fold on training rows only.
- **Dummy baseline**: a model that ignores the features (predicts the mean / most frequent class). Real models must beat it.
- **Macro-F1**: F1 score computed per class, then averaged so small classes count as much as big ones.
- **Stratified**: every split keeps the same class proportions.

## Design notes (useful when explaining it)

- **`portable.py`** holds the cleaning/preprocessing code. The library imports it, and the code generator pastes it word for word into the generated script. That is why the script reproduces the library's scores (the tests check this to 6 decimals).
- **All thresholds are in `config.py`.** Seed is fixed to 42.
- Cleaning rules are decided on the 80% development rows only; the holdout cannot influence them.
- Models are **ranked by CV score only**. The holdout score and gap are shown in the rating as a final check.
- The saved model in the generated script is re-trained on **all** rows (the printed scores come from the CV/holdout evaluation before that).
- The rating thresholds (`min_gain_over_baseline`, `max_cv_holdout_gap`, `max_ci_halfwidth`, `max_train_cv_gap`) are reasonable defaults I chose, not scientific constants. Change them in `Config`.

## Layout

```
autoassist/  config.py  decision_log.py  portable.py  ingestion.py  task.py  profiler.py  planner.py
             models.py  evaluation.py  menu.py  codegen.py  report.py  core.py  cli.py  __main__.py
tests/test_autoassist.py     demo.py     pyproject.toml     requirements.txt
```
