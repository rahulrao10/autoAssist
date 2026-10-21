"""core.py -- the AutoAssist class (fit / menu / export) and the one-call run() function."""
from pathlib import Path

from . import evaluation, ingestion, menu, planner, profiler, report, task
from .codegen import generate_script, write_script
from .config import Config
from .decision_log import DecisionLog
from .models import get_models
from .portable import select_and_cast

TOTAL_STEPS = 7


class AutoAssist:
    """Use step by step:   ai = AutoAssist("price"); ai.fit("data.csv"); ai.menu(); ai.export()"""

    def __init__(self, target, config=None, verbose=True, input_fn=input):
        self.target = target
        self.config = config or Config()
        self.verbose = verbose
        self.input_fn = input_fn
        self.log = DecisionLog(verbose)
        self.results = []
        self.chosen = None
        self.script_path = None

    # ------------------------------------------------------------------ steps 1-5
    def fit(self, data):
        cfg, log = self.config, self.log

        # ---- 1. LOAD + VALIDATE
        log.section(1, TOTAL_STEPS, "LOAD + VALIDATE")
        df, self.source = ingestion.load_table(data, log)
        ingestion.validate_target(df, self.target)
        df = ingestion.remove_missing_target_rows(df, self.target, cfg, log)
        y = df[self.target]
        features, self.datetime_columns = ingestion.expand_dates(df.drop(columns=[self.target]), cfg, log)

        # ---- 2. TASK DETECTION
        log.section(2, TOTAL_STEPS, "TASK DETECTION")
        self.task, self.task_reason = task.detect_task(y, cfg)
        log.add("task", "Task: %s" % self.task, self.task_reason)

        # ---- 3. DIAGNOSIS + CLEANING (holdout is split off FIRST so it cannot influence the rules)
        log.section(3, TOTAL_STEPS, "DATA DIAGNOSIS + CLEANING")
        X_dev_raw, X_hold_raw, self.y_dev, self.y_hold, self.stratified = \
            evaluation.split_holdout(features, y, self.task, cfg, log)
        profiles = profiler.profile_columns(X_dev_raw, cfg)
        self.plan = planner.make_plan(profiles, cfg, log)
        self.plan.datetime_columns = self.datetime_columns
        X_all = select_and_cast(features, self.plan.numeric, self.plan.categorical)
        self.X_dev, self.X_hold = X_all.loc[X_dev_raw.index], X_all.loc[X_hold_raw.index]
        self._print_summary(len(df) , features)

        # ---- 4. TRAINING + SCORING
        log.section(4, TOTAL_STEPS, "MODEL TRAINING + SCORING")
        self.scoring, self.metric_label = evaluation.choose_metric(self.task, self.y_dev, cfg, log)
        cv, self.cv_folds = evaluation.make_cv(self.task, self.y_dev, cfg, log)
        specs = get_models(self.task, cfg)
        self.results = evaluation.evaluate_models(specs, self.plan, self.X_dev, self.y_dev,
                                                  self.X_hold, self.y_hold, self.scoring, cv, log)
        if not self.results:
            raise RuntimeError("No model could be trained.")

        # ---- 5. MODEL-LEVEL RATING
        log.section(5, TOTAL_STEPS, "MODEL RATING")
        evaluation.rate_models(self.results, cfg)
        log.say(menu.format_table(self.results, self.metric_label))
        return self

    def _print_summary(self, n_rows, features):
        """Plain-language list of what changed in the user's data."""
        p, log = self.plan, self.log
        log.say("")
        log.say("  WHAT CHANGED IN YOUR DATA")
        log.say("  - rows used: %d" % n_rows)
        log.say("  - columns dropped: %s" % (list(p.dropped) or "none"))
        log.say("  - date columns expanded: %s" % (p.datetime_columns or "none"))
        log.say("  - kept: %d numeric, %d one-hot, %d frequency-encoded" % (len(p.numeric), len(p.low_card), len(p.high_card)))
        log.say("  - values were imputed (median / most frequent) inside each CV fold; nothing else was changed")

    # ------------------------------------------------------------------ step 6
    def menu(self, choice=None, interactive=True):
        """Show the ranked table and pick a model. Returns the chosen ModelResult."""
        self.log.section(6, TOTAL_STEPS, "MODEL MENU")
        if choice is None and interactive:
            self.log.say(menu.format_menu(self.results))
        self.chosen = menu.choose_model(self.results, choice, self.input_fn, interactive, self.log.say)
        return self.chosen

    # ------------------------------------------------------------------ step 7
    def export(self, path=None, if_exists=None, interactive=True):
        """Write the standalone script. Returns the path it was written to."""
        if self.chosen is None:
            self.chosen = self.results[0]
        self.log.section(7, TOTAL_STEPS, "CODE GENERATION")
        default = Path("autoassist_output") / ("%s_pipeline.py" % self.chosen.spec.key)
        if path is None:
            path = default
            if interactive:
                typed = self.input_fn("Where should the script be saved? [%s]: " % default).strip()
                path = typed or default
        code = generate_script(self)
        self.script_path = write_script(code, path, if_exists, interactive, self.input_fn, self.log.say)
        return self.script_path

    # ------------------------------------------------------------------ step 8
    def save_reports(self, output_dir="autoassist_output"):
        jp, hp = report.save_reports(self, output_dir)
        self.log.say("  Saved decision log: %s" % jp)
        self.log.say("  Saved report:       %s" % hp)
        return jp, hp


def run(data, target, model=None, script_path=None, output_dir="autoassist_output",
        if_exists=None, interactive=True, config=None, verbose=True, input_fn=input):
    """The whole flow in one call.
    model        -> rank number or key (e.g. 2 or 'random_forest'); None = ask (or top model if not interactive)
    script_path  -> where to write the script; None = ask (or default if not interactive)
    interactive  -> False means never ask anything"""
    ai = AutoAssist(target, config, verbose, input_fn)
    ai.fit(data)
    ai.menu(choice=model, interactive=interactive)
    ai.export(script_path, if_exists, interactive)
    ai.log.section(8, TOTAL_STEPS + 1, "SAVE LOG + REPORT")
    ai.save_reports(output_dir)
    return ai
