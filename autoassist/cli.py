"""cli.py -- command line:  python -m autoassist data.csv target"""
import argparse

from .config import Config
from .core import run


def main(argv=None):
    p = argparse.ArgumentParser(prog="autoassist", description="Lightweight AutoML for tabular supervised learning")
    p.add_argument("csv", help="path to the CSV file")
    p.add_argument("target", help="name of the column to predict")
    p.add_argument("--model", help="rank number or key of the model (skips the menu)")
    p.add_argument("--script", dest="script_path", help="where to write the generated script (skips the question)")
    p.add_argument("--if-exists", choices=["new", "append", "overwrite"], help="what to do if the script file exists")
    p.add_argument("--output-dir", default="autoassist_output", help="folder for decision_log.json and report.html")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-interactive", action="store_true", help="never ask questions; use defaults")
    a = p.parse_args(argv)
    run(a.csv, a.target, model=a.model, script_path=a.script_path, output_dir=a.output_dir,
        if_exists=a.if_exists, interactive=not a.no_interactive, config=Config(seed=a.seed))


if __name__ == "__main__":
    main()
