"""demo.py -- creates two demo CSVs in demo_data/ and runs AutoAssist on both.

    python demo.py                 runs both with no questions (fast)
    python demo.py --interactive   lets you use the menu and type the script path yourself
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

import autoassist

OUT = Path("demo_data")


def make_messy_classification():
    """Breast-cancer data made deliberately messy so every cleaning rule has something to do."""
    rng = np.random.default_rng(0)
    bc = load_breast_cancer(as_frame=True)
    cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "worst radius", "worst concavity"]
    df = bc.data[cols].copy()
    df.columns = [c.replace(" ", "_") for c in cols]
    df["diagnosis"] = bc.target.map({0: "malignant", 1: "benign"})          # text target
    # make it imbalanced: keep only 50 malignant rows (~12%) -> macro-F1 will be chosen
    keep = df.index[df["diagnosis"] == "benign"].tolist() + \
        rng.choice(df.index[df["diagnosis"] == "malignant"], 50, replace=False).tolist()
    df = df.loc[keep].sample(frac=1, random_state=0).reset_index(drop=True)
    n = len(df)
    df["patient_id"] = rng.permutation(np.arange(1000, 1000 + n))           # ID column
    df["study_site"] = "Site A"                                             # constant column
    df["lab_notes_score"] = np.where(rng.random(n) < 0.85, np.nan, rng.normal(size=n))  # 85% missing
    start = pd.Timestamp("2020-01-01")
    df["exam_date"] = [(start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(0, 1500, n)]
    df["region"] = rng.choice(["north", "south", "east", "west"], n)        # low-cardinality category
    df["clinic_code"] = ["C%02d" % c for c in rng.integers(0, 30, n)]       # high-cardinality category
    for c in ["mean_radius", "mean_texture", "worst_concavity"]:            # random gaps in numbers
        df.loc[rng.random(n) < 0.08, c] = np.nan
    df.loc[rng.random(n) < 0.05, "region"] = np.nan                         # gaps in a category
    df.loc[rng.choice(n, 5, replace=False), "diagnosis"] = np.nan           # rows with missing target
    return df


def make_regression():
    d = load_diabetes(as_frame=True)
    df = d.data.copy()
    df["progression"] = d.target
    return df


def main():
    interactive = "--interactive" in sys.argv
    OUT.mkdir(exist_ok=True)
    make_messy_classification().to_csv(OUT / "messy_classification.csv", index=False)
    make_regression().to_csv(OUT / "regression.csv", index=False)
    print("Created demo_data/messy_classification.csv and demo_data/regression.csv")
    for name, target in [("messy_classification", "diagnosis"), ("regression", "progression")]:
        print("\n" + "#" * 70 + "\n# DEMO: %s\n" % name + "#" * 70)
        autoassist.run(str(OUT / ("%s.csv" % name)), target=target, interactive=interactive,
                       script_path=None if interactive else "demo_output/%s_pipeline.py" % name,
                       if_exists="overwrite", output_dir="demo_output/%s" % name)


if __name__ == "__main__":
    main()
