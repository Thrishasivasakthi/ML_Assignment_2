"""
make_readme_tables.py - print README section d straight from the CSV files.

    python make_readme_tables.py              # comparison table + leaders
    python make_readme_tables.py --baseline   # the untuned table
    python make_readme_tables.py --delta      # tuned minus baseline
    python make_readme_tables.py --all        # all three
    python make_readme_tables.py --verify     # check README section d

WHY THIS FILE EXISTS
--------------------
README section d is worth marks, and every number in it has to match
metrics.csv exactly.  Typing 30 figures by hand guarantees that at least one
will eventually drift - the previous version of this README already quoted a
correlation of 0.911 where the data says about 0.90.  Reading the CSV and
printing Markdown removes the transcription step entirely: paste what comes
out.

--verify closes the loop the other way.  It extracts every four-decimal
figure from section d of the committed README and reports any that is not
present in metrics.csv, metrics_baseline.csv or train_scores.csv, so a
stale number cannot survive a retrain.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from config import METRIC_KEYS, MODEL_LABELS

BASE_DIR = Path(__file__).resolve().parent

HEADINGS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}


def load(name):
    path = BASE_DIR / name
    if not path.exists():
        sys.exit(f"{name} not found - run `python train.py` first.")
    return pd.read_csv(path).set_index("model")


def markdown_table(frame):
    """The exact table format the assignment brief specifies."""
    lines = [
        "| ML Model Name | " + " | ".join(HEADINGS[m] for m in METRIC_KEYS) + " |",
        "|---" + "|---:" * len(METRIC_KEYS) + "|",
    ]
    for label in [MODEL_LABELS[k] for k in MODEL_LABELS]:
        if label not in frame.index:
            continue
        row = frame.loc[label]
        cells = " | ".join(f"{float(row[m]):.4f}" for m in METRIC_KEYS)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def leaders_note(frame):
    """'Best value per column: ...' - generated, so it cannot go stale.

    The frame is indexed by model LABEL (that is how metrics.csv is written),
    so idxmax() already returns the display name.
    """
    parts = []
    for metric in METRIC_KEYS:
        col = frame[metric].dropna()
        if col.empty:
            continue
        parts.append(f"{HEADINGS[metric]} — {col.idxmax()}")
    return "*Best value per column: " + "; ".join(parts) + ".*"


def verify_readme():
    """Every 0.xxxx in README section d must exist in a source file."""
    readme = BASE_DIR / "README.md"
    if not readme.exists():
        sys.exit("README.md not found.")

    text = readme.read_text(encoding="utf-8")
    match = re.search(r"^## d\..*?(?=^## )", text, re.S | re.M)
    if not match:
        sys.exit("Could not find section d in README.md.")
    section = match.group(0)

    known = set()
    for name in ("metrics.csv", "metrics_baseline.csv", "train_scores.csv"):
        frame = load(name)
        for value in frame.to_numpy().ravel():
            try:
                known.add(f"{float(value):.4f}")
            except (TypeError, ValueError):
                pass

    # The before/after table quotes differences: derived, not stored, so
    # compute them exactly the way make_readme_tables.py prints them.
    try:
        delta = load("metrics.csv") - load("metrics_baseline.csv")
        for value in delta.to_numpy().ravel():
            known.add(f"{float(value):.4f}")
            known.add(f"{abs(float(value)):.4f}")
    except (TypeError, ValueError, KeyError):
        pass

    quoted = set(re.findall(r"\b0\.\d{4}\b", section))
    unknown = sorted(quoted - known)

    print(f"README section d quotes {len(quoted)} distinct 4-dp figures.")
    if unknown:
        print("NOT found in metrics.csv / metrics_baseline.csv / train_scores.csv:")
        for value in unknown:
            print(f"  {value}")
        return 1
    print("All of them match a value in the source files.")

    # Membership is not enough: a cell swapped with another model's would
    # still "exist" somewhere.  Render each expected table and require the
    # README to contain that exact block of text.
    tuned = load("metrics.csv")
    baseline = load("metrics_baseline.csv")
    blocks = {
        "comparison table (tuned)": markdown_table(tuned),
        "comparison table (baseline)": markdown_table(baseline),
        "before/after table": markdown_table((tuned - baseline).round(4)),
    }
    failures = 0
    # Normalise whitespace first: the README wraps prose but markdown table
    # rows stay on one line, so a flattened copy is still exact for tables.
    flat = re.sub(r"\s+", " ", section)
    for name, block in blocks.items():
        present = re.sub(r"\s+", " ", block) in flat
        print(f"  [{'PASS' if present else 'FAIL'}] {name} matches its CSV exactly")
        if not present:
            failures += 1

    leaders = leaders_note(tuned)
    present = re.sub(r"\s+", " ", leaders) in flat
    print(f"  [{'PASS' if present else 'FAIL'}] best-per-column line matches")
    if not present:
        failures += 1

    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="untuned table")
    parser.add_argument("--delta", action="store_true", help="tuned minus baseline")
    parser.add_argument("--all", action="store_true", help="everything")
    parser.add_argument("--verify", action="store_true", help="check README section d")
    args = parser.parse_args()

    if args.verify:
        sys.exit(verify_readme())

    if not (args.baseline or args.delta or args.all):
        args.all = True

    tuned = load("metrics.csv")

    if args.baseline or args.all:
        print("<!-- metrics_baseline.csv (sklearn defaults) -->")
        print(markdown_table(load("metrics_baseline.csv")))
        print()

    if args.all:
        print("<!-- metrics.csv (tuned) -->")
        print(markdown_table(tuned))
        print()
        print(leaders_note(tuned))
        print()

    if args.delta or args.all:
        baseline = load("metrics_baseline.csv")
        delta = tuned - baseline
        print("<!-- tuned minus baseline -->")
        print(markdown_table(delta.round(4)))


if __name__ == "__main__":
    main()
