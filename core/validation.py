"""
core.validation - pre-flight checks on an uploaded test CSV.

WHY THIS EXISTS
---------------
Two of the six required metrics are fragile:

  * ROC AUC is UNDEFINED when only one class is present.  sklearn raises
    rather than returning NaN, so a single-class CSV produced a traceback
    instead of an explanation.
  * MCC and the confusion matrix are meaningless for the same reason.

The spec also requires the uploaded test CSV to carry every input feature
AND the target column - this module checks exactly that, so a malformed
upload fails with a message the user can act on instead of a red exception
box that looks like a broken application.

Nothing here touches the models, so it is safe to call before scoring.
"""

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    coerce_target,
)

MIN_ROWS = 2

# How many offending values to name in a message, so a file with a thousand
# bad rows does not produce a thousand-line error.
SAMPLE = 5


def validate_dataset(df):
    """Return a list of human-readable problems.  An empty list == usable.

    Checks are ordered from "nothing else is checkable" outward, so each
    early return only fires when continuing would be pointless.

    Column ORDER and extra columns are deliberately not checked: the app
    selects features by name, so a shuffled file and a file with three
    surplus columns are both fine - and saying otherwise would be a false
    alarm.
    """
    if not isinstance(df, pd.DataFrame):
        return ["The file could not be parsed as a table."]

    problems = []

    # ---------------------------------------------------------- structure
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        problems.append(
            "Missing feature column(s): " + ", ".join(f"`{c}`" for c in missing)
        )

    if TARGET not in df.columns:
        problems.append(
            f"Missing target column `{TARGET}`. The spec requires the test CSV "
            "to contain the labels as well as the features, because the app "
            "evaluates against them."
        )
        return problems

    if len(df) == 0:
        problems.append(
            "The file has a header but no data rows, so there is nothing to "
            "evaluate."
        )
        return problems

    if len(df) < MIN_ROWS:
        problems.append(
            f"Need at least {MIN_ROWS} rows to evaluate; the file has {len(df)}."
        )
        return problems

    # ------------------------------------------------------ target column
    # Accepts 0/1, 0.0/1.0, True/False and yes/no in either case - see
    # config.coerce_target.  Anything left unread is reported by value, so
    # the user can see WHICH rows the app refused to interpret.
    tgt = coerce_target(df[TARGET])

    if tgt.isna().any():
        bad = df[TARGET][tgt.isna()].astype(str).str.strip()
        shown = ", ".join(f"`{v}`" for v in bad.unique()[:SAMPLE])
        more = "" if bad.nunique() <= SAMPLE else ", ..."
        problems.append(
            f"`{TARGET}` has {int(tgt.isna().sum())} value(s) that are not "
            f"readable as a purchase label: {shown}{more}. "
            "Expected 0 or 1 (also accepts 0.0/1.0, True/False, yes/no)."
        )
        return problems

    values = set(tgt.unique().tolist())
    if not values.issubset({0, 1}):
        problems.append(
            f"`{TARGET}` must contain only 0 and 1; found {sorted(values)}."
        )
        return problems

    if len(values) < 2:
        only = int(values.pop())
        label = "1 (will purchase)" if only else "0 (no purchase)"
        problems.append(
            f"`{TARGET}` contains only {label}. ROC AUC and the confusion "
            "matrix need both classes present to be defined."
        )

    # ----------------------------------------------------- numeric columns
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        # values that were non-null on input but failed to coerce are bad
        non_numeric = int(coerced.isna().sum() - df[col].isna().sum())
        if non_numeric > 0:
            problems.append(f"Column `{col}` has {non_numeric} non-numeric value(s).")
        elif np.isinf(coerced.fillna(0.0)).any():
            problems.append(f"Column `{col}` contains infinite values.")

    # -------------------------------------------------- categorical columns
    # These are one-hot encoded, so a blank becomes its own literal category
    # ("nan") and silently pollutes the feature space.
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            problems.append(
                f"Column `{col}` has {int(df[col].isna().sum())} missing value(s)."
            )

    return problems


def dataset_summary(df):
    """Small dict of facts the UI shows next to the validation status.

    The target goes through coerce_target rather than a raw value_counts:
    a CSV whose Revenue is written True/False or yes/no would otherwise
    count zero of each class and report a 0.0% conversion rate.
    """
    tgt = coerce_target(df[TARGET])
    n_pos = int((tgt == 1).sum())
    n_neg = int((tgt == 0).sum())
    total = int(len(df))
    return {
        "rows": total,
        "features": len(FEATURES),
        "purchasers": n_pos,
        "non_purchasers": n_neg,
        "conversion_rate": (n_pos / total) if total else 0.0,
        # The yardstick accuracy is judged against: a classifier that always
        # answers "No purchase" simply takes the larger class.  Computed from
        # the loaded file rather than quoted, so an uploaded CSV with a
        # different class balance gets the baseline that belongs to IT.
        "always_no_accuracy": (max(n_pos, n_neg) / total) if total else 0.0,
        "numeric_features": len(NUMERIC_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
    }
