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

from config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

MIN_ROWS = 2


def validate_dataset(df):
    """Return a list of human-readable problems.  An empty list == usable.

    Checks are ordered from "nothing else is checkable" outward, so each
    early return only fires when continuing would be pointless.
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

    if len(df) < MIN_ROWS:
        problems.append(
            f"Need at least {MIN_ROWS} rows to evaluate; the file has {len(df)}."
        )
        return problems

    # ------------------------------------------------------ target column
    # Coerce first: a round-trip through Excel often turns 0/1 into 0.0/1.0,
    # which must not be rejected as an invalid label.
    tgt = pd.to_numeric(df[TARGET], errors="coerce")

    if tgt.isna().any():
        problems.append(
            f"`{TARGET}` has {int(tgt.isna().sum())} value(s) that are not "
            "numeric. Expected 0 or 1."
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
    """Small dict of facts the UI shows next to the validation status."""
    counts = df[TARGET].value_counts()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    total = int(len(df))
    return {
        "rows": total,
        "features": len(FEATURES),
        "purchasers": n_pos,
        "non_purchasers": n_neg,
        "conversion_rate": (n_pos / total) if total else 0.0,
        "numeric_features": len(NUMERIC_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
    }
