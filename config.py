"""
config.py - single source of truth for the project.

WHY THIS FILE EXISTS
--------------------
train.py and app.py must agree EXACTLY on:
  * which columns are features vs. target
  * which features are numeric vs. categorical
  * how categorical columns are cleaned before encoding

If that logic is copy-pasted into both files it will eventually drift
apart, and the Streamlit app will silently produce WRONG predictions
(no crash, just garbage metrics). Importing one shared module is the
cheap insurance policy.

This module is therefore imported FIRST by every entry point, so the
environment fix below runs before joblib is used.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Environment fix - must run before joblib/loky first counts CPU cores.
#
# Windows 11 removed `wmic`, which loky shells out to.  The probe raises,
# gets caught, and prints a 2 KB traceback to stderr.  Results are
# unaffected but the noise pollutes the Streamlit logs and the BITS Lab
# screenshot.  Seeding loky's cache short-circuits the probe entirely.
#
# Wrapped in try/except because this is an OPTIMISATION of an internal
# joblib detail.  If a joblib upgrade renames the cache attribute or moves
# the module, the application must still run - it just loses the log
# silence it was buying.  A cosmetic fix must never be able to take the
# app down.
# ---------------------------------------------------------------------------
try:
    import os as _os

    from joblib.externals.loky.backend import context as _loky_ctx

    if getattr(_loky_ctx, "physical_cores_cache", None) is None:
        _loky_ctx.physical_cores_cache = _os.cpu_count() or 4
except Exception:
    pass
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------- target
TARGET = "Revenue"

# ------------------------------------------------- numeric (continuous)
# These 10 are genuine quantities -> StandardScaler is meaningful on them.
NUMERIC_FEATURES = [
    "Administrative",              # count of admin pages visited
    "Administrative_Duration",     # time on admin pages
    "Informational",               # count of informational pages
    "Informational_Duration",      # time on informational pages
    "ProductRelated",              # count of product pages
    "ProductRelated_Duration",     # time on product pages
    "BounceRates",                 # Google Analytics bounce rate
    "ExitRates",                   # Google Analytics exit rate
    "PageValues",                  # avg value of a page in the session
    "SpecialDay",                  # closeness to a special day (0..1)
]

# ------------------------------------------- categorical (nominal codes)
# WARNING: four of these arrive as int64 but are CATEGORY IDs, not
# quantities.  TrafficType=20 is NOT "twice" TrafficType=10.  Treating
# them as numbers imposes a fake ordering + fake scale on the model.
# They MUST be one-hot encoded.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",   # nominal code stored as int64
    "Browser",            # nominal code stored as int64
    "Region",             # nominal code stored as int64
    "TrafficType",        # nominal code stored as int64
    "VisitorType",
    "Weekend",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ------------------------------------------------------------- constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_KEYS = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}

METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]


# Spellings of "this session did not purchase" and "...did", accepted in the
# Revenue column in either case.  Deliberately short: a label nobody would
# plausibly write is rejected and REPORTED rather than guessed at, because a
# silent guess turns a mis-typed file into silently wrong metrics.
_FALSE_WORDS = {"0", "0.0", "false", "f", "no", "n"}
_TRUE_WORDS = {"1", "1.0", "true", "t", "yes", "y"}


def coerce_target(series):
    """Map the Revenue column onto {0.0, 1.0, NaN}, whatever spelling arrived.

    WHY THIS EXISTS
    ---------------
    The same column reaches this app as 0/1, as 0.0/1.0 after a round-trip
    through Excel, as True/False if the file was written by pandas, and as
    yes/no if a human typed it.  `pd.to_numeric` alone handles the first two
    and rejects the rest as non-numeric, which would refuse a perfectly
    understandable file.

    Rules, in order:
      * bool dtype          -> True/False become 1/0.
      * numeric 0 or 1      -> kept (this covers 0, 1, 0.0, 1.0).
      * anything else       -> lower-cased, stripped, matched against the
                               word lists above.
      * everything else     -> NaN, so the caller reports the offending
                               values instead of crashing or guessing.

    Returns a float Series aligned to the input.  NaN means "could not be
    read as a label", never "missing at random" - core.validation turns any
    NaN into a message naming the values it could not read.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8").astype("float64")

    out = pd.Series(np.nan, index=series.index, dtype="float64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ok = numeric.notna() & numeric.isin([0, 1])
    out[numeric_ok] = numeric[numeric_ok].astype("float64")

    # Still unread: try the word spellings, but only where the column was
    # not simply empty - an empty cell is a missing label, not a bad one.
    pending = out.isna() & series.notna()
    if pending.any():
        words = series[pending].astype(str).str.strip().str.lower()
        out[pending & words.isin(_TRUE_WORDS)] = 1.0
        out[pending & words.isin(_FALSE_WORDS)] = 0.0

    return out


def prepare_features(df):
    """Select the feature columns and cast every categorical one to str.

    WHY CAST TO str?
    ----------------
    OperatingSystems / Browser / Region / TrafficType arrive as int64 and
    Weekend arrives as bool.  Casting them all to str guarantees that
    OneHotEncoder sees IDENTICAL categories when it is fitted on the
    training data and when the app later transforms an uploaded CSV -
    even if a round-trip through Excel changes 2 -> 2.0 and silently
    creates an unseen category.
    """
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X
