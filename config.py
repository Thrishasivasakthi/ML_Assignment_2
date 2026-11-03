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

# ---------------------------------------------------------------------------
# Environment fix - must run before joblib/loky first counts CPU cores.
#
# Windows 11 removed `wmic`, which loky shells out to.  The probe raises,
# gets caught, and prints a 2 KB traceback to stderr.  Results are
# unaffected but the noise pollutes the Streamlit logs and the BITS Lab
# screenshot.  Seeding loky's cache short-circuits the probe entirely.
# ---------------------------------------------------------------------------
import os as _os

from joblib.externals.loky.backend import context as _loky_ctx

if getattr(_loky_ctx, "physical_cores_cache", None) is None:
    _loky_ctx.physical_cores_cache = _os.cpu_count() or 4
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
