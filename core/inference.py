"""
core.inference - loading saved artifacts and scoring them.

HARD RULE
---------
There is NO .fit() call anywhere in this module.  That is deliberate: the
spec requires the app to evaluate WITHOUT retraining on uploaded data.
Loading, transforming and predicting - that is the entire job.

train.py fits the preprocessor and the five classifiers once and writes
them out; this module only ever reads them back.
"""

from pathlib import Path

# config is imported BEFORE joblib so the loky environment fix in config.py
# runs before anything here can touch a serialized estimator.
from config import MODEL_KEYS, prepare_features

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.joblib"
MODEL_DIR = BASE_DIR / "model"
BUNDLED_TEST_CSV = BASE_DIR / "test_data.csv"


def model_path(key):
    """Absolute path of one saved classifier."""
    return MODEL_DIR / f"{key}.joblib"


def load_artifacts():
    """Return (preprocessor, {model_key: fitted classifier}).

    Callers wrap this in st.cache_resource: Streamlit re-executes the whole
    script on every widget interaction, so without caching each click would
    re-read roughly 8 MB of joblib from disk.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = {key: joblib.load(model_path(key)) for key in MODEL_KEYS}
    return preprocessor, models


def predict(model, preprocessor, df):
    """Return (y_pred, y_proba) for one model over an already-loaded CSV.

    y_proba is the column for class 1, which is what ROC AUC needs.
    """
    X = prepare_features(df)
    X_t = preprocessor.transform(X)
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]
    return y_pred, y_proba
