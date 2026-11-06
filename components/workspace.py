"""
components.workspace - page bootstrap, data gate and shared evaluation.

Every page follows the same three steps:

    boot()             page config + install the design system
    open_workspace()   sidebar (upload + model dropdown) + validation gate
    evaluate_all()     score the five saved models on the loaded data

Funneling all three through one module is what guarantees the figure shown
on Overview is the identical figure shown on Error Analysis and the figure
written into metrics.csv - there is exactly one scoring path.

WHY ARTIFACTS ARE CACHED WITH A PLAIN GLOBAL
--------------------------------------------
st.cache_resource() would also work, but decorating a function executes the
decorator at import time, and st.set_page_config() must be the first
Streamlit command a page issues.  Python caches imported modules in
sys.modules across Streamlit reruns, so a module-level global survives every
rerun of the script anyway - and it is cleared automatically if the file
watcher reloads the module, which is precisely when we would want the models
re-read from disk.
"""

import json

import pandas as pd
import streamlit as st

from components.layout import empty_state, inject_css
from components.sidebar import render_sidebar
# config is reached through core, which installs the joblib/loky fix first.
from config import FEATURES, MODEL_KEYS, MODEL_LABELS, TARGET, coerce_target
from core.evaluation import comparison_frame, score_all
from core.facts import dataset_facts as collect_facts
from core.inference import BASE_DIR, PREPROCESSOR_PATH, load_artifacts, model_path
from core.validation import dataset_summary, validate_dataset

_ARTIFACTS = None


def boot(title):
    """First call on every page: page config, then the stylesheet."""
    st.set_page_config(page_title=title, initial_sidebar_state="expanded")
    inject_css()


def get_artifacts():
    """Return (preprocessor, {key: classifier}), loaded once per process."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()
    return _ARTIFACTS


def _files_stamp(*paths):
    """mtime+size of the data files, used as a cache key.

    st.cache_data keys on the function body and its arguments, not on the
    contents of a file the function reads.  Passing the stamp means that
    re-running train.py invalidates the cached numbers immediately instead
    of leaving the app serving the previous run's figures until someone
    restarts the server.
    """
    parts = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            parts.append(f"{path.name}:missing")
        else:
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _cached_facts(stamp):
    """Correlations, split sizes and class ratios, computed from disk."""
    return collect_facts()


def facts():
    """Cached wrapper - call this from a page."""
    return _cached_facts(
        _files_stamp(
            BASE_DIR / "raw" / "online_shoppers_intention.csv",
            BASE_DIR / "train_data.csv",
            BASE_DIR / "test_data.csv",
        )
    )


@st.cache_data(show_spinner=False)
def train_scores_map(stamp):
    """{model key: {train_accuracy, test_accuracy, gap}} from train_scores.csv.

    Returns {} when the file is absent so a missing artefact degrades to
    "no overfitting sentence" rather than a red exception box.
    """
    path = BASE_DIR / "train_scores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    key_of = {label: key for key, label in MODEL_LABELS.items()}
    out = {}
    for _, row in df.iterrows():
        key = key_of.get(row.get("model"))
        if key is None:
            continue
        out[key] = {
            "train_accuracy": float(row["train_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
            "gap": float(row["gap"]),
        }
    return out


def train_scores():
    """Cached wrapper - call this wherever an overfitting claim is made."""
    return train_scores_map(_files_stamp(BASE_DIR / "train_scores.csv"))


@st.cache_data(show_spinner=False)
def _cached_best_params(stamp):
    path = BASE_DIR / "best_params.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_params():
    """The tuning record train.py wrote: search definition + winners.

    Read from disk rather than restated, so the Methodology page quotes the
    same scoring function, fold count and grids the search actually used.
    """
    return _cached_best_params(_files_stamp(BASE_DIR / "best_params.json"))


@st.cache_data(show_spinner=False)
def _cached_baseline(stamp):
    path = BASE_DIR / "metrics_baseline.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("model")


def baseline_metrics():
    """The untuned (sklearn defaults) table, for before/after reporting.

    Returns None when metrics_baseline.csv is absent, so callers can skip
    the comparison rather than crash on it.
    """
    return _cached_baseline(_files_stamp(BASE_DIR / "metrics_baseline.csv"))


def open_workspace(validate=True):
    """Render the sidebar and gate the page behind a usable dataset.

    Returns (df, source_name, selected_key).  Stops the page when nothing
    is loaded, or - unless validate=False - when the CSV would produce
    meaningless metrics.  Data Workspace passes validate=False because
    showing the problems IS that page's job.
    """
    df, source_name, selected_key = render_sidebar()

    if df is None:
        empty_state(
            "Upload <b>test_data.csv</b> from the panel on the left, or click "
            "<b>Reset to bundled test_data.csv</b> to load the copy that ships "
            "with this repository. The upload stays loaded while you move "
            "between pages."
        )
        st.stop()

    if validate:
        problems = validate_dataset(df)
        if problems:
            st.error("**This CSV cannot be scored**")
            for p in problems:
                st.markdown(f"- {p}")
            st.caption(
                "Expected columns: "
                + ", ".join(f"`{c}`" for c in FEATURES)
                + f", plus the target `{TARGET}`."
            )
            st.stop()

    return df, source_name, selected_key


def evaluate_all(df):
    """Score every saved classifier on one dataset.

    Returns a dict:
        metrics     {key: {metric: float} | None}
        preds       {key: array | None}
        probas      {key: array | None}   probability of class 1
        y_true      the ground-truth labels
        comparison  wide DataFrame indexed by model KEY
        summary     row/feature/class counts
        failed      [key] for any model that could not be scored

    A model that raises yields None rather than NaN, so one broken row
    cannot silently poison the comparison table - the UI shows an em dash.

    The heavy part is cached on two things: the CONTENTS of the dataframe
    (st.cache_data hashes a DataFrame by value, so a different file is a
    different key while a page change is not) and the mtime+size of the
    saved models.  The second argument matters because the cache is keyed on
    arguments only - without it, retraining while the server is up would
    leave the app serving the previous run's numbers until someone
    restarted it.
    """
    stamp = _files_stamp(
        PREPROCESSOR_PATH, *[model_path(key) for key in MODEL_KEYS]
    )
    return _score_dataset(df, stamp)


@st.cache_data(show_spinner=False)
def _score_dataset(df, artifact_stamp):
    """The single scoring path.  See evaluate_all() for the cache rationale."""
    preprocessor, models = get_artifacts()
    # Labels are coerced, not assumed: 0/1, 0.0/1.0, True/False and yes/no
    # all reach here as the same 0/1 integers.  Anything unreadable would
    # already have been rejected by validate_dataset on this page.
    y_true = coerce_target(df[TARGET]).astype("int64").values

    metrics, preds, probas = score_all(
        models, preprocessor, df, y_true, MODEL_KEYS
    )

    return {
        "metrics": metrics,
        "preds": preds,
        "probas": probas,
        "y_true": y_true,
        "comparison": comparison_frame(metrics, MODEL_KEYS),
        "summary": dataset_summary(df),
        "failed": [k for k in MODEL_KEYS if metrics.get(k) is None],
    }


def require_scored(result, selected_key):
    """Stop the page with a clear message if the selected model failed."""
    if result["metrics"].get(selected_key) is None:
        st.error(
            f"**{MODEL_LABELS[selected_key]} could not be scored on this "
            "dataset.** Choose another classifier, or upload a CSV with both "
            "target classes present."
        )
        st.stop()


def best_by(result, metric):
    """Key of the leading model on one metric, ignoring any that failed."""
    frame = result["comparison"]
    values = frame[metric].dropna()
    if values.empty:
        return None
    return values.idxmax()
