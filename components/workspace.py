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

import pandas as pd
import streamlit as st

from components.layout import empty_state, inject_css
from components.sidebar import render_sidebar
# config is reached through core, which installs the joblib/loky fix first.
from config import FEATURES, MODEL_KEYS, MODEL_LABELS, TARGET
from core.evaluation import comparison_frame, score_all
from core.inference import load_artifacts
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
            "<b>Use bundled test_data.csv</b> to load the copy that ships with "
            "this repository. The upload stays loaded while you move between "
            "pages."
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
        y_true      the ground-truth labels
        comparison  wide DataFrame indexed by model KEY
        summary     row/feature/class counts
        failed      [key] for any model that could not be scored

    A model that raises yields None rather than NaN, so one broken row
    cannot silently poison the comparison table - the UI shows an em dash.
    """
    preprocessor, models = get_artifacts()
    y_true = df[TARGET].astype(int).values

    metrics, preds = score_all(models, preprocessor, df, y_true, MODEL_KEYS)

    return {
        "metrics": metrics,
        "preds": preds,
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
