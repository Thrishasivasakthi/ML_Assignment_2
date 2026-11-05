"""
app.py - Session2Shop, Overview page and entry point for ML Assignment 2.

    streamlit run app.py

WHAT THIS APP DOES (spec Section 6, 4 marks)
--------------------------------------------
  a. Dataset upload option (CSV)                     [sidebar, every page]
  b. Model selection dropdown                        [sidebar, every page]
  c. Display of evaluation metrics                   [section 01 below]
  d. Confusion matrix / classification report        [section 02 below]

All four are co-located on this page on purpose: the submission contains a
single screenshot of the running application, and a requirement spread across
four pages cannot be evidenced in one frame.

HARD CONSTRAINT FROM THE SPEC
-----------------------------
"The app must load the saved model, calculate the evaluation metrics, and
display the results WITHOUT retraining the model on the uploaded test data."

There is no .fit() call anywhere in this repository outside train.py.
Artifacts written once by train.py are loaded, applied and scored - that is
the entire job of this application.

PAGE MAP
--------
    app.py                          01 Overview        (this file)
    pages/02_Model_Explorer.py      02 per-classifier detail
    pages/03_Model_Benchmarking.py  03 five-model comparison
    pages/04_Error_Analysis.py      04 matrix, false alarms, missed buyers
    pages/05_Data_Workspace.py      05 upload, validation, preview
    pages/06_Methodology.py         06 dataset, preprocessing, metrics

The stylesheet and page chrome live in components/, the scoring path in core/;
this file is composition only.
"""

import streamlit as st

# components.workspace pulls in core, which installs the joblib/loky fix
# before any artifact is touched.
from components.charts import comparison_bars, metric_selector
from components.layout import eyebrow, page_header, rule, section, stat_strip
from components.metric_cards import baseline_note, kpi_tiles, metric_cards
from components.notes import METRIC_DISAGREEMENT
from components.tables import comparison_table, confusion_block, report_table
from components.workspace import (
    best_by,
    boot,
    evaluate_all,
    open_workspace,
    require_scored,
)
from config import MODEL_LABELS
from core.evaluation import confusion_counts

# Always-No baseline on this dataset: the yardstick accuracy is judged against.
ALWAYS_NO_ACCURACY = 0.844

boot("Session2Shop | Overview")

page_header(
    "M.Tech AIML &nbsp;/&nbsp; Machine Learning &nbsp;/&nbsp; Assignment 2",
    "Session2Shop",
    "Predicting whether an e-commerce browsing session ends in a purchase, "
    "from 17 behavioural features. Load a held-out test set, choose a "
    "classifier, and read its six evaluation metrics.",
)

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)
require_scored(result, selected_key)

metrics = result["metrics"][selected_key]
y_true = result["y_true"]
y_pred = result["preds"][selected_key]
summary = result["summary"]
frame = result["comparison"]

tn, fp, fn, tp = confusion_counts(y_true, y_pred)

# ------------------------------------------------------------------ masthead
stat_strip(
    [
        ("Rows", f"{summary['rows']:,}"),
        ("Features", str(summary["features"])),
        (
            "Purchasers",
            f"{summary['purchasers']:,} ({summary['conversion_rate'] * 100:.1f}%)",
        ),
        (
            "Non-purchasers",
            f"{summary['non_purchasers']:,} "
            f"({100 - summary['conversion_rate'] * 100:.1f}%)",
        ),
        ("Source", source_name),
    ]
)

mcc_leader = best_by(result, "mcc")
leader_mcc = float(frame.loc[mcc_leader, "mcc"]) if mcc_leader else float("nan")

kpi_tiles(
    [
        {
            "label": "Sessions evaluated",
            "value": f"{summary['rows']:,}",
            "sub": "held-out rows the saved models scored",
        },
        {
            "label": "Purchase conversion",
            "value": f"{summary['conversion_rate'] * 100:.1f}%",
            "sub": f"{summary['purchasers']:,} sessions ended in a purchase",
        },
        {
            "label": "Leading classifier",
            "value": MODEL_LABELS[mcc_leader] if mcc_leader else "&mdash;",
            "sub": f"MCC {leader_mcc:.4f} - the imbalance-honest metric",
            "accent": True,
            "text": True,
        },
        {
            "label": "Real buyers missed",
            "value": f"{fn:,}",
            "sub": (
                f"of {summary['purchasers']:,} actual purchasers "
                f"({fn / max(summary['purchasers'], 1) * 100:.1f}%)"
            ),
        },
    ]
)

# ------------------------------------------------------- 01 headline metrics
st.markdown(
    section("01", MODEL_LABELS[selected_key], "six metrics on the loaded dataset"),
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:.86rem;color:#6B7280;margin-top:.3rem">'
    "Computed from the <b>saved</b> model and the <b>saved</b> preprocessor. "
    "Nothing was retrained on this data.</div>",
    unsafe_allow_html=True,
)

metric_cards(metrics)
baseline_note()

st.caption(
    f"**Accuracy {metrics['accuracy']:.4f}** against an always-No baseline of "
    f"**84.4%** - a lead of {(metrics['accuracy'] - ALWAYS_NO_ACCURACY) * 100:+.1f} "
    f"points. MCC **{metrics['mcc']:.4f}** is the figure that survives the "
    "imbalance."
)

# ------------------------------------------------------- 02 error analysis
st.markdown(
    section("02", "Where it goes wrong", "confusion matrix and per-class report"),
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.7rem">Confusion matrix</div>',
        unsafe_allow_html=True,
    )
    confusion_block(y_true, y_pred)
with right:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.7rem">Classification report</div>',
        unsafe_allow_html=True,
    )
    report_table(y_true, y_pred)

st.caption(
    f"**{tp:,}** true positives and **{tn:,}** true negatives; "
    f"**{fp:,}** false alarms and **{fn:,}** real purchasers missed. "
    "On a 15.6% positive class the second pair is what costs money."
)

# ------------------------------------------------------- 03 all five models
st.markdown(
    section("03", "All five models", "same data, same saved weights"),
    unsafe_allow_html=True,
)

bar_metric = metric_selector(default="mcc", key="overview_compare_metric")
comparison_bars(frame, selected_key, bar_metric)

st.markdown(
    '<div class="s2s-eyebrow" style="margin:1.6rem 0 .8rem">'
    "Full comparison &mdash; best in green, weakest in red</div>",
    unsafe_allow_html=True,
)
comparison_table(frame, selected_key)

if mcc_leader:
    st.caption(
        f"**{MODEL_LABELS[mcc_leader]}** leads on MCC "
        f"({leader_mcc:.4f}) - the metric that stays honest on an imbalanced "
        "dataset. Bar lengths run from zero to one, so no gap is exaggerated."
    )

with st.expander("Why the metrics disagree"):
    st.markdown(METRIC_DISAGREEMENT)

st.markdown(rule(), unsafe_allow_html=True)
st.markdown(
    '<div class="s2s-eyebrow">M.Tech AIML &nbsp;/&nbsp; Machine Learning '
    "&nbsp;/&nbsp; Assignment 2</div>",
    unsafe_allow_html=True,
)
