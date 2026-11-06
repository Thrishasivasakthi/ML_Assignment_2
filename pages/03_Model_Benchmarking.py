"""
pages/03_Model_Benchmarking.py - all five classifiers on identical data.

The comparison the assignment asks for, presented as a benchmark rather than
a table dump: pick the metric, read the bars, see the leaderboard, then see
which classifier owns each metric.

Every bar is scaled from zero to one.  Scaling from the observed minimum
would draw the weakest model as a near-full bar and exaggerate every gap,
which is the sort of chart that invites awkward questions in a submission.
"""

import streamlit as st

from components.charts import comparison_bars, metric_selector, ranking_list
from components.layout import page_header, rule, section, stat_strip
from components.notes import metric_disagreement
from components.tables import comparison_table, metric_leaders_table
from components.workspace import best_by, boot, evaluate_all, open_workspace, require_scored
from config import METRIC_KEYS, MODEL_LABELS

boot("Session2Shop | Model Benchmarking")

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)
require_scored(result, selected_key)

frame = result["comparison"]
summary = result["summary"]

page_header(
    "Model Benchmarking",
    "Five classifiers, one held-out set",
    "Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM "
    f"scored on exactly the same {summary['rows']:,} held-out sessions with "
    "exactly the weights train.py saved. Nothing here is refitted.",
)

# Leader on each metric - answers "who wins what" at a glance.
winners = {mk: best_by(result, mk) for mk in METRIC_KEYS}
distinct = sorted({k for k in winners.values() if k})

stat_strip(
    [
        ("Models scored", f"{len([k for k in result['metrics'] if result['metrics'][k]])} of 5"),
        ("Metric leaders", str(len(distinct))),
        ("Overall winner (MCC)", MODEL_LABELS[winners["mcc"]] if winners["mcc"] else "-"),
        ("Best AUC", MODEL_LABELS[winners["auc"]] if winners["auc"] else "-"),
        ("Rows scored", f"{result['summary']['rows']:,}"),
    ]
)

# ------------------------------------------------------------ 01 choose metric
st.markdown(
    section("01", "Compare by", "bar lengths run from zero to one"),
    unsafe_allow_html=True,
)
bar_metric = metric_selector(default="mcc", key="benchmark_metric")
comparison_bars(frame, selected_key, bar_metric)

st.caption(
    f"Sorted view below is by **{bar_metric.upper() if bar_metric in ('auc', 'mcc') else bar_metric}**. "
    "Switch the metric and the ordering changes - that disagreement is the "
    "point of reporting six metrics instead of one."
)

# ------------------------------------------------------------- 02 leaderboard
st.markdown(
    section("02", "Leaderboard", "ranked on the selected metric"),
    unsafe_allow_html=True,
)
ranking_list(frame, bar_metric, selected_key)

# --------------------------------------------------------- 03 full comparison
st.markdown(
    section(
        "03",
        "Full comparison",
        "six metrics, best in green and weakest in red",
    ),
    unsafe_allow_html=True,
)
comparison_table(frame, selected_key)

st.markdown(rule(), unsafe_allow_html=True)

# ------------------------------------------------------------ 04 who wins what
st.markdown(
    section("04", "Who wins what", "no single model takes every metric"),
    unsafe_allow_html=True,
)
metric_leaders_table(frame)

winner_mcc = winners["mcc"]
winner_auc = winners["auc"]
if winner_mcc and winner_auc and winner_mcc != winner_auc:
    st.caption(
        f"**{MODEL_LABELS[winner_auc]}** has the best AUC "
        f"({float(frame.loc[winner_auc, 'auc']):.4f}) but **"
        f"{MODEL_LABELS[winner_mcc]}** has the best MCC "
        f"({float(frame.loc[winner_mcc, 'mcc']):.4f}). Those are different "
        "questions: AUC measures ranking, MCC measures the agreement between "
        "prediction and truth at the 0.5 cut-off."
    )

# --------------------------------------------------------- 05 why they differ
st.markdown(
    section("05", "Why the metrics disagree", "and which one to trust here"),
    unsafe_allow_html=True,
)
st.markdown(metric_disagreement(frame))

st.markdown(
    '<div class="s2s-baseline"><b>Why MCC is the deciding metric:</b> a model '
    f"that always answers &ldquo;No purchase&rdquo; reaches "
    f"<b>{summary['always_no_accuracy'] * 100:.1f}% accuracy</b> "
    "and <b>0 MCC</b>. MCC cannot be inflated that way, which is why the "
    "assignment requires it alongside accuracy rather than in place of it.</div>",
    unsafe_allow_html=True,
)

st.markdown(rule(), unsafe_allow_html=True)
