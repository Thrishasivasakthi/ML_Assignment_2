"""
pages/02_Model_Explorer.py - deep dive on a single classifier.

The assignment requires a model-selection dropdown; that lives in the shared
sidebar so it is present on every page, and this is the page that pays it
off - it explains what the chosen classifier actually did, not just what it
scored.

Sections
    01  the six required metrics
    02  how to read this particular model
    03  where it stands against the other four
    04  how it was configured at training time
"""

import streamlit as st

from components.charts import comparison_bars, metric_selector
from components.layout import page_header, rule, section, stat_strip
from components.metric_cards import baseline_note, metric_cards
from components.notes import MODEL_NOTES
from components.tables import comparison_table
from components.workspace import (
    best_by,
    boot,
    evaluate_all,
    get_artifacts,
    open_workspace,
    require_scored,
)
from config import METRIC_KEYS, MODEL_KEYS, MODEL_LABELS

# Hyperparameters worth surfacing; anything not present on the estimator is
# skipped rather than shown as a blank row.
PARAM_FOCUS = {
    "logistic_regression": ["penalty", "C", "solver", "max_iter", "class_weight"],
    "decision_tree": ["criterion", "max_depth", "min_samples_split",
                      "min_samples_leaf", "ccp_alpha"],
    "knn": ["n_neighbors", "weights", "metric", "p", "algorithm"],
    "naive_bayes": ["var_smoothing"],
    "svm": ["C", "kernel", "gamma", "shrinking", "probability"],
}

boot("Session2Shop | Model Explorer")

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)
require_scored(result, selected_key)

label = MODEL_LABELS[selected_key]
note = MODEL_NOTES[selected_key]
frame = result["comparison"]
metrics = result["metrics"][selected_key]
y_true, y_pred = result["y_true"], result["preds"][selected_key]

page_header(
    "Model Explorer",
    label,
    note["interpretation"],
)

# Rank of this model on each metric (1 = best of five), computed once.
ranks = {}
for mk in METRIC_KEYS:
    ordered = frame[mk].dropna().sort_values(ascending=False)
    ranks[mk] = (list(ordered.index).index(selected_key) + 1, len(ordered)) \
        if selected_key in ordered.index else (None, len(ordered))

mcc_rank = ranks["mcc"][0]
best_key = best_by(result, "mcc")

stat_strip(
    [
        ("Architecture", note["family"]),
        ("MCC rank", f"{mcc_rank} of {ranks['mcc'][1]}" if mcc_rank else "-"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("AUC", f"{metrics['auc']:.4f}"),
        ("MCC", f"{metrics['mcc']:.4f}"),
    ]
)

# ------------------------------------------------------------ 01 six metrics
st.markdown(section("01", "The six required metrics", label), unsafe_allow_html=True)
metric_cards(metrics)
baseline_note()

st.caption(
    f"Read against the always-No baseline of **84.4% accuracy**. This model's "
    f"lead is **{(metrics['accuracy'] - 0.844) * 100:+.1f} points**, and its "
    f"MCC is **{metrics['mcc']:.4f}** against the baseline's **0**."
)

# ------------------------------------------------------ 02 how to read it
st.markdown(
    section("02", "How to read this model", "strengths, limits, and why"),
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">Strongest at</div>'
        f'<div style="font-size:.95rem;color:#111827;font-weight:600;'
        f'margin-bottom:.9rem">{note["best"]}</div>'
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">Weakest at</div>'
        f'<div style="font-size:.95rem;color:#111827;font-weight:600">'
        f'{note["worst"]}</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">Rank among the five</div>',
        unsafe_allow_html=True,
    )
    comparison_bars(frame, selected_key, "mcc")

st.markdown(rule(), unsafe_allow_html=True)
st.markdown(
    '<div class="s2s-eyebrow" style="margin-bottom:.7rem">Rank on each metric</div>',
    unsafe_allow_html=True,
)

head = (
    "<tr><th class='l'>Metric</th><th>Value</th><th>Rank</th>"
    "<th>Held by</th></tr>"
)
rows = []
for mk in METRIC_KEYS:
    rank, total = ranks[mk]
    leader = frame[mk].dropna()
    leader_key = leader.idxmax() if not leader.empty else None
    rank_txt = f"{rank} of {total}" if rank else "-"
    mark = " best" if rank == 1 else ""
    rows.append(
        f"<tr><td class='l'>{mk.upper() if mk in ('auc', 'mcc') else mk.capitalize()}</td>"
        f"<td class='num'>{metrics[mk]:.4f}</td>"
        f"<td class='num{mark}'>{rank_txt}</td>"
        f"<td class='l'>{MODEL_LABELS.get(leader_key, '') if leader_key else '-'}</td></tr>"
    )
st.markdown(
    f'<table class="s2s-tbl">{head}{"".join(rows)}</table>', unsafe_allow_html=True
)

# -------------------------------------------------------- 03 configuration
st.markdown(
    section("03", "Training configuration", "as fitted by train.py"),
    unsafe_allow_html=True,
)

preprocessor, models = get_artifacts()
estimator = models[selected_key]
params = estimator.get_params()

cfg_rows = []
for name in PARAM_FOCUS[selected_key]:
    if name not in params:
        continue
    value = params[name]
    cfg_rows.append(
        f"<tr><td class='l'>{name}</td>"
        f"<td class='num l'>{value}</td></tr>"
    )

st.markdown(
    '<table class="s2s-tbl">'
    "<tr><th class='l'>Hyperparameter</th><th class='l'>Value</th></tr>"
    + "".join(cfg_rows)
    + "</table>",
    unsafe_allow_html=True,
)

st.caption(
    "17 input features become **74 columns** after preprocessing: 10 numeric "
    "columns standardised, 64 one-hot columns from the 7 nominal categories. "
    "The preprocessor was fitted on the training split only."
)

# ------------------------------------------------------ 04 against the rest
st.markdown(
    section("04", "Against the other four", "same data, same saved weights"),
    unsafe_allow_html=True,
)
bar_metric = metric_selector(default="mcc", key="explorer_compare_metric")
comparison_bars(frame, selected_key, bar_metric)
st.markdown(
    '<div class="s2s-eyebrow" style="margin:1.6rem 0 .8rem">'
    "Full comparison</div>",
    unsafe_allow_html=True,
)
comparison_table(frame, selected_key)

if best_key and best_key != selected_key:
    st.caption(
        f"**{MODEL_LABELS[best_key]}** currently leads on MCC "
        f"({float(frame.loc[best_key, 'mcc']):.4f})."
    )

st.markdown(rule(), unsafe_allow_html=True)
