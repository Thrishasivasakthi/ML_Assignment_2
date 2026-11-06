"""
pages/04_Error_Analysis.py - what the classifier gets wrong, and what it costs.

On a dataset where only a small minority of sessions end in a purchase, the
two error types are not equivalent:

    false alarm   a non-buyer was targeted as a buyer  -> wasted spend
    missed buyer  a real purchaser was ignored         -> lost revenue

This page separates them explicitly rather than folding both into "wrong".
"""

import streamlit as st

from components.layout import page_header, rule, section, stat_strip
from components.metric_cards import metric_cards
from components.tables import confusion_block, report_table
from components.workspace import boot, evaluate_all, open_workspace, require_scored
from config import MODEL_KEYS, MODEL_LABELS
from core.evaluation import confusion_counts

boot("Session2Shop | Error Analysis")

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)
require_scored(result, selected_key)

label = MODEL_LABELS[selected_key]

y_true = result["y_true"]
y_pred = result["preds"][selected_key]
metrics = result["metrics"][selected_key]
summary = result["summary"]
frame = result["comparison"]

tn, fp, fn, tp = confusion_counts(y_true, y_pred)
n_pos = max(summary["purchasers"], 1)
n_neg = max(summary["non_purchasers"], 1)

page_header(
    "Error Analysis",
    f"{label} - errors and their cost",
    "A confusion matrix is not four numbers; it is two different kinds of "
    "mistake with two different price tags. The figures below come from the "
    "saved classifier applied to the held-out set.",
)

stat_strip(
    [
        ("False alarms", f"{fp:,} ({fp / n_neg * 100:.1f}% of non-buyers)"),
        ("Missed buyers", f"{fn:,} ({fn / n_pos * 100:.1f}% of buyers)"),
        ("Caught", f"{tp:,} ({tp / n_pos * 100:.1f}% of buyers)"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
    ]
)

# ------------------------------------------------------------- 01 the matrix
st.markdown(
    section("01", "Confusion matrix", f"{label} on {summary['rows']:,} held-out rows"),
    unsafe_allow_html=True,
)
confusion_block(y_true, y_pred)

# ------------------------------------------------------- 02 what it costs
st.markdown(
    section("02", "What each error costs", "the two mistakes are not equal"),
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<div class="s2s-eyebrow is-accent" style="margin-bottom:.5rem">'
        "False alarms &mdash; wasted spend</div>"
        f'<div class="s2s-kpi"><div class="k">Non-buyers targeted as buyers</div>'
        f'<div class="v">{fp:,}</div>'
        f"<div class='s'>That is {fp / n_neg * 100:.1f}% of every genuine "
        f"non-purchaser in the set. Precision "
        f"<b>{metrics['precision']:.4f}</b> is the share of these flags that "
        "were correct.</div></div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="s2s-eyebrow is-accent" style="margin-bottom:.5rem">'
        "Missed buyers &mdash; lost revenue</div>"
        f'<div class="s2s-kpi"><div class="k">Real purchasers not caught</div>'
        f'<div class="v">{fn:,}</div>'
        f"<div class='s'>That is {fn / n_pos * 100:.1f}% of every actual "
        f"purchaser in the set. Recall <b>{metrics['recall']:.4f}</b> is the "
        "share of buyers this model did find.</div></div>",
        unsafe_allow_html=True,
    )

st.caption(
    f"Because buyers are only "
    f"**{summary['conversion_rate'] * 100:.1f}%** of rows, the two columns "
    "have very different sizes - which is exactly why accuracy alone cannot "
    "tell these models apart."
)

# ------------------------------------------------------------ 03 the report
st.markdown(
    section("03", "Classification report", "per-class precision, recall, F1"),
    unsafe_allow_html=True,
)
left, right = st.columns([1, 1.15], gap="large")
with left:
    report_table(y_true, y_pred)
with right:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">Reading it</div>'
        '<div style="font-size:.9rem;color:#4B5563;line-height:1.65">'
        f"<b>Precision</b> answers: when this model says &ldquo;will "
        "purchase&rdquo;, how often is it right? "
        f"<b>{metrics['precision']:.4f}</b><br><br>"
        f"<b>Recall</b> answers: of the sessions that really did purchase, "
        f"how many did it find? <b>{metrics['recall']:.4f}</b><br><br>"
        f"<b>F1</b> is the harmonic mean of the two, "
        f"<b>{metrics['f1']:.4f}</b>, and punishes a model that is strong on "
        "one only because it is hopeless on the other.<br><br>"
        f"<b>Support</b> is how many rows actually belong to each class: "
        f"{summary['purchasers']:,} purchasers against "
        f"{summary['non_purchasers']:,} non-purchasers.</div>",
        unsafe_allow_html=True,
    )

st.markdown(rule(), unsafe_allow_html=True)

# --------------------------------------------------- 04 all five error profiles
st.markdown(
    section(
        "04",
        "Error profile of all five",
        "who buys the most precision, and who pays for it",
    ),
    unsafe_allow_html=True,
)

head = (
    "<tr><th class='l'>Model</th><th>False alarms</th><th>Missed buyers</th>"
    "<th>Precision</th><th>Recall</th><th>F1</th></tr>"
)
rows = []
for key in MODEL_KEYS:
    m = result["metrics"].get(key)
    p = result["preds"].get(key)
    if m is None or p is None:
        rows.append(
            f"<tr><td class='l'>{MODEL_LABELS[key]}</td>"
            "<td class='num' colspan='5'>&mdash;</td></tr>"
        )
        continue
    _, c_fp, c_fn, _ = confusion_counts(y_true, p)
    sel = ' class="sel"' if key == selected_key else ""
    rows.append(
        f"<tr{sel}><td class='l'>{MODEL_LABELS[key]}</td>"
        f"<td class='num'>{c_fp:,}</td>"
        f"<td class='num'>{c_fn:,}</td>"
        f"<td class='num'>{m['precision']:.4f}</td>"
        f"<td class='num'>{m['recall']:.4f}</td>"
        f"<td class='num'>{m['f1']:.4f}</td></tr>"
    )
st.markdown(
    f'<table class="s2s-tbl">{head}{"".join(rows)}</table>', unsafe_allow_html=True
)

# The caption below is generated from this table rather than written down:
# which model recalls the most buyers, which flags the least reliably and
# which balances the two all move whenever the models are retrained.
scored = {k: result["metrics"][k] for k in MODEL_KEYS if result["metrics"].get(k)}
recall_by = {k: m["recall"] for k, m in scored.items()}
prec_by = {k: m["precision"] for k, m in scored.items()}
f1_by = {k: m["f1"] for k, m in scored.items()}
recall_leader = max(recall_by, key=recall_by.get)
precision_laggard = min(prec_by, key=prec_by.get)
f1_leader = max(f1_by, key=f1_by.get)

st.caption(
    (
        f"**{MODEL_LABELS[recall_leader]}** sits at the extreme of this "
        f"table: it catches the most real purchasers (recall "
        f"{recall_by[recall_leader]:.4f}) while its own purchase flags are "
        f"the least reliable (precision {prec_by[recall_leader]:.4f}). "
    )
    if recall_leader == precision_laggard
    else (
        f"**{MODEL_LABELS[recall_leader]}** has the highest recall in this "
        f"table ({recall_by[recall_leader]:.4f}); "
        f"**{MODEL_LABELS[precision_laggard]}** has the lowest precision "
        f"({prec_by[precision_laggard]:.4f}). "
    )
    + f"**{MODEL_LABELS[f1_leader]}** holds the best balance of the two, "
    f"with the highest F1 ({f1_by[f1_leader]:.4f})."
)

# ------------------------------------------------------------ 05 the numbers
st.markdown(
    section("05", "The six metrics again", f"for {label}"),
    unsafe_allow_html=True,
)
metric_cards(metrics)

st.markdown(rule(), unsafe_allow_html=True)
