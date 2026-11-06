"""
components.tables - confusion matrix, classification report, comparison table.

All three read from core.evaluation so the numbers cannot disagree with the
ones metrics.csv was written from.  A metric that failed to score renders as
an em dash rather than the string "nan".
"""

import pandas as pd
import streamlit as st

from components.layout import METRIC_LABELS
from config import MODEL_KEYS, MODEL_LABELS
from core.evaluation import NEG_LABEL, POS_LABEL, confusion_counts, report_dict

EM_DASH = "&mdash;"

# Columns whose extremes get painted.  Precision and Recall are excluded on
# purpose - see comparison_table()'s docstring.
HIGHLIGHTED_METRICS = frozenset({"accuracy", "auc", "f1", "mcc"})

# +1 means higher is better, -1 means lower is better.  All six of the
# assignment's metrics are higher-is-better; the map exists so the table
# stays correct if that ever stops being true.
DIRECTION = {name: 1 for name in METRIC_LABELS}


def confusion_block(y_true, y_pred):
    """2x2 matrix: correct cells in emerald, errors in rose."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    def cell(value, ok, note):
        cls = "s2s-cm-ok" if ok else "s2s-cm-err"
        return (
            f'<div class="s2s-cm-c {cls}"><b>{value:,}</b>'
            f"<span>{note}</span></div>"
        )

    st.markdown(
        '<div class="s2s-cm">'
        '<div class="s2s-cm-h"></div>'
        f'<div class="s2s-cm-h">Pred {NEG_LABEL}</div>'
        f'<div class="s2s-cm-h">Pred {POS_LABEL}</div>'
        f'<div class="s2s-cm-r">Actual<br>{NEG_LABEL}</div>'
        f"{cell(tn, True, 'correct')}"
        f"{cell(fp, False, 'false alarm')}"
        f'<div class="s2s-cm-r">Actual<br>{POS_LABEL}</div>'
        f"{cell(fn, False, 'missed')}"
        f"{cell(tp, True, 'correct')}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"**{tn + tp:,}** correct of **{tn + fp + fn + tp:,}**"
        f" &nbsp;·&nbsp; **{fp:,}** false alarms"
        f" &nbsp;·&nbsp; **{fn:,}** missed purchases"
    )
    return tn, fp, fn, tp


def report_table(y_true, y_pred):
    """Per-class precision / recall / F1 / support."""
    rep = report_dict(y_true, y_pred)
    head = (
        "<tr><th class='l'>Class</th><th>Precision</th><th>Recall</th>"
        "<th>F1</th><th>Support</th></tr>"
    )
    rows = []
    for name in (NEG_LABEL, POS_LABEL):
        r = rep[name]
        rows.append(
            f"<tr><td class='l'>{name}</td>"
            f"<td class='num'>{r['precision']:.4f}</td>"
            f"<td class='num'>{r['recall']:.4f}</td>"
            f"<td class='num'>{r['f1-score']:.4f}</td>"
            f"<td class='num'>{int(r['support']):,}</td></tr>"
        )
    w = rep["weighted avg"]
    rows.append(
        "<tr><td class='l' style='color:#6B7280'>Weighted avg</td>"
        f"<td class='num'>{w['precision']:.4f}</td>"
        f"<td class='num'>{w['recall']:.4f}</td>"
        f"<td class='num'>{w['f1-score']:.4f}</td>"
        f"<td class='num'>{int(w['support']):,}</td></tr>"
    )
    st.markdown(
        f'<table class="s2s-tbl">{head}{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )


def metric_leaders_table(frame):
    """Which classifier wins each of the six metrics, and by how much."""
    head = "<tr><th class='l'>Metric</th><th>Leader</th><th>Value</th></tr>"
    rows = []
    for col_name in frame.columns:
        col = frame[col_name].dropna()
        if col.empty:
            continue
        key = col.idxmax()
        rows.append(
            f"<tr><td class='l'>{METRIC_LABELS[col_name]}</td>"
            f"<td class='l' style='color:#0F766E;font-weight:600'>"
            f"{MODEL_LABELS.get(key, key)}</td>"
            f"<td class='num best'>{float(col.max()):.4f}</td></tr>"
        )
    st.markdown(
        f'<table class="s2s-tbl">{head}{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )


def comparison_table(comparison, selected_key, label_of=None, legend=True):
    """Six-metric table across all five models.

    Best value in green, weakest in red, selected row tinted.  comparison is
    indexed by model KEY; label_of maps a key to its display name.

    WHICH COLUMNS GET COLOURED
    --------------------------
    Only accuracy, AUC, F1 and MCC.  Precision and Recall are left plain
    because on a 15.6% positive class either one can be pushed up by simply
    flagging more sessions as purchases - painting the extreme green would
    reward exactly the behaviour the required metric set exists to catch.
    Read the pair together instead, which is what the legend says.

    The direction of "better" is looked up rather than assumed, so adding a
    lower-is-better metric later cannot silently paint its worst value green.
    """
    cols = list(comparison.columns)
    head = (
        "<tr><th class='l'>Model</th>"
        + "".join(f"<th>{METRIC_LABELS[c]}</th>" for c in cols)
        + "</tr>"
    )

    # skipna is the default, so a failed model does not poison best/worst.
    def extreme(column, best_side):
        values = comparison[column].dropna()
        if values.empty:
            return None
        # direction > 0 -> higher is better; < 0 -> lower is better.
        direction = DIRECTION.get(column, 1)
        target = values.max() if (best_side == (direction > 0)) else values.min()
        return target

    best = {c: extreme(c, True) for c in cols}
    worst = {c: extreme(c, False) for c in cols}

    def name_for(key):
        if label_of:
            return label_of(key)
        return MODEL_LABELS.get(key, key)

    rows = []
    for key in MODEL_KEYS:
        if key not in comparison.index:
            continue
        cls = ' class="sel"' if key == selected_key else ""
        cells = []
        for c in cols:
            v = comparison.loc[key, c]
            if pd.isna(v):
                cells.append(f"<td class='num'>{EM_DASH}</td>")
                continue
            fv = float(v)
            k = ""
            if c in HIGHLIGHTED_METRICS and best[c] is not None and fv == best[c]:
                k = " best"
            elif c in HIGHLIGHTED_METRICS and worst[c] is not None and fv == worst[c]:
                k = " worst"
            cells.append(f"<td class='num{k}'>{fv:.4f}</td>")
        rows.append(f"<tr{cls}><td class='l'>{name_for(key)}</td>{''.join(cells)}</tr>")

    st.markdown(
        f'<table class="s2s-tbl">{head}{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )

    if legend:
        st.caption(
            "**Green** is the best value in a column, **red** the weakest - "
            "for Accuracy, AUC, F1 and MCC only. Precision and Recall are "
            "left uncoloured: either can be raised on an imbalanced dataset "
            "simply by flagging more sessions as purchases, so the pair is "
            "meant to be read together rather than ranked."
        )
