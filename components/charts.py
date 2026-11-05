"""
components.charts - comparison and distribution visuals.

All bars are scaled from ZERO to one.  Scaling from the observed minimum
draws the weakest model as a near-full bar and exaggerates every gap, which
is precisely the sort of chart that invites awkward questions in a
submission.  The exact figure is printed beside every bar regardless.
"""

import pandas as pd
import streamlit as st

from components.layout import METRIC_LABELS
from config import METRIC_KEYS, MODEL_KEYS, MODEL_LABELS

EM_DASH = "&mdash;"


def comparison_bars(comparison, selected_key, metric):
    """Horizontal bars for one metric; the selected model is accent-filled.

    comparison is indexed by model KEY (see core.evaluation.comparison_frame).
    """
    rows = []
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        v = comparison.loc[key, metric] if key in comparison.index else pd.NA
        on = " on" if key == selected_key else ""

        if pd.isna(v):
            width, shown = 0.0, EM_DASH
        else:
            fv = float(v)
            width = max(0.0, min(100.0, fv * 100.0))
            shown = f"{fv:.4f}"

        rows.append(
            f'<div class="s2s-bar{on}">'
            f'<div class="nm">{label}</div>'
            f'<div class="tk"><div class="fl" style="width:{width:.1f}%"></div></div>'
            f'<div class="vl">{shown}</div>'
            "</div>"
        )
    st.markdown(f'<div class="s2s-bars">{"".join(rows)}</div>', unsafe_allow_html=True)


def metric_selector(default="mcc", key="comparison_metric"):
    """Dropdown for choosing which metric the bars and ranking use."""
    return st.selectbox(
        "Metric",
        options=list(METRIC_KEYS),
        format_func=lambda k: METRIC_LABELS[k],
        index=list(METRIC_KEYS).index(default),
        key=key,
        label_visibility="collapsed",
    )


def ranking_list(frame, metric, selected_key):
    """Ordered leaderboard on one metric, best first, selected model tinted."""
    col = frame[metric].dropna().sort_values(ascending=False)
    rows = []
    for i, (key, value) in enumerate(col.items(), start=1):
        on = " on" if key == selected_key else ""
        rows.append(
            f'<div class="s2s-bar{on}" '
            'style="grid-template-columns:30px minmax(0,1fr) 64px">'
            f'<div class="nm" style="font-family:ui-monospace,Menlo,monospace;'
            f'color:#9CA3AF;font-size:.8rem">{i}</div>'
            f'<div class="nm">{MODEL_LABELS.get(key, key)}</div>'
            f'<div class="vl">{float(value):.4f}</div>'
            "</div>"
        )
    if not rows:
        rows.append('<div class="s2s-bar"><div class="nm">No model scored.</div></div>')
    st.markdown(f'<div class="s2s-bars">{" ".join(rows)}</div>', unsafe_allow_html=True)


def class_distribution(summary):
    """Two-bar split of purchasers vs non-purchasers.

    Makes the 15.6% positive rate visible, which is the single fact that
    explains why accuracy flatters every model here.
    """
    total = max(summary["rows"], 1)
    pos_pct = 100.0 * summary["purchasers"] / total
    neg_pct = 100.0 * summary["non_purchasers"] / total

    def bar(label, pct, count, accent):
        fill = "#0F766E" if accent else "#D1D5DB"
        return (
            f'<div class="s2s-bar on" style="grid-template-columns:150px minmax(0,1fr) 118px">'
            f'<div class="nm" style="color:#374151;font-weight:600">{label}</div>'
            f'<div class="tk" style="height:24px">'
            f'<div class="fl" style="width:{pct:.1f}%;background:{fill}"></div></div>'
            f'<div class="vl" style="text-align:right">{count:,} &middot; {pct:.1f}%</div>'
            "</div>"
        )

    html = (
        '<div class="s2s-bars">'
        + bar("Will purchase", pos_pct, summary["purchasers"], True)
        + bar("No purchase", neg_pct, summary["non_purchasers"], False)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
