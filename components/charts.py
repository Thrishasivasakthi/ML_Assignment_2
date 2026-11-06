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


def roc_overlay(result, selected_key):
    """All five ROC curves on one set of axes, selected model emphasised.

    ROC is drawn from the probabilities computed alongside the metrics rather
    than from a second scoring pass, so the AUC printed in the legend is the
    same number the comparison table prints.  Two colours only - accent teal
    for the classifier under inspection, slate for the field - because five
    competing hues would say nothing about which line is which; the legend
    carries the model names and their AUC.
    """
    probas = result.get("probas") or {}
    y_true = result["y_true"]
    available = [k for k in MODEL_KEYS if probas.get(k) is not None]

    if not available:
        st.caption(
            "No classifier produced probabilities on this dataset, so there "
            "is no curve to draw."
        )
        return
    if len(set(y_true)) < 2:
        st.caption(
            "ROC is undefined when only one class is present in the loaded "
            "rows, so no curve is drawn."
        )
        return

    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never opens a window
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # What a random ranker would achieve - the reference every curve is read
    # against, so a curve that hugs it is visibly worthless.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#CBD5E1")

    for key in MODEL_KEYS:
        if key not in available:
            continue
        fpr, tpr, _ = roc_curve(y_true, probas[key])
        auc = float(result["metrics"][key]["auc"])
        if key == selected_key:
            ax.plot(fpr, tpr, color="#0F766E", lw=2.6, zorder=3,
                    label=f"{MODEL_LABELS[key]}  {auc:.4f}")
        else:
            ax.plot(fpr, tpr, color="#94A3B8", lw=1.4, alpha=0.95, zorder=2,
                    label=f"{MODEL_LABELS[key]}  {auc:.4f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate", fontsize=9, color="#6B7280")
    ax.set_ylabel("True positive rate", fontsize=9, color="#6B7280")
    ax.grid(color="#F3F4F6", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#E5E7EB")
    ax.tick_params(colors="#6B7280", labelsize=8)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color("#374151")
    ax.set_title(
        "ROC - the classifier under inspection is drawn in teal",
        fontsize=9.5, color="#6B7280", loc="left", pad=10,
    )

    st.pyplot(fig, clear_figure=True)


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

    Makes the minority positive rate visible, which is the single fact that
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
