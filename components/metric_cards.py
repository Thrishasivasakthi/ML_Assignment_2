"""
components.metric_cards - the six required metrics, and overview KPI tiles.

The six cards are the assignment's §Step 6 "display of evaluation metrics"
requirement, so they appear on Overview as well as on Model Explorer.
"""

import streamlit as st

from components.layout import BASELINE_ACCURACY, METRIC_INFO, METRIC_LABELS


def metric_cards(metrics):
    """Render the six metric cards.

    metrics: dict keyed by metric name -> float (from core.evaluation).
    """
    cards = "".join(
        '<div class="s2s-card">'
        f'<div class="lab">{METRIC_LABELS[k]}</div>'
        f'<div class="val">{metrics[k]:.4f}</div>'
        f'<div class="dsc">{METRIC_INFO[k]}</div>'
        "</div>"
        for k in metrics
    )
    st.markdown(f'<div class="s2s-cards">{cards}</div>', unsafe_allow_html=True)


def baseline_note():
    """Why accuracy alone is not enough on an 84.4% imbalanced dataset."""
    st.markdown(
        '<div class="s2s-baseline"><b>Baseline context:</b> a classifier that '
        f"always predicts &ldquo;No purchase&rdquo; scores "
        f"<b>{BASELINE_ACCURACY * 100:.1f}% accuracy</b> on this dataset "
        "without learning anything &mdash; which is why recall, F1 and MCC "
        "are reported alongside accuracy.</div>",
        unsafe_allow_html=True,
    )


def kpi_tiles(tiles):
    """Overview KPI row.

    tiles: list of dicts with keys
        label  - uppercase eyebrow
        value  - big figure (str, already formatted)
        sub    - small caption underneath
        accent - optional bool; True renders the tile in solid teal
        text   - optional bool; True renders the value in serif rather than mono
    """
    cells = []
    for t in tiles:
        cls = "s2s-kpi"
        if t.get("accent"):
            cls += " is-accent"
        vcls = "v text" if t.get("text") else "v"
        cells.append(
            f'<div class="{cls}">'
            f'<div class="k">{t["label"]}</div>'
            f'<div class="{vcls}">{t["value"]}</div>'
            f'<div class="s">{t.get("sub", "")}</div>'
            "</div>"
        )
    st.markdown(f'<div class="s2s-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)
