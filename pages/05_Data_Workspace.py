"""
pages/05_Data_Workspace.py - upload, validate and inspect the test CSV.

Unlike the other pages this one deliberately does NOT stop when the data is
invalid: reporting what is wrong with a file is the entire purpose of a data
workspace.  It only stops when nothing has been loaded at all.

The spec requires the uploaded test CSV to contain all input features and
the target column, because the application evaluates against those labels.
That interpretation is kept deliberately - this is an evaluation tool, not a
general prediction service.
"""

import pandas as pd
import streamlit as st

from components.charts import class_distribution
from components.layout import chip, page_header, rule, section, stat_strip
from components.sidebar import BUNDLED_SOURCE, render_sidebar
from components.workspace import boot, get_artifacts
from config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from core.inference import BUNDLED_TEST_CSV
from core.validation import dataset_summary, validate_dataset

PREVIEW_ROWS = 20

boot("Session2Shop | Data Workspace")

df, source_name, selected_key = render_sidebar()

page_header(
    "Data Workspace",
    "Test dataset",
    "Upload the held-out CSV, confirm it is structurally sound, and inspect "
    "what is inside it before any model is asked to score it.",
)

if df is None:
    st.markdown(
        '<div class="s2s-empty">'
        '<div class="s2s-eyebrow">No dataset loaded</div>'
        "<p>Use the panel on the left: either drop in a CSV, or click "
        "<b>Reset to bundled test_data.csv</b> to load the copy that ships with "
        "this repository. The file must contain all 17 feature columns plus "
        f"the <b>{TARGET}</b> target column.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------- validation
problems = validate_dataset(df)
summary = dataset_summary(df)

st.markdown(
    section("01", "Validation status", "checked before any model runs"),
    unsafe_allow_html=True,
)

if problems:
    st.error(f"**{len(problems)} problem(s) found** - this file cannot be scored")
    for p in problems:
        st.markdown(f"- {p}")
else:
    st.success(
        f"**Valid.** All {len(FEATURES)} feature columns present, "
        f"`{TARGET}` contains both classes and is numeric, and every numeric "
        "column parses cleanly."
    )

st.caption(
    "Expected columns: "
    + ", ".join(f"`{c}`" for c in FEATURES)
    + f", plus the target `{TARGET}`."
)

# --------------------------------------------------------------- summary
st.markdown(
    section("02", "What is in the file", "shape and target balance"),
    unsafe_allow_html=True,
)

stat_strip(
    [
        ("Source", source_name),
        ("Rows", f"{summary['rows']:,}"),
        ("Features", f"{summary['features']} ({summary['numeric_features']} numeric + "
                     f"{summary['categorical_features']} categorical)"),
        ("Purchasers", f"{summary['purchasers']:,}"),
        ("Conversion", f"{summary['conversion_rate'] * 100:.2f}%"),
    ]
)

class_distribution(summary)

st.caption(
    f"An always-&ldquo;No&rdquo; classifier would score "
    f"**{100 - summary['conversion_rate'] * 100:.1f}% accuracy** on this file "
    "without learning anything - which is why accuracy is never reported alone."
)

# ------------------------------------------------------------ column inventory
st.markdown(
    section("03", "Column inventory", "how the 17 features are treated"),
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">'
        f"Numeric &mdash; {len(NUMERIC_FEATURES)} columns, standardised</div>"
        '<div style="font-size:.87rem;color:#4B5563;line-height:1.8">'
        + "<br>".join(chip(c) for c in NUMERIC_FEATURES)
        + "</div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="s2s-eyebrow" style="margin-bottom:.6rem">'
        f"Categorical &mdash; {len(CATEGORICAL_FEATURES)} columns, one-hot encoded</div>"
        '<div style="font-size:.87rem;color:#4B5563;line-height:1.8">'
        + "<br>".join(chip(c) for c in CATEGORICAL_FEATURES)
        + "</div>",
        unsafe_allow_html=True,
    )

st.caption(
    "The four nominal identifiers (OperatingSystems, Browser, Region, "
    "TrafficType) arrive as integers but are category labels, not quantities - "
    "so they are encoded rather than scaled."
)

# ---------------------------------------------------------------- preview
st.markdown(
    section("04", "Preview", f"first {min(PREVIEW_ROWS, len(df))} rows"),
    unsafe_allow_html=True,
)
st.dataframe(
    df.head(PREVIEW_ROWS),
    use_container_width=True,
    height=360,
)

# ------------------------------------------------------------- utilities
st.markdown(rule(), unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.download_button(
        "Download this CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=(source_name or "uploaded.csv").replace(".csv", "") + "_copy.csv",
        mime="text/csv",
        use_container_width=True,
        help="Keeps the exact file currently loaded, so the same data can be "
             "rescored later.",
    )
with right:
    # The only chip() call site that was going through st.caption, which
    # escapes HTML by default - so this line printed its <span> tags as
    # literal text.  unsafe_allow_html=True is the whole fix: every other
    # call site already passes it to st.markdown.
    st.caption(
        f"Loaded from {chip(source_name or '-')}.<br>"
        f"Bundled reference: {chip('test_data.csv')}"
        + ("" if source_name == BUNDLED_SOURCE else ""),
        unsafe_allow_html=True,
    )

st.markdown(rule(), unsafe_allow_html=True)
