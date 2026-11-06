"""
components.sidebar - the shared control rail.

Called at the top of every page so the dataset and the selected classifier
are identical wherever the user navigates.

Two pieces of cross-page state live here:

    st.session_state["df"]             the loaded dataset
    st.session_state["source_name"]    where it came from
    st.session_state["model_choice"]   which classifier is under inspection

Streamlit's session_state is per-SESSION, not per-page, so a value written
on Overview is still present on Methodology.  That is exactly why loading
must write into session_state rather than a local variable: st.button
returns True for exactly one run, and without this the dataset would vanish
the moment any other widget was touched.

On a first visit with nothing in session_state the bundled test_data.csv is
loaded automatically, so the app opens on real numbers instead of an empty
rail; the file that ends up loaded is always shown in the page's stat strip
by name, and the reset button returns to it from any uploaded file.

One caveat, discovered by testing rather than reading: a key passed to
selectbox(key=...) does NOT make the same journey.  The dataset survived
page navigation while the selected model reset to SVM.  Plain keys
survive; widget-bound keys are scoped to the page that created them.  The
classifier selection therefore lives in the plain key "model_choice" and
is mirrored into a widget key around each call - see render_sidebar().
"""

import pandas as pd
import streamlit as st

from components.layout import chip, eyebrow, rule
from config import MODEL_KEYS, MODEL_LABELS
from core.inference import BUNDLED_TEST_CSV
from core.validation import validate_dataset

BUNDLED_SOURCE = "test_data.csv (bundled)"


def render_sidebar():
    """Render the shared sidebar.  Returns (df, source_name, selected_key)."""
    with st.sidebar:
        st.markdown(eyebrow("Dataset"), unsafe_allow_html=True)

        # The reset control sits ABOVE the uploader on purpose.  To clear an
        # upload you have to empty the uploader's own widget key, and
        # Streamlit refuses that once the widget has been instantiated in
        # this run - so the button has to be rendered first.
        reset = st.button(
            "Reset to bundled test_data.csv",
            key="use_bundled_button",
            use_container_width=True,
            help="Clears any uploaded file and returns to the copy shipped "
                 "with the repository.",
        )
        if reset:
            st.session_state["dataset_uploader"] = None
            if BUNDLED_TEST_CSV.exists():
                st.session_state["df"] = pd.read_csv(BUNDLED_TEST_CSV)
                st.session_state["source_name"] = BUNDLED_SOURCE

        uploaded = st.file_uploader(
            "Test dataset (CSV)",
            type=["csv"],
            key="dataset_uploader",
            help="test_data.csv from the repository: all 17 feature columns "
                 "plus the Revenue target column.",
        )

        # FIRST VISIT: load the bundled file so the app is usable the moment
        # it opens, instead of presenting an empty rail and a dead page.
        # An upload below still takes precedence on later runs.
        if st.session_state.get("df") is None and BUNDLED_TEST_CSV.exists():
            st.session_state["df"] = pd.read_csv(BUNDLED_TEST_CSV)
            st.session_state["source_name"] = BUNDLED_SOURCE

        # ------------------------------------------------ load / persist
        if uploaded is not None:
            if st.session_state.get("source_name") != uploaded.name:
                try:
                    st.session_state["df"] = pd.read_csv(uploaded)
                    st.session_state["source_name"] = uploaded.name
                except Exception as exc:
                    st.error(f"Could not read that CSV: `{exc}`")

        df = st.session_state.get("df")
        source_name = st.session_state.get("source_name")

        # ------------------------------------------------ validation flag
        if df is None:
            st.caption("No dataset loaded yet.")
        else:
            problems = validate_dataset(df)
            if problems:
                st.warning(
                    f"**{len(problems)} problem(s)** - open Data Workspace "
                    "for details."
                )
            else:
                st.success(f"Validated - {len(df):,} rows")

        st.markdown(rule(), unsafe_allow_html=True)
        st.markdown(eyebrow("Classifier"), unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # WHY THIS IS TWO KEYS RATHER THAN ONE
        #
        # st.session_state["df"] survives a page change because nothing binds
        # it to a widget. A key handed to selectbox(key=...) does NOT survive
        # the same navigation: pick Naive Bayes on Overview, open Model
        # Explorer, and it comes back as SVM. The widget's own state is scoped
        # to the page that created it, so the value has to live in a plain
        # key and be copied both directions around the widget.
        #
        #   model_choice     plain key   - survives navigation, source of truth
        #   _model_widget    widget key  - Streamlit's, seeded from the above
        #
        # Seeding must happen BEFORE the widget is instantiated (Streamlit
        # forbids mutating a widget key afterwards), but ONLY when the widget
        # key is absent. Seeding it unconditionally on every run overwrites
        # the value the user just chose before the widget can report it, which
        # snaps the dropdown straight back to SVM the moment it is changed.
        # Absent means "we just arrived on this page" - restore from the plain
        # key. Present means "this is the live choice" - leave it alone.
        # ------------------------------------------------------------------
        if "model_choice" not in st.session_state:
            st.session_state["model_choice"] = "svm"
        if "_model_widget" not in st.session_state:
            st.session_state["_model_widget"] = st.session_state["model_choice"]

        selected_key = st.selectbox(
            "Model",
            options=MODEL_KEYS,
            format_func=lambda k: MODEL_LABELS[k],
            key="_model_widget",
            label_visibility="collapsed",
            help="Which saved classifier to evaluate. Loading a model never "
                 "retrains it.",
        )
        st.session_state["model_choice"] = selected_key

        st.markdown(rule(), unsafe_allow_html=True)
        st.markdown(
            eyebrow("Mode")
            + '<div style="font-size:.82rem;color:#374151;margin-top:.5rem">'
            "<b>Inference only.</b> Models and the preprocessor were fitted "
            f"once by {chip('train.py')}. This page only loads, transforms "
            "and scores.</div>",
            unsafe_allow_html=True,
        )

    return df, source_name, selected_key
