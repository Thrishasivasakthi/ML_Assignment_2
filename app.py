"""
app.py - Streamlit front end for ML Assignment 2 (M.Tech AIML).

    streamlit run app.py

WHAT THIS APP DOES (spec Section 6, 4 marks)
--------------------------------------------
  a. Dataset upload option (CSV)                    [1 mark]
  b. Model selection dropdown                       [1 mark]
  c. Display of evaluation metrics                  [1 mark]
  d. Confusion matrix / classification report       [1 mark]

HARD CONSTRAINT FROM THE SPEC
-----------------------------
"The app must load the saved model, calculate the evaluation metrics, and
display the results WITHOUT retraining the model on the uploaded test data."

There is no .fit() call anywhere in this file.  Artifacts written by
train.py are loaded, applied, and scored - that is the whole job.

DESIGN NOTE
-----------
The visual language is a small, deliberate system rather than per-widget
styling:

  palette     ink / slate neutrals + ONE accent (deep teal)
  type        sans for prose, mono for every number (tabular-nums)
  hierarchy   eyebrow label -> headline -> hairline rule -> content
  sections    numbered 01 / 02 / 03 so the page reads top-to-bottom
  emoji       none - typographic labels carry the structure instead

The accent is teal on purpose: indigo/violet is the default palette of
generated Streamlit apps and reads as unoriginal.
"""

from pathlib import Path

# config is imported FIRST on purpose: it installs the joblib/loky
# environment fix before this file loads any .joblib artifact.
from config import (
    FEATURES,
    MODEL_KEYS,
    MODEL_LABELS,
    TARGET,
    prepare_features,
)

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.joblib"
MODEL_DIR = BASE_DIR / "model"
BUNDLED_TEST_CSV = BASE_DIR / "test_data.csv"

NEG_LABEL = "No purchase"
POS_LABEL = "Will purchase"

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}

METRIC_INFO = {
    "accuracy": "Share of rows classified correctly",
    "auc": "Ranking quality, threshold-free",
    "precision": "Of sessions flagged as buyers, the share that really were",
    "recall": "Of real buyers, how many we caught",
    "f1": "Harmonic mean of precision and recall",
    "mcc": "Correlation between prediction and truth",
}

# A model that always answers "No purchase" - the yardstick every metric
# on this imbalanced dataset has to be judged against.
BASELINE_ACCURACY = 0.844

st.set_page_config(
    page_title="Session2Shop | Purchase Intent Classifier",
    page_icon="\u25c6",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Design system
# ===========================================================================
CSS = """
<style>
/* ---------- primitives ---------------------------------------------- */
.s2s-eyebrow {
    font-size: .66rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #9CA3AF;
}
.s2s-eyebrow.is-accent { color: #0F766E; }
.s2s-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.s2s-rule { border: 0; border-top: 1px solid #E5E7EB; margin: 1.15rem 0; }

/* ---------- masthead ------------------------------------------------- */
/* Display type: serif, deliberately.  Body and widget text are sans
   (set in .streamlit/config.toml) and every number is mono, so the page
   runs a three-typeface system - display / text / data - instead of
   Streamlit's single default face. */
.s2s-masthead h1 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    font-size: 2.8rem; font-weight: 700; letter-spacing: -.02em;
    color: #111827; margin: .55rem 0 .45rem; line-height: 1.06;
}
.s2s-lede {
    font-size: 1.02rem; color: #4B5563; max-width: 64ch;
    line-height: 1.6; margin: 0;
}
.s2s-stats { display: flex; flex-wrap: wrap; gap: 2.6rem; margin-top: .2rem; }
.s2s-stat .k {
    font-size: .62rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #9CA3AF;
}
.s2s-stat .v {
    font-size: 1.12rem; font-weight: 700; color: #111827; margin-top: .3rem;
    font-variant-numeric: tabular-nums;
}

/* ---------- section headers ----------------------------------------- */
.s2s-sec {
    display: flex; align-items: baseline; gap: .8rem;
    flex-wrap: wrap; margin-top: 2.4rem;
}
.s2s-sec .n {
    font-size: .78rem; font-weight: 700; color: #0F766E;
    font-family: ui-monospace, Menlo, monospace; letter-spacing: .04em;
}
.s2s-sec h2 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    font-size: 1.4rem; font-weight: 700; letter-spacing: -.01em;
    color: #111827; margin: 0;
}
.s2s-sec .sub { color: #6B7280; font-size: .88rem; font-style: italic; }

/* Inline chip.  NB: a raw code tag CANNOT be used in this app's HTML - Streamlit's
   markdown parser rewrites it into a full stCode BLOCK component (with a
   copy-to-clipboard button) that breaks the sentence across lines, and no
   amount of CSS on the inner element fixes it.  A span with our own class is
   the only reliable route to an inline chip. */
.s2s-code {
    display: inline;
    background: #FFFFFF; border: 1px solid #D1D5DB;
    border-radius: 4px; padding: 1px 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .84em; color: #374151; white-space: nowrap;
}

/* ---------- empty state ---------------------------------------------- */
.s2s-empty {
    background: #F7F8FA; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 1.5rem 1.7rem; max-width: 62ch;
}
.s2s-empty p {
    font-size: .95rem; color: #4B5563; line-height: 1.65;
    margin: .6rem 0 0;
}

/* ---------- metric cards -------------------------------------------- */
.s2s-cards {
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 11px; margin: 1.1rem 0 .3rem;
}
.s2s-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 15px 14px 14px; box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.s2s-card .lab {
    font-size: .61rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: #6B7280;
}
.s2s-card .val {
    font-family: ui-monospace, Menlo, monospace; font-size: 1.5rem;
    font-weight: 600; color: #111827; margin-top: .6rem; letter-spacing: -.02em;
    font-variant-numeric: tabular-nums;
}
.s2s-card .dsc {
    font-size: .68rem; color: #9CA3AF; margin-top: .55rem; line-height: 1.45;
    min-height: 2.9em;
}
.s2s-baseline {
    font-size: .8rem; color: #6B7280; background: #F7F8FA;
    border: 1px solid #E5E7EB; border-left: 3px solid #0F766E;
    border-radius: 0 8px 8px 0; padding: .6rem .9rem; display: inline-block;
}

/* ---------- confusion matrix ---------------------------------------- */
.s2s-cm {
    display: grid; grid-template-columns: auto repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.s2s-cm-h {
    font-size: .62rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #9CA3AF; text-align: center;
    display: flex; align-items: flex-end; justify-content: center;
    padding-bottom: 2px;
}
.s2s-cm-r {
    font-size: .74rem; font-weight: 600; color: #6B7280; text-align: right;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 6px; line-height: 1.25;
}
.s2s-cm-c {
    border-radius: 9px; padding: 17px 8px; text-align: center;
    border: 1px solid transparent;
}
.s2s-cm-c b {
    display: block; font-family: ui-monospace, Menlo, monospace;
    font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums;
}
.s2s-cm-c span {
    font-size: .6rem; letter-spacing: .1em; text-transform: uppercase;
    opacity: .75; display: block; margin-top: 3px;
}
.s2s-cm-ok  { background: #ECFDF5; border-color: #A7F3D0; color: #065F46; }
.s2s-cm-err { background: #FFF1F2; border-color: #FECDD3; color: #9F1239; }

/* ---------- tables --------------------------------------------------- */
.s2s-tbl { width: 100%; border-collapse: collapse; font-size: .87rem; }
.s2s-tbl th {
    font-size: .61rem; font-weight: 700; letter-spacing: .11em;
    text-transform: uppercase; color: #6B7280; padding: 0 8px 9px;
    border-bottom: 1px solid #E5E7EB; text-align: right; white-space: nowrap;
}
.s2s-tbl th.l, .s2s-tbl td.l { text-align: left; }
.s2s-tbl td {
    padding: 10px 8px; border-bottom: 1px solid #F3F4F6; text-align: right;
    font-variant-numeric: tabular-nums; color: #374151;
}
.s2s-tbl td.num { font-family: ui-monospace, Menlo, monospace; font-size: .84rem; }
.s2s-tbl tr.sel td { background: #F0FDFA; }
.s2s-tbl tr.sel td.l { color: #0F766E; font-weight: 700; }
.s2s-tbl td.best { color: #047857; font-weight: 700; }
.s2s-tbl td.worst { color: #BE123C; }

/* ---------- comparison bars ------------------------------------------ */
.s2s-bars { display: flex; flex-direction: column; gap: 9px; }
.s2s-bar {
    display: grid; grid-template-columns: 178px minmax(0, 1fr) 62px;
    align-items: center; gap: 14px;
}
.s2s-bar .nm { font-size: .85rem; color: #6B7280; }
.s2s-bar.on .nm { color: #111827; font-weight: 700; }
.s2s-bar .tk {
    height: 20px; background: #F3F4F6; border-radius: 5px; overflow: hidden;
}
.s2s-bar .fl { height: 100%; background: #D1D5DB; border-radius: 5px; }
.s2s-bar.on .fl { background: #0F766E; }
.s2s-bar .vl {
    font-family: ui-monospace, Menlo, monospace; font-size: .84rem;
    text-align: right; color: #374151; font-variant-numeric: tabular-nums;
}
.s2s-bar.on .vl { color: #0F766E; font-weight: 700; }

/* ---------- sidebar -------------------------------------------------- */
section[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
section[data-testid="stSidebar"] .s2s-eyebrow { margin-top: .2rem; }
div[data-testid="stButton"] > button {
    border-radius: 8px; border: 1px solid #E5E7EB; background: #FFFFFF;
    color: #374151; font-weight: 600; font-size: .83rem; padding: .45rem 1rem;
}
div[data-testid="stButton"] > button:hover {
    border-color: #0F766E; color: #0F766E; background: #F0FDFA;
}
div[data-testid="stFileUploader"] {
    background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 10px;
}

/* ---------- responsive -------------------------------------------------
   Streamlit's block container keeps 80px padding per side, so usable content
   width ~= viewport - 256 (sidebar) - 160 (padding).  A half-screen 912px
   window therefore leaves only ~486px, which crushes a six-across card row to
   72px per card and makes the matrix cells overflow.  Breakpoints are keyed to
   the content width each layout actually receives:

     viewport >= 1400  ->  content >= ~985px   -> 6 cards across
     1100 - 1400       ->  3 cards across
     < 1100            ->  compact matrix + bars (matrix sits in a fixed
                           Streamlit column, so it cannot shrink on its own)
     < 760             ->  2 cards across
*/
@media (max-width: 1400px) {
    .s2s-cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 1100px) {
    /* Section 02's columns are too narrow side-by-side: the 5-column
       classification report overflows its column and is clipped off-screen
       (measured 364px of table in a 228px column). Stack them so the report
       gets the full content width instead. */
    [data-testid="stHorizontalBlock"] { row-gap: 1.75rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* The 7-column comparison table measured 511px against 486px of content. */
    .s2s-tbl { font-size: .82rem; }
    .s2s-tbl td.num { font-size: .8rem; }

    .s2s-cm { gap: 5px; }
    .s2s-cm-c { padding: 13px 4px; }
    .s2s-cm-c b { font-size: 1.2rem; }
    .s2s-cm-c span { font-size: .55rem; letter-spacing: .06em; }
    .s2s-cm-r { font-size: .67rem; padding-right: 3px; }
    .s2s-cm-h { font-size: .57rem; letter-spacing: .06em; }

    .s2s-bar { grid-template-columns: 128px minmax(0, 1fr) 56px; gap: 9px; }
    .s2s-bar .nm { font-size: .79rem; }

    .s2s-masthead h1 { font-size: 2.2rem; }
}
@media (max-width: 760px) {
    .s2s-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .s2s-stats { gap: 1.4rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
# Artifacts + scoring (inference only - no .fit() below this line)
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the fitted preprocessor and all five trained classifiers.

    st.cache_resource matters: Streamlit re-executes the whole script on
    every widget interaction, so without caching each click would re-read
    ~8 MB of joblib files from disk.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODEL_KEYS}
    return preprocessor, models


def validate_upload(df):
    """Return human-readable problems; empty list means the CSV is usable."""
    problems = []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        problems.append(
            "Missing feature column(s): " + ", ".join(f"`{c}`" for c in missing)
        )
    if TARGET not in df.columns:
        problems.append(f"Missing target column `{TARGET}` - no labels, no metrics.")
    if len(df) < 2:
        problems.append("The file needs at least 2 rows to score.")
    return problems


def score_model(model, preprocessor, df):
    """Transform -> predict -> score.  Pure inference, never refitted."""
    X = prepare_features(df)
    y = df[TARGET].astype(int).values
    X_t = preprocessor.transform(X)

    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
    }
    return metrics, y, y_pred


# ===========================================================================
# Render helpers
# ===========================================================================
def eyebrow(text, accent=False):
    cls = "s2s-eyebrow is-accent" if accent else "s2s-eyebrow"
    return f'<div class="{cls}">{text}</div>'


def section(number, title, subtitle=""):
    sub = f'<span class="sub">{subtitle}</span>' if subtitle else ""
    return (
        f'<div class="s2s-sec"><span class="n">{number}</span>'
        f"<h2>{title}</h2>{sub}</div>"
    )


def metric_cards(metrics):
    cards = "".join(
        '<div class="s2s-card">'
        f'<div class="lab">{METRIC_LABELS[k]}</div>'
        f'<div class="val">{metrics[k]:.4f}</div>'
        f'<div class="dsc">{METRIC_INFO[k]}</div>'
        "</div>"
        for k in metrics
    )
    st.markdown(f'<div class="s2s-cards">{cards}</div>', unsafe_allow_html=True)


def confusion_block(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def cell(value, ok, note):
        cls = "s2s-cm-ok" if ok else "s2s-cm-err"
        return f'<div class="s2s-cm-c {cls}"><b>{value:,}</b><span>{note}</span></div>'

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


def report_table(y_true, y_pred):
    rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[NEG_LABEL, POS_LABEL],
        output_dict=True,
        zero_division=0,
    )
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


def comparison_bars(comparison, selected_key, metric):
    """Horizontal bars for one metric; the selected model is accent-filled."""
    rows = []
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        v = float(comparison.loc[label, metric])
        # Scaled from ZERO, not from the observed minimum.  Min-scaling draws
        # the weakest model as a near-full bar and exaggerates every gap -
        # precisely the sort of chart that invites awkward questions in a
        # submission.  The exact figure sits beside each bar regardless.
        width = max(0.0, min(100.0, v * 100.0))
        on = " on" if key == selected_key else ""
        rows.append(
            f'<div class="s2s-bar{on}">'
            f'<div class="nm">{label}</div>'
            f'<div class="tk"><div class="fl" style="width:{width:.1f}%"></div></div>'
            f'<div class="vl">{v:.4f}</div>'
            "</div>"
        )
    st.markdown(f'<div class="s2s-bars">{"".join(rows)}</div>', unsafe_allow_html=True)


def comparison_table(comparison, selected_key):
    cols = list(comparison.columns)
    head = "<tr><th class='l'>Model</th>" + "".join(
        f"<th>{METRIC_LABELS[c]}</th>" for c in cols
    ) + "</tr>"

    best = {c: comparison[c].max() for c in cols}
    worst = {c: comparison[c].min() for c in cols}

    rows = []
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        cls = ' class="sel"' if key == selected_key else ""
        cells = []
        for c in cols:
            v = float(comparison.loc[label, c])
            k = ""
            if v == best[c]:
                k = " best"
            elif v == worst[c]:
                k = " worst"
            cells.append(f"<td class='num{k}'>{v:.4f}</td>")
        rows.append(f"<tr{cls}><td class='l'>{label}</td>{''.join(cells)}</tr>")

    st.markdown(
        f'<table class="s2s-tbl">{head}{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown(eyebrow("Input"), unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Test dataset (CSV)",
        type=["csv"],
        help="test_data.csv from the repository: all 17 features plus the "
        "Revenue target column.",
    )
    use_bundled = st.button(
        "Use bundled test_data.csv",
        use_container_width=True,
        help="Loads the copy shipped in the repository so the app can be "
        "demonstrated without downloading anything first.",
    )

    st.markdown('<hr class="s2s-rule">', unsafe_allow_html=True)
    st.markdown(eyebrow("Classifier"), unsafe_allow_html=True)
    selected_key = st.selectbox(
        "Model",
        options=MODEL_KEYS,
        format_func=lambda k: MODEL_LABELS[k],
        index=MODEL_KEYS.index("svm"),
        label_visibility="collapsed",
    )

    st.markdown(
        '<hr class="s2s-rule">'
        + eyebrow("Mode")
        + '<div class="s2s-prose" style="font-size:.82rem;color:#374151;margin-top:.5rem">'
        "<b>Inference only.</b> Models and the preprocessor were fitted "
        'once by <span class="s2s-code">train.py</span>. This page only loads, '
        "transforms and scores.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Main
# ===========================================================================
st.markdown(
    '<div class="s2s-masthead">'
    + eyebrow("M.Tech AIML &nbsp;/&nbsp; Machine Learning &nbsp;/&nbsp; Assignment 2", True)
    + "<h1>Session2Shop</h1>"
    + '<p class="s2s-lede">Predicting whether an e-commerce browsing session '
    "ends in a purchase, from 17 behavioural features. Upload a held-out test "
    "set, choose a classifier, and read its six evaluation metrics.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---- load ----------------------------------------------------------------
df = None
source_name = None
BUNDLED_SOURCE = "test_data.csv (shipped in the repository)"

# The dataset lives in st.session_state on purpose.  st.button returns True
# for exactly ONE run, so without this the bundled CSV would load and then
# vanish the moment any other widget was touched - which is precisely how
# the app gets demonstrated (load, then switch models to compare).
if uploaded is not None:
    if st.session_state.get("source_name") != uploaded.name:
        try:
            st.session_state["df"] = pd.read_csv(uploaded)
            st.session_state["source_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Could not read that CSV: `{exc}`")
elif use_bundled and BUNDLED_TEST_CSV.exists():
    if st.session_state.get("source_name") != BUNDLED_SOURCE:
        st.session_state["df"] = pd.read_csv(BUNDLED_TEST_CSV)
        st.session_state["source_name"] = BUNDLED_SOURCE

df = st.session_state.get("df")
source_name = st.session_state.get("source_name")

if df is None:
    st.markdown(
        '<hr class="s2s-rule">'
        '<div class="s2s-empty">'
        + '<div class="s2s-eyebrow">No dataset loaded</div>'
        + '<p>Upload <span class="s2s-code">test_data.csv</span> from the panel '
        "on the left, or click <b>Use bundled test_data.csv</b> to load the "
        "copy that ships with this repository.</p>"
        + "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---- validate ------------------------------------------------------------
problems = validate_upload(df)
if problems:
    st.error("**This CSV cannot be scored**")
    for p in problems:
        st.markdown(f"- {p}")
    st.caption(
        "Expected columns: "
        + ", ".join(f"`{c}`" for c in FEATURES)
        + f", plus the target `{TARGET}`."
    )
    st.stop()

# ---- masthead stats ------------------------------------------------------
preprocessor, models = load_artifacts()
counts = df[TARGET].value_counts()
n_pos = int(counts.get(1, 0))
n_neg = int(counts.get(0, 0))
pos_pct = 100 * n_pos / max(len(df), 1)

st.markdown(
    '<div class="s2s-stats">'
    f'<div class="s2s-stat"><div class="k">Rows</div><div class="v">{len(df):,}</div></div>'
    f'<div class="s2s-stat"><div class="k">Features</div><div class="v">{len(FEATURES)}</div></div>'
    f'<div class="s2s-stat"><div class="k">Purchasers</div>'
    f'<div class="v">{n_pos:,} ({pos_pct:.1f}%)</div></div>'
    f'<div class="s2s-stat"><div class="k">Non-purchasers</div>'
    f'<div class="v">{n_neg:,} ({100 - pos_pct:.1f}%)</div></div>'
    f'<div class="s2s-stat"><div class="k">Source</div>'
    f'<div class="v" style="font-size:.92rem">{source_name}</div></div>'
    "</div>"
    '<hr class="s2s-rule">',
    unsafe_allow_html=True,
)

# ---- score selected model ------------------------------------------------
try:
    metrics, y_true, y_pred = score_model(
        models[selected_key], preprocessor, df
    )
except ValueError as exc:
    st.error(f"Scoring failed: `{exc}`")
    st.stop()

st.markdown(
    section(
        "01",
        MODEL_LABELS[selected_key],
        "scored on the loaded dataset",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:.86rem;color:#6B7280;margin-top:.3rem">'
    "Computed from the <b>saved</b> model and the <b>saved</b> preprocessor. "
    "Nothing was retrained on this data.</div>",
    unsafe_allow_html=True,
)
metric_cards(metrics)

st.markdown(
    f'<div class="s2s-baseline"><b>Baseline context:</b> a classifier that '
    f"always predicts &ldquo;{NEG_LABEL}&rdquo; scores "
    f"<b>{BASELINE_ACCURACY * 100:.1f}% accuracy</b> on this dataset without "
    "learning anything &mdash; which is why recall, F1 and MCC are reported "
    "alongside accuracy.</div>",
    unsafe_allow_html=True,
)

# ---- 02 error analysis ---------------------------------------------------
st.markdown(
    section("02", "Error analysis", "where the model is right and wrong"),
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

# ---- 03 comparison -------------------------------------------------------
st.markdown(
    section("03", "All five models", "same data, same saved weights"),
    unsafe_allow_html=True,
)

comparison_rows = []
for key in MODEL_KEYS:
    try:
        m, _, _ = score_model(models[key], preprocessor, df)
    except ValueError:
        m = {k: float("nan") for k in METRIC_LABELS}
    comparison_rows.append({"Model": MODEL_LABELS[key], **m})
comparison = pd.DataFrame(comparison_rows).set_index("Model")

st.markdown(
    '<div class="s2s-eyebrow" style="margin:1.1rem 0 .8rem">Compare by</div>',
    unsafe_allow_html=True,
)
bar_metric = st.selectbox(
    "Metric",
    options=list(METRIC_LABELS.keys()),
    format_func=lambda k: METRIC_LABELS[k],
    index=list(METRIC_LABELS.keys()).index("mcc"),
    label_visibility="collapsed",
)
comparison_bars(comparison, selected_key, bar_metric)

st.markdown(
    '<div class="s2s-eyebrow" style="margin:1.6rem 0 .8rem">'
    "Full comparison &mdash; best in green, weakest in red</div>",
    unsafe_allow_html=True,
)
comparison_table(comparison, selected_key)

best_key = max(MODEL_KEYS, key=lambda k: comparison.loc[MODEL_LABELS[k], "mcc"])
st.caption(
    f"**{MODEL_LABELS[best_key]}** leads on MCC "
    f"({comparison.loc[MODEL_LABELS[best_key], 'mcc']:.4f}) - the metric that "
    "stays honest on an imbalanced dataset. Bar lengths run from zero to "
    "one, so no gap is exaggerated."
)

with st.expander("How to read these six metrics"):
    st.markdown(
        """
- **Accuracy** - share of rows classified correctly. On this dataset a
  "always predict no purchase" baseline already reaches ~84%, so accuracy
  alone flatters every model.
- **AUC** - how well the model *ranks* sessions by purchase likelihood,
  independent of the 0.5 cut-off.
- **Precision** - of the sessions flagged as buyers, the fraction that
  really were.
- **Recall** - of the real buyers, the fraction caught. Missed buyers are
  lost revenue, so this is often the business number.
- **F1** - harmonic mean of precision and recall.
- **MCC** - correlation between prediction and truth, from -1 to +1. It
  stays meaningful under class imbalance, which is why the assignment
  requires it alongside accuracy.
"""
    )

st.markdown('<hr class="s2s-rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="s2s-eyebrow">M.Tech AIML &nbsp;/&nbsp; Machine Learning '
    "&nbsp;/&nbsp; Assignment 2</div>",
    unsafe_allow_html=True,
)
