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

POS_LABEL = "Will purchase"
NEG_LABEL = "No purchase"

# Short plain-English captions for the six metrics.  Keeps the dashboard
# readable for a reviewer who does not remember what MCC is.
METRIC_INFO = {
    "accuracy": "Overall correct predictions",
    "auc": "Ranking quality, threshold-free",
    "precision": "Of those predicted buyers, how many were",
    "recall": "Of real buyers, how many we caught",
    "f1": "Harmonic mean of precision & recall",
    "mcc": "Balanced score for imbalanced data",
}

st.set_page_config(
    page_title="Session2Shop - Purchase Intent Classifier",
    page_icon="\U0001f6cd\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling.  Deliberately hand-rolled rather than a stock template -
# the assignment explicitly warns that copy-paste Streamlit templates may
# be penalised.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .s2s-hero {
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 55%, #0891b2 100%);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 18px;
    }
    .s2s-hero h1 { color:#fff; margin:0 0 6px 0; font-size:1.95rem; font-weight:800; }
    .s2s-hero p  { color:#e0e7ff; margin:0; font-size:.95rem; opacity:.95; }

    .s2s-grid {
        display:grid;
        grid-template-columns:repeat(6, minmax(0,1fr));
        gap:12px;
        margin:6px 0 22px 0;
    }
    .s2s-card {
        background:#0f172a;
        border:1px solid #334155;
        border-top:3px solid #6366f1;
        border-radius:14px;
        padding:15px 14px;
    }
    .s2s-label { color:#94a3b8; font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
    .s2s-value { color:#f8fafc; font-size:1.6rem; font-weight:750; margin-top:7px; font-variant-numeric:tabular-nums; }
    .s2s-desc  { color:#64748b; font-size:.7rem; margin-top:7px; line-height:1.4; }

    .s2s-cm { display:grid; grid-template-columns:auto repeat(2, 108px); gap:7px; align-items:stretch; }
    .s2s-cm-h { color:#94a3b8; font-size:.72rem; font-weight:700; display:flex; align-items:center; justify-content:center; text-align:center; }
    .s2s-cm-r { color:#cbd5e1; font-size:.75rem; font-weight:600; display:flex; align-items:center; padding-right:8px; justify-content:flex-end; text-align:right; }
    .s2s-cm-c { border-radius:11px; padding:16px 8px; text-align:center; background:#1e293b; border:1px solid #334155; }
    .s2s-cm-c b { display:block; font-size:1.5rem; font-weight:750; color:#f8fafc; font-variant-numeric:tabular-nums; }
    .s2s-cm-c span { font-size:.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em; }
    .s2s-cm-good { background:#064e3b !important; border-color:#059669 !important; }
    .s2s-cm-bad  { background:#4c0519 !important; border-color:#e11d48 !important; }

    .s2s-pill {
        display:inline-block; background:#1e293b; border:1px solid #334155;
        color:#a5b4fc; border-radius:999px; padding:4px 13px;
        font-size:.72rem; font-weight:600; margin-right:7px;
    }
    div[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pretrained models...")
def load_artifacts():
    """Load the fitted preprocessor and all five trained classifiers.

    st.cache_resource matters: Streamlit re-executes the whole script on
    every widget interaction.  Without caching, each click would re-read
    ~8 MB of joblib files from disk and the app would feel broken.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODEL_KEYS}
    return preprocessor, models


# ---------------------------------------------------------------------------
# Validation + scoring  (NO .fit() ANYWHERE BELOW - inference only)
# ---------------------------------------------------------------------------
def validate_upload(df):
    """Return a list of human-readable problems, empty if the CSV is usable."""
    problems = []
    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        problems.append(
            "Missing required feature column(s): "
            + ", ".join(f"`{c}`" for c in missing_features)
        )
    if TARGET not in df.columns:
        problems.append(
            f"Missing target column `{TARGET}` - metrics cannot be computed "
            "without the true labels."
        )
    if len(df) < 2:
        problems.append("The file needs at least 2 rows to score.")
    return problems


def score_all(model, preprocessor, df):
    """Apply the saved preprocessor + saved model, then compute the 6 metrics.

    This is pure inference: transform -> predict -> score.  The model is
    never refitted on the uploaded data, exactly as the spec requires.
    """
    X = prepare_features(df)                 # shared with train.py via config
    y = df[TARGET].astype(int).values
    X_t = preprocessor.transform(X)

    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]  # P(class = 1)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
    }
    return metrics, y, y_pred


def metric_cards(metrics):
    cards = "".join(
        f'<div class="s2s-card">'
        f'<div class="s2s-label">{key.upper()}</div>'
        f'<div class="s2s-value">{metrics[key]:.4f}</div>'
        f'<div class="s2s-desc">{METRIC_INFO[key]}</div>'
        f"</div>"
        for key in metrics
    )
    st.markdown(f'<div class="s2s-grid">{cards}</div>', unsafe_allow_html=True)


def confusion_matrix_block(y_true, y_pred):
    """Confusion matrix drawn as a styled 2x2 grid (no plotting library)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def cell(value, is_correct, note):
        cls = "s2s-cm-good" if is_correct else "s2s-cm-bad"
        return f'<div class="s2s-cm-c {cls}"><b>{value:,}</b><span>{note}</span></div>'

    grid = (
        '<div class="s2s-cm">'
        '<div class="s2s-cm-h"></div>'
        f'<div class="s2s-cm-h">Pred: {NEG_LABEL}</div>'
        f'<div class="s2s-cm-h">Pred: {POS_LABEL}</div>'
        f'<div class="s2s-cm-r">Actual<br>{NEG_LABEL}</div>'
        f"{cell(tn, True, 'correct')}"
        f"{cell(fp, False, 'false alarm')}"
        f'<div class="s2s-cm-r">Actual<br>{POS_LABEL}</div>'
        f"{cell(fn, False, 'missed')}"
        f"{cell(tp, True, 'correct')}"
        "</div>"
    )
    st.markdown(grid, unsafe_allow_html=True)

    total = tn + fp + fn + tp
    st.caption(
        f"**{tn + tp:,}** correct out of **{total:,}** · "
        f"**{fp:,}** false alarms · **{fn:,}** missed purchases"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001f4e6 Data")
    uploaded = st.file_uploader(
        "Upload the held-out test CSV",
        type=["csv"],
        help="test_data.csv from the repository - it contains all 17 "
        "features plus the Revenue target column.",
    )
    use_bundled = st.button(
        "\u26a1 Use the bundled test_data.csv",
        use_container_width=True,
        help="Convenience shortcut so the app can be demonstrated without "
        "downloading anything first.",
    )

    st.markdown("---")
    st.markdown("### \U0001f916 Model")
    selected_key = st.selectbox(
        "Choose a classifier",
        options=MODEL_KEYS,
        format_func=lambda k: MODEL_LABELS[k],
        index=MODEL_KEYS.index("svm"),
    )

    st.markdown("---")
    st.markdown(
        '<span class="s2s-pill">No retraining</span>'
        '<span class="s2s-pill">Inference only</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Models and the preprocessor were fitted once by `train.py`. "
        "This app only loads, transforms and scores."
    )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="s2s-hero">
  <h1>Session2Shop \U0001f6cd\ufe0f</h1>
  <p>Predicting whether an e-commerce browsing session ends in a purchase,
     from 17 behavioural features. Upload a test CSV, pick a model, and read
     its six evaluation metrics.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---- load the data ---------------------------------------------------------
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:  # malformed file should not crash the app
        st.error(f"Could not read that CSV: `{exc}`")
elif use_bundled and BUNDLED_TEST_CSV.exists():
    df = pd.read_csv(BUNDLED_TEST_CSV)
    source_name = "test_data.csv (bundled with the repository)"

if df is None:
    st.info(
        "\U0001f4c1 **Upload `test_data.csv`** in the sidebar - or click "
        "*Use the bundled test_data.csv* - to score the models."
    )
    st.markdown(
        """
#### What this app reports

| | |
|---|---|
| **Dataset** | Online Shoppers Purchasing Intention |
| **Task** | Binary classification - will the session convert? |
| **Features** | 17 (10 continuous + 7 categorical) |
| **Models** | Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM |
| **Metrics** | Accuracy, AUC, Precision, Recall, F1, MCC |
"""
    )
    st.stop()

# ---- validate --------------------------------------------------------------
problems = validate_upload(df)
if problems:
    st.error("**This CSV cannot be scored:**")
    for p in problems:
        st.markdown(f"- {p}")
    st.caption(
        "Expected columns: "
        + ", ".join(f"`{c}`" for c in FEATURES)
        + f", plus the target `{TARGET}`."
    )
    st.stop()

# ---- dataset strip ---------------------------------------------------------
preprocessor, models = load_artifacts()
class_counts = df[TARGET].value_counts()
n_pos = int(class_counts.get(1, 0))
n_neg = int(class_counts.get(0, 0))
pos_pct = 100 * n_pos / max(len(df), 1)

st.markdown(
    f'<span class="s2s-pill">{len(df):,} rows</span>'
    f'<span class="s2s-pill">{len(FEATURES)} features</span>'
    f'<span class="s2s-pill">{n_pos:,} purchasers ({pos_pct:.1f}%)</span>'
    f'<span class="s2s-pill">source: {source_name}</span>',
    unsafe_allow_html=True,
)

# ---- (b) model dropdown result + (c) metrics -------------------------------
st.markdown(f"### \U0001f4ca {MODEL_LABELS[selected_key]}")
st.caption(
    "Scores below are computed on the uploaded rows using the **saved** "
    "model - nothing was retrained."
)

try:
    metrics, y_true, y_pred = score_all(models[selected_key], preprocessor, df)
except ValueError as exc:
    st.error(f"Scoring failed: `{exc}`")
    st.stop()

metric_cards(metrics)

# ---- (d) confusion matrix + classification report --------------------------
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("#### \U0001f9f9 Confusion matrix")
    confusion_matrix_block(y_true, y_pred)

with right:
    st.markdown("#### \U0001f4cb Classification report")
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[NEG_LABEL, POS_LABEL],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True, height=248)

# ---- all five models side by side (spec: "results of different models
#      on your test data should be visible on the streamlit app") -----------
st.markdown("#### \U0001f501 All five models on this data")
st.caption(
    "The model selected in the sidebar is highlighted. Every row is scored "
    "with that model's own saved weights."
)

comparison_rows = []
for key in MODEL_KEYS:
    try:
        m, _, _ = score_all(models[key], preprocessor, df)
    except ValueError:
        m = {k: float("nan") for k in METRIC_INFO}
    comparison_rows.append({"Model": MODEL_LABELS[key], **m})

comparison = pd.DataFrame(comparison_rows).set_index("Model").round(4)

st.dataframe(
    comparison.style.highlight_max(axis=0, color="#064e3b")
    .highlight_min(axis=0, color="#4c0519")
    .format("{:.4f}"),
    use_container_width=True,
)
st.caption(
    "\U0001f7e2 highest and \U0001f534 lowest value per column. "
    "Note that a model which always predicts 'No purchase' would still "
    "reach ~84% accuracy here - which is why MCC and F1 matter."
)

with st.expander("\U0001f4a1 How to read these numbers"):
    st.markdown(
        f"""
- **Accuracy** - share of rows classified correctly. On this imbalanced
  dataset a "always predict no purchase" baseline already scores about
  84%, so accuracy alone flatters every model.
- **AUC** - how well the model *ranks* sessions by purchase likelihood,
  independent of the 0.5 cut-off.
- **Precision** - of the sessions flagged as buyers, the fraction that
  really were.
- **Recall** - of the real buyers, the fraction we managed to catch.
  Missed buyers are lost revenue, so this is often the business number.
- **F1** - harmonic mean of precision and recall.
- **MCC** - correlation between prediction and truth; ranges -1 to +1
  and stays honest under class imbalance, which is why it is required
  alongside accuracy.
"""
    )

st.divider()
st.caption(
    "M.Tech (AIML) · Machine Learning · Assignment 2 · "
    "Models trained by `train.py`, served here without retraining."
)
