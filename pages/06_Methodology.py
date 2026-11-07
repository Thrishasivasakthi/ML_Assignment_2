"""
pages/06_Methodology.py - dataset, preprocessing, models, metrics.

This page exists so the work can be explained without reading source code.
Everything stated here matches train.py and config.py; nothing is asserted
that the repository does not do.

It also carries the single most important claim in the application: the app
never refits anything.
"""

import streamlit as st

from components.layout import chip, page_header, rule, section, stat_strip
from components.notes import MODEL_FAMILY
from components.workspace import (
    baseline_metrics,
    best_params,
    boot,
    evaluate_all,
    facts,
    get_artifacts,
    open_workspace,
)
from config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_KEYS,
    MODEL_LABELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from core.inference import encoded_width

# The offline suite, for the length of the list printed in section 07.  The
# count on this page is len(CHECKS) rather than a literal, so it cannot be
# left behind when a check is added - which is how "17" survived two new
# checks.  Importing it runs nothing: the suite only executes under
# `if __name__ == "__main__"`.
from tests.check_pipeline import CHECKS

# Per-feature descriptions.  The numeric comments mirror config.py so the two
# cannot drift; the categorical ones are documented here because config.py
# only flags how they are encoded.
FEATURE_DESCRIPTIONS = {
    "Administrative": "Count of administrative pages visited in the session.",
    "Administrative_Duration": "Total time on administrative pages.",
    "Informational": "Count of informational pages visited.",
    "Informational_Duration": "Total time on informational pages.",
    "ProductRelated": "Count of product-related pages visited.",
    "ProductRelated_Duration": "Total time on product-related pages.",
    "BounceRates": "Average bounce rate of the entry pages (Google Analytics).",
    "ExitRates": "Average exit rate across the pages visited (Google Analytics).",
    "PageValues": "Average value of the pages visited, weighted by transaction.",
    "SpecialDay": "Closeness of the visit to a special day, from 0 to 1.",
    "Month": "Calendar month of the visit.",
    "OperatingSystems": "Operating system identifier (nominal code).",
    "Browser": "Browser identifier (nominal code).",
    "Region": "Geographic region identifier (nominal code).",
    "TrafficType": "Traffic source type identifier (nominal code).",
    "VisitorType": "Returning visitor, new visitor, or other.",
    "Weekend": "Whether the visit fell on a weekend.",
}

boot("Session2Shop | Methodology")

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)

summary = result["summary"]
frame = result["comparison"]
background = facts()
splits = background["splits"]
balance = background["balance"]
correlations = background["correlations"]

# Construction is repr() of the SAVED estimator, so this table cannot drift
# away from model/<key>.joblib after a retrain - it is read from the file
# the application actually loads.
preprocessor, models = get_artifacts()
model_config = {key: repr(models[key]) for key in MODEL_KEYS}
n_numeric, n_onehot, n_total = encoded_width(preprocessor)

split_total = max(splits["train"] + splits["test"], 1)
train_pct = 100.0 * splits["train"] / split_total
test_pct = 100.0 * splits["test"] / split_total

# Search definition, read from the record train.py wrote.
search = (best_params() or {}).get("search") or {}
cv_folds = search.get("cv_folds", "?")
scoring = search.get("scoring", "?")

# The untuned table, for the before/after clause in section 04.
baseline = baseline_metrics()

page_header(
    "Methodology",
    "How these results were produced",
    "Dataset, feature handling, the train/test protocol, the five "
    "classifiers and the six metrics - in the order train.py performs them.",
)

stat_strip(
    [
        ("Raw rows", f"{splits['raw']:,}"),
        ("After de-duplication", f"{splits['deduplicated']:,}"),
        ("Training split", f"{splits['train']:,} ({train_pct:.0f}%)"),
        ("Held-out split", f"{splits['test']:,} ({test_pct:.0f}%)"),
        ("Seed", f"{RANDOM_STATE}"),
    ]
)

# --------------------------------------------------------------- 01 dataset
st.markdown(
    section("01", "Dataset", "Online Shoppers Purchasing Intention"),
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:.93rem;color:#374151;line-height:1.7;max-width:74ch">'
    "The dataset describes one year of visits to an e-commerce website. Each "
    "row is a single browsing session characterised by the pages visited, the "
    "time spent, and Google Analytics metrics such as bounce rate and page "
    "value. The target column <b>Revenue</b> marks whether the session ended "
    "in a purchase, making this a binary classification problem with a "
    "real business question behind it: <i>which sessions convert?</i></div>"
    '<div class="s2s-baseline" style="margin-top:1rem">'
    "<b>Source:</b> Sakar, C. &amp; Kastro, Y. (2018). "
    "<i>Online Shoppers Purchasing Intention Dataset</i>. "
    "UCI Machine Learning Repository. DOI 10.24432/C5F88Q. "
    "Licensed CC BY 4.0. Also published on Kaggle as "
    "<i>imakash3011/online-shoppers-purchasing-intention-dataset</i>.</div>",
    unsafe_allow_html=True,
)

st.caption(
    f"Chosen because it clears the brief's floor of 12 features and 500 "
    f"instances ({len(FEATURES)} features, {splits['raw']:,} rows), is a "
    "genuine classification problem rather than a regression relabelled, and "
    "carries a class imbalance that makes accuracy alone misleading - which "
    "is why the brief asks for MCC."
)

# -------------------------------------------------------------- 02 features
st.markdown(
    section(
        "02",
        f"The {len(FEATURES)} features",
        f"{len(NUMERIC_FEATURES)} scaled, {len(CATEGORICAL_FEATURES)} encoded",
    ),
    unsafe_allow_html=True,
)

head = "<tr><th class='l'>Feature</th><th class='l'>Treatment</th><th class='l'>Description</th></tr>"
rows = []
for name in NUMERIC_FEATURES:
    rows.append(
        f"<tr><td class='l'>{chip(name)}</td>"
        "<td class='l' style='color:#6B7280'>StandardScaler</td>"
        f"<td class='l' style='color:#6B7280'>{FEATURE_DESCRIPTIONS[name]}</td></tr>"
    )
for name in CATEGORICAL_FEATURES:
    rows.append(
        f"<tr><td class='l'>{chip(name)}</td>"
        "<td class='l' style='color:#6B7280'>OneHotEncoder</td>"
        f"<td class='l' style='color:#6B7280'>{FEATURE_DESCRIPTIONS[name]}</td></tr>"
    )
st.markdown(
    f'<table class="s2s-tbl">{head}{"".join(rows)}</table>', unsafe_allow_html=True
)

st.caption(
    "Four of the categorical columns arrive as integers but are nominal "
    "identifiers: `Browser=2` is not twice `Browser=1`. Treating them as "
    "numbers would impose a fake ordering and a fake scale, so they are "
    "one-hot encoded instead."
)

# --------------------------------------------------------- 03 preprocessing
st.markdown(
    section("03", "Preprocessing and split", "fit on training data only"),
    unsafe_allow_html=True,
)
split_call = (
    f"train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, "
    "stratify=y)"
)

PREPROCESS_STEPS = "".join(
    [
        f"<b>1. Remove exact duplicates.</b> {splits['raw']:,} raw rows "
        f"become {splits['deduplicated']:,}; {splits['duplicates']} duplicated "
        "sessions are dropped.<br>",
        "<b>2. Split before any fitting.</b> ",
        chip(split_call, wrap=True),
        f" gives {splits['train']:,} training and {splits['test']:,} "
        f"held-out rows, with the "
        f"{100 * (1 - balance['train']):.1f} / "
        f"{100 * balance['train']:.1f} class ratio preserved in both.<br>",
        f"<b>3. Fit the preprocessor on the training split only.</b> "
        f"{n_numeric} numeric columns are standardised; "
        f"{len(CATEGORICAL_FEATURES)} categorical columns are one-hot "
        "encoded with ",
        chip('handle_unknown="ignore"'),
        " so a previously unseen category in an uploaded file becomes an "
        f"all-zero block instead of an error. The result is <b>{n_total} "
        "columns</b>.",
        "<br><b>4. Transform the held-out split</b> with that same fitted "
        "preprocessor - never refit it on test data.<br>",
        "<b>5. Tune and fit the five classifiers.</b> GridSearchCV runs "
        f"{cv_folds}-fold stratified cross-validation over the training rows "
        f"only, scoring {scoring}; the winning parameters are then used for "
        "one fit, and the held-out rows are scored exactly once.",
    ]
)

st.markdown(
    '<div style="font-size:.93rem;color:#374151;line-height:1.8">'
    + PREPROCESS_STEPS
    + "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Splitting first, then fitting the preprocessor, is what keeps the held-out "
    "set honest: statistics used to scale the test rows were never allowed to "
    "see them."
)

# ---------------------------------------------------------------- 04 models
st.markdown(
    section("04", "The five classifiers", "as constructed in train.py"),
    unsafe_allow_html=True,
)
head = "<tr><th class='l'>Model</th><th class='l'>Construction</th><th class='l'>Character</th></tr>"
rows = []

for key in MODEL_KEYS:
    sel = ' class="sel"' if key == selected_key else ""
    rows.append(
        f"<tr{sel}><td class='l'>{MODEL_LABELS[key]}</td>"
        f"<td class='l'>{chip(model_config[key], wrap=True)}</td>"
        f"<td class='l' style='color:#6B7280'>{MODEL_FAMILY[key]}</td></tr>"
    )
st.markdown(
    f'<table class="s2s-tbl">{head}{"".join(rows)}</table>', unsafe_allow_html=True
)

# Every figure in this caption is read back from the repository: the two
# correlations from the de-duplicated raw data, the smoothing value from
# best_params.json, and the two accuracies from the two metric files.
nb_smooth = (best_params().get("best_params") or {}).get("naive_bayes", {}).get(
    "var_smoothing", float("nan")
)
# frame is indexed by model KEY, metrics_baseline.csv by display label - the
# two frames are deliberately keyed differently, so the Naive Bayes row is
# addressed through both spellings rather than one assumed to work for both.
NB = "naive_bayes"
nb_auc_after = float(frame.loc[NB, "auc"])
nb_mcc_after = float(frame.loc[NB, "mcc"])
best_mcc = float(frame["mcc"].max())

if baseline is not None and MODEL_LABELS[NB] in baseline.index:
    nb_acc_before = float(baseline.loc[MODEL_LABELS[NB], "accuracy"])
    nb_acc_after = float(frame.loc[NB, "accuracy"])
    smoothing_clause = (
        f" Raising `var_smoothing` to {nb_smooth} lifts accuracy from "
        f"{nb_acc_before:.4f} at sklearn's default to {nb_acc_after:.4f} and "
        f"AUC to {nb_auc_after:.4f} - a large gain that still leaves MCC at "
        f"{nb_mcc_after:.4f} against the best model's {best_mcc:.4f}. "
        "Smoothing mitigates the violated assumption; it does not remove it."
    )
else:
    smoothing_clause = (
        f" Its AUC is {nb_auc_after:.4f} and its MCC {nb_mcc_after:.4f}, "
        f"against the best model's {best_mcc:.4f}."
    )

st.caption(
    f"Gaussian Naive Bayes assumes its features are independent, and they "
    f"are not: `BounceRates` and `ExitRates` correlate at "
    f"{correlations['BounceRates~ExitRates']:.3f}, and `ProductRelated` with "
    f"`ProductRelated_Duration` at "
    f"{correlations['ProductRelated~ProductRelated_Duration']:.3f}, so the "
    f"same evidence is counted twice." + smoothing_clause
)

# --------------------------------------------------------------- 05 metrics
st.markdown(
    section("05", "Evaluation metrics", "the six the brief requires"),
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:.93rem;color:#374151;line-height:1.8">'
    "<b>Accuracy</b> - share of rows labelled correctly. Misleading on its "
    f"own here: always answering &ldquo;No&rdquo; already reaches "
    f"{summary['always_no_accuracy'] * 100:.1f}%.<br>"
    "<b>AUC</b> - area under the ROC curve. Measures how well the model "
    "ranks sessions by purchase likelihood, independent of the 0.5 cut-off.<br>"
    "<b>Precision</b> - of the sessions flagged as purchases, the share that "
    "really were.<br>"
    "<b>Recall</b> - of the sessions that really were purchases, the share "
    "the model found.<br>"
    "<b>F1</b> - harmonic mean of precision and recall.<br>"
    "<b>MCC</b> - Matthews correlation coefficient, from -1 to +1. It is the "
    "only one of the six that cannot be inflated by the class imbalance, "
    "which is why this study treats it as the deciding metric.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------- 06 no retraining
st.markdown(
    section("06", "No retraining", "the constraint the brief sets"),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="s2s-empty" style="max-width:74ch">'
    '<div class="s2s-eyebrow is-accent">Inference only</div>'
    "<p>There is no "
    + chip(".fit()")
    + " call anywhere in the application. "
    + f"{chip('train.py')} fits the preprocessor and the five classifiers once "
    + f"and writes them to {chip('artifacts/')} and {chip('model/')}. The "
    + f"application only ever calls {chip('load')}, {chip('transform')}, "
    + f"{chip('predict')} and {chip('predict_proba')} - so uploading a "
    + "different CSV changes the rows being scored, never the model doing "
    + "the scoring.</p>"
    + "</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ 07 reproduce
st.markdown(
    section("07", "Reproducing these numbers", "from a clean checkout"),
    unsafe_allow_html=True,
)
st.code(
    "git clone https://github.com/Thrishasivasakthi/ML_Assignment_2\n"
    "cd ML_Assignment_2\n"
    "pip install -r requirements.txt\n"
    "python train.py            # refits everything, rewrites metrics.csv\n"
    f"python tests/check_pipeline.py   # {len(CHECKS)} offline checks, no browser\n"
    "streamlit run app.py       # launches this application",
    language="bash",
)

st.caption(
    f"Seed {RANDOM_STATE} is set on the split and on every stochastic "
    "estimator, so a clean run reproduces metrics.csv exactly. The "
    f"offline suite above is {len(CHECKS)} checks and must end in "
    '"0 failed"; it needs no browser and re-runs nothing - it only reads '
    "the artifacts already committed. The application does not need any "
    "of this step - it loads those artifacts directly."
)

st.markdown(rule(), unsafe_allow_html=True)
