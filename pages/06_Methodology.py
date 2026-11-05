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
from components.notes import MODEL_NOTES
from components.workspace import boot, evaluate_all, get_artifacts, open_workspace
from config import (
    CATEGORICAL_FEATURES,
    MODEL_KEYS,
    MODEL_LABELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)

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

MODEL_CONFIG = {
    "logistic_regression": "LogisticRegression(max_iter=1000, random_state=42)",
    "decision_tree": "DecisionTreeClassifier(random_state=42) - grown fully, unpruned",
    "knn": "KNeighborsClassifier(n_neighbors=5)",
    "naive_bayes": "GaussianNB() - sklearn defaults",
    "svm": "SVC(kernel='rbf', probability=True, random_state=42)",
}

boot("Session2Shop | Methodology")

df, source_name, selected_key = open_workspace(validate=True)
result = evaluate_all(df)

summary = result["summary"]

page_header(
    "Methodology",
    "How these results were produced",
    "Dataset, feature handling, the train/test protocol, the five "
    "classifiers and the six metrics - in the order train.py performs them.",
)

stat_strip(
    [
        ("Raw rows", "12,330"),
        ("After de-duplication", "12,205"),
        ("Training split", "9,764 (80%)"),
        ("Held-out split", "2,441 (20%)"),
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
    "Chosen because it clears the brief's floor of 12 features and 500 "
    "instances (17 features, 12,330 rows), is a genuine classification "
    "problem rather than a regression relabelled, and carries a class "
    "imbalance that makes accuracy alone misleading - which is why the "
    "brief asks for MCC."
)

# -------------------------------------------------------------- 02 features
st.markdown(
    section("02", "The 17 features", "ten scaled, seven encoded"),
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
        "<b>1. Remove exact duplicates.</b> 12,330 raw rows become 12,205; "
        "125 duplicated sessions are dropped.<br>",
        "<b>2. Split before any fitting.</b> ",
        chip(split_call, wrap=True),
        " gives 9,764 training and 2,441 held-out rows, with the "
        "84.4 / 15.6 class ratio preserved in both.<br>",
        "<b>3. Fit the preprocessor on the training split only.</b> Ten numeric "
        "columns are standardised; seven categorical columns are one-hot "
        "encoded with ",
        chip('handle_unknown="ignore"'),
        " so a previously unseen category in an uploaded file becomes an "
        "all-zero block instead of an error. The result is <b>74 columns</b>.",
        "<br><b>4. Transform the held-out split</b> with that same fitted "
        "preprocessor - never refit it on test data.<br>",
        "<b>5. Fit the five classifiers</b> on the transformed training rows, "
        "then score the held-out rows once.",
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
        f"<td class='l'>{chip(MODEL_CONFIG[key])}</td>"
        f"<td class='l' style='color:#6B7280'>{MODEL_NOTES[key]['family']}</td></tr>"
    )
st.markdown(
    f'<table class="s2s-tbl">{head}{"".join(rows)}</table>', unsafe_allow_html=True
)

st.caption(
    "Gaussian Naive Bayes is kept at sklearn's defaults. It scores poorly "
    "because its independence assumption is violated by this data - "
    "`BounceRates` and `ExitRates` correlate at 0.911 - "
    "not because it was implemented incorrectly. Its AUC of 0.7498 shows the "
    "ranking is still sound; only the 0.5 threshold is broken."
)

# --------------------------------------------------------------- 05 metrics
st.markdown(
    section("05", "Evaluation metrics", "the six the brief requires"),
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:.93rem;color:#374151;line-height:1.8">'
    "<b>Accuracy</b> - share of rows labelled correctly. Misleading on its "
    "own here: always answering &ldquo;No&rdquo; already reaches 84.4%.<br>"
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
    "streamlit run app.py       # launches this application",
    language="bash",
)

st.caption(
    f"Seed {RANDOM_STATE} is set on the split and on every stochastic "
    "estimator, so a clean run reproduces metrics.csv exactly. The "
    "application does not need this step - it loads the artifacts that are "
    "already committed."
)

st.markdown(rule(), unsafe_allow_html=True)
