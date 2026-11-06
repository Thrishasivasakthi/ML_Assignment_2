"""
train.py - Phase 1 (preprocessing) + Phase 2 (tune, train & evaluate 5 models).

    python train.py

OUTPUTS
-------
  artifacts/preprocessor.joblib      fitted ColumnTransformer
  train_data.csv / test_data.csv     splits (original column form) - NEVER
                                     allowed to change (see write_split)
  model/<key>.joblib                 5 tuned classifiers
  figures/confusion_matrix_<key>.png confusion matrices for the README
  metrics.csv                        the tuned 5 x 6 comparison table
  metrics_baseline.csv               the same table at sklearn defaults
  best_params.json                   hyperparameters the search selected
  train_scores.csv                   train vs test accuracy, tuned models

DESIGN RULE ENFORCED HERE
-------------------------
Split FIRST -> fit the preprocessor on X_train ONLY -> transform both.
Fitting the scaler on the whole dataset would leak test-set means and
standard deviations into training, producing optimistically biased scores.

The hyperparameter search obeys the same rule one level up: GridSearchCV
runs on X_train_t only, with StratifiedKFold over the training rows. The
held-out set is touched exactly once per model, at final scoring time.
"""

import json
import warnings
from pathlib import Path

# config is imported FIRST on purpose: it installs the joblib/loky
# environment fix (Windows 11 removed `wmic`) before anything below
# touches joblib.  See the comment at the top of config.py.
from config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRIC_KEYS,
    MODEL_KEYS,
    MODEL_LABELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    prepare_features,
)

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe: required on BITS Lab / Streamlit Cloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "raw" / "online_shoppers_intention.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "model"
FIGURES_DIR = BASE_DIR / "figures"
REQUIREMENTS = BASE_DIR / "requirements.txt"


# --------------------------------------------------------------- tuning
#
# SEARCH SCORING = f1.  WHY NOT roc_auc?
# ---------------------------------------
# 1. f1 is one of the six metrics the assignment requires, and it is the
#    only one of them driven by the two error costs this problem actually
#    has: a missed purchaser (lost revenue) against a false alarm (wasted
#    retargeting spend).  Precision and recall are the two sides of that
#    coin, and f1 is their harmonic mean.
# 2. The application makes its decision at the 0.5 cut-off.  Optimising AUC
#    optimises the RANKING and can select a hyperparameter that wins on AUC
#    while losing on the thresholded metrics - which is exactly the
#    Logistic-Regression-takes-AUC / SVM-takes-MCC split already visible in
#    metrics.csv.  Tuning on f1 tunes what is actually reported.
# 3. The honest counter-argument, recorded rather than hidden: f1 is
#    threshold-dependent and noisier under cross-validation than roc_auc.
#    StratifiedKFold keeps the ~300 purchasers in X_train spread across the
#    five folds (~60 each), which is what makes the estimate usable at all.
#
# The search is deliberately SMALL: a handful of values per knob, fitted on
# X_train_t only.  A large grid would fit the CV noise of a 15.6% positive
# class and would take minutes on the SVM.
SCORING = "f1"
CV_FOLDS = 5

SEARCH_GRIDS = {
    # C is the inverse regularisation strength - the single knob that trades
    # fit against overfit.  class_weight="balanced" multiplies each class by
    # its inverse frequency, which is the standard answer to an 84/16 split.
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    },
    # max_depth / min_samples_leaf are the two pruning controls: together
    # they are what stops the fully-grown tree memorising its 9,764 rows.
    "decision_tree": {
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 20],
        "class_weight": [None, "balanced"],
    },
    # k must be odd so a binary vote cannot tie; distance weighting lets a
    # close neighbour outvote a far one instead of counting them equally.
    "knn": {
        "n_neighbors": [3, 5, 9, 15, 25],
        "weights": ["uniform", "distance"],
    },
    # var_smoothing is added to every feature variance so no dimension is
    # treated as perfectly certain.  Log grid from sklearn's default 1e-9
    # up to 1.0 - nine decades, because the default is known to be far too
    # small for this data.
    "naive_bayes": {
        "var_smoothing": list(np.logspace(-9, 0, 10)),
    },
    # C and gamma are the RBF pair (margin width against fit detail);
    # gamma="scale" is sklearn's default = 1 / (n_features * Var(X)).
    "svm": {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", 0.01, 0.001],
        "class_weight": [None, "balanced"],
    },
}


def assert_pinned_sklearn():
    """Refuse to write joblib unless sklearn matches requirements.txt.

    A pickle records which scikit-learn built it.  Loading it under another
    version can fail outright or, worse, score slightly differently - which
    would put the deployed app out of step with metrics.csv and with every
    number quoted in README section d.  The pin lives in requirements.txt so
    there is one source of truth to keep in step.
    """
    import sklearn

    pinned = None
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry.lower().startswith("scikit-learn=="):
            pinned = entry.split("==", 1)[1].strip()
            break

    if pinned is None:
        raise RuntimeError(
            "requirements.txt has no 'scikit-learn==' pin, so the version "
            "that wrote the pickles cannot be proven."
        )
    if sklearn.__version__ != pinned:
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} is installed but "
            f"requirements.txt pins {pinned}. Refusing to save joblib files "
            "that this environment could not reload."
        )
    print(f"  scikit-learn {sklearn.__version__} matches requirements.txt")


def write_split(frame, path):
    """Persist a train/test split, refusing to alter one that already exists.

    The brief makes test_data.csv and train_data.csv a hard constraint: their
    content must never change.  Writing them silently on every run would mean
    a future edit to the split logic quietly redefines the held-out set - and
    every reported metric with it.  Comparing first turns that silent drift
    into a loud failure.  Line endings are normalised before comparison so
    the guard still holds on a checkout that .gitattributes has forced to LF.
    """
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")

    if path.exists():
        current = path.read_bytes().replace(b"\r\n", b"\n")
        if current != payload:
            raise RuntimeError(
                f"{path.name} would change. The split is a hard constraint - "
                "test_data.csv and train_data.csv must stay byte-identical in "
                "content. Revert the change that altered the split instead."
            )
        print(f"  {path.name:<16} unchanged ({len(frame):,} rows)")
        return

    path.write_bytes(payload)
    print(f"  {path.name:<16} written  ({len(frame):,} rows)")


# ---------------------------------------------------------------- loading


def load_and_clean():
    """Load the raw CSV and apply the cleaning decisions we verified."""
    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)

    # DECISION: drop duplicates.
    # 125 duplicate rows exist; if they straddle the train/test split the
    # model has already "seen" its test rows -> inflated scores.
    df = df.drop_duplicates().reset_index(drop=True)

    # DECISION: target as 0/1 integers rather than bool.
    df[TARGET] = df[TARGET].astype(int)

    print("=" * 74)
    print("PHASE 1 :: LOAD & CLEAN")
    print("=" * 74)
    print(f"  raw rows                 : {n_raw}")
    print(f"  duplicate rows removed   : {n_raw - len(df)}")
    print(f"  rows after dedup         : {len(df)}   (requirement >= 500)")
    print(f"  missing values remaining : {int(df.isna().sum().sum())}")
    print(f"  features                 : {len(FEATURES)}   (requirement >= 12)")
    pct = df[TARGET].value_counts(normalize=True) * 100
    print(f"  class balance            : {pct[0]:.1f}% / {pct[1]:.1f}%  (IMBALANCED)")
    return df


# ---------------------------------------------------------- preprocessing


def build_preprocessor():
    """Scaler for real quantities, one-hot for category codes.

    handle_unknown='ignore' is NOT optional: without it, an uploaded CSV
    containing any category the model never saw at training time raises
    a ValueError and the deployed app dies on first use.
    """
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # never silently leak an unlisted column
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------- models


def build_models():
    """The five classifiers required by the assignment.

    Every model gets the SAME preprocessed X_train / X_test, so the
    comparison is apples-to-apples - the spec requires all five to run
    on the same dataset under the same conditions.

    Notes on the choices:
      * LogisticRegression(max_iter=1000) - the default 100 iterations can
        fail to converge and emit a ConvergenceWarning; 1000 removes the
        doubt.
      * GaussianNB, NOT MultinomialNB - MultinomialNB requires
        non-negative features, but StandardScaler mean-centers the data
        and therefore guarantees negatives.  Gaussian is the correct
        variant for continuous inputs.
      * SVC(probability=True) - WITHOUT this flag SVC has no
        predict_proba, roc_auc_score() fails, and you lose a required
        metric.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred, y_proba):
    """All six metrics required by the assignment.

    This is a BINARY problem, so sklearn's default average='binary' is
    correct for precision / recall / f1 - it scores the POSITIVE class
    (Revenue = 1), which is exactly the minority class we care about.
    Had this been multi-class we would have had to pass
    average='weighted' explicitly.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def jsonable(value):
    """numpy scalars and containers -> plain Python, so best_params.json loads."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def tune_one(key, model, X_train_t, y_train):
    """Stratified 5-fold GridSearchCV over TRAIN rows only.

    Returns (best_params, cv_f1).  refit=False: the winning combination is
    returned as parameters and the final estimator is fitted through exactly
    the same code path as the untuned baseline, so the only difference
    between metrics_baseline.csv and metrics.csv is the hyperparameters.
    """
    search_estimator = clone(model)
    if key == "svm":
        # scoring="f1" needs predict() only, and predict() is computed from
        # the decision function, which Platt scaling does not touch.
        # probability=True would run a second internal 5-fold calibration
        # inside every one of the 90 candidate fits for no benefit here.
        search_estimator.set_params(probability=False)

    search = GridSearchCV(
        estimator=search_estimator,
        param_grid=SEARCH_GRIDS[key],
        scoring=SCORING,
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        refit=False,
        n_jobs=1,  # sequential: deterministic, and no loky workers to spawn
    )
    search.fit(X_train_t, y_train)

    best = {k: jsonable(v) for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def plot_confusion_matrix(y_true, y_pred, label, out_path):
    """Confusion matrix PNG - used in the README / report."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No purchase", "Purchase"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main():
    # ============================ GUARD ================================
    # Refuse to write any joblib until we know this interpreter can reload
    # what it is about to produce.
    assert_pinned_sklearn()

    # ============================ PHASE 1 ============================
    df = load_and_clean()

    X = prepare_features(df)
    y = df[TARGET]

    # SPLIT FIRST - stratify keeps the 84.4/15.6 ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # FIT ON TRAIN ONLY - no leakage.
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print()
    print("=" * 74)
    print("PHASE 1 :: SPLIT & PREPROCESS")
    print("=" * 74)
    print(f"  X_train : {X_train.shape}   minority: {y_train.mean() * 100:.1f}%")
    print(f"  X_test  : {X_test.shape}   minority: {y_test.mean() * 100:.1f}%")
    print(f"  encoded features         : {X_train_t.shape[1]}   (requirement >= 12)")
    print(f"  preprocessor fitted on   : X_train only (no leakage)")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    # Saved in ORIGINAL column form so the app can replay the exact same
    # transforms through the saved preprocessor - without retraining.
    test_out = X_test.copy()
    test_out[TARGET] = y_test.values

    train_out = X_train.copy()
    train_out[TARGET] = y_train.values

    print()
    print("=" * 74)
    print("SPLIT FILES  (hard constraint: their content must never change)")
    print("=" * 74)
    write_split(test_out, BASE_DIR / "test_data.csv")
    write_split(train_out, BASE_DIR / "train_data.csv")

    MODEL_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # ======================= PHASE 2a :: BASELINE ==========================
    # Untuned sklearn defaults, scored first so that every later claim of
    # "tuning improved X" has an untouched number to be measured against.
    print()
    print("=" * 74)
    print("PHASE 2a :: BASELINE  (sklearn defaults, no tuning)")
    print("=" * 74)

    models = build_models()
    baseline_rows = []
    baseline_train = {}

    for key in MODEL_KEYS:
        model = models[key]
        label = MODEL_LABELS[key]

        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = label
        baseline_rows.append(metrics)
        baseline_train[label] = accuracy_score(y_train, model.predict(X_train_t))

        print(
            f"  {label:<24} "
            f"acc={metrics['accuracy']:.4f}  "
            f"auc={metrics['auc']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"mcc={metrics['mcc']:.4f}"
        )

    baseline = pd.DataFrame(baseline_rows).set_index("model")[METRIC_KEYS]
    baseline.to_csv(BASE_DIR / "metrics_baseline.csv")
    print(f"\n  retained -> metrics_baseline.csv ({len(baseline)} rows)")

    # ======================= PHASE 2b :: TUNING ===========================
    print()
    print("=" * 74)
    print(
        f"PHASE 2b :: TUNING  (GridSearchCV, {CV_FOLDS}-fold stratified, "
        f"scoring={SCORING})"
    )
    print("=" * 74)

    tuned_rows = []
    tuned_train = {}
    best_params = {}
    cv_scores = {}

    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        candidates = int(np.prod([len(v) for v in SEARCH_GRIDS[key]]))

        print(f"\n  {label}  ({candidates} candidates x {CV_FOLDS} folds)")
        best, cv_f1 = tune_one(key, models[key], X_train_t, y_train)
        best_params[key] = best
        cv_scores[key] = cv_f1
        print(f"    best {SCORING} on training CV : {cv_f1:.4f}")
        print(f"    selected                    : {best}")

        # Refit through the SAME path as the baseline - the only difference
        # between the two runs is the hyperparameters.
        model = clone(models[key]).set_params(**best)
        model.fit(X_train_t, y_train)

        y_pred = model.predict(X_test_t)
        # All five expose predict_proba, so AUC is computed uniformly
        # from the probability of the POSITIVE class.
        y_proba = model.predict_proba(X_test_t)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = label
        tuned_rows.append(metrics)

        # Train accuracy is captured purely to expose OVERFITTING - it is
        # the raw material for the graded "observations" table.
        tuned_train[label] = accuracy_score(y_train, model.predict(X_train_t))

        joblib.dump(model, MODEL_DIR / f"{key}.joblib")
        plot_confusion_matrix(
            y_test, y_pred, label, FIGURES_DIR / f"confusion_matrix_{key}.png"
        )

        print(
            f"    held-out       "
            f"acc={metrics['accuracy']:.4f}  "
            f"auc={metrics['auc']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"mcc={metrics['mcc']:.4f}"
        )

    # ---------------- comparison table (README section d) -------------
    results = pd.DataFrame(tuned_rows).set_index("model")
    results = results[METRIC_KEYS]
    results.to_csv(BASE_DIR / "metrics.csv")

    search_record = {
        "search": {
            "scoring": SCORING,
            "cv_folds": CV_FOLDS,
            "cv_splitter": (
                f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, "
                f"random_state={RANDOM_STATE})"
            ),
            "searched_on": "X_train_t only - the held-out set is never seen",
            "grids": jsonable(SEARCH_GRIDS),
        },
        "best_params": jsonable(best_params),
        "cv_f1": jsonable(cv_scores),
    }
    (BASE_DIR / "best_params.json").write_text(
        json.dumps(search_record, indent=2) + "\n", encoding="utf-8"
    )
    print("\n  retained -> best_params.json")

    print()
    print("=" * 74)
    print("COMPARISON TABLE - TUNED  (-> README section d, 5 marks)")
    print("=" * 74)
    print(results.round(4).to_string())

    print()
    print("=" * 74)
    print("BEFORE / AFTER  (tuned minus baseline; positive is an improvement)")
    print("=" * 74)
    print((results - baseline).round(4).to_string())

    print()
    print("=" * 74)
    print("OVERFITTING CHECK  (train acc vs test acc)  -> observations table")
    print("=" * 74)
    for label in [MODEL_LABELS[k] for k in MODEL_KEYS]:
        gap = tuned_train[label] - results.loc[label, "accuracy"]
        base_gap = baseline_train[label] - baseline.loc[label, "accuracy"]
        flag = "  <-- OVERFIT" if gap > 0.05 else ""
        base_flag = "  <-- OVERFIT" if base_gap > 0.05 else ""
        print(
            f"  {label:<24} tuned train={tuned_train[label]:.4f} "
            f"test={results.loc[label, 'accuracy']:.4f} gap={gap:+.4f}{flag}"
        )
        print(
            f"  {'':<24} base  train={baseline_train[label]:.4f} "
            f"test={baseline.loc[label, 'accuracy']:.4f} "
            f"gap={base_gap:+.4f}{base_flag}"
        )

    # RETAIN the training scores.  Previously they were printed and lost, so
    # the generalisation gap quoted in README section d had no file behind
    # it.  Writing them next to metrics.csv means the claim can be checked
    # from the repository instead of taken on trust, and the test column is
    # read back from metrics.csv rather than recomputed - if the two ever
    # disagreed, the file would show it immediately.  Both runs are kept:
    # "the untuned Decision Tree memorised its training data" is a claim
    # about the BASELINE, so it needs the baseline's train score, not the
    # tuned one.
    overfit = pd.DataFrame(
        {
            "model": [MODEL_LABELS[k] for k in MODEL_KEYS],
            "train_accuracy": [tuned_train[MODEL_LABELS[k]] for k in MODEL_KEYS],
            "test_accuracy": [
                results.loc[MODEL_LABELS[k], "accuracy"] for k in MODEL_KEYS
            ],
            "baseline_train_accuracy": [
                baseline_train[MODEL_LABELS[k]] for k in MODEL_KEYS
            ],
            "baseline_test_accuracy": [
                baseline.loc[MODEL_LABELS[k], "accuracy"] for k in MODEL_KEYS
            ],
        }
    )
    overfit["gap"] = overfit["train_accuracy"] - overfit["test_accuracy"]
    overfit["baseline_gap"] = (
        overfit["baseline_train_accuracy"] - overfit["baseline_test_accuracy"]
    )
    overfit.to_csv(BASE_DIR / "train_scores.csv", index=False)
    print(f"\n  retained -> train_scores.csv ({len(overfit)} rows)")

    # ---------------- requirement checks -----------------------------
    print()
    print("=" * 74)
    print("REQUIREMENT CHECK")
    print("=" * 74)
    total_rows = len(train_out) + len(test_out)
    print(f"  total rows {total_rows} >= 500  : {'PASS' if total_rows >= 500 else 'FAIL'}")
    n_feat = X_train_t.shape[1]
    print(f"  features   {n_feat} >= 12   : {'PASS' if n_feat >= 12 else 'FAIL'}")
    print(f"  models     {len(models)} == 5      : {'PASS' if len(models) == 5 else 'FAIL'}")
    print(f"  metrics    {len(METRIC_KEYS)} == 6      : {'PASS' if len(METRIC_KEYS) == 6 else 'FAIL'}")
    print(f"  splits     unchanged (write_split)  : PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
