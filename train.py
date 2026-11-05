"""
train.py - Phase 1 (preprocessing) + Phase 2 (train & evaluate 5 models).

    python train.py

OUTPUTS
-------
  artifacts/preprocessor.joblib      fitted ColumnTransformer
  train_data.csv / test_data.csv     splits (original column form)
  model/<key>.joblib                 5 trained classifiers
  figures/confusion_matrix_<key>.png confusion matrices for the README
  metrics.csv                        the 5 x 6 comparison table

DESIGN RULE ENFORCED HERE
-------------------------
Split FIRST -> fit the preprocessor on X_train ONLY -> transform both.
Fitting the scaler on the whole dataset would leak test-set means and
standard deviations into training, producing optimistically biased scores.
"""

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

matplotlib.use("Agg")  # headless-safe: required on BITS Lab / Streamlit Cloud
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "raw" / "online_shoppers_intention.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "model"
FIGURES_DIR = BASE_DIR / "figures"


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
    test_out.to_csv(BASE_DIR / "test_data.csv", index=False)

    train_out = X_train.copy()
    train_out[TARGET] = y_train.values
    train_out.to_csv(BASE_DIR / "train_data.csv", index=False)

    # ============================ PHASE 2 ============================
    print()
    print("=" * 74)
    print("PHASE 2 :: TRAIN & EVALUATE 5 MODELS")
    print("=" * 74)

    MODEL_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    models = build_models()
    rows = []
    train_accs = {}

    for key in MODEL_KEYS:
        model = models[key]
        label = MODEL_LABELS[key]

        model.fit(X_train_t, y_train)

        y_pred = model.predict(X_test_t)
        # All five expose predict_proba, so AUC is computed uniformly
        # from the probability of the POSITIVE class.
        y_proba = model.predict_proba(X_test_t)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = label
        rows.append(metrics)

        # Train accuracy is captured purely to expose OVERFITTING - it is
        # the raw material for the graded "observations" table.
        train_accs[label] = accuracy_score(y_train, model.predict(X_train_t))

        joblib.dump(model, MODEL_DIR / f"{key}.joblib")
        plot_confusion_matrix(y_test, y_pred, label, FIGURES_DIR / f"confusion_matrix_{key}.png")

        print(
            f"  {label:<24} "
            f"acc={metrics['accuracy']:.4f}  "
            f"auc={metrics['auc']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"mcc={metrics['mcc']:.4f}"
        )

    # ---------------- comparison table (README section d) -------------
    results = pd.DataFrame(rows).set_index("model")
    results = results[METRIC_KEYS]
    results.to_csv(BASE_DIR / "metrics.csv")

    print()
    print("=" * 74)
    print("COMPARISON TABLE  (-> README section d, 5 marks)")
    print("=" * 74)
    print(results.round(4).to_string())

    print()
    print("=" * 74)
    print("OVERFITTING CHECK  (train acc vs test acc)  -> observations table")
    print("=" * 74)
    for label in [MODEL_LABELS[k] for k in MODEL_KEYS]:
        gap = train_accs[label] - results.loc[label, "accuracy"]
        flag = "  <-- OVERFIT" if gap > 0.05 else ""
        print(
            f"  {label:<24} train={train_accs[label]:.4f}  "
            f"test={results.loc[label, 'accuracy']:.4f}  gap={gap:+.4f}{flag}"
        )

    # RETAIN the training scores.  Previously they were printed and lost, so
    # the "+0.1450 Decision Tree gap" quoted in README section d had no file
    # behind it.  Writing them next to metrics.csv means the claim can be
    # checked from the repository instead of taken on trust, and the test
    # column is read back from metrics.csv rather than recomputed - if the
    # two ever disagreed, the file would show it immediately.
    overfit = pd.DataFrame(
        {
            "model": [MODEL_LABELS[k] for k in MODEL_KEYS],
            "train_accuracy": [train_accs[MODEL_LABELS[k]] for k in MODEL_KEYS],
            "test_accuracy": [results.loc[MODEL_LABELS[k], "accuracy"] for k in MODEL_KEYS],
        }
    )
    overfit["gap"] = overfit["train_accuracy"] - overfit["test_accuracy"]
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
    print("=" * 74)


if __name__ == "__main__":
    main()
