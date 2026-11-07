"""
tests/cv_pipeline_check.py - does preprocessing INSIDE the CV fold change
which hyperparameters get selected?

    python tests/cv_pipeline_check.py        # about 3-4 minutes (SVM grid)

WHY THIS EXISTS
---------------
train.py fits the preprocessor on ALL of X_train BEFORE GridSearchCV
runs.  The scaler means, the scaler variances and the one-hot column set
are therefore computed once from the whole training split, so every CV
validation fold inside the search contributes to them.  The held-out test
set is never involved - X_test_t is transformed once, after the search, by
a preprocessor that never saw it - so metrics.csv is unaffected.  It is
still a real ordering question, and the honest way to answer it is to
measure it rather than to assert it.

This script repeats the same five searches with the preprocessor as the
first step of a Pipeline fitted on raw X_train, so it is refitted on every
fold, and compares the hyperparameters that search selects against the
ones train.py recorded in best_params.json.

Everything else is held constant on purpose: the same SEARCH_GRIDS, the
same StratifiedKFold(5, shuffle=True, random_state=42), the same "f1"
scoring, n_jobs=1, and the same probability=False shortcut on the SVM
search that train.py's tune_one() takes - scoring "f1" needs predict()
only, and predict() comes from the decision function, which Platt scaling
does not touch, so it cannot change which parameter wins.

IT WRITES NOTHING.  No artifact, no metrics file, no best_params.json, no
figure: the only output is printed.  It is deliberately kept OUT of
tests/check_pipeline.py because the SVM grid alone takes minutes, which
would make the offline suite unusable as a quick check.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from config import (  # noqa: E402
    MODEL_KEYS,
    MODEL_LABELS,
    RANDOM_STATE,
    TARGET,
    prepare_features,
)
from train import (  # noqa: E402
    CV_FOLDS,
    SCORING,
    SEARCH_GRIDS,
    build_models,
    build_preprocessor,
    jsonable,
)


def load_training_split():
    """Recover (X_train, y_train) exactly as train.py split them.

    train_data.csv IS the training split: train.py writes X_train with the
    target column appended, unchanged.  Reading it back and running
    prepare_features() on it rebuilds the frame the search ran on - the
    categorical columns come back as the same strings OneHotEncoder saw -
    without re-splitting, re-downloading, or touching a committed
    artifact.
    """
    df = pd.read_csv(REPO / "train_data.csv")
    return prepare_features(df), df[TARGET].astype("int64")


def pipeline_search(key, model, X, y):
    """The recorded search, with the preprocessor moved inside the fold.

    Returns (selected_params, cv_f1) as plain Python.
    """
    classifier = clone(model)
    if key == "svm":
        # The same shortcut train.py's tune_one() takes, for the same
        # reason: it cannot affect which parameter wins, and without it
        # every candidate fit would run a second internal 5-fold
        # calibration.
        classifier.set_params(probability=False)

    pipe = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("clf", classifier)]
    )
    search = GridSearchCV(
        estimator=pipe,
        # SEARCH_GRIDS is written for a bare classifier; on a Pipeline
        # every parameter has to name the step it belongs to.
        param_grid={
            f"clf__{name}": values for name, values in SEARCH_GRIDS[key].items()
        },
        scoring=SCORING,
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        refit=False,  # nothing beyond the search itself gets fitted
        n_jobs=1,
    )
    search.fit(X, y)

    chosen = {
        name.split("__", 1)[1]: value
        for name, value in search.best_params_.items()
    }
    return jsonable(chosen), float(search.best_score_)


def main():
    print("=" * 74)
    print("Cross-validation preprocessing order")
    print("=" * 74)
    print("train.py fits the preprocessor on all of X_train before the")
    print("search, so the CV validation folds contribute to the scaling")
    print("and encoding statistics.  This repeats the same five searches")
    print("with the preprocessor INSIDE the fold and compares what each")
    print("one selects against best_params.json.")
    print()

    recorded = json.loads(
        (REPO / "best_params.json").read_text(encoding="utf-8")
    )
    X, y = load_training_split()
    models = build_models()
    print(f"  train_data.csv -> X {X.shape}   y {y.shape}   "
          f"{y.mean() * 100:.1f}% positive")
    print()

    pipeline_cv = {}
    matched = 0
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        started = time.perf_counter()
        chosen, cv_f1 = pipeline_search(key, models[key], X, y)
        pipeline_cv[key] = cv_f1
        elapsed = time.perf_counter() - started

        expected = recorded["best_params"][key]
        same = chosen == expected
        matched += int(same)

        print(f"  {label}")
        print(f"    cv_f1 inside fold : {cv_f1:.4f}   "
              f"(recorded {recorded['cv_f1'][key]:.4f}, {elapsed:.0f}s)")
        print(f"    selected          : {'MATCH' if same else 'DIFFERS'} {chosen}")
        if not same:
            print(f"    recorded          : {expected}")
        print()

    drift = max(
        abs(pipeline_cv[k] - recorded["cv_f1"][k]) for k in MODEL_KEYS
    )

    print("=" * 74)
    print(f"  largest cv_f1 movement across the five searches: {drift:.4f}")
    print("=" * 74)
    if matched == len(MODEL_KEYS):
        print(f"PASS: all {matched} of {len(MODEL_KEYS)} models selected identical")
        print("hyperparameters, so the reported results in metrics.csv are")
        print("unchanged.")
        print("=" * 74)
        return 0

    print(
        f"FAIL: {len(MODEL_KEYS) - matched} of {len(MODEL_KEYS)} models selected"
        " different hyperparameters."
    )
    print("=" * 74)
    return 1


if __name__ == "__main__":
    sys.exit(main())
