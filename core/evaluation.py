"""
core.evaluation - the six required metrics and the five-model comparison.

Every number the app shows is produced by the functions below, so the
interface and the README cannot drift apart: metrics.csv is written by
train.py using these same definitions.

The metric list is fixed by the assignment brief:
accuracy, AUC, precision, recall, F1, MCC.
"""

import pandas as pd
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

from config import METRIC_KEYS, MODEL_KEYS
from core.inference import predict

NEG_LABEL = "No purchase"
POS_LABEL = "Will purchase"


def compute_metrics(y_true, y_pred, y_proba):
    """The six metrics, in the order the assignment lists them."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def confusion_counts(y_true, y_pred):
    """(tn, fp, fn, tp) with labels pinned to [0, 1].

    labels= matters: if a batch happened to contain only one class,
    confusion_matrix would return a 1x1 matrix and ravel() would fail.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def report_dict(y_true, y_pred):
    """classification_report as a dict, pinned to both class labels."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[NEG_LABEL, POS_LABEL],
        output_dict=True,
        zero_division=0,
    )


def score_one(model, preprocessor, df, y_true):
    """Six metrics plus predictions for a single classifier."""
    y_pred, y_proba = predict(model, preprocessor, df)
    return compute_metrics(y_true, y_pred, y_proba), y_pred


def score_all(models, preprocessor, df, y_true, labels):
    """Score every saved classifier on the same data.

    Returns ({key: metrics|None}, {key: y_pred|None}).

    A model that fails yields None rather than NaN: an exception inside one
    row must not silently poison the whole comparison table.  The UI renders
    None as an em dash and tells the user which model failed.
    """
    metrics, preds = {}, {}
    for key in labels:
        try:
            m, p = score_one(models[key], preprocessor, df, y_true)
            metrics[key], preds[key] = m, p
        except Exception:
            metrics[key], preds[key] = None, None
    return metrics, preds


def comparison_frame(metrics_by_key, labels):
    """Turn {key: metrics} into the wide table the UI and README both show.

    Missing entries become pd.NA so they render as an em dash instead of
    the literal string 'nan'.
    """
    rows = []
    for key in labels:
        m = metrics_by_key.get(key)
        if m is None:
            rows.append({"model": key, **{k: pd.NA for k in METRIC_KEYS}})
        else:
            rows.append({"model": key, **m})
    return pd.DataFrame(rows).set_index("model")
