"""
components.notes - interpretation of each classifier, GENERATED from scores.

WHY THIS IS A FUNCTION AND NOT A DICTIONARY OF PROSE
----------------------------------------------------
It used to be a dictionary of hand-written sentences with numbers typed
into them.  Two of them were already wrong: the KNN note claimed it "misses
more real buyers than any model except Logistic Regression" while KNN in
fact had the LOWEST recall of the five, and the SVM note claimed its worst
metric was "Nothing", which overclaims for any model.  A sentence written
once and quoted forever will drift away from metrics.csv the moment the
models are retrained - and in a viva the examiner only has to open one file
to find the contradiction.

So: the ranks, the values, the generalisation gap and the two correlations
are all computed here, from the same frame the comparison table renders.
A statement that is generated cannot disagree with the table it sits next
to.  Only the structural facts - which family a model belongs to, and the
mechanism that limits it - remain literal, because those do not depend on
the scores.

Callers pass:
    frame         comparison DataFrame indexed by model KEY (5 x 6)
    train_scores  {key: {"train_accuracy", "test_accuracy", "gap"}}
    correlations  {"A~B": pearson_r} from core.facts
"""

from components.layout import METRIC_LABELS
from config import METRIC_KEYS, MODEL_LABELS

# Static: architecture, not performance.  Safe to hard-code.
MODEL_FAMILY = {
    "logistic_regression": "Linear decision boundary",
    "decision_tree": "Axis-aligned splits, pruned",
    "knn": "Instance-based, k neighbours",
    "naive_bayes": "Gaussian, feature-independent",
    "svm": "RBF kernel, max-margin",
}

# Static: WHY each model behaves as it does.  Deliberately contains no
# metric values - those come from frame - so it cannot go stale.
MECHANISM = {
    "logistic_regression": (
        "One weight per column and no interaction terms, so it draws a "
        "straight boundary: it ranks sessions well but cannot bend around "
        "combinations of features."
    ),
    "decision_tree": (
        "A deeper tree can carve out pockets of the training rows that do "
        "not exist in new data, which is why depth and leaf size are the "
        "two knobs that decide whether it generalises."
    ),
    "knn": (
        "It answers with the majority vote of the nearest sessions in the "
        "scaled feature space, so with purchasers in the minority a "
        "session's neighbours are usually non-purchasers."
    ),
    "naive_bayes": (
        "Its features are treated as independent, but they are not: "
        "BounceRates and ExitRates correlate at {bounce_exit:.3f} and "
        "ProductRelated with ProductRelated_Duration at "
        "{product_duration:.3f}, so the same evidence is counted twice "
        "and the posterior is pushed off balance."
    ),
    "svm": (
        "A maximum-margin boundary in the kernel space; class_weight "
        "changes how hard the margin fights for the minority class, which "
        "moves precision and recall against each other."
    ),
}


def _ranks(frame, key):
    """{metric: (rank, n)} where rank 1 is best of the scored models.

    Ties share a rank: a model is beaten only by models that score strictly
    higher, so "5 of 5" genuinely means last and not merely "listed fifth".
    """
    ranks = {}
    for metric in METRIC_KEYS:
        col = frame[metric].dropna() if metric in frame.columns else []
        if key not in getattr(col, "index", []):
            ranks[metric] = (None, len(col))
            continue
        ranks[metric] = (1 + int((col > col[key]).sum()), len(col))
    return ranks


def _extreme(ranks, metrics, pick_best):
    """The metric this model is strongest/weakest on, ties broken by order."""
    scored = [m for m in metrics if ranks[m][0] is not None]
    if not scored:
        return None
    if pick_best:
        return min(scored, key=lambda m: (ranks[m][0], metrics.index(m)))
    return max(scored, key=lambda m: (ranks[m][0], metrics.index(m)))


def model_note(key, frame, train_scores, correlations=None):
    """Build the interpretation dict for one model from the live scores.

    Returns {"family", "best", "worst", "interpretation"}.
    """
    family = MODEL_FAMILY[key]
    correlations = correlations or {}

    ranks = _ranks(frame, key)
    if ranks["mcc"][0] is None:
        # The model could not be scored on this dataset; say so instead of
        # inventing an interpretation from an empty frame.
        return {
            "family": family,
            "best": METRIC_LABELS["mcc"],
            "worst": METRIC_LABELS["mcc"],
            "interpretation": (
                "This classifier could not be scored on the loaded dataset, "
                "so there is no metric to interpret. Upload a CSV in which "
                "both classes are present."
            ),
        }

    label = MODEL_LABELS[key]

    def value(metric):
        return float(frame.loc[key, metric])

    def rank_text(metric):
        rank, total = ranks[metric]
        return f"{rank} of {total}"

    best_metric = _extreme(ranks, METRIC_KEYS, pick_best=True)
    worst_metric = _extreme(ranks, METRIC_KEYS, pick_best=False)

    best_rank = ranks[best_metric][0]
    worst_rank = ranks[worst_metric][0]
    total = ranks[best_metric][1]

    if best_rank == 1:
        best = (
            f"{METRIC_LABELS[best_metric]} {value(best_metric):.4f} "
            f"- best of the {total}"
        )
    else:
        best = (
            f"{METRIC_LABELS[best_metric]} {value(best_metric):.4f} "
            f"- rank {best_rank} of {total}"
        )

    if worst_rank == total:
        worst = (
            f"{METRIC_LABELS[worst_metric]} {value(worst_metric):.4f} "
            f"- lowest of the {total}"
        )
    else:
        worst = (
            f"{METRIC_LABELS[worst_metric]} {value(worst_metric):.4f} "
            f"- rank {worst_rank} of {total}"
        )

    # ---------------------------------------------------------------- prose
    mcc_rank, mcc_total = ranks["mcc"]
    sentences = [
        (
            f"{label} ranks {mcc_rank} of {mcc_total} on MCC "
            f"({value('mcc'):.4f}), the metric that stays honest when one "
            f"class dominates, and {rank_text('f1')} on F1 "
            f"({value('f1'):.4f})."
        ),
        (
            f"It is best of the {total} on {METRIC_LABELS[best_metric]} at "
            f"{value(best_metric):.4f}."
            if best_rank == 1
            else (
                f"Its strongest showing is {METRIC_LABELS[best_metric]} at "
                f"{value(best_metric):.4f}, rank {best_rank} of {total}."
            )
        ),
        (
            f"It is lowest of the {total} on "
            f"{METRIC_LABELS[worst_metric]} at {value(worst_metric):.4f}."
            if worst_rank == total
            else (
                f"Its weakest showing is {METRIC_LABELS[worst_metric]} at "
                f"{value(worst_metric):.4f}, rank {worst_rank} of {total}."
            )
        ),
    ]

    # Overfitting, read from train_scores.csv rather than remembered.
    scores = (train_scores or {}).get(key)
    if scores:
        gap = scores["gap"]
        if gap > 0.05:
            sentences.append(
                f"It scores {scores['train_accuracy']:.4f} on the rows it "
                f"was fitted to against {scores['test_accuracy']:.4f} on "
                f"held-out rows, a gap of {gap:+.4f} - it fits the training "
                "data better than it predicts new data."
            )
        else:
            sentences.append(
                f"Training accuracy {scores['train_accuracy']:.4f} against "
                f"{scores['test_accuracy']:.4f} held out (gap "
                f"{gap:+.4f}), so it generalises about as well as it fits."
            )

    mechanism = MECHANISM[key]
    if key == "naive_bayes":
        mechanism = mechanism.format(
            bounce_exit=correlations.get("BounceRates~ExitRates", float("nan")),
            product_duration=correlations.get(
                "ProductRelated~ProductRelated_Duration", float("nan")
            ),
        )
    sentences.append(mechanism)

    return {
        "family": family,
        "best": best,
        "worst": worst,
        "interpretation": " ".join(sentences),
    }


# The assignment requires an explanation of why metrics disagree.  The
# general argument is fixed; the closing sentence is generated, because
# "which is exactly what happens here" is only true while the AUC leader
# and the MCC leader really are different models.
def metric_disagreement(frame=None):
    text = (
        "Accuracy, AUC and MCC answer three different questions, so they are "
        "allowed to disagree. AUC asks how well the model *ranks* sessions "
        "and ignores the 0.5 cut-off entirely. Accuracy asks how many rows "
        "were labelled correctly and is dominated by whichever class is "
        "larger. MCC asks how well prediction and truth move together and "
        "is the only one of the three that stays honest when one class "
        "dominates. A model can therefore win on ranking while losing on "
        "classification."
    )
    if frame is None:
        return text

    auc_col = frame["auc"].dropna() if "auc" in frame.columns else []
    mcc_col = frame["mcc"].dropna() if "mcc" in frame.columns else []
    if auc_col.empty or mcc_col.empty:
        return text

    auc_leader, mcc_leader = auc_col.idxmax(), mcc_col.idxmax()
    if auc_leader != mcc_leader:
        text += (
            f" Which is exactly what happens here: "
            f"{MODEL_LABELS[auc_leader]} leads on AUC "
            f"({auc_col.max():.4f}) while {MODEL_LABELS[mcc_leader]} leads "
            f"on MCC ({mcc_col.max():.4f})."
        )
    else:
        text += (
            f" On the loaded data {MODEL_LABELS[mcc_leader]} leads on both "
            f"AUC ({auc_col.max():.4f}) and MCC ({mcc_col.max():.4f}), so "
            "here the two questions happen to agree."
        )
    return text
