"""
components.notes - explanatory content that more than one page needs.

Kept out of the pages so the business reading of each classifier is written
once and stays consistent between Overview, Model Explorer and Methodology.
The judgements below are derived from metrics.csv, not asserted from memory.
"""

# Keyed by model KEY (config.MODEL_KEYS).
MODEL_NOTES = {
    "logistic_regression": {
        "family": "Linear decision boundary",
        "best": "Ranking and precision",
        "worst": "Recall",
        "interpretation": (
            "The most disciplined of the five. It produces the highest AUC in "
            "the study, meaning it orders sessions by purchase likelihood better "
            "than any other model here, and when it does flag a buyer it is right "
            "three times out of four. The cost of that discipline is caution: it "
            "declines to call most genuine purchasers, so it leaves revenue on "
            "the table."
        ),
    },
    "decision_tree": {
        "family": "Axis-aligned splits, fully grown",
        "best": "Recall among the interpretable models",
        "worst": "Calibration and generalisation",
        "interpretation": (
            "A single unpruned tree that memorised its training data - it scores "
            "1.0000 on the rows it was fitted to and 0.8550 on held-out ones, a "
            "gap of 0.1450. On new data it casts a wider net than Logistic "
            "Regression, catching more real purchasers, but it also raises more "
            "false alarms, which is what drags its precision down."
        ),
    },
    "knn": {
        "family": "Instance-based, k neighbours",
        "best": "Precision relative to its recall",
        "worst": "Recall of the five",
        "interpretation": (
            "Predicts by looking at the nearest sessions in the scaled feature "
            "space. Because purchasers are the minority class, a visitor's "
            "neighbours are usually non-purchasers, so the model answers 'No' "
            "conservatively. It has the lowest recall in the study and therefore "
            "misses more real buyers than any model except Logistic Regression."
        ),
    },
    "naive_bayes": {
        "family": "Gaussian, feature-independent",
        "best": "Recall",
        "worst": "Precision",
        "interpretation": (
            "Not a coding error - a structural one. It labels almost every "
            "session as a purchase, so it catches 97.91% of real buyers while "
            "82% of its purchase flags are wrong. Its AUC is still 0.7498, so "
            "the underlying ranking is reasonable; the 0.5 threshold is what is "
            "broken. The cause is that its features are not independent: "
            "BounceRates and ExitRates correlate at 0.911, so the model counts "
            "the same evidence twice and inflates its posterior."
        ),
    },
    "svm": {
        "family": "RBF kernel, max-margin",
        "best": "Accuracy, F1 and MCC together",
        "worst": "Nothing across the six metrics",
        "interpretation": (
            "The best-balanced model in the study. It leads on accuracy, F1 and "
            "MCC simultaneously, which matters because those three cannot be "
            "gamed by the class imbalance the way accuracy alone can. It sits "
            "slightly behind Logistic Regression on AUC - a better ranker, but a "
            "worse thresholded classifier."
        ),
    },
}

# The assignment requires an explanation of why metrics disagree.  This is the
# general argument; pages instantiate it with actual numbers.
METRIC_DISAGREEMENT = (
    "Accuracy, AUC and MCC answer three different questions, so they are "
    "allowed to disagree. AUC asks how well the model *ranks* sessions and "
    "ignores the 0.5 cut-off entirely. Accuracy asks how many rows were "
    "labelled correctly and is dominated by whichever class is larger. MCC "
    "asks how well prediction and truth move together and is the only one of "
    "the three that stays honest when the classes are 84/16. A model can "
    "therefore win on ranking while losing on classification - which is "
    "exactly what happens here."
)
