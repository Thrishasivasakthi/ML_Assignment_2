"""
core.facts - every background figure the interface and the README quote,
computed from this repository's own files.

WHY THIS EXISTS
---------------
The interface and the README used to state several numbers from memory:
the always-"No" baseline, the 12,330 -> 12,205 de-duplication, the size of
each split, and the two feature correlations used to argue that Naive Bayes's
independence assumption fails.  One of those correlations was quoted as
0.911 when the data says about 0.90.  A number that only exists in prose
cannot be checked; a number read back from raw/, train_data.csv and
test_data.csv can be, every time the app starts.

Nothing here touches the models, so it is safe to call anywhere.
"""

from pathlib import Path

import pandas as pd

from config import TARGET

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "raw" / "online_shoppers_intention.csv"
TRAIN_CSV = BASE_DIR / "train_data.csv"
TEST_CSV = BASE_DIR / "test_data.csv"

# The two pairs quoted as evidence that Naive Bayes double-counts evidence.
# Both are pairs of features that measure overlapping things: bounce rate
# and exit rate are two views of the same "user left quickly" signal, and
# product page count and product page duration are two views of the same
# "user browsed a lot" signal.
CORRELATION_PAIRS = [
    ("BounceRates", "ExitRates"),
    ("ProductRelated", "ProductRelated_Duration"),
]


def _positive_share(path):
    """Fraction of rows whose Revenue is 1, read from one CSV."""
    series = pd.read_csv(path, usecols=[TARGET])[TARGET]
    if not len(series):
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def dataset_facts():
    """All background numbers, in one pass over the three data files."""
    raw_df = pd.read_csv(RAW_CSV)
    dedup_df = raw_df.drop_duplicates()

    train_rows = len(pd.read_csv(TRAIN_CSV, usecols=[TARGET]))
    test_rows = len(pd.read_csv(TEST_CSV, usecols=[TARGET]))
    raw_rows = len(raw_df)

    return {
        # Pearson r for each CORRELATION_PAIRS entry, on the de-duplicated
        # population the split was drawn from - so the figure does not
        # depend on which half of the split you happen to open.
        "correlations": {
            f"{a}~{b}": float(dedup_df[a].corr(dedup_df[b]))
            for a, b in CORRELATION_PAIRS
        },
        "splits": {
            "raw": raw_rows,
            "duplicates": raw_rows - len(dedup_df),
            "deduplicated": len(dedup_df),
            "train": train_rows,
            "test": test_rows,
        },
        # Class ratio of each population, used where the prose is about the
        # SPLIT rather than about whatever file the user happens to have
        # loaded (those places read core.validation.dataset_summary instead).
        "balance": {
            "deduplicated": float((dedup_df[TARGET] == 1).mean()),
            "train": _positive_share(TRAIN_CSV),
            "test": _positive_share(TEST_CSV),
        },
    }
