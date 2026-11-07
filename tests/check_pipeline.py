"""
tests/check_pipeline.py - checks that need no browser and no extra package.

Run from anywhere:

    python tests/check_pipeline.py

Exit status 0 means every check passed.  Only packages already pinned in
requirements.txt are imported - no pytest - so this runs in exactly the
environment the application runs in, including a fresh virtualenv.

WHAT IT PROTECTS
----------------
1. The saved artifacts still reproduce metrics.csv to 1e-9.  A retrain that
   silently changed a number, or a metrics.csv edited by hand, fails here.
2. The validation layer behaves on the malformed files a real submission
   gets handed: missing columns, no target, one class, non-numeric values,
   blanks in a numeric column, a header with no rows - and ACCEPTS the files
   it must accept: shuffled columns, surplus columns, and Revenue spelled
   0/1, 0.0/1.0, True/False or yes/no.
3. Nothing outside train.py and the CV comparison script fits anything.
   Checked with the AST, not with a text search, because a text search
   counts the words ".fit()" inside comments and docstrings and would
   report false positives.  The application itself - app.py, config.py,
   pages/, components/, core/ - must never appear, whatever the exemption
   list says.
"""

import ast
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import (  # noqa: E402  (path set up above)
    FEATURES,
    METRIC_KEYS,
    MODEL_KEYS,
    MODEL_LABELS,
    NUMERIC_FEATURES,
    TARGET,
    coerce_target,
)
from core.evaluation import score_all  # noqa: E402
from core.inference import load_artifacts  # noqa: E402
from core.validation import dataset_summary, validate_dataset  # noqa: E402

TOLERANCE = 1e-9

# The only two files allowed to fit, compared by NAME so a copy of either
# in a subdirectory would still be caught:
#   train.py              the single writer of model/ and artifacts/
#   cv_pipeline_check.py  a dev-only script whose whole job is to re-run
#                         the five searches with the preprocessor inside
#                         the CV fold.  It is not imported by the
#                         application, writes nothing, and is kept out of
#                         this suite because it takes minutes.
# Everything else - and every file in the application - must be clean.
FIT_ALLOWED = {"train.py", "cv_pipeline_check.py"}

# Attribute names that learn parameters from data.  fit_transform is
# included because it fits on the way through - forgetting it would leave
# a preprocessor that could be re-fitted from the app.
FIT_METHODS = {"fit", "fit_transform", "partial_fit"}

PASSED, FAILED = [], []


def check(name):
    """Decorator: run a function, record PASS/FAIL, never raise out."""
    def wrap(fn):
        def run():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - we report, not re-raise
                FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"  [FAIL] {name}")
                print(f"         {type(exc).__name__}: {exc}")
                return
            PASSED.append(name)
            print(f"  [PASS] {name}")
        run.__name__ = f"run_{fn.__name__}"
        return run
    return wrap


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# 1. the artifacts reproduce the reported numbers
# ---------------------------------------------------------------------------
@check("saved artifacts reproduce metrics.csv within 1e-9")
def artifacts_match_metrics():
    path = REPO / "metrics.csv"
    require(path.exists(), "metrics.csv is missing")
    expected = pd.read_csv(path).set_index("model")

    data_path = REPO / "test_data.csv"
    require(data_path.exists(), "test_data.csv is missing")
    df = pd.read_csv(data_path)

    preprocessor, models = load_artifacts()
    y_true = coerce_target(df[TARGET]).astype("int64").values
    metrics, preds, probas = score_all(models, preprocessor, df, y_true, MODEL_KEYS)

    require(
        sorted(metrics) == sorted(MODEL_KEYS),
        f"score_all returned {sorted(metrics)}, expected {sorted(MODEL_KEYS)}",
    )

    worst = 0.0
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        require(label in expected.index, f"{label} missing from metrics.csv")
        require(metrics[key] is not None, f"{label} failed to score")
        require(preds[key] is not None, f"{label} produced no predictions")
        require(probas[key] is not None, f"{label} produced no probabilities")
        for metric in METRIC_KEYS:
            got = float(metrics[key][metric])
            want = float(expected.loc[label, metric])
            worst = max(worst, abs(got - want))
            require(
                abs(got - want) <= TOLERANCE,
                f"{label}.{metric}: artifacts give {got!r}, metrics.csv says "
                f"{want!r} (|diff| {abs(got - want):.3e} > {TOLERANCE:.0e})",
            )
    print(f"         worst deviation across 30 cells: {worst:.3e}")


@check("exactly five models are saved and reported")
def five_models():
    model_dir = REPO / "model"
    require(model_dir.is_dir(), "model/ directory is missing")
    saved = sorted(p.stem for p in model_dir.glob("*.joblib"))
    require(
        saved == sorted(MODEL_KEYS),
        f"model/ holds {saved}, expected {sorted(MODEL_KEYS)}",
    )
    frame = pd.read_csv(REPO / "metrics.csv")
    require(len(frame) == 5, f"metrics.csv has {len(frame)} rows, expected 5")
    require(
        list(frame.columns) == ["model"] + METRIC_KEYS,
        f"metrics.csv columns are {list(frame.columns)}",
    )


# ---------------------------------------------------------------------------
# 2. validation: files that must be refused
# ---------------------------------------------------------------------------
def base_frame():
    """A known-good frame, so every mutation below changes one thing."""
    df = pd.read_csv(REPO / "test_data.csv")
    problems = validate_dataset(df)
    require(problems == [], f"the bundled test file should be clean: {problems}")
    return df


@check("missing feature column is reported by name")
def missing_feature():
    df = base_frame().drop(columns=["PageValues"])
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("`PageValues`" in problems[0], f"message does not name the column: {problems[0]}")


@check("missing Revenue column is reported")
def missing_target():
    df = base_frame().drop(columns=[TARGET])
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require(TARGET in problems[0], f"message does not name the target: {problems[0]}")


@check("single-class target is refused (AUC would be undefined)")
def single_class():
    df = base_frame().copy()
    df[TARGET] = 0
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("only" in problems[0] and "AUC" in problems[0], problems[0])


@check("non-numeric feature value is reported with its column")
def non_numeric():
    df = base_frame().copy()
    # astype(object) first: assigning a string into a float64 column is a
    # pandas FutureWarning, and a warning in the test output reads like a
    # failure to anyone skimming it.
    df["BounceRates"] = df["BounceRates"].astype(object)
    df.loc[df.index[0], "BounceRates"] = "not-a-number"
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("`BounceRates`" in problems[0], problems[0])


@check("missing numeric value is reported with its column")
def missing_numeric():
    df = base_frame().copy()
    df.loc[df.index[3], "PageValues"] = float("nan")
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("`PageValues`" in problems[0], f"message does not name the column: {problems[0]}")
    require(
        "1 missing value(s)" in problems[0],
        f"message does not count the holes: {problems[0]}",
    )
    # The point of the check: refuse, do not fill.  A validator that
    # imputed would leave this frame looking usable and quietly change the
    # scores the assignment is graded on.
    require(
        pd.isna(df.loc[df.index[3], "PageValues"]),
        "the validator filled in the hole it was asked to report",
    )


@check("header with no data rows is refused")
def header_only():
    df = base_frame().head(0)
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("no data rows" in problems[0], problems[0])


@check("unreadable Revenue value is reported with the value itself")
def unreadable_target():
    df = base_frame().copy()
    df[TARGET] = df[TARGET].astype(object)
    df.loc[df.index[0], TARGET] = "banana"
    problems = validate_dataset(df)
    require(len(problems) == 1, f"expected one problem, got {problems}")
    require("banana" in problems[0], f"message does not show the bad value: {problems[0]}")
    require(coerce_target(df[TARGET]).isna().sum() == 1, "coerce_target did not flag it")


# ---------------------------------------------------------------------------
# 3. validation: files that must be accepted
# ---------------------------------------------------------------------------
@check("the bundled test_data.csv validates with no problems")
def bundled_clean():
    # The other way round: the file the submission actually ships must not
    # trip the new missing-value rule.  Read fresh rather than through
    # base_frame() so this check stands on its own if the helpers change.
    df = pd.read_csv(REPO / "test_data.csv")
    problems = validate_dataset(df)
    require(problems == [], f"the bundled test file should be clean: {problems}")
    for col in NUMERIC_FEATURES:
        require(
            int(df[col].isna().sum()) == 0,
            f"test_data.csv has blanks in `{col}`",
        )


@check("shuffled column order is accepted unchanged")
def shuffled_columns():
    df = base_frame()
    shuffled = df[list(reversed(df.columns))]
    require(
        list(shuffled.columns) != list(df.columns),
        "test setup: reversing the columns did not change the order",
    )
    problems = validate_dataset(shuffled)
    require(problems == [], f"a shuffled file was refused: {problems}")


@check("surplus columns are accepted and ignored")
def extra_columns():
    df = base_frame().copy()
    df["SessionId"] = range(len(df))
    df["Notes"] = "typed by hand"
    problems = validate_dataset(df)
    require(problems == [], f"a file with extra columns was refused: {problems}")


@check("Revenue as 0/1 integers is accepted")
def revenue_int():
    df = base_frame().copy()
    df[TARGET] = df[TARGET].astype("int64")
    require(validate_dataset(df) == [], "0/1 Revenue was refused")
    require(int(coerce_target(df[TARGET]).sum()) > 0, "positive count lost")


@check("Revenue as 0.0/1.0 floats (Excel round-trip) is accepted")
def revenue_float():
    df = base_frame().copy()
    df[TARGET] = df[TARGET].astype("float64")
    require(validate_dataset(df) == [], "0.0/1.0 Revenue was refused")
    require(
        int(coerce_target(df[TARGET]).sum()) == int(df[TARGET].sum()),
        "float coercion changed the number of purchasers",
    )


@check("Revenue as True/False is accepted")
def revenue_bool():
    df = base_frame().copy()
    df[TARGET] = df[TARGET].astype(bool)
    problems = validate_dataset(df)
    require(problems == [], f"True/False Revenue was refused: {problems}")
    want = int(base_frame()[TARGET].sum())
    require(
        int(coerce_target(df[TARGET]).sum()) == want,
        f"bool coercion gave {int(coerce_target(df[TARGET]).sum())} purchasers, expected {want}",
    )


@check("Revenue as yes/no in mixed case is accepted")
def revenue_words():
    df = base_frame().copy()
    df[TARGET] = df[TARGET].map({0: "No", 1: "Yes"})
    problems = validate_dataset(df)
    require(problems == [], f"yes/no Revenue was refused: {problems}")
    want = int(base_frame()[TARGET].sum())
    require(
        int(coerce_target(df[TARGET]).sum()) == want,
        "word coercion changed the number of purchasers",
    )


@check("summary counts the same purchasers whatever the spelling")
def summary_consistent():
    df = base_frame()
    want = dataset_summary(df)
    for spelling, mutate in (
        ("float", lambda d: d.astype("float64")),
        ("bool", lambda d: d.astype(bool)),
        ("words", lambda d: d.map({0: "no", 1: "y"})),
    ):
        alt = df.copy()
        alt[TARGET] = mutate(alt[TARGET])
        got = dataset_summary(alt)
        require(
            got["purchasers"] == want["purchasers"],
            f"{spelling}: summary says {got['purchasers']}, expected {want['purchasers']}",
        )
        require(
            got["non_purchasers"] == want["non_purchasers"],
            f"{spelling}: summary says {got['non_purchasers']} non-purchasers",
        )


# ---------------------------------------------------------------------------
# 4. nothing outside train.py learns from data
# ---------------------------------------------------------------------------
def fit_calls(allowed=()):
    """Every .fit/.fit_transform call outside `allowed`, as 'path:line:code'."""
    allowed = set(allowed)
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        parts = set(rel.parts)
        if parts & {"__pycache__", ".git", ".venv", "venv", "node_modules"}:
            continue
        if rel.name in allowed:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{rel}: does not parse ({exc})")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr in FIT_METHODS:
                offenders.append(f"{rel}:{node.lineno}: .{func.attr}(")
    return offenders


def top_level(item):
    """'tests\\cv_pipeline_check.py:12: .fit(' -> 'tests'.

    'train.py:4: .fit(' -> 'train.py'.  The path part is everything before
    the first colon; separators are normalised because this runs on
    Windows as well as on Linux.
    """
    return item.split(":", 1)[0].replace("\\", "/").split("/")[0]


@check("only train.py and the CV comparison script fit (AST scan)")
def no_fit_outside_train():
    offenders = fit_calls(allowed=FIT_ALLOWED)
    require(
        offenders == [],
        "these files fit something:\n         " + "\n         ".join(offenders),
    )


@check("the fit scan is not passing vacuously")
def scan_actually_works():
    # If the scanner never matched anything, the check above would pass on
    # an empty list no matter what.  Dropping every exemption must therefore
    # FIND the fits that are really there - proof the detector looks at real
    # calls rather than, say, at comments.
    everything = fit_calls(allowed=())
    require(
        len(everything) >= 4,
        f"expected train.py's fits to be detected, found {everything}",
    )
    require(
        any(item.startswith("train.py:") for item in everything),
        "with train.py unexempted its fits were not detected: "
        + str(everything),
    )
    # ...and however generous the exemption list gets, the application
    # must never be one of the files that fits.  That is the property this
    # pair of checks exists to protect: the application scores with
    # artifacts somebody else fitted.
    in_application = [
        item for item in everything if top_level(item) not in {"train.py", "tests"}
    ]
    require(
        in_application == [],
        "something in the application fits: " + str(in_application),
    )


# ---------------------------------------------------------------------------
# The suite, as a list.
#
# Deliberately at module level rather than inside main(): the Methodology
# page prints the LENGTH of this list next to the command, so the number on
# screen is read from the same object the run iterates over.  Typing the
# count into two files is exactly how a suite grows to 19 checks while the
# README still says 17.
CHECKS = (
    artifacts_match_metrics,
    five_models,
    missing_feature,
    missing_target,
    single_class,
    non_numeric,
    missing_numeric,
    header_only,
    unreadable_target,
    bundled_clean,
    shuffled_columns,
    extra_columns,
    revenue_int,
    revenue_float,
    revenue_bool,
    revenue_words,
    summary_consistent,
    no_fit_outside_train,
    scan_actually_works,
)


def main():
    print("=" * 72)
    print("Session2Shop offline pipeline checks")
    print(f"repository: {REPO}")
    print("=" * 72)

    for fn in CHECKS:
        fn()

    print("=" * 72)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
