# Session2Shop — Purchase Intent Classifier

**M.Tech (AIML) · Machine Learning · Assignment 2**

A Streamlit application that predicts whether an e-commerce browsing session
ends in a purchase, using five classifiers trained on the same data and
evaluated with the six metrics the assignment specifies.

---

## Live Streamlit app

**`<STREAMLIT_URL>`**

Deployed from this repository on Streamlit Community Cloud, on Python 3.11.
The build serves the committed `model/*.joblib` and
`artifacts/preprocessor.joblib`: a CSV uploaded to it is scored, never
trained on.

## BITS Virtual Lab

Screenshots of the application running on the BITS Virtual Lab are in the
submitted PDF. This repository carries the code and its outputs only.

---

## a. Problem statement

An e-commerce platform wants to know, from a visitor's browsing behaviour
alone, whether a session will end in a purchase — so that attention, offers
and retargeting spend can be aimed at the sessions most likely to convert.

Framed as machine learning, this is a **binary classification problem over 17
behavioural features** captured per session: how many pages of each type were
visited, how long was spent on them, and what the Google Analytics metrics
(bounce rate, exit rate, page value) recorded during the visit. The target
column, `Revenue`, marks whether the session produced a transaction.

Two properties of the problem shape everything that follows:

1. **The classes are badly skewed.** Only 15.6% of sessions end in a purchase.
   A classifier that answers "No purchase" for every visitor is therefore
   **84.4% accurate while being completely useless**. Accuracy on its own
   cannot separate a real model from a trivial one, which is why this
   assignment requires MCC and AUC alongside it.

2. **The two errors are not equivalent.** Missing a genuine purchaser is lost
   revenue; flagging a non-purchaser as a buyer is wasted spend. A single
   number cannot represent both, so the confusion matrix is treated as a
   first-class result rather than a footnote.

This project implements five classifiers — Logistic Regression, Decision Tree,
K-Nearest Neighbour, Gaussian Naive Bayes and Support Vector Machine — on one
dataset, evaluates all six metrics on one held-out test set, and serves the
results through a Streamlit application that **loads saved models and never
retrains on uploaded data**.

---

## b. Dataset description

### Online Shoppers Purchasing Intention Dataset

| Property | Value |
|---|---|
| **Repository** | UCI Machine Learning Repository, dataset 468 |
| **URL** | <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| **DOI** | `10.24432/C5F88Q` |
| **Also published on** | Kaggle — `imakash3011/online-shoppers-purchasing-intention-dataset` |
| **Licence** | CC BY 4.0 |
| **Citation** | Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. |
| **Original size** | 12,330 rows × 17 feature columns + 1 target column |
| **Target** | `Revenue` — binary, whether the session ended in a purchase |
| **Class balance** | 15.6% purchasers / 84.4% non-purchasers |
| **Task** | Binary classification |

Each row is a single browsing session over one year of visits to an e-commerce
site. Features describe *what the visitor did* (pages visited, time spent) and
*what the analytics measured* (bounce rate, exit rate, page value), plus the
context of the visit (month, device, traffic source, weekend flag).

### The 17 features

**Numeric — 10 columns, standardised with `StandardScaler`**

| Feature | Description |
|---|---|
| `Administrative` | Count of administrative pages visited |
| `Administrative_Duration` | Total time on administrative pages |
| `Informational` | Count of informational pages visited |
| `Informational_Duration` | Total time on informational pages |
| `ProductRelated` | Count of product-related pages visited |
| `ProductRelated_Duration` | Total time on product-related pages |
| `BounceRates` | Average bounce rate of the entry pages (Google Analytics) |
| `ExitRates` | Average exit rate across the pages visited (Google Analytics) |
| `PageValues` | Average value of the pages visited, weighted by transaction |
| `SpecialDay` | Closeness of the visit to a special day, 0 to 1 |

**Categorical — 7 columns, one-hot encoded with `OneHotEncoder`**

| Feature | Description |
|---|---|
| `Month` | Calendar month of the visit |
| `OperatingSystems` | Operating system identifier (nominal code) |
| `Browser` | Browser identifier (nominal code) |
| `Region` | Geographic region identifier (nominal code) |
| `TrafficType` | Traffic source type identifier (nominal code) |
| `VisitorType` | Returning visitor, new visitor, or other |
| `Weekend` | Whether the visit fell on a weekend |

`OperatingSystems`, `Browser`, `Region` and `TrafficType` arrive as integers
but are **nominal identifiers, not quantities** — `Browser=2` is not "twice"
`Browser=1`. They are one-hot encoded rather than scaled, because treating
them as numbers would impose a fake ordering and a fake scale on the model.

### Preprocessing and split

| Step | Detail |
|---|---|
| 1. De-duplicate | 12,330 raw rows → **12,205** after removing 125 exact duplicates |
| 2. Split first | `train_test_split(test_size=0.2, random_state=42, stratify=y)` → **9,764 train / 2,441 held-out**, preserving the 84.4/15.6 ratio in both |
| 3. Fit preprocessor on training rows only | `ColumnTransformer`: 10 numeric → `StandardScaler`; 7 categorical → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`; `remainder="drop"` |
| 4. Transform held-out rows | With the same fitted preprocessor — never refit on test data |
| 5. Output width | **74 columns** (10 scaled + 64 one-hot) |

Splitting **before** fitting the preprocessor is what keeps the held-out set
honest: the statistics used to scale the test rows were never allowed to see
them. `handle_unknown="ignore"` means a category that never appeared in
training becomes an all-zero block rather than an error, so an uploaded CSV
with an unseen month or browser still scores.

### Why this dataset

It clears the brief's floor of **12 features and 500 instances** (17 and
12,330); it is a genuine classification problem with a real business question
behind it; and — decisively — its class imbalance makes accuracy alone
misleading, which is precisely the situation in which the required metric set
(AUC, F1 and especially MCC) earns its place.

---

## c. GitHub Repository Link

**<https://github.com/Thrishasivasakthi/ML_Assignment_2>**

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point — **Overview** page |
| `pages/02_Model_Explorer.py` | Per-classifier detail and training configuration |
| `pages/03_Model_Benchmarking.py` | Five-model comparison, leaderboard, metric leaders |
| `pages/04_Error_Analysis.py` | Confusion matrix, false alarms vs missed buyers |
| `pages/05_Data_Workspace.py` | Upload, validation, column inventory, preview |
| `pages/06_Methodology.py` | Dataset, preprocessing, models, metrics, reproduction |
| `core/inference.py` | Loads saved artifacts and scores them — **no `.fit()` anywhere** |
| `core/evaluation.py` | The six metrics and the five-model comparison |
| `core/validation.py` | Pre-flight checks on an uploaded CSV |
| `core/facts.py` | Background figures (split sizes, class ratios, correlations) computed from the data files |
| `components/` | Design system, sidebar, cards, charts, tables |
| `config.py` | Single source of truth: column lists, `prepare_features()`, seed |
| `train.py` | Training pipeline — the only file that fits anything |
| `make_readme_tables.py` | Prints the tables in section `d` straight from the CSVs, and verifies them |
| `tests/check_pipeline.py` | Offline check: artifacts reproduce `metrics.csv`, validation edge cases, no fitting outside `train.py` |
| `model/*.joblib` | Five saved classifiers |
| `artifacts/preprocessor.joblib` | Fitted `ColumnTransformer` |
| `test_data.csv` | Held-out test set (2,441 × 18) the app accepts |
| `train_data.csv` | Training split, for reference |
| `metrics.csv` | The tuned comparison table below, as written by `train.py` |
| `metrics_baseline.csv` | The same five models at sklearn defaults — the "before" column of the before/after table |
| `best_params.json` | Search definition and the winning hyperparameters per model |
| `train_scores.csv` | Train vs test accuracy for both runs — the source of the `+0.1450` Decision Tree gap quoted below |
| `figures/` | Five confusion-matrix plots |
| `requirements.txt` | Every imported package, exact-pinned |
| `.gitattributes` | Line-ending policy: text stored as LF, CSVs checked out as LF everywhere, joblib and PNG never touched |
| `.gitignore` | Keeps `__pycache__/`, `uci_download.zip`, the brief and the working notes out of the repository |
| `.streamlit/config.toml` | Theme and toolbar configuration |

---

## d. Models used

All five models are implemented on the **same dataset**, trained on the **same
training split**, and evaluated on the **same held-out test set** of 2,441
rows. Every figure below is read from a file `train.py` writes —
`metrics.csv` (tuned), `metrics_baseline.csv` (untuned) and
`train_scores.csv` (train vs test accuracy). Reprint all three tables with
`python make_readme_tables.py --all`.

### How the models were tuned

Each model was tuned with `GridSearchCV` using **5-fold
`StratifiedKFold(shuffle=True, random_state=42)` over `X_train_t` only** —
the training split, never the held-out split. The search scores on **`f1`**,
and the held-out set is scored exactly once, after the search. The grids and
the winning parameters are recorded in `best_params.json`.

**Why `f1` rather than `roc_auc`.** `f1` is one of the six metrics the
assignment requires and the one built from the two error costs this problem
actually has — a missed purchaser against a false alarm. The application
also decides at the 0.5 cut-off, so tuning on AUC would optimise a ranking
that is thresholded anyway and could pick a setting that wins on AUC while
losing on F1 and MCC. The counter-argument is recorded rather than hidden:
`f1` is threshold-dependent and noisier under cross-validation than
`roc_auc`, which is why the folds are stratified — about 300 purchasers in
`X_train`, roughly 60 per fold.

The grids were kept small on purpose (4 × 2 for Logistic Regression,
4 × 3 × 2 for the tree, 5 × 2 for KNN, 9 decades for Naive Bayes,
3 × 3 × 2 for SVM), because a large grid on a 15.6% positive class fits the
cross-validation noise rather than the signal.

### Comparison table — tuned (`metrics.csv`, what the app reports)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8587 | 0.9099 | 0.5325 | 0.7932 | 0.6372 | 0.5706 |
| Decision Tree | 0.9021 | 0.9263 | 0.7085 | 0.6361 | 0.6703 | 0.6143 |
| KNN | 0.8693 | 0.7681 | 0.6245 | 0.4136 | 0.4976 | 0.4380 |
| Naive Bayes | 0.8161 | 0.8418 | 0.4419 | 0.6675 | 0.5318 | 0.4371 |
| Support Vector Machine | 0.8718 | 0.9084 | 0.5670 | 0.7644 | 0.6511 | 0.5842 |

*Best value per column: Accuracy — Decision Tree; AUC — Decision Tree;
Precision — Decision Tree; Recall — Logistic Regression; F1 — Decision Tree;
MCC — Decision Tree.*

### Comparison table — untuned, sklearn defaults (`metrics_baseline.csv`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8550 | 0.7435 | 0.5337 | 0.5812 | 0.5564 | 0.4705 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.2773 | 0.7498 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8733 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

### Before and after (tuned − baseline)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | -0.0283 | 0.0104 | -0.2199 | 0.3796 | 0.1034 | 0.0674 |
| Decision Tree | 0.0471 | 0.1828 | 0.1748 | 0.0550 | 0.1140 | 0.1438 |
| KNN | -0.0041 | -0.0277 | -0.0377 | 0.0236 | 0.0067 | -0.0055 |
| Naive Bayes | 0.5387 | 0.0920 | 0.2664 | -0.3115 | 0.2340 | 0.2996 |
| Support Vector Machine | -0.0238 | 0.0352 | -0.1691 | 0.2461 | 0.0428 | 0.0229 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The search chose `C=0.01` with `class_weight='balanced'`. Recall rises from 0.4136 to 0.7932, the highest of the five, while precision falls from 0.7524 to 0.5325 and accuracy slips from 0.8869 to 0.8587 — the model was moved along the precision/recall trade-off rather than made better at both. F1 still improves from 0.5338 to 0.6372, MCC from 0.5032 to 0.5706 and AUC from 0.8996 to 0.9099. Train and test accuracy differ by +0.0008 (`train_scores.csv`), so it generalises about as well as it fits. |
| **Decision Tree** | The search chose `max_depth=5` and `min_samples_leaf=20`, leaving `class_weight` at `None`, so the gain comes from pruning alone. The untuned tree was grown without restriction: 1.0000 on the rows it was fitted to against 0.8550 on held-out rows, a gap of +0.1450, against 0.9082 and 0.9021 for the pruned tree, a gap of +0.0061 — pruning removed the memorisation instead of trading it away. It now leads on five of the six metrics: accuracy 0.9021, AUC 0.9263, precision 0.7085, F1 0.6703 and MCC 0.6143. Recall is its weakest showing at 0.6361, fourth of five, which is the price of a conservative tree. |
| **KNN** | The search kept `n_neighbors=3` with `weights='uniform'`, and on this data tuning bought almost nothing: F1 moves from 0.4909 to 0.4976 while MCC falls from 0.4435 to 0.4380 and AUC from 0.7958 to 0.7681. The search optimised `f1`, and the other metrics moved against it — a single-metric objective is allowed to do that, and reporting all six is what makes it visible. Its weakest showing is F1 at 0.4976, last of the five, and its train/test gap of +0.0488 is the largest of the tuned models. |
| **Naive Bayes** | The search chose `var_smoothing=0.1`, nine decades above sklearn's default of `1e-9`, and that one change lifts accuracy from 0.2773 to 0.8161, F1 from 0.2978 to 0.5318 and MCC from 0.1375 to 0.4371. The default was so bad for a structural reason: `BounceRates` and `ExitRates` correlate at 0.902 and `ProductRelated` with `ProductRelated_Duration` at 0.860 on the de-duplicated data (computed by `core/facts.py`), so the independence assumption counts the same evidence twice. Smoothing widens every variance and soaks up much of that double counting, which is why it recovers so much. It only mitigates the violated assumption, though — MCC still ends at 0.4371 against the best model's 0.6143. |
| **Support Vector Machine** | The search chose `C=5.0`, `gamma=0.01` and `class_weight='balanced'`, which shifts it toward catching buyers: recall rises from 0.5183 to 0.7644 while precision falls from 0.7361 to 0.5670. Accuracy therefore drops from 0.8955 to 0.8718, but F1 improves from 0.6083 to 0.6511, MCC from 0.5614 to 0.5842 and AUC from 0.8733 to 0.9084. It is second on accuracy, AUC, F1 and MCC and third on precision and recall — never last on anything. Train and test accuracy differ by +0.0111, so it is not overfitting. |
| **Overall, Winner for your dataset?** | **Decision Tree, decided on MCC = 0.6143.** MCC is the deciding metric because an always-"No purchase" classifier reaches 84.4% accuracy with an MCC of 0, so MCC cannot be gamed by the imbalance; the tree also takes accuracy (0.9021), AUC (0.9263), precision (0.7085) and F1 (0.6703), five of the six metrics. **Caveat:** it wins over SVM by 0.6143 against 0.5842 MCC on one split with `random_state=42`, so a different split or a different search space could reorder the top three. Naive Bayes is reported at its tuned setting and still trails at 0.4371. |

### Why the metrics disagree

The three headline metrics answer different questions, so they are *allowed* to
disagree:

- **AUC** asks how well the model *ranks* sessions and ignores the 0.5 cut-off
  entirely.
- **Accuracy** asks how many rows were labelled correctly, and is dominated by
  whichever class is larger — hence the 84.4% trivial baseline.
- **MCC** asks how well prediction and truth move together, and is the only one
  of the three that stays honest under an 84/16 split.

On the tuned table all three happen to agree — Decision Tree leads AUC,
accuracy and MCC — but that is a property of this result, not a rule. The
untuned table above shows the disagreement plainly: Logistic Regression takes
AUC (0.8996) while Support Vector Machine takes MCC (0.5614).

---

## Reproducing these numbers

```bash
git clone https://github.com/Thrishasivasakthi/ML_Assignment_2
cd ML_Assignment_2
pip install -r requirements.txt
python train.py            # refits everything, rewrites metrics*.csv
python tests/check_pipeline.py   # 19 offline checks, no browser needed
python make_readme_tables.py --all   # reprints the three tables above
streamlit run app.py       # launches the application
```

`tests/check_pipeline.py` asserts that the saved artifacts reproduce every
cell of `metrics.csv` to within 1e-9, that the validation layer refuses the
malformed files it should refuse and accepts shuffled, surplus-column and
`0.0/1.0`-spelled ones, and - by parsing the source with `ast` rather than
searching for text - that no file outside `train.py` fits anything. It
imports only packages already pinned in `requirements.txt`, so it needs no
extra environment.

`random_state = 42` is set on the split, on the cross-validation splitter and
on every stochastic estimator, so a clean run reproduces `metrics.csv`,
`metrics_baseline.csv`, `best_params.json`, `train_scores.csv` and
`figures/` exactly. `train.py` refuses to run if scikit-learn does not match
the pin in `requirements.txt`, and refuses to overwrite `train_data.csv` or
`test_data.csv` if their content would change.

**The application does not need this step.** It loads the artifacts already
committed in `model/` and `artifacts/`.

## Application features (spec §Step 6)

| Requirement | Where |
|---|---|
| Dataset upload (CSV) | Sidebar, present on every page; also the Data Workspace |
| Model selection dropdown | Sidebar, present on every page |
| Display of evaluation metrics | Overview §01, Model Explorer §01, Error Analysis §05 |
| Confusion matrix / classification report | Overview §02, Error Analysis §01 and §03 |

All four are co-located on the **Overview** page as well as being expanded on
their own pages, so a single screenshot evidences every requirement.

**No retraining.** There is no `.fit()` call anywhere outside `train.py`.
The application only ever calls `load`, `transform`, `predict` and
`predict_proba`, so uploading a different CSV changes the rows being scored —
never the model doing the scoring.

## Metric definitions

| Metric | Meaning |
|---|---|
| Accuracy | Share of rows labelled correctly — misleading alone on an 84/16 split |
| AUC | Area under the ROC curve; ranking quality, independent of the 0.5 cut-off |
| Precision | Of sessions flagged as purchases, the share that really were |
| Recall | Of sessions that really were purchases, the share the model found |
| F1 | Harmonic mean of precision and recall |
| MCC | Matthews correlation coefficient, −1 to +1; cannot be inflated by imbalance |
