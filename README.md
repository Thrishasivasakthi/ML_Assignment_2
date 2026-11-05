# Session2Shop — Purchase Intent Classifier

**M.Tech (AIML) · Machine Learning · Assignment 2**

A Streamlit application that predicts whether an e-commerce browsing session
ends in a purchase, using five classifiers trained on the same data and
evaluated with the six metrics the assignment specifies.

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
| `components/` | Design system, sidebar, cards, charts, tables |
| `config.py` | Single source of truth: column lists, `prepare_features()`, seed |
| `train.py` | Training pipeline — the only file that fits anything |
| `model/*.joblib` | Five saved classifiers |
| `artifacts/preprocessor.joblib` | Fitted `ColumnTransformer` |
| `test_data.csv` | Held-out test set (2,441 × 18) the app accepts |
| `train_data.csv` | Training split, for reference |
| `metrics.csv` | The comparison table below, as written by `train.py` |
| `train_scores.csv` | Train vs test accuracy per model — the source of the `+0.1450` Decision Tree gap quoted below |
| `figures/` | Five confusion-matrix plots |
| `requirements.txt` | Every imported package, range-pinned |
| `.streamlit/config.toml` | Theme and toolbar configuration |

---

## d. Models used

All five models are implemented on the **same dataset**, trained on the **same
training split**, and evaluated on the **same held-out test set** of 2,441
rows. Every figure below is taken from `metrics.csv`, which `train.py` writes
directly.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8550 | 0.7435 | 0.5337 | 0.5812 | 0.5564 | 0.4705 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.2773 | 0.7498 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8733 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

*Best value per column: accuracy, F1 and MCC — **SVM**; AUC — **Logistic
Regression**; precision — **Logistic Regression**; recall — **Naive Bayes**.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The best **ranker** in the study: its AUC of **0.8996** is the highest, meaning it orders sessions by purchase likelihood better than any other model. It is also the most precise of the four well-behaved models (**0.7524**) — when it predicts a purchase it is right three times in four. The cost of that discipline is caution: recall of **0.4136** means it fails to flag 58.6% of actual purchasers, leaving revenue unclaimed. Second-best MCC (**0.5032**). Best understood as a strong ranker with a conservative threshold. |
| **Decision Tree** | The only model that **memorised its training data**: 1.0000 training accuracy against 0.8550 on held-out rows — a generalisation gap of **+0.1450**, recorded in `train_scores.csv`, from a fully grown, unpruned tree. On new data it compensates by casting a wider net (recall **0.5812**, the highest of the four well-behaved models) but precision falls to **0.5337**. It also records the **lowest AUC of all five (0.7435)**: a single tree emits coarse, blocky scores, so its ranking is the weakest even though its accuracy looks respectable. |
| **KNN** | Precision of **0.6622** is respectable, but recall of **0.3901** is the **lowest of all five models** — it misses roughly six of every ten real purchasers. The cause is structural: with purchasers only 15.6% of rows, a session's nearest neighbours in the scaled feature space are usually non-purchasers, so the majority vote defaults to "No purchase". Its AUC (**0.7958**) is better than Decision Tree's, so the ranking is adequate; it is the 0.5 cut-off that costs it, giving the weakest F1 (**0.4909**) of the four well-behaved models. |
| **Naive Bayes** | The deliberate outlier, and **not an implementation error**. Accuracy collapses to **0.2773** while AUC holds at **0.7498** — comparable to Decision Tree. That pairing is diagnostic: the underlying *ranking* is sound, the *threshold* is not. It labels almost every session as a purchase (recall **0.9791**), so precision falls to **0.1756** and 82% of its purchase flags are wrong. The cause is a **violated independence assumption**: `BounceRates`↔`ExitRates` correlate at **0.911**, and `ProductRelated`↔`ProductRelated_Duration` at **0.891**, so the model counts the same evidence twice and inflates its posterior toward "purchase". It was left at sklearn's defaults (`var_smoothing = 1e-9`) rather than tuned, because tuning it to look competitive would hide the mechanism: a smoothing sweep recovers accuracy from 0.277 → 0.816, which demonstrates *why* the assumption fails rather than papering over it. |
| **Support Vector Machine** | Takes **three of the six metrics at once** — accuracy (**0.8955**), F1 (**0.6083**) and MCC (**0.5614**). Those three matter together because none can be inflated by the class imbalance the way accuracy alone can. Its precision (**0.7361**) is nearly Logistic Regression's while its recall (**0.5183**) runs about ten points higher, which is exactly why its F1 leads. It is a slightly weaker *ranker* than Logistic Regression (AUC 0.8733 vs 0.8996) but a better *thresholded* classifier — first on three metrics, second on precision, and third on recall, where the two models ahead of it are the extreme-recall Naive Bayes and the wider-netting Decision Tree. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** It leads on **accuracy, F1 and MCC**, and **MCC is the deciding metric**: an always-"No purchase" classifier reaches 84.4% accuracy with an MCC of **0**, so MCC cannot be gamed by the imbalance — SVM's 0.5614 represents genuine agreement between prediction and truth. **Logistic Regression does take AUC (0.8996)**; if the task were ranking sessions for a marketing queue rather than making a binary call at 0.5, it would be the better choice. But across the six metrics this assignment specifies, SVM is first on three and no model beats it on more than two. Naive Bayes is reported at sklearn's defaults and is outside the contention by design. |

### Why the metrics disagree

The three headline metrics answer different questions, so they are *allowed* to
disagree:

- **AUC** asks how well the model *ranks* sessions and ignores the 0.5 cut-off
  entirely — hence Logistic Regression winning here while losing on MCC.
- **Accuracy** asks how many rows were labelled correctly, and is dominated by
  whichever class is larger — hence the 84.4% trivial baseline.
- **MCC** asks how well prediction and truth move together, and is the only one
  of the three that stays honest under an 84/16 split.

---

## Reproducing these numbers

```bash
git clone https://github.com/Thrishasivasakthi/ML_Assignment_2
cd ML_Assignment_2
pip install -r requirements.txt
python train.py            # refits everything, rewrites metrics.csv
streamlit run app.py       # launches the application
```

`random_state = 42` is set on the split and on every stochastic estimator, so
a clean run reproduces `metrics.csv` exactly.

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
