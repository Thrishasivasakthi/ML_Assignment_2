# ML Assignment 2 — M.Tech (AIML) · Machine Learning

**Marks:** 15 | **Deadline:** 08-Nov-2026, 23:59 PM (NO extensions, NO draft submissions)
**Status:** 🟦 Not started

---

## 🎯 What is actually being graded (reverse-engineered from the PDF)

| Bundle | Marks | What the examiner looks at |
|---|---|---|
| Model implementation + GitHub upload | 10 | 5 models × 6 metrics (5) · dataset description (1) · repo with all files (1) · observations table (3) |
| Streamlit app development | 4 | CSV upload (1) · model dropdown (1) · metrics display (1) · confusion matrix/report (1) |
| BITS Virtual Lab execution | 1 | ONE screenshot |
| **Total** | **15** | |

**Submission = ONE single PDF**, in this exact order:
1. GitHub repository link (code + `requirements.txt` + `README.md` + test data CSV)
2. Live Streamlit app link (must open an interactive frontend)
3. BITS Virtual Lab execution screenshot
4. The full README content (must also be pasted inside the PDF)

---

## Phase 0 — Decisions & setup (Day 1)

- [x] **0.1 Pick the dataset.** ✅ **SELECTED: Online Shoppers Purchasing Intention**
  - [x] ≥ **12 features** → **17 features** ✓
  - [x] ≥ **500 instances** → **12,330 rows** ✓
  - [x] Classification (binary **or** multi-class) → **binary** (`Revenue`) ✓
  - [x] Source: Kaggle or UCI only → **both** ✓
    - Kaggle: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset
    - UCI: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset (DOI: 10.24432/C5F88Q)
  - [x] ⚠️ **Not a "famous default"** → not Iris/Titanic/BreastCancer/Wine/MNIST ✓
  - [x] Has a real, describable **problem statement** ✓ → *"Will this browsing session end in a purchase?"*
- [x] **0.2 Check for a target column** → `Revenue` (bool), genuine supervised classification ✓
  - Class balance: **84.5% No (10,422) / 15.5% Yes (1,908)** — imbalanced ⚠️
- [x] **0.3 Download and place raw data**
  - File: `raw/online_shoppers_intention.csv` (1,072,063 bytes)
  - Verified: 12,330 × 18 (17 features + target), **no missing values**, **125 duplicate rows** (to handle in Phase 1)
  - Cite in README: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI ML Repository. CC BY 4.0
- [ ] **0.4 BITS Virtual Lab — strategy agreed: build locally, run there only at the end for the screenshot**
  - **Develop and test entirely on this machine** (Python 3.11.4 / sklearn 1.5.2 / Streamlit 1.46.1 — all verified working)
  - **Final step only:** copy the finished project to BITS Lab, run it once, capture the **ONE screenshot** [1 mark]
  - ⚠️ **Do not leave this to the last hour** — if lab access is broken, email **csislabsupport@wilp.bits-pilani.ac.in** and **raj.rohan@wilp.bits-pilani.ac.in**, subject: `NSP1 ML Assignment 2: BITS Lab issue`
  - ⚠️ The screenshot must show the assignment *executing* on the lab, not a login page
- [ ] **0.5 Create local git repo** and make an initial commit (commit history is reviewed for anti-plagiarism)

---

## Phase 1 — Data understanding & preprocessing (Day 1–2)

**✅ VERIFIED FACTS (already measured on `raw/online_shoppers_intention.csv` — do not re-derive):**
- Shape **12,330 × 18** → **17 features + 1 target (`Revenue`)**
- **No missing values** anywhere
- **125 duplicate rows — all of them `Revenue=False`** (10,422 → 10,297 negatives; positives unchanged at 1,908)
- **No constant columns** → nothing to drop
- Class balance **84.5% / 15.5%** → 84.4% / 15.6% after dedup
- **Zero negative values** in the 10 continuous columns
- Target `Revenue` and `Weekend` are **bool**; `Month`, `VisitorType` are **object**
- ⚠️ **4 columns are nominal codes stored as `int64`:** `OperatingSystems`(8), `Browser`(13), `Region`(9), `TrafficType`(20) — they are IDs, not quantities

- [x] **1.1** `df.head()`, `df.shape`, `df.info()`, `df.describe()` — shape/info verified above
- [x] **1.2** Check class balance → **84.5% / 15.5%, imbalanced** ⚠️ (this is why MCC is in the rubric)
- [x] **1.3** Missing values → **none, no imputation needed**
- [x] **1.4** Duplicates → **125, dropped** → 12,205 rows (≥500 ✅)
- [x] **1.5** Encode categoricals — ✅ **DONE via `ColumnTransformer` in `train.py`**
  - `StandardScaler` → the **10 continuous** features
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` → `Month`, `VisitorType`, `Weekend` **+ the 4 nominal int codes**
  - Result: **64 one-hot columns → 74 encoded features total** (≥12 ✅)
- [x] **1.6** **Split FIRST, then fit preprocessor on train only** → `test_size=0.2, stratify=y, random_state=42`
  - `X_train (9764, 17)` / `X_test (2441, 17)`, minority class **15.6% in both** ✅
- [x] **1.7** Fit `ColumnTransformer` on **X_train only**, transform both — **no leakage**
- [x] **1.8** Saved **`test_data.csv`** (2441 × 18, original column form + target)
  - ✅ Round-trip verified: saved preprocessor re-transforms the saved CSV → `(2441, 74)`, zero NaN/inf
- [x] **1.9** Sanity check → **12,205 rows ≥ 500** and **74 features ≥ 12** — **BOTH PASS**

**📁 Phase 1 files created:**
- `config.py` — shared single source of truth (column lists, constants, `prepare_features()`)
- `train.py` — load → clean → split → fit preprocessor → save
- `artifacts/preprocessor.joblib` — fitted ColumnTransformer
- `test_data.csv` — held-out split the Streamlit app will accept
- `train_data.csv` — training split (reproducibility)

---

## Phase 2 — Train & evaluate the 5 models (Day 2–3) ✅ **COMPLETE**

- [x] **2.1** All 5 estimators implemented in `build_models()`:
  - [x] `LogisticRegression(max_iter=1000, random_state=42)`
  - [x] `DecisionTreeClassifier(random_state=42)`
  - [x] `KNeighborsClassifier(n_neighbors=5)`
  - [x] **`GaussianNB()`** — MultinomialNB rejected: `StandardScaler` mean-centers → guarantees negatives → MultinomialNB invalid
  - [x] `SVC(kernel='rbf', probability=True, random_state=42)` ← `probability=True` present, AUC works
- [x] **2.2** All five fit on the **same** `X_train_t` / `y_train` ✅
- [x] **2.3** All **6 metrics** computed on the same `X_test_t` ✅ (binary → default `average='binary'` correctly scores the minority class)
- [x] **2.4** Results stored in **`metrics.csv`** → this IS the README comparison table
- [x] **2.5** Confusion matrices saved → `figures/confusion_matrix_*.png` (5 files)
- [x] **2.6** Sanity-checked — see findings below
- [x] **2.7** All 5 models saved → `model/*.joblib` ✅
- [x] **2.8** Preprocessing artifact saved → `artifacts/preprocessor.joblib` ✅
- [x] ~~2.9~~ **→ moved to Phase 7** (BITS Lab run is a final step only)

### 📊 FINAL RESULTS — the graded comparison table (README §d)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8869 | **0.8996** | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8550 | 0.7435 | 0.5337 | **0.5812** | 0.5564 | 0.4705 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.2773 | 0.7498 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | **0.8955** | 0.8733 | 0.7361 | 0.5183 | **0.6083** | **0.5614** |

**→ Winner: Support Vector Machine** (best Accuracy, F1 and MCC — 3 of 6 metrics, and MCC is the most trustworthy under 84/16 imbalance). *Logistic Regression takes AUC.*

### 🔍 Findings to write into the observations table (README §e, 3 marks)

1. **Naive Bayes collapse is real, not a bug** — its mean P(purchase) = **0.873** vs true base rate **0.156**. Root cause: features violate NB's independence assumption (`BounceRates↔ExitRates` = **0.911**, `ProductRelated↔ProductRelated_Duration` = **0.891**), so the model double-counts the same evidence. **Proven** by a `var_smoothing` sweep: accuracy recovers 0.277 → 0.816 as smoothing goes 1e-9 → 1e-1. Note its **AUC is still 0.7498** — the *ranking* is fine, only the threshold is broken.
2. **Decision Tree overfits** — train acc **1.0000** vs test **0.8550**, gap **+0.1450**. Unpruned tree memorised the training set.
3. **Accuracy is misleading here** — a naive "always predict No" classifier already scores **84.4%**. SVM's 89.55% is only +5.1pts over that, but its **MCC = 0.5614** vs the dummy's **0**. This is exactly why the rubric demands MCC.
4. **LogReg has the best AUC (0.8996) but loses on every thresholded metric** — it ranks sessions better, but its default 0.5 cut-off is wrong for an 84/16 split. Recall only 0.4136.
5. **Recall is the industry-relevant weakness** — LogReg and KNN miss ~60% of actual purchasers (0.41 / 0.39).

**Clean run verified:** `python train.py` → exit 0, **stderr 0 bytes** (joblib/loky `wmic` warning fixed — Windows 11 removed `wmic`).

---

## Phase 3 — GitHub repository (Day 3–4)

- [x] **3.1** Required structure created — local repo initialised on branch **`main`**:
  ```
  project-folder/
  ├── app.py              ← Phase 4 (pending)
  ├── requirements.txt    ✅
  ├── README.md           ← Phase 6 (pending)
  ├── test_data.csv       ✅  2,441 × 18
  ├── model/              ✅  5 × .joblib
  ├── config.py           ✅  shared config (own naming)
  ├── train.py            ✅  source code
  ├── artifacts/          ✅  preprocessor.joblib
  ├── figures/            ✅  5 confusion matrices
  └── raw/                ✅  source dataset
  ```
- [x] **3.2** `requirements.txt` written — **6 packages, all actually imported**:
  `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`
  - Each entry names the file that needs it; floor/ceiling ranges so Streamlit Cloud can always resolve
  - `seaborn` deliberately **omitted** — we use sklearn's `ConfusionMatrixDisplay`, so it would be an unused dependency
  - *(PDF: "Missing dependencies are the #1 cause of deployment failure")*
- [x] **3.3** **3 meaningful commits** made (history is reviewed):
  ```
  2fef9f9  chore: project scaffolding - requirements.txt and gitignore
  d89623e  feat: shared config and preprocessing pipeline
  f53b54c  feat: train and evaluate five classifiers
  ```
  *(further commits land with Phase 4 and Phase 6 — the history shows real incremental development)*
- [x] **3.4** Own naming throughout — `config.py` / `prepare_features()` / `NUMERIC_FEATURES` / `build_models()`, plus a distinctive repo name rather than a generic `ml-assignment-2`
- [x] **3.5** **PUSHED & VERIFIED** → https://github.com/Thrishasivasakthi/ML_Assignment_2
  - Branch `main`, local == remote at `f53b54c`, **20 tracked files**
  - Auth fixed by purging a stale `git-credential` entry so Git Credential Manager re-prompted
  - ⏳ Still to add: `app.py` (Phase 4) and `README.md` (Phase 6) — each gets its own commit
  - ✅ Repo is **public** (required by Streamlit Community Cloud free tier)

---

## Phase 4 — Streamlit app (Day 4–5) · 4 marks ✅ **COMPLETE**

- [x] **4.1** `st.file_uploader(...)` — **test data only**, CSV, in the sidebar ✅ **tested with a real file upload**
- [x] **4.2** `st.selectbox` across all 5 saved models ✅ (renders + bound to `models[selected_key]`)
- [x] **4.3** Loads **saved** model + preprocessor via `joblib.load`, cached with `st.cache_resource`
- [x] **4.4** Transform → predict → score against the CSV's target column ✅
- [x] **4.5** **NO retraining** — zero `.fit()` calls in `app.py` ✅
  - **Proven:** all **30 metrics** (5 models × 6) from the app reproduce `train.py`'s `metrics.csv` **exactly**
- [x] **4.6** All **6 evaluation metrics** displayed as custom cards ✅
- [x] **4.7** **Confusion matrix** (custom HTML grid) **+** classification report ✅
- [x] **4.8** Column validation → friendly `st.error()` listing exactly what's missing ✅
- [x] **4.9** **Customised UI** — hand-rolled CSS: gradient hero, metric cards, styled confusion matrix, pills, tooltips. *No stock template.*
- [x] **4.10** Tested locally — `streamlit run app.py`, **HTTP 200, no errors in log**

### Extra (beyond the spec, directly required by it)
- [x] **"Results of different models on your test data should be visible"** → **all-five-models comparison table**, max/min highlighted per column
- [x] "How to read these numbers" explainer for the six metrics
- [x] Bundled-data shortcut so an examiner can demo the app without downloading anything

### 🧪 Test evidence
| Test | Result |
|---|---|
| Syntax + import | ✅ |
| Headless launch, HTTP 200 | ✅ |
| CSV upload → 2,441 rows / 17 features / 382 purchasers | ✅ |
| Six metric cards match `metrics.csv` | ✅ |
| All 30 metrics match `train.py` exactly | ✅ **zero drift** |
| Confusion matrix + classification report render | ✅ |
| All-five-models table renders | ✅ |
| `train.py` after refactor | ✅ exit 0, **stderr 0 bytes** |
| Model dropdown switching | ⚠️ not automatable — **verify by hand** (2 clicks) |

---

## Phase 5 — Deployment on Streamlit Community Cloud (Day 5) · free

- [ ] **5.1** Go to https://streamlit.io/cloud → sign in with GitHub
- [ ] **5.2** New App → select repo → branch `main` → select `app.py` → Deploy
- [ ] **5.3** Verify the live link **opens an interactive frontend**
- [ ] **5.4** Upload `test_data.csv` in the app and confirm metrics render for **each of the 5 models**
- [ ] **5.5** Check logs for import errors and fix `requirements.txt` if anything fails

---

## Phase 6 — README.md (Day 6) · 7 of the 15 marks live here

Required structure (§Step 5 — follow exactly):

- [ ] **a.** Problem statement
- [ ] **b.** Dataset description `[1 mark]`
- [ ] **c.** GitHub Repository Link `[1 mark]`
- [ ] **d.** Models used → **Comparison Table** `[5 marks = 1 per model × 6 metrics]`

  | ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
  |---|---|---|---|---|---|---|
  | Logistic Regression | | | | | | |
  | Decision Tree | | | | | | |
  | KNN | | | | | | |
  | Naive Bayes | | | | | | |
  | Support Vector Machine | | | | | | |

- [ ] **e.** Observations table `[3 marks]`

  | ML Model Name | Observation about model performance |
  |---|---|
  | Logistic Regression | |
  | Decision Tree | |
  | KNN | |
  | Naive Bayes | |
  | Support Vector Machine | |
  | **Overall, Winner for your dataset?** | |

- [ ] **f.** Paste the **entire README content into the submitted PDF** (mandatory)

---

## Phase 7 — Final submission (Day 6–7, before 08-Nov 23:59)

- [ ] **7.0 ⭐ BITS VIRTUAL LAB RUN + SCREENSHOT [1 mark]** — *the agreed plan: everything above was built locally; this is the only lab step*
  - [ ] Copy the finished project folder onto BITS Virtual Lab
  - [ ] Install deps from `requirements.txt` there
  - [ ] Run `train.py` → then `streamlit run app.py` → exercise the upload + all 5 models
  - [ ] Capture the **ONE screenshot** showing the assignment **executing** on the lab (not a login/blank page)
- [ ] **7.1** GitHub repo link works
- [ ] **7.2** Streamlit app link opens correctly
- [ ] **7.3** App loads without errors
- [ ] **7.4** All 4 required app features implemented
- [ ] **7.5** README.md updated **and** included in the PDF
- [ ] **7.6** Assemble the single PDF in the required order:
  1. GitHub repository link
  2. Live Streamlit app link
  3. **BITS Lab screenshot**
  4. Full README content
- [ ] **7.7** **Press SUBMIT** (no draft submissions are accepted)

---

## ⚠️ Anti-plagiarism watchlist

- [ ] GitHub **commit history** will be reviewed → commit incrementally, not once
- [ ] **Identical repo structure & variable names** may be flagged → use your own naming
- [ ] **Copy-paste Streamlit templates** without customization may be penalized → build your own layout
- [ ] **Same dataset + same model + same outputs** across students → pick a distinctive dataset
- [ ] AI tools = **learning support only**, not direct copy-paste → understand every line you submit
