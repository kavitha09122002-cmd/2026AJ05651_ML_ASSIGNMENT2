# Dry Bean Variety Classification - ML Assignment 2

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP  
**Student:** `KAVITHA S` | **BITS ID:** `2026AJ05651`  
**Live Streamlit app:** `<paste your https://....streamlit.app link here after deploying>`

---

## a. Problem statement

Dry beans are sold in many varieties that look alike to the human eye, and sorting them by hand is slow and
inconsistent. The goal is to build and compare five classical machine-learning classifiers that identify the
**variety of a dry bean** (a 7-class problem) from 16 geometric measurements taken from a photograph of the bean
(area, perimeter, axis lengths, roundness, shape factors, ...). The best-performing model is exposed through an
interactive Streamlit app where a user uploads a test CSV, picks a model, and sees the evaluation metrics and
confusion matrix without any retraining.

## b. Dataset description  `[1 mark]`

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository - [Dry Bean Dataset (id 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Citation | Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B |
| License | CC BY 4.0 |
| Task | Multi-class classification (7 bean varieties) |
| Features | 16 numeric: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1` ... `ShapeFactor4` |
| Target | `Class` - Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira |
| Minimum requirements | >= 12 features and >= 500 instances - both met |

Statistics computed from the data actually used for training (auto-filled by `train_models.py`):

<!-- DATASET_STATS_START -->
- **Instances used:** 13543 (raw file: 13611; 68 exact duplicate rows removed, 0 missing cells)
- **Features:** 16 numeric columns | **Target:** `Class` (7 classes)
- **Split:** 10834 train / 2709 test (stratified 80/20)
- **Strongly correlated feature pairs (correlation above 0.9):** 21

| Class | Samples | Share |
|---|---:|---:|
| Dermason | 3546 | 26.2% |
| Sira | 2636 | 19.5% |
| Seker | 2027 | 15.0% |
| Horoz | 1860 | 13.7% |
| Cali | 1630 | 12.0% |
| Barbunya | 1322 | 9.8% |
| Bombay | 522 | 3.9% |
<!-- DATASET_STATS_END -->

**Pre-processing:** exact duplicate rows were removed *before* splitting (so identical beans cannot appear in
both train and test); features are standardised with `StandardScaler` inside the scikit-learn `Pipeline` for the
scale-sensitive models (Logistic Regression, KNN, SVM); the target is label-encoded. The scaler lives inside each
saved model, so the app needs no separate preprocessing.

## c. GitHub repository link  `[1 mark]`

https://github.com/kavitha09122002-cmd/2026AJ05651_ML_ASSIGNMENT2

```
2026AJ05651_ML_ASSIGNMENT2/
|-- app.py               Streamlit app (upload CSV, choose model, metrics, confusion matrix)
|-- train_models.py      trains + tunes + evaluates the 5 models, saves everything
|-- data_loader.py       loads / cleans the UCI Dry Bean data
|-- evaluation.py        metric + plotting helpers shared by training and the app
|-- requirements.txt
|-- README.md
|-- test_data.csv        20 % hold-out split (features + Class) used for evaluation and the app
|-- model/               logistic_regression / decision_tree / knn / naive_bayes / svm .joblib,
|                        label_encoder.joblib, metadata.json
|-- results/             metrics_comparison.csv, confusion-matrix PNGs, EDA figures
`-- .streamlit/config.toml
```

## d. Models used  `[5 marks]`

All five models were trained on the **same** stratified 80 / 20 split (`random_state = 2026`). Hyper-parameters of
each model were chosen by 5-fold stratified cross-validated grid search on the training portion only (scoring:
macro-F1); the test split was never used for tuning. Metrics below are on the held-out test split
(`test_data.csv`). Precision, Recall and F1 are **macro-averaged** over the 7 classes; AUC is one-vs-rest macro
ROC-AUC computed from predicted probabilities (the SVM uses `probability=True`).

### Comparison table

<!-- RESULTS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.9251 | 0.9947 | 0.9371 | 0.9373 | 0.9371 | 0.9094 |
| **Decision Tree** | 0.9096 | 0.9813 | 0.9236 | 0.9217 | 0.9225 | 0.8906 |
| **KNN** | 0.9228 | 0.9882 | 0.9374 | 0.9364 | 0.9369 | 0.9067 |
| **Naive Bayes** | 0.8527 | 0.9828 | 0.8574 | 0.8559 | 0.8557 | 0.8222 |
| **Support Vector Machine** | 0.9280 | 0.9939 | 0.9407 | 0.9397 | 0.9402 | 0.9129 |
<!-- RESULTS_TABLE_END -->

### Observations on model performance  `[3 marks]`

<!-- OBSERVATIONS_START -->
| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy 92.51%, macro-F1 0.937, MCC 0.909 (rank 2 of 5 by MCC). It draws linear boundaries on standardised features, so it cannot bend around classes that overlap non-linearly (best C = 1). Train-vs-test accuracy gap is 0.0 points (it generalises well). Weakest class: Sira (recall 87.48%); most frequent mistake: Dermason predicted as Sira (46 cases). |
| **Decision Tree** | Accuracy 90.96%, macro-F1 0.922, MCC 0.891 (rank 4 of 5 by MCC). Axis-aligned splits need no scaling and stay interpretable; depth and leaf size were tuned by CV (depth = 12, min leaf = 10) to limit overfitting. Train-vs-test accuracy gap is 3.5 points (it generalises well). Weakest class: Sira (recall 85.01%); most frequent mistake: Sira predicted as Dermason (57 cases). |
| **KNN** | Accuracy 92.28%, macro-F1 0.937, MCC 0.907 (rank 3 of 5 by MCC). It classifies by neighbourhood vote on standardised features (best k = 9, weights = distance); it keeps the whole training set, so prediction cost grows with data size. Train accuracy is 100% by construction with distance weighting (each training point is its own nearest neighbour), so the 7.7-point train-vs-test gap overstates overfitting; CV macro-F1 (0.936) closely matches test macro-F1 (0.937). Weakest class: Sira (recall 85.96%); most frequent mistake: Sira predicted as Dermason (54 cases). |
| **Naive Bayes** | Accuracy 85.27%, macro-F1 0.856, MCC 0.822 (rank 5 of 5 by MCC). It assumes independent, Gaussian-distributed features, but 21 feature pairs have a correlation above 0.9 (the size measures such as Area, Perimeter and ConvexArea), which violates that assumption and typically costs accuracy. Train-vs-test accuracy gap is 0.2 points (it generalises well). Weakest class: Barbunya (recall 67.17%); most frequent mistake: Barbunya predicted as Cali (68 cases). |
| **Support Vector Machine** | Accuracy 92.80%, macro-F1 0.940, MCC 0.913 (rank 1 of 5 by MCC). The RBF kernel learns smooth non-linear boundaries once features are standardised (best C = 10, gamma = 0.1). Train-vs-test accuracy gap is 1.4 points (it generalises well). Weakest class: Sira (recall 87.10%); most frequent mistake: Sira predicted as Dermason (50 cases). |
| **Overall winner for this dataset** | **Winner: Support Vector Machine.** It has the highest MCC (0.913), the metric that is most robust to class imbalance, and also scores Accuracy 92.80%, AUC 0.9939 and macro-F1 0.940 on the held-out test set. Runner-up: Logistic Regression (MCC 0.909). |
<!-- OBSERVATIONS_END -->

## How to run

```bash
pip install -r requirements.txt

# 1. train all models (downloads the UCI data automatically; writes model/, test_data.csv, results/)
python train_models.py

# 2. launch the app
streamlit run app.py
```

If the machine cannot reach the UCI website, download `dry+bean+dataset.zip` manually, extract
`Dry_Bean_Dataset.xlsx` into a `data/` folder and run `train_models.py` again - it picks the local file first.

## Streamlit app features

- **CSV upload** of test data (or one click to use the bundled `test_data.csv`) with validation of columns / labels
- **Model selection dropdown** for the five saved classifiers
- **Evaluation metrics** - Accuracy, AUC, Precision, Recall, F1, MCC (macro or weighted averaging)
- **Confusion matrix** (counts or row-normalised) and a full **classification report**
- **Compare all models** tab with a highlighted table, bar chart and CSV download
- Row-level **predictions** tab with confidence and a misclassified-only filter
- Models are loaded from `model/` - nothing is retrained on the uploaded data

## Deployment

Deployed on Streamlit Community Cloud from the `main` branch with `app.py` as the entry point.
`requirements.txt` pins `scikit-learn` to the same version used for training so the saved models load correctly.
