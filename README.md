# Customer (Expresso) Churn Prediction

## Overview
This project aims to predict **customer churn** (whether a customer will leave the service) using machine learning techniques.  
Churn prediction is crucial for telecom and subscription-based businesses because it allows proactive retention campaigns, reduces revenue loss, and lowers acquisition costs.

We experiment with three families of models:
- **Logistic Regression** (linear baseline, sensitive to scaling)
- **Random Forest** (tree-based ensemble, robust to noise and imbalance)
- **Gradient Boosting** (XGBoost/LightGBM, powerful non-linear learners)

We also evaluate multiple **feature selection strategies** (Filter, Wrapper, Embedded) and compare their impact on predictive performance.

---

##  Team
- **Sara**  
- **Mai**  
- **Yining**

---

## Project Structure
```bash
AI7101-Project/
│
├── data/ # Datasets
│ ├── raw/ # Original input data (Train.csv, Test.csv, etc.)
│ ├── processed/ # Preprocessed splits (train/val/test)
│
├── notebooks/ # Notebooks for step-by-step workflow
│ ├── 01_data-exploration.ipynb
│ ├── 02_data-preprocessing.ipynb
│ ├── 03_data-evaluation.ipynb
│
├── src/churn/ # Source code
│ ├── experiments/ # Experiment scripts (FS + CV runs)
│ │ ├── logistic_regression_experiments.py
│ │ ├── random_forest_experiments.py
│ │ ├── xgboost_experiments.py
│ │
│ ├── train/ # Training scripts for baseline models
│ │ ├── train_logistic_regression.py
│ │ ├── train_random_forest.py
│ │ ├── train_xgboost.py
│ │
│ ├── eda.py # Exploratory data analysis helpers
│ ├── eval.py # Unified evaluation (ROC, PR, confusion, importances)
│ ├── utils.py # Shared utilities (splits, seeding, metrics)
│ ├── init.py
│
├── models/ # Saved models + metrics (joblib/json)
├── figs/ # Exported figures (ROC/PR curves, feature importances)
├── README.md # Documentation
├── requirements.txt # Dependencies
└── .gitattributes / .gitignore
```

---
## Dataset
- **Source**: Telecom churn dataset provided for AI7101 coursework.
- **Size**: ~2.1M records, 18 features.
- **Target**: `CHURN` (binary: 0 = stay, 1 = churn).
- **Important preprocessing decisions**:
  - Dropped features with >90% missing (`ZONE1`, `ZONE2`), or no variance (`MRG`).
  - Imputed **numeric** features with median, **categorical** with `"Unknown"`.
  - Encoded:
    - `REGION` → one-hot encoding.
    - `TOP_PACK` → frequency encoding.
    - `TENURE` → ordinal mapping.
  - Feature engineering:
    - Ratios such as `REVENUE / MONTANT`, `DATA_VOLUME / REGULARITY`.
    - Differences such as `REVENUE - MONTANT`.
    - Log-transform of skewed numeric variables.

---

## Exploratory Data Analysis (EDA)
- Visualized **target distribution** → Imbalanced dataset.
- **Missing values** analysis (bar plots).
- **Categorical vs churn** distributions (count plots).
- **Numeric distributions** (histograms + boxplots).
- **Correlations** (heatmap: `REVENUE` and `ARPU_SEGMENT` are collinear).
- Applied **log transforms** for skewed features.

---

## Modeling
### Algorithms
- **Logistic Regression** (baseline, requires scaling).
- **Random Forest** (tree ensemble).
- **XGBoost / Gradient Boosting** (powerful boosting model).

### Feature Selection Methods
- **Filter**: `SelectKBest` (ANOVA F-test).
- **Wrapper**: Recursive Feature Elimination (RFE).
- **Embedded**: SelectFromModel (based on feature importance or L1 regularization).

### Evaluation
- Proper **train/val/test split**:
  - Train 70%, Validation 15%, Test 15% (stratified).
- **Cross-validation** for robustness.
- Metrics reported:
  - ROC AUC
  - PR AUC
  - Precision, Recall, F1-score
- Threshold optimized on validation by **F1-score**.

---

## Results
- **Random Forest (tuned baseline)**: ROC AUC ≈ 0.93, PR AUC ≈ 0.70, F1 ≈ 0.69.
- Feature selection (filter, wrapper, embedded) gave **marginal improvements**.
- **Logistic Regression**: Competitive baseline (ROC AUC ≈ 0.92) but lower F1 due to imbalance.
- **XGBoost**: Best performance overall (ROC AUC ≈ 0.9315, PR AUC ≈ 0.706, F1 ≈ 0.68).

📊 Figures (saved in `/figs`):
- ROC and PR curves (Validation & Test).
- Confusion matrices.
- Feature importance plots.

---

## How to Run
### 1. Install dependencies
```bash
conda create -n churn python=3.11
conda activate churn
pip install -r requirements.txt
```
---

### 2. Preprocess data

```bash
jupyter notebook notebooks/02_data-preprocessing.ipynb
```
---

### 3. Train models

Random Forest:
```bash
python -m src.churn.random_forest_experiments --data_dir data/processed --models_dir models --cv 3 --jobs 2 --run_filter --run_wrapper --run_embedded
```

Logistic Regression:
```bash
python -m src.churn.logistic_regression_experiments --data_dir data/processed --models_dir models --cv 3 --jobs 2 --run_filter --run_wrapper --run_embedded
```

XGBoost:
```bash
python -m src.churn.xgboost_experiments --data_dir data/processed --models_dir models --cv 3 --jobs 2 --run_filter --run_wrapper --run_embedded
```

---

### 4. Evaluate models
Open:
```bash
notebooks/03_data-evaluation.ipynb
```
to generate classification reports, confusion matrices, ROC/PR curves, and feature importance plots.

---

### Contributors:
* Mai Chau
* Sara Alhajeri
* Yining Ma