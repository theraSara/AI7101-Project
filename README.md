# Customer (Expresso) Churn Prediction

##  Overview
This project aims to **predict customer churn** in the telecom domain using **machine learning techniques**.  
Customer churn — when customers stop using a service — leads to **major revenue loss** and **higher acquisition costs**.  
By predicting churners early, companies can take **proactive retention actions** and reduce costs.

We applied three models:
- **Logistic Regression** (baseline, interpretable)
- **Random Forest** (ensemble, robust)
- **Gradient Boosting** (state-of-the-art for tabular data)

---

##  Team
- **Sara**  
- **Mai**  
- **Yining**

---

##  Project Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Checked dataset distribution and imbalance  
   - Visualized missing values  
   - Explored relationships between features and churn  
   - Dropped redundant features (e.g., ARPU segment 100% correlated with revenue)

2. **Data Preprocessing**  
   - Dropped non-informative columns (`user_id`, `MRG`, `ZONE1`, `ZONE2`)  
   - Split data into **70% Train / 15% Validation / 15% Test** (stratified by churn)  
   - Imputed missing values  
     - Numeric → **median**  
     - Categorical → **"Unknown"**  
   - Encoding  
     - `TENURE` → ordinal mapping  
     - `REGION` → one-hot  
     - `TOP_PACK` → frequency encoding (to handle high cardinality)  
   - Feature Engineering  
     - Ratios: revenue efficiency, engagement intensity, loyalty indicators  
     - Log transform of skewed numeric variables

3. **Modeling**  
   - Logistic Regression (with scaling)  
   - Random Forest (tuned with GridSearchCV)  
   - Gradient Boosting (tuned with GridSearchCV)  
   - Random Forest pipelines with **feature selection methods**:  
     - Filter (SelectKBest)  
     - Wrapper (RFE)  
     - Embedded (SelectFromModel)  

4. **Evaluation**  
   - Metrics: **ROC AUC, PR AUC, F1-score** (threshold tuned on F1)  
   - Cross-validation: stratified k-fold (3–5 folds)  
   - Compared validation and test set performance
   
