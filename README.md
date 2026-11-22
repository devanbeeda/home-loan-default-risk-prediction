# Home Loan Default Risk Prediction

An end-to-end machine learning project that predicts which home loan applicants are likely to default. It combines **7 data tables** into one customer-level dataset, compares multiple classification models on a highly imbalanced target, and selects a **recall-focused decision threshold** so that more defaulters are caught.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Key Results](#key-results)
- [Key Insights](#key-insights)
- [Challenges and Solutions](#challenges-and-solutions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Before Production Use](#before-production-use)
- [Author](#author)

---

## Problem Statement

Lending to customers who cannot repay is costly for banks, but rejecting good customers loses business. This project builds a model that estimates default risk from an applicant's profile and their credit history, so a risk team can review the highest-risk applications first.

**Objectives**
1. Prepare a complete data analysis report.
2. Identify factors associated with loan default and customer segments with different risk levels.
3. Compare multiple classification models and recommend one based on measured performance.
4. Document data challenges, the techniques used, and the reasons for choosing them.

**Target:** `TARGET` (1 = defaulter, 0 = non-defaulter)

## Dataset

The data comes as 7 CSV files describing each applicant and their history:

| File | Description |
|---|---|
| `application_train.csv` | Main table: one row per loan application, with the target |
| `bureau.csv` | Customer's previous credits from other institutions |
| `bureau_balance.csv` | Monthly balance history of those credits |
| `previous_application.csv` | Earlier loan applications with the bank |
| `POS_CASH_balance.csv` | Monthly point-of-sale and cash loan balances |
| `credit_card_balance.csv` | Monthly credit card balances |
| `installments_payments.csv` | Repayment history of previous credits |

| Property | Value |
|---|---|
| Main table | 307,511 records, 122 columns |
| After merging history | 307,511 records, **209 columns**, 0 duplicate customer IDs |
| Class balance | 91.93% non-defaulters, **8.07% defaulters** |
| Modelling sample | 100,000 records (reproducible), 80:20 stratified split |
| Test set | 20,000 records (18,385 non-defaulters, 1,615 defaulters) |

> The data files are not included in this repository because of their size. Place them in a `data` folder before running the notebook.

## Workflow

```
7 CSV files -> Chunked aggregation per customer -> Merge on SK_ID_CURR -> EDA
-> Preprocessing pipeline -> 5 baseline models -> Hyperparameter tuning
-> Repeated cross-validation -> Threshold selection -> Interpretation
```

1. **Data inspection:** checked schemas, missing values, and the target distribution.
2. **EDA:** analysed default rates by category and the correlation of numeric features with the target.
3. **Feature engineering:** converted day-based columns into age and tenure features and created affordability ratios, with division-by-zero protection.
4. **Historical aggregation:** read the 6 supporting tables in chunks, aggregated them to one row per customer, and left-joined them onto the application table, which preserved every application and avoided duplicate rows.
5. **Preprocessing pipeline:** median imputation for numeric features, most-frequent imputation for categorical features, one-hot encoding, and standard scaling. It was fitted only on training data, which prevents data leakage.
6. **Model comparison:** Dummy baseline, Logistic Regression, Decision Tree, Random Forest, and Extra Trees, with class weights because of the imbalance.
7. **Tuning and validation:** `RandomizedSearchCV` (3-fold stratified, PR-AUC scoring) on Logistic Regression and Extra Trees, followed by repeated stratified cross-validation.
8. **Threshold selection:** chose the decision threshold on a validation split using F2-score, then evaluated once on the untouched test set.

## Key Results

### Baseline model comparison

| Model | Key finding |
|---|---|
| Dummy Classifier | 91.9% accuracy, but detects **zero** defaulters |
| Random Forest | 89.7% accuracy, but only 16.6% recall on defaulters |
| **Logistic Regression** | **Best PR-AUC (0.231)** and ROC-AUC (0.757), with 67.7% recall |

Accuracy is misleading here, since a model that predicts "no default" for everyone scores 91.9%. Models were therefore selected on **PR-AUC and recall**.

### Final tuned Logistic Regression (untouched test set)

| Threshold | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| 0.50 (default) | 0.476 | 0.024 | 0.046 | n/a |
| **0.10 (validation-selected)** | **0.190** | **0.598** | **0.288** | **0.418** |

| Metric | Value |
|---|---|
| Test PR-AUC | 0.226 |
| Test ROC-AUC | 0.750 |
| Cross-validated PR-AUC | 0.204 (std 0.011) |
| Cross-validated ROC-AUC | 0.734 (std 0.010) |

At the default 0.50 threshold the model catches only 2.4% of defaulters. Lowering it to **0.10** catches **59.75%** of them, at the cost of more false alarms. The right threshold depends on the bank's cost of a missed default compared with the cost of a manual review.

## Key Insights

- **External risk scores matter most.** `EXT_SOURCE_3` (-0.179), `EXT_SOURCE_2` (-0.160), and `EXT_SOURCE_1` (-0.155) have the strongest correlations with default.
- **Repayment behaviour is informative.** Installment-payment aggregates (`INST_AMT_PAYMENT_MEAN`, `INST_AMT_INSTALMENT_MEAN`) were among the most influential model features.
- **Previous credit history helps.** Aggregates from previous applications added signal beyond the application form.
- **Some property data is very sparse.** Fields such as `COMMONAREA_AVG`, `COMMONAREA_MEDI`, and `COMMONAREA_MODE` are about 69.87% missing.

> These are associations, not proof of causation.

## Challenges and Solutions

| Challenge | Technique used |
|---|---|
| Large files | Chunked CSV processing to reduce memory use |
| Different table grains | Aggregate each table to customer level before joining, which prevents duplicated rows |
| Missing values | Median and most-frequent imputation, fitted on training data only |
| Categorical variables | One-hot encoding with unknown-category handling |
| Class imbalance | Stratified splits, class weights, PR-AUC, and recall |
| Data leakage | Train/test split before fitting any preprocessing |
| Threshold choice | Compared precision and recall across thresholds on validation data |
| Interpretability | Logistic Regression coefficients to see the direction of influence |

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modelling | Scikit-learn (pipelines, Logistic Regression, Random Forest, Extra Trees, RandomizedSearchCV) |
| Model saving | Joblib |
| Environment | Jupyter Notebook |

## Project Structure

```
.
├── Home_Loan_Default_Risk_Management.ipynb   # Full analysis and modelling
├── requirements.txt                          # Python dependencies
├── .gitignore
└── README.md
```

The notebook saves the final model and its threshold to `home_loan_default_risk_final_model.joblib` when it runs.

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/devanbeeda/home-loan-default-risk-prediction.git
cd home-loan-default-risk-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the 7 CSV files in a data folder, then open the notebook
jupyter notebook
```

In the notebook's configuration cell, set `PROJECT_PATH` to the folder that holds your CSV files. To run faster, keep the 100,000-record sample. To use all rows, set `SAMPLE_SIZE = None`, which needs more RAM.

## Limitations

- Results come from a **100,000-record sample**, not the full merged dataset, so performance may change on the full population.
- Precision at the 0.10 threshold is about 19%, so most flagged applicants are not defaulters. The model is built for screening, not for final decisions.
- The data has no clear application-date field, so an out-of-time validation was not possible.
- Only numeric history is aggregated, and categorical history from the supporting tables is not used.
- Feature relationships are associations, not causal effects.

## Before Production Use

This model should not be used as an automatic approval or rejection system. A bank would first need to:
1. Validate on a later, out-of-time cohort.
2. Calibrate predicted probabilities.
3. Review subgroup performance and fairness with compliance teams.
4. Agree the threshold based on the cost of missed defaults and review capacity.
5. Monitor drift and retrain regularly.
6. Keep a human review and appeal process.

## Author

**Devan Beeda**
[LinkedIn](https://www.linkedin.com/in/devanbeeda/) | [GitHub](https://github.com/devanbeeda) | devanbeeda@gmail.com
