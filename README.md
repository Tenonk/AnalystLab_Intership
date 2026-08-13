# Telco Customer Churn — Data Preprocessing & Feature Engineering

**AnalystLab Africa — Machine Learning Internship Programme**
Week 2: Data Preprocessing & Feature Engineering for Machine Learning

## Overview

This project prepares the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) for machine learning by building a full preprocessing pipeline: data cleaning, feature engineering, encoding, scaling, outlier treatment, and feature selection. The output is a clean, model-ready CSV dataset for a future churn-prediction model.

## Business Questions

1. What data quality issues exist in the dataset?
2. Which features require preprocessing?
3. Which encoding techniques are most appropriate?
4. Which scaling method should be applied?
5. Are there significant outliers that need treatment?
6. Which features contribute most to solving the business problem?
7. Is the dataset ready for model training?

Full context and answers are in [`reports/Business_Understanding_Report.pdf`](reports/Business_Understanding_Report.pdf) and [`reports/Data_Preprocessing_Report.pdf`](reports/Data_Preprocessing_Report.pdf).

## Repository Structure

```
.
├── README.md
├── .gitignore
├── notebook/
│   ├── EDA (2).ipynb            # Week 1 exploratory data analysis
│   └── prepro_feat_engin.ipynb  # Week 2 preprocessing pipeline (Parts 1–6)
├── data/
│   ├── Telco-Customer-Churn.csv              # Original raw dataset
│   └── Telco-Customer-processed-Churn.csv    # Final ML-ready dataset
├── Image/
│   ├── boxplot_univariate.png   # Outlier check (tenure, MonthlyCharges, TotalCharges)
│   ├── hisplot_univariate.png   # Distribution histograms
│   ├── hisplot_bivariate.png    # Bivariate distributions
│   ├── bar_bivariate.png        # Categorical features vs. churn
│   ├── violin_bivariate.png     # Numerical distributions by churn
│   ├── univariate_cat.png       # Categorical feature distributions
│   ├── target_distribution.png  # Churn class balance
│   ├── correlation.png          # Correlation heatmap
│   ├── pairplot.png             # Pairwise feature relationships
│   └── tenure_groups.png        # Customer count & churn rate by tenure group
└── reports/
    ├── Business_Understanding_Report.pdf
    └── Data_Preprocessing_Report.pdf
```

## Pipeline Summary

| Step | What was done |
|---|---|
| **Inspection** | 7,043 rows × 21 columns, 0 duplicates, 3 numerical / 18 categorical features |
| **Cleaning** | Fixed `TotalCharges` mistyped as text (11 hidden blanks → `NaN`); encoded `Churn` target to 0/1 |
| **Feature engineering** | Added `tenure_group` (6 ordinal 12-month bins) and `is_new_customer` (binary flag, 0–12 months) |
| **Outlier treatment** | IQR method (1.5×IQR) on `tenure`, `MonthlyCharges`, `TotalCharges` → 7,043 → 7,032 rows |
| **Feature selection** | Correlation heatmap (dropped `TotalCharges`); Chi² + Cramér's V on categorical features (dropped `gender`, `PhoneService`, `MultipleLines`); Variance Threshold on encoded features (0 dropped) |
| **Encoding & scaling** | `StandardScaler` on `tenure`/`MonthlyCharges`; One-Hot Encoding on nominal categories; Ordinal Encoding on `tenure_group` |
| **Output** | `Telco-Customer-processed-Churn.csv` — 7,032 rows × 40 columns, fully numeric, ready for modeling |

## Key Findings

- Customers in their first 12 months churn at **~47%**, far above the overall average — the main driver behind the `is_new_customer` feature.
- `TotalCharges` is mathematically redundant with `tenure` × `MonthlyCharges`, so it was dropped to avoid multicollinearity.
- `MultipleLines` is statistically associated with churn (p = 0.0036) but the effect is negligible in practice (Cramér's V = 0.04), so it was still dropped for dimensionality reduction.

## How to Run

```bash
git clone <repo-url>
cd <repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook "notebook/prepro_feat_engin.ipynb"
```

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`

## Author
TENON KONE
Junior Machine Learning Engineer — AnalystLab Africa Internship Programme

---
*Part of the #AnalystLabAfrica Machine Learning Internship Programme.*
