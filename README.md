# Credit Risk Scorecard & Application Approval Simulator

**Author:** Denzel C. Seals | Senior Analyst - Coporate Development  
**Domain:** Consumer Credit Risk — Underwriting & Model Validation  
**Stack:** Python · Scikit-learn · Pandas · Plotly · Streamlit  
**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 150K applicants)

---

## Overview

This project builds an end-to-end credit risk scorecard using the industry-standard
Weight of Evidence (WOE) / Information Value (IV) framework, validated with KS statistic
and Gini coefficient, and deployed as an interactive Streamlit application.

The Streamlit app allows a credit analyst or risk manager to:
- Adjust the approval score cut-off via slider
- Instantly see projected **approval rate**, **bad rate among approved**, and **expected loss**
- Explore the **score distribution** by outcome and the **KS / ROC validation** curves

---

## Business Context

A scorecard translates raw applicant attributes into a single points-based score.
The score drives a binary approval decision at a chosen cut-off threshold. This project
replicates the core workflow a credit risk team would follow when building a new
application scorecard from historical performance data:

1. **Feature selection** via Information Value (IV) ranking
2. **WOE binning** to handle non-linearity and missing values
3. **Logistic regression** on WOE-transformed features
4. **Scorecard scaling** — convert log-odds to points (PDO methodology)
5. **Cut-off strategy** — balance approval rate vs. expected loss
6. **Model validation** — KS, Gini, ROC curve

---

## Project Structure

```
credit-risk-scorecard/
│
├── data/
│   ├── raw/                        # Drop cs-training.csv here (from Kaggle)
│   └── processed/                  # WOE-transformed features, train/test splits
│
├── src/
│   ├── preprocessing.py            # Cleaning, imputation, train/test split
│   ├── woe_binning.py              # WOE/IV calculation and binning pipeline
│   ├── scorecard.py                # Logistic regression + points scaling
│   ├── validation.py               # KS, Gini, ROC, score distribution
│   └── visualization.py            # Reusable Plotly chart functions
│
├── notebooks/
│   └── scorecard_development.ipynb # Full model development narrative
│
├── app/
│   └── streamlit_app.py            # Interactive approval simulator
│
├── outputs/
│   ├── charts/                     # Exported validation charts
│   └── model/                      # Saved model artifacts (joblib)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/DSeals12/credit-risk-scorecard.git
cd credit-risk-scorecard
pip install -r requirements.txt

# 2. Add data
# Download cs-training.csv from https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place it at: data/raw/cs-training.csv

# 3. Run the notebook (model development)
jupyter notebook notebooks/scorecard_development.ipynb

# 4. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Key Outputs

| Output | Description |
|---|---|
| WOE/IV Table | Feature ranking by predictive power |
| Score Distribution | Histogram of scores by good/bad outcome |
| KS Curve | Separation between good and bad score distributions |
| ROC Curve | AUC with Gini coefficient |
| Approval Simulator | Interactive cut-off → approval rate, bad rate, expected loss |

---

## Model Performance (on held-out test set)

| Metric | Value |
|---|---|
| AUC-ROC | ~0.86 |
| KS Statistic | ~0.38 |
| Gini Coefficient | ~0.72 |

*Actual values will vary slightly based on random seed and binning choices.*

---

## Methodology Notes

- **WOE binning**: 10-bin monotonic binning with manual override capability for non-monotonic features
- **IV threshold**: Features with IV < 0.02 excluded (weak predictors); IV > 0.50 flagged for review (potential data leakage)
- **Scorecard scaling**: Base score 600, PDO (Points to Double Odds) = 20 — industry standard
- **Missing values**: Imputed to a dedicated "Missing" WOE bin rather than mean imputation — preserves missing-as-signal
- **Train/test split**: 70/30 stratified by target variable

---

## Skills Demonstrated

- Weight of Evidence (WOE) and Information Value (IV) feature engineering
- Logistic regression scorecard development with points scaling (PDO methodology)
- Model validation: KS statistic, Gini coefficient, ROC/AUC
- Cut-off strategy analysis balancing approval rate vs. credit loss
- End-to-end ML pipeline from raw data to deployed Streamlit application

📊 [View rendered notebook with charts](https://nbviewer.org/github/DSeals12/Credit-Risk-Scorecard/blob/main/notebooks/scorecard_development.ipynb)
