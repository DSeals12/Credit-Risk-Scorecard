"""
preprocessing.py
----------------
Loads, cleans, and splits the Give Me Some Credit dataset.

Target variable: SeriousDlqin2yrs (1 = bad, 0 = good)
150,000 applicants, 10 features.

Usage:
    from src.preprocessing import load_and_clean, split_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Feature Metadata ──────────────────────────────────────────────────────────

FEATURE_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age":                                  "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd_30_59_cnt",
    "DebtRatio":                            "debt_ratio",
    "MonthlyIncome":                        "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":      "open_credit_lines",
    "NumberOfTimes90DaysLate":              "dpd_90_cnt",
    "NumberRealEstateLoansOrLines":         "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd_60_89_cnt",
    "NumberOfDependents":                   "dependents",
}

TARGET = "SeriousDlqin2yrs"
RENAMED_TARGET = "bad_flag"

# Features where values > threshold are capped (extreme outliers)
CAP_RULES = {
    "revolving_util":   1.0,     # utilization > 100% is data error
    "debt_ratio":       5.0,     # debt ratio > 500% is extreme
    "dpd_30_59_cnt":    10,
    "dpd_60_89_cnt":    10,
    "dpd_90_cnt":       10,
    "open_credit_lines": 40,
    "real_estate_loans": 10,
    "dependents":        8,
}

# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_and_clean(path: str = "data/raw/cs-training.csv") -> pd.DataFrame:
    """
    Load the Give Me Some Credit training dataset, clean, and rename columns.

    Steps:
    1. Drop the index column Kaggle includes
    2. Rename columns to snake_case
    3. Cap extreme outliers
    4. Impute missing values to -1 (will be handled as a separate WOE bin)
    5. Drop duplicates

    Returns a cleaned DataFrame ready for WOE binning.
    """
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download cs-training.csv from https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "and place it at data/raw/cs-training.csv"
        )

    df = pd.read_csv(fpath, index_col=0)
    print(f"Loaded: {len(df):,} rows, {df.shape[1]} columns")

    # Rename target and features
    df = df.rename(columns={TARGET: RENAMED_TARGET})
    df = df.rename(columns=FEATURE_NAMES)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df):,}")

    # Report missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values:")
        for col, cnt in missing.items():
            print(f"  {col}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Cap outliers
    for col, cap in CAP_RULES.items():
        if col in df.columns:
            n_capped = (df[col] > cap).sum()
            if n_capped > 0:
                df[col] = df[col].clip(upper=cap)

    # Impute missing to -1 (sentinel for "missing" WOE bin)
    # This preserves the missing-as-signal pattern rather than imputing to mean
    df = df.fillna(-1)

    print(f"\nFinal dataset: {len(df):,} rows")
    print(f"Bad rate: {df[RENAMED_TARGET].mean()*100:.2f}%")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    """
    Stratified 70/30 train/test split on bad_flag.
    Returns X_train, X_test, y_train, y_test.
    """
    features = [c for c in df.columns if c != RENAMED_TARGET]
    X = df[features]
    y = df[RENAMED_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"Train: {len(X_train):,} rows | Bad rate: {y_train.mean()*100:.2f}%")
    print(f"Test:  {len(X_test):,} rows  | Bad rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick EDA summary table for all features."""
    features = [c for c in df.columns if c != RENAMED_TARGET]
    rows = []
    for f in features:
        rows.append({
            "feature":   f,
            "dtype":     str(df[f].dtype),
            "missing":   (df[f] == -1).sum(),
            "pct_miss":  f"{(df[f] == -1).mean()*100:.1f}%",
            "mean":      df[f][df[f] != -1].mean(),
            "min":       df[f][df[f] != -1].min(),
            "max":       df[f][df[f] != -1].max(),
            "bad_rate_corr": df[f].corr(df[RENAMED_TARGET]),
        })
    return pd.DataFrame(rows).set_index("feature").round(4)
