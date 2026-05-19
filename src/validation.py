"""
validation.py
-------------
Model validation metrics for the credit risk scorecard.

Metrics:
  - KS Statistic  : max separation between good/bad cumulative distributions
  - Gini Coefficient : 2 * AUC - 1
  - ROC / AUC
  - Score distribution by outcome
  - Population Stability Index (PSI) for monitoring

Usage:
    from src.validation import validate_scorecard, compute_psi
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_ks(y_true: pd.Series, p_bad: np.ndarray) -> Tuple[float, float]:
    """
    Compute KS statistic and the score at which it occurs.

    KS = max(cumulative bad rate - cumulative good rate) across all thresholds.
    Values: 0.2–0.3 acceptable, 0.3–0.4 good, >0.4 strong.

    Returns: (ks_statistic, ks_threshold)
    """
    df = pd.DataFrame({"p_bad": p_bad, "y": y_true}).sort_values("p_bad", ascending=False)
    df["cum_bad"]  = df["y"].cumsum() / df["y"].sum()
    df["cum_good"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    df["ks"]       = abs(df["cum_bad"] - df["cum_good"])

    ks_idx   = df["ks"].idxmax()
    ks_stat  = df.loc[ks_idx, "ks"]
    ks_thresh = df.loc[ks_idx, "p_bad"]

    return round(float(ks_stat), 4), round(float(ks_thresh), 4)


def compute_gini(y_true: pd.Series, p_bad: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Range: 0 (random) to 1 (perfect)."""
    auc  = roc_auc_score(y_true, p_bad)
    gini = 2 * auc - 1
    return round(gini, 4)


def compute_auc(y_true: pd.Series, p_bad: np.ndarray) -> float:
    return round(roc_auc_score(y_true, p_bad), 4)


def full_validation_report(
    y_true: pd.Series,
    p_bad: np.ndarray,
    scores: np.ndarray,
    dataset_label: str = "Test",
) -> Dict:
    """
    Run all validation metrics and return as a dictionary.
    """
    ks, ks_thresh = compute_ks(y_true, p_bad)
    gini = compute_gini(y_true, p_bad)
    auc  = compute_auc(y_true, p_bad)

    report = {
        "dataset":    dataset_label,
        "n":          len(y_true),
        "bad_rate":   round(y_true.mean() * 100, 2),
        "auc":        auc,
        "gini":       gini,
        "ks":         ks,
        "score_mean": round(float(scores.mean()), 1),
        "score_std":  round(float(scores.std()), 1),
        "score_min":  int(scores.min()),
        "score_max":  int(scores.max()),
    }

    print(f"\n{'='*45}")
    print(f"  Validation Report — {dataset_label}")
    print(f"{'='*45}")
    print(f"  Records:        {report['n']:,}")
    print(f"  Bad rate:       {report['bad_rate']}%")
    print(f"  AUC-ROC:        {report['auc']}")
    print(f"  Gini:           {report['gini']}")
    print(f"  KS Statistic:   {report['ks']}")
    print(f"  Score range:    {report['score_min']} – {report['score_max']}")
    print(f"  Score mean±std: {report['score_mean']} ± {report['score_std']}")

    return report


# ── ROC Curve Data ────────────────────────────────────────────────────────────

def roc_curve_data(y_true: pd.Series, p_bad: np.ndarray) -> pd.DataFrame:
    """Return FPR, TPR, thresholds as a DataFrame for plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, p_bad)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


# ── KS Curve Data ─────────────────────────────────────────────────────────────

def ks_curve_data(y_true: pd.Series, p_bad: np.ndarray) -> pd.DataFrame:
    """Return cumulative good/bad distributions sorted by score for KS plot."""
    df = pd.DataFrame({"p_bad": p_bad, "y": y_true}).sort_values("p_bad", ascending=False).reset_index(drop=True)
    df["pct_population"] = (df.index + 1) / len(df) * 100
    df["cum_bad"]        = df["y"].cumsum() / df["y"].sum() * 100
    df["cum_good"]       = (1 - df["y"]).cumsum() / (1 - df["y"]).sum() * 100
    df["ks"]             = abs(df["cum_bad"] - df["cum_good"])
    return df


# ── Score Distribution ────────────────────────────────────────────────────────

def score_distribution(scores: np.ndarray, y_true: pd.Series, n_bins: int = 20) -> pd.DataFrame:
    """
    Bin scores into n_bins and compute good/bad counts and bad rate per bin.
    Used for score distribution histogram in the Streamlit app.
    """
    df = pd.DataFrame({"score": scores, "y": y_true})
    df["bin"] = pd.cut(df["score"], bins=n_bins)
    df["bin_mid"] = df["bin"].apply(lambda x: round((x.left + x.right) / 2))

    agg = (
        df.groupby("bin_mid")["y"]
        .agg(total="count", bad="sum")
        .reset_index()
    )
    agg["good"]     = agg["total"] - agg["bad"]
    agg["bad_rate"] = (agg["bad"] / agg["total"] * 100).round(2)
    return agg


# ── Population Stability Index (PSI) ─────────────────────────────────────────

def compute_psi(
    scores_train: np.ndarray,
    scores_new: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """
    PSI measures score distribution shift between development and current population.

    PSI < 0.10  : No significant change
    PSI 0.10–0.25: Moderate shift — investigate
    PSI > 0.25  : Major shift — model may need recalibration

    Returns (psi_total, bin_level_detail)
    """
    bins = np.percentile(scores_train, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1
    bins[-1] += 1

    def bin_pcts(scores):
        counts = np.histogram(scores, bins=bins)[0]
        return counts / counts.sum()

    pct_train = bin_pcts(scores_train)
    pct_new   = bin_pcts(scores_new)

    eps = 1e-6
    psi_bins = (pct_new - pct_train) * np.log((pct_new + eps) / (pct_train + eps))
    psi_total = psi_bins.sum()

    detail = pd.DataFrame({
        "bin_low":   bins[:-1].round(0).astype(int),
        "bin_high":  bins[1:].round(0).astype(int),
        "pct_train": (pct_train * 100).round(2),
        "pct_new":   (pct_new   * 100).round(2),
        "psi_bin":   psi_bins.round(4),
    })

    return round(float(psi_total), 4), detail
