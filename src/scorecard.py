"""
scorecard.py
------------
Logistic regression scorecard with PDO (Points to Double Odds) scaling.

Industry convention:
  - Base score: 600 (score assigned at base odds)
  - Base odds:  50:1 (good:bad) — i.e., 1 bad per 50 goods
  - PDO:        20   — score increases by 20 for every doubling of odds

This produces scores roughly in the 300–850 range, consistent with
FICO-style consumer credit scores.

Usage:
    from src.scorecard import CreditScorecard
    sc = CreditScorecard()
    sc.fit(X_train_woe, y_train)
    scores = sc.score(X_test_woe)
    points_table = sc.points_table()
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ── PDO Scaling Parameters ────────────────────────────────────────────────────

BASE_SCORE = 600
BASE_ODDS  = 50       # good:bad ratio at base score
PDO        = 20       # points to double the odds

# Derived scaling constants
FACTOR  = PDO / np.log(2)
OFFSET  = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


class CreditScorecard:
    """
    Logistic regression scorecard with PDO points scaling.
    Expects WOE-transformed features as input.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.C        = C
        self.max_iter = max_iter
        self.seed     = seed
        self.model_   = None
        self.features_: list = []
        self.coef_df_: Optional[pd.DataFrame] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_woe: pd.DataFrame, y: pd.Series) -> "CreditScorecard":
        """
        Fit logistic regression on WOE-transformed features.
        No StandardScaler needed — WOE values are already on a comparable scale.
        """
        self.features_ = list(X_woe.columns)

        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.seed,
            solver="lbfgs",
        )
        self.model_.fit(X_woe, y)

        # Store coefficient table
        self.coef_df_ = pd.DataFrame({
            "feature":     self.features_,
            "coefficient": self.model_.coef_[0],
        }).sort_values("coefficient", ascending=False).reset_index(drop=True)

        print("Model fitted.")
        print(f"  Features: {len(self.features_)}")
        print(f"  Intercept: {self.model_.intercept_[0]:.4f}")

        return self

    # ── Score ─────────────────────────────────────────────────────────────────

    def predict_proba_bad(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Return probability of being a bad applicant (class=1)."""
        return self.model_.predict_proba(X_woe[self.features_])[:, 1]

    def score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """
        Convert predicted probability to credit score using PDO scaling.
        Higher score = lower risk (better applicant) — consistent with FICO convention.
        """
        p_bad  = self.predict_proba_bad(X_woe)
        # Clamp to avoid log(0)
        p_bad  = np.clip(p_bad, 1e-6, 1 - 1e-6)
        odds   = (1 - p_bad) / p_bad   # good:bad odds
        scores = OFFSET + FACTOR * np.log(odds)
        return np.round(scores).astype(int)

    # ── Points Table ──────────────────────────────────────────────────────────

    def points_table(self, binner=None) -> pd.DataFrame:
        """
        Build the scorecard points table — the deliverable that risk teams use.
        For each feature × bin, shows the WOE and corresponding points contribution.

        If binner (WOEBinner) is provided, includes bin-level bad rates.
        """
        rows = []
        intercept_points = OFFSET + FACTOR * self.model_.intercept_[0]

        for feat, coef in zip(self.features_, self.model_.coef_[0]):
            if binner and feat in binner.bin_stats_:
                bins = binner.bin_stats_[feat]
                for _, bin_row in bins.iterrows():
                    woe    = bin_row["woe"]
                    points = round(-(coef * woe * FACTOR))
                    rows.append({
                        "feature":   feat,
                        "bin":       bin_row["bin"],
                        "woe":       round(woe, 4),
                        "bad_rate":  round(bin_row["bad_rate"] * 100, 2),
                        "pct_total": round(bin_row["pct_total"] * 100, 2),
                        "points":    points,
                    })
            else:
                rows.append({
                    "feature":   feat,
                    "bin":       "N/A",
                    "woe":       None,
                    "bad_rate":  None,
                    "pct_total": None,
                    "points":    round(-(coef * FACTOR)),
                })

        df = pd.DataFrame(rows)
        df.attrs["base_points"] = round(intercept_points)
        return df

    # ── Cut-off Analysis ──────────────────────────────────────────────────────

    def cutoff_analysis(
        self,
        X_woe: pd.DataFrame,
        y: pd.Series,
        avg_balance: float = 2400,
        loss_given_default: float = 0.75,
    ) -> pd.DataFrame:
        """
        For each possible score cut-off, compute:
          - Approval rate
          - Bad rate among approved
          - Expected loss per approved account
          - Expected loss as % of balance
        """
        scores  = self.score(X_woe)
        p_bad   = self.predict_proba_bad(X_woe)

        results = []
        cutoffs = range(int(scores.min()), int(scores.max()) + 1, 5)

        for cutoff in cutoffs:
            approved_mask = scores >= cutoff
            n_approved    = approved_mask.sum()
            n_total       = len(scores)

            if n_approved == 0:
                continue

            approved_bad_rate = y[approved_mask].mean()
            approval_rate     = n_approved / n_total
            expected_loss     = approved_bad_rate * avg_balance * loss_given_default

            results.append({
                "cutoff":            cutoff,
                "approval_rate":     round(approval_rate * 100, 2),
                "bad_rate_approved": round(approved_bad_rate * 100, 2),
                "expected_loss":     round(expected_loss, 2),
                "expected_loss_pct": round(approved_bad_rate * loss_given_default * 100, 2),
                "n_approved":        int(n_approved),
            })

        return pd.DataFrame(results)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str = "outputs/model/scorecard.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Scorecard saved to {path}")

    @classmethod
    def load(cls, path: str = "outputs/model/scorecard.joblib") -> "CreditScorecard":
        return joblib.load(path)
