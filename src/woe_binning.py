"""
woe_binning.py
--------------
Weight of Evidence (WOE) binning and Information Value (IV) calculation.

WOE transforms raw feature values into a single numeric that encodes the
log-odds of being a bad applicant within each bin. This handles:
  - Non-linear relationships between features and default risk
  - Missing values (treated as a dedicated bin)
  - Outliers (handled at capping stage in preprocessing)

IV thresholds (industry standard):
  < 0.02  — Useless
  0.02–0.1 — Weak
  0.1–0.3  — Medium
  0.3–0.5  — Strong
  > 0.5   — Suspicious (possible data leakage)

Usage:
    from src.woe_binning import WOEBinner
    binner = WOEBinner(n_bins=10)
    binner.fit(X_train, y_train)
    X_train_woe = binner.transform(X_train)
    print(binner.iv_summary())
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class WOEBinner:
    """
    Fits WOE bins on training data and transforms any DataFrame.
    Handles continuous features via quantile binning and missing sentinel (-1).
    """

    def __init__(self, n_bins: int = 10, min_bin_pct: float = 0.05):
        """
        n_bins      : number of quantile bins for continuous features
        min_bin_pct : minimum fraction of records per bin (merges small bins)
        """
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.bin_stats_: Dict[str, pd.DataFrame] = {}
        self.iv_: Dict[str, float] = {}
        self.features_: List[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinner":
        """Compute WOE and IV for each feature in X."""
        self.features_ = list(X.columns)
        total_bad  = y.sum()
        total_good = (1 - y).sum()

        for feat in self.features_:
            stats = self._compute_woe(X[feat], y, total_bad, total_good)
            self.bin_stats_[feat] = stats
            self.iv_[feat] = stats["iv_bin"].sum()

        return self

    def _compute_woe(
        self,
        series: pd.Series,
        y: pd.Series,
        total_bad: float,
        total_good: float,
    ) -> pd.DataFrame:
        """Compute bin-level WOE and IV contribution for one feature."""
        df = pd.DataFrame({"x": series, "y": y})

        # Separate missing sentinel from real values
        missing_mask = df["x"] == -1
        df_valid   = df[~missing_mask].copy()
        df_missing = df[missing_mask].copy()

        # Quantile binning on valid values
        try:
            df_valid["bin"] = pd.qcut(
                df_valid["x"],
                q=self.n_bins,
                duplicates="drop",
            ).astype(str)
        except Exception:
            # Fallback: too few unique values — treat as categorical
            df_valid["bin"] = df_valid["x"].astype(str)

        # Add missing bin
        if len(df_missing) > 0:
            df_missing["bin"] = "Missing"
            df_all = pd.concat([df_valid, df_missing])
        else:
            df_all = df_valid

        # Aggregate by bin
        stats = (
            df_all.groupby("bin")["y"]
            .agg(total="count", bad="sum")
            .reset_index()
        )
        stats["good"] = stats["total"] - stats["bad"]

        # WOE = ln(dist_good / dist_bad)
        # Add small epsilon to avoid division by zero / log(0)
        eps = 0.5
        stats["dist_bad"]  = (stats["bad"]  + eps) / (total_bad  + eps * len(stats))
        stats["dist_good"] = (stats["good"] + eps) / (total_good + eps * len(stats))
        stats["woe"]       = np.log(stats["dist_good"] / stats["dist_bad"])
        stats["iv_bin"]    = (stats["dist_good"] - stats["dist_bad"]) * stats["woe"]

        # Add pct of total
        stats["pct_total"] = stats["total"] / len(df_all)
        stats["bad_rate"]  = stats["bad"] / stats["total"]

        return stats[["bin", "total", "pct_total", "bad", "good", "bad_rate", "woe", "iv_bin"]]

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace raw feature values with their WOE scores."""
        X_woe = X.copy().astype(float)

        for feat in self.features_:
            if feat not in self.bin_stats_:
                continue
            stats = self.bin_stats_[feat]
            X_woe[feat] = X[feat].apply(lambda v: self._lookup_woe(v, feat, stats))

        return X_woe

    def _lookup_woe(self, value, feat: str, stats: pd.DataFrame) -> float:
        """Map a single value to its WOE bin score."""
        if value == -1:
            row = stats[stats["bin"] == "Missing"]
            if len(row):
                return row["woe"].values[0]
            # No missing bin observed in training — return 0 (neutral)
            return 0.0

        # Find which interval bin this value falls in
        for _, row in stats[stats["bin"] != "Missing"].iterrows():
            bin_label = row["bin"]
            try:
                # Parse pandas interval string e.g. "(0.1, 0.5]"
                bin_label = bin_label.strip()
                left_open  = bin_label[0] == "("
                right_open = bin_label[-1] == ")"
                parts = bin_label[1:-1].split(", ")
                lo, hi = float(parts[0]), float(parts[1])

                in_left  = value > lo if left_open  else value >= lo
                in_right = value < hi if right_open else value <= hi

                if in_left and in_right:
                    return row["woe"]
            except Exception:
                # Categorical bin — exact match
                if str(value) == bin_label:
                    return row["woe"]

        # Value outside all observed bins — return nearest extreme bin WOE
        non_missing = stats[stats["bin"] != "Missing"]
        if len(non_missing):
            return non_missing.iloc[-1]["woe"]
        return 0.0

    # ── Summary ───────────────────────────────────────────────────────────────

    def iv_summary(self) -> pd.DataFrame:
        """Return features ranked by Information Value with interpretation."""
        rows = []
        for feat, iv in self.iv_.items():
            if iv < 0.02:
                strength = "Useless"
            elif iv < 0.10:
                strength = "Weak"
            elif iv < 0.30:
                strength = "Medium"
            elif iv < 0.50:
                strength = "Strong"
            else:
                strength = "Suspicious — check for leakage"

            rows.append({"feature": feat, "iv": round(iv, 4), "predictive_power": strength})

        return (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def bin_detail(self, feature: str) -> pd.DataFrame:
        """Return full bin-level WOE stats for a single feature."""
        if feature not in self.bin_stats_:
            raise ValueError(f"Feature '{feature}' not found. Run fit() first.")
        return self.bin_stats_[feature].copy()

    def selected_features(self, min_iv: float = 0.02, max_iv: float = 0.50) -> List[str]:
        """Return features within the acceptable IV range for modeling."""
        return [
            f for f, iv in self.iv_.items()
            if min_iv <= iv <= max_iv
        ]
