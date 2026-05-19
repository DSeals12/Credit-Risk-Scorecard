"""
visualization.py
----------------
Reusable Plotly chart functions for the credit risk scorecard project.
All functions return go.Figure objects and optionally save to outputs/charts/.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE   = "#2E86AB"
RED    = "#C73E1D"
GREEN  = "#44BBA4"
GRAY   = "#AAAAAA"
AMBER  = "#F18F01"


# ── IV Bar Chart ──────────────────────────────────────────────────────────────

def plot_iv_ranking(iv_summary: pd.DataFrame, save: bool = True) -> go.Figure:
    """Horizontal bar chart of features ranked by Information Value."""
    df = iv_summary.sort_values("iv")
    colors = []
    for iv in df["iv"]:
        if iv < 0.02:   colors.append("#DDDDDD")
        elif iv < 0.10: colors.append(AMBER)
        elif iv < 0.30: colors.append(BLUE)
        elif iv < 0.50: colors.append(GREEN)
        else:           colors.append(RED)

    fig = go.Figure(go.Bar(
        x=df["iv"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=df["iv"].round(3),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>IV: %{x:.4f}<extra></extra>",
    ))

    fig.add_vline(x=0.02, line_dash="dash", line_color=GRAY, annotation_text="Weak (0.02)")
    fig.add_vline(x=0.10, line_dash="dash", line_color=BLUE, annotation_text="Medium (0.10)")
    fig.add_vline(x=0.30, line_dash="dash", line_color=GREEN, annotation_text="Strong (0.30)")

    fig.update_layout(
        title="Feature Information Value (IV) Ranking",
        xaxis_title="Information Value",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=420, margin=dict(l=180, r=60, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")

    if save:
        fig.write_image(str(OUTPUT_DIR / "iv_ranking.png"), scale=2)
    return fig


# ── WOE Bin Plot ──────────────────────────────────────────────────────────────

def plot_woe_bins(bin_detail: pd.DataFrame, feature_name: str, save: bool = True) -> go.Figure:
    """Bar + line combo: bar = % population per bin, line = WOE."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=bin_detail["bin"].astype(str),
        y=bin_detail["pct_total"] * 100,
        name="% Population",
        marker_color=BLUE,
        opacity=0.6,
        hovertemplate="Bin: %{x}<br>Population: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=bin_detail["bin"].astype(str),
        y=bin_detail["woe"],
        name="WOE",
        mode="lines+markers",
        line=dict(color=RED, width=2),
        marker=dict(size=8),
        hovertemplate="Bin: %{x}<br>WOE: %{y:.4f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title=f"WOE Binning — {feature_name}",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        height=380, legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(title_text="% Population", secondary_y=False, showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="WOE", secondary_y=True, showgrid=False)

    if save:
        fname = OUTPUT_DIR / f"woe_bins_{feature_name}.png"
        fig.write_image(str(fname), scale=2)
    return fig


# ── Score Distribution ────────────────────────────────────────────────────────

def plot_score_distribution(score_dist: pd.DataFrame, save: bool = True) -> go.Figure:
    """Grouped bar: good vs bad count per score bin, with bad rate overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=score_dist["bin_mid"], y=score_dist["good"],
        name="Good", marker_color=BLUE, opacity=0.7,
        hovertemplate="Score: %{x}<br>Good: %{y:,}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=score_dist["bin_mid"], y=score_dist["bad"],
        name="Bad", marker_color=RED, opacity=0.7,
        hovertemplate="Score: %{x}<br>Bad: %{y:,}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=score_dist["bin_mid"], y=score_dist["bad_rate"],
        name="Bad Rate %", mode="lines+markers",
        line=dict(color=AMBER, width=2), marker=dict(size=6),
        hovertemplate="Score: %{x}<br>Bad Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title="Score Distribution by Outcome",
        barmode="stack",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=420, margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_xaxes(title_text="Credit Score", showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Count", secondary_y=False, showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Bad Rate (%)", secondary_y=True, showgrid=False, ticksuffix="%")

    if save:
        fig.write_image(str(OUTPUT_DIR / "score_distribution.png"), scale=2)
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(roc_data: pd.DataFrame, auc: float, gini: float, save: bool = True) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(color=GRAY, dash="dash", width=1),
        name="Random (AUC=0.50)", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=roc_data["fpr"], y=roc_data["tpr"],
        mode="lines", line=dict(color=BLUE, width=2.5),
        name=f"Scorecard (AUC={auc}, Gini={gini})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        fill="tozeroy", fillcolor="rgba(46,134,171,0.08)",
    ))

    fig.update_layout(
        title=f"ROC Curve — AUC: {auc} | Gini: {gini}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=440, margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.45, y=0.10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", range=[0, 1])
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", range=[0, 1])

    if save:
        fig.write_image(str(OUTPUT_DIR / "roc_curve.png"), scale=2)
    return fig


# ── KS Curve ──────────────────────────────────────────────────────────────────

def plot_ks_curve(ks_data: pd.DataFrame, ks_stat: float, save: bool = True) -> go.Figure:
    """Cumulative good/bad distributions with KS gap highlighted."""
    ks_row = ks_data.loc[ks_data["ks"].idxmax()]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ks_data["pct_population"], y=ks_data["cum_bad"],
        mode="lines", name="Cumulative Bad %",
        line=dict(color=RED, width=2),
    ))

    fig.add_trace(go.Scatter(
        x=ks_data["pct_population"], y=ks_data["cum_good"],
        mode="lines", name="Cumulative Good %",
        line=dict(color=BLUE, width=2),
    ))

    # KS annotation
    fig.add_shape(type="line",
        x0=ks_row["pct_population"], x1=ks_row["pct_population"],
        y0=ks_row["cum_good"], y1=ks_row["cum_bad"],
        line=dict(color=AMBER, width=2, dash="dot"),
    )
    fig.add_annotation(
        x=ks_row["pct_population"],
        y=(ks_row["cum_good"] + ks_row["cum_bad"]) / 2,
        text=f"KS = {ks_stat:.3f}",
        showarrow=True, arrowhead=2,
        font=dict(color=AMBER, size=13, family="Arial"),
    )

    fig.update_layout(
        title=f"KS Curve — KS Statistic: {ks_stat}",
        xaxis_title="% of Population (sorted by score, riskiest first)",
        yaxis_title="Cumulative %",
        yaxis_ticksuffix="%",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=440, margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

    if save:
        fig.write_image(str(OUTPUT_DIR / "ks_curve.png"), scale=2)
    return fig


# ── Cut-off Strategy Chart ────────────────────────────────────────────────────

def plot_cutoff_analysis(cutoff_df: pd.DataFrame, save: bool = True) -> go.Figure:
    """Dual-axis: approval rate and bad rate vs. score cut-off."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=cutoff_df["cutoff"], y=cutoff_df["approval_rate"],
        mode="lines", name="Approval Rate %",
        line=dict(color=BLUE, width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=cutoff_df["cutoff"], y=cutoff_df["bad_rate_approved"],
        mode="lines", name="Bad Rate (Approved) %",
        line=dict(color=RED, width=2),
    ), secondary_y=True)

    fig.update_layout(
        title="Cut-off Strategy: Approval Rate vs. Bad Rate",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=420, margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(x=0.5, y=0.95),
    )
    fig.update_xaxes(title_text="Score Cut-off", showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Approval Rate (%)", secondary_y=False, ticksuffix="%")
    fig.update_yaxes(title_text="Bad Rate Among Approved (%)", secondary_y=True, ticksuffix="%")

    if save:
        fig.write_image(str(OUTPUT_DIR / "cutoff_analysis.png"), scale=2)
    return fig
