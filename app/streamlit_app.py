"""
streamlit_app.py
----------------
Interactive Credit Risk Scorecard & Approval Simulator

Launch with: streamlit run app/streamlit_app.py

Requires the notebook to have been run first so model artifacts exist at:
  outputs/model/scorecard.joblib
  outputs/model/binner.joblib
  outputs/model/cutoff_analysis.parquet
  outputs/model/score_dist.parquet
  outputs/model/roc_data.parquet
  outputs/model/ks_data.parquet
  outputs/model/validation_report.parquet
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="📊",
    layout="wide",
)

# ── Load Artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    base = Path("outputs/model")
    sc      = joblib.load(base / "scorecard.joblib")
    binner  = joblib.load(base / "binner.joblib")
    cutoff  = pd.read_parquet(base / "cutoff_analysis.parquet")
    score_d = pd.read_parquet(base / "score_dist.parquet")
    roc_d   = pd.read_parquet(base / "roc_data.parquet")
    ks_d    = pd.read_parquet(base / "ks_data.parquet")
    val     = pd.read_parquet(base / "validation_report.parquet")
    return sc, binner, cutoff, score_d, roc_d, ks_d, val

try:
    sc, binner, cutoff_df, score_dist, roc_data, ks_data, val_report = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE  = "#2E86AB"
RED   = "#C73E1D"
GREEN = "#44BBA4"
AMBER = "#F18F01"
GRAY  = "#AAAAAA"

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Credit Risk Scorecard")
st.markdown(
    "**Dataset:** Give Me Some Credit (Kaggle) · "
    "**Model:** Logistic Regression + WOE Binning · "
    "**Author:** Denzel C. Seals"
)
st.markdown("---")

if not artifacts_loaded:
    st.error(
        f"⚠️ Model artifacts not found. Run the notebook first to generate model outputs.\n\n"
        f"Error: {load_error}"
    )
    st.info(
        "1. Open `notebooks/scorecard_development.ipynb`\n"
        "2. Run all cells\n"
        "3. Return here and refresh"
    )
    st.stop()

# ── Pull validation stats ─────────────────────────────────────────────────────
test_row = val_report[val_report["dataset"] == "Test"].iloc[0]

# ── Section 1: Model Performance Summary ─────────────────────────────────────

st.subheader("Model Performance — Test Set")

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC",       f"{test_row['auc']:.3f}")
col2.metric("Gini",          f"{test_row['gini']:.3f}")
col3.metric("KS Statistic",  f"{test_row['ks']:.3f}")
col4.metric("Bad Rate",      f"{test_row['bad_rate']:.1f}%")

st.markdown("---")

# ── Section 2: Approval Simulator ─────────────────────────────────────────────

st.subheader("⚙️ Approval Cut-off Simulator")
st.markdown(
    "Adjust the score cut-off below. Applicants scoring **at or above** the cut-off are approved. "
    "See how approval rate, bad rate, and expected loss respond in real time."
)

score_min = int(cutoff_df["cutoff"].min())
score_max = int(cutoff_df["cutoff"].max())
default_cutoff = int(cutoff_df.loc[(cutoff_df["approval_rate"] - 70).abs().idxmin(), "cutoff"])

cutoff_val = st.slider(
    "Score Cut-off",
    min_value=score_min,
    max_value=score_max,
    value=default_cutoff,
    step=5,
    help="Higher cut-off = stricter approval = lower approval rate, lower bad rate",
)

# Lookup metrics at selected cut-off
row = cutoff_df[cutoff_df["cutoff"] == cutoff_val]
if len(row) == 0:
    row = cutoff_df.iloc[(cutoff_df["cutoff"] - cutoff_val).abs().argsort()[:1]]
row = row.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Approval Rate",       f"{row['approval_rate']:.1f}%")
c2.metric("Bad Rate (Approved)", f"{row['bad_rate_approved']:.1f}%")
c3.metric("Exp. Loss / Account", f"${row['expected_loss']:,.0f}")
c4.metric("Accounts Approved",   f"{row['n_approved']:,}")

# Cut-off strategy chart
fig_cutoff = make_subplots(specs=[[{"secondary_y": True}]])

fig_cutoff.add_trace(go.Scatter(
    x=cutoff_df["cutoff"], y=cutoff_df["approval_rate"],
    mode="lines", name="Approval Rate %",
    line=dict(color=BLUE, width=2),
), secondary_y=False)

fig_cutoff.add_trace(go.Scatter(
    x=cutoff_df["cutoff"], y=cutoff_df["bad_rate_approved"],
    mode="lines", name="Bad Rate (Approved) %",
    line=dict(color=RED, width=2),
), secondary_y=True)

# Vertical line at selected cut-off
fig_cutoff.add_vline(
    x=cutoff_val,
    line_dash="dash", line_color=AMBER, line_width=2,
    annotation_text=f"Cut-off: {cutoff_val}",
    annotation_font_color=AMBER,
)

fig_cutoff.update_layout(
    height=360,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Arial", size=12),
    legend=dict(x=0.5, y=0.95),
    margin=dict(l=60, r=60, t=20, b=60),
)
fig_cutoff.update_xaxes(title_text="Score Cut-off", showgrid=True, gridcolor="#EEEEEE")
fig_cutoff.update_yaxes(title_text="Approval Rate (%)", secondary_y=False, ticksuffix="%")
fig_cutoff.update_yaxes(title_text="Bad Rate Among Approved (%)", secondary_y=True, ticksuffix="%")

st.plotly_chart(fig_cutoff, use_container_width=True)
st.markdown("---")

# ── Section 3: Score Distribution ─────────────────────────────────────────────

st.subheader("Score Distribution by Outcome")
st.markdown("Scores for good applicants (blue) skew right; bad applicants (red) skew left. Greater separation = stronger model.")

fig_dist = make_subplots(specs=[[{"secondary_y": True}]])

fig_dist.add_trace(go.Bar(
    x=score_dist["bin_mid"], y=score_dist["good"],
    name="Good", marker_color=BLUE, opacity=0.7,
), secondary_y=False)

fig_dist.add_trace(go.Bar(
    x=score_dist["bin_mid"], y=score_dist["bad"],
    name="Bad", marker_color=RED, opacity=0.7,
), secondary_y=False)

fig_dist.add_trace(go.Scatter(
    x=score_dist["bin_mid"], y=score_dist["bad_rate"],
    name="Bad Rate %", mode="lines+markers",
    line=dict(color=AMBER, width=2), marker=dict(size=5),
), secondary_y=True)

fig_dist.add_vline(
    x=cutoff_val,
    line_dash="dash", line_color=AMBER, line_width=2,
    annotation_text=f"Cut-off: {cutoff_val}",
    annotation_font_color=AMBER,
)

fig_dist.update_layout(
    barmode="stack", height=380,
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Arial", size=12),
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=60, r=60, t=20, b=60),
)
fig_dist.update_xaxes(title_text="Credit Score", showgrid=True, gridcolor="#EEEEEE")
fig_dist.update_yaxes(title_text="Count", secondary_y=False, showgrid=True, gridcolor="#EEEEEE")
fig_dist.update_yaxes(title_text="Bad Rate (%)", secondary_y=True, ticksuffix="%")

st.plotly_chart(fig_dist, use_container_width=True)
st.markdown("---")

# ── Section 4: Validation Curves (side by side) ───────────────────────────────

st.subheader("Model Validation")
left, right = st.columns(2)

# ROC
with left:
    st.markdown("**ROC Curve**")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(color=GRAY, dash="dash"), name="Random"))
    fig_roc.add_trace(go.Scatter(
        x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
        line=dict(color=BLUE, width=2.5),
        name=f"AUC={test_row['auc']} | Gini={test_row['gini']}",
        fill="tozeroy", fillcolor="rgba(46,134,171,0.08)",
    ))
    fig_roc.update_layout(
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(l=50,r=20,t=20,b=50),
        legend=dict(x=0.35, y=0.08),
    )
    fig_roc.update_xaxes(title_text="FPR", showgrid=True, gridcolor="#EEE", range=[0,1])
    fig_roc.update_yaxes(title_text="TPR", showgrid=True, gridcolor="#EEE", range=[0,1])
    st.plotly_chart(fig_roc, use_container_width=True)

# KS
with right:
    st.markdown("**KS Curve**")
    ks_row = ks_data.loc[ks_data["ks"].idxmax()]
    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(
        x=ks_data["pct_population"], y=ks_data["cum_bad"],
        mode="lines", name="Cumulative Bad %", line=dict(color=RED, width=2),
    ))
    fig_ks.add_trace(go.Scatter(
        x=ks_data["pct_population"], y=ks_data["cum_good"],
        mode="lines", name="Cumulative Good %", line=dict(color=BLUE, width=2),
    ))
    fig_ks.add_shape(type="line",
        x0=ks_row["pct_population"], x1=ks_row["pct_population"],
        y0=ks_row["cum_good"], y1=ks_row["cum_bad"],
        line=dict(color=AMBER, width=2, dash="dot"),
    )
    fig_ks.add_annotation(
        x=ks_row["pct_population"],
        y=(ks_row["cum_good"] + ks_row["cum_bad"]) / 2,
        text=f"KS = {test_row['ks']:.3f}",
        showarrow=True, arrowhead=2,
        font=dict(color=AMBER, size=12),
    )
    fig_ks.update_layout(
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(l=50,r=20,t=20,b=50),
    )
    fig_ks.update_xaxes(title_text="% of Population (riskiest first)", showgrid=True, gridcolor="#EEE")
    fig_ks.update_yaxes(title_text="Cumulative %", showgrid=True, gridcolor="#EEE", ticksuffix="%")
    st.plotly_chart(fig_ks, use_container_width=True)

st.markdown("---")

# ── Section 5: Scorecard Points Table ────────────────────────────────────────

st.subheader("Scorecard Points Table")
st.markdown(
    "Each feature × bin combination is assigned a point value. "
    "An applicant's total score is the sum of all bin points plus the base score."
)

try:
    points_table = sc.points_table(binner=binner)
    st.dataframe(
        points_table.style.format({
            "woe":       "{:.4f}",
            "bad_rate":  "{:.2f}%",
            "pct_total": "{:.1f}%",
            "points":    "{:+d}",
        }).background_gradient(subset=["points"], cmap="RdYlGn"),
        use_container_width=True,
        height=400,
    )
except Exception as e:
    st.warning(f"Points table unavailable: {e}")

st.markdown("---")
st.caption(
    "Built by Denzel C. Seals · "
    "Data: Give Me Some Credit (Kaggle) · "
    "Model: Logistic Regression + WOE/IV · "
    "github.com/DSeals12"
)
