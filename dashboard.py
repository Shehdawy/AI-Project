"""
dashboard.py — Real-Time GUI Dashboard (Streamlit)
Shows live player stats, ML prediction, clustering, and performance graphs.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import SkillClassifier, build_features, FEATURE_COLS
from clustering import PlayerClustering
from visualization import (
    plot_reaction_time_trend,
    plot_accuracy_over_time,
    plot_improvement,
    plot_difficulty_distribution,
    plot_confusion_matrix,
)

CSV_PATH = Path(__file__).parent / "dataset.csv"
PLOT_DIR = Path(__file__).parent / "plots"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Aim Trainer — Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0f0f19;
    color: #dde;
  }
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: bold;
    color: #e03c5a;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .skill-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    font-weight: bold;
    letter-spacing: 2px;
  }
  .skill-Beginner { background: #1a3a1a; color: #50dc78; border: 1px solid #50dc78; }
  .skill-Average  { background: #3a2a1a; color: #f5a623; border: 1px solid #f5a623; }
  .skill-Pro      { background: #3a1a1a; color: #e03c5a; border: 1px solid #e03c5a; }
  .section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
    margin-bottom: 14px;
  }
  [data-testid="stSidebar"] { background-color: #0d0d18; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3)
def load_data():
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    return df if len(df) > 0 else None


def compute_stats(df: pd.DataFrame, session_id=None) -> dict:
    if session_id and session_id != "All Sessions":
        df = df[df["session_id"] == session_id]
    total_shots = len(df)
    total_hits  = df["hit"].sum()
    accuracy    = total_hits / total_shots if total_shots > 0 else 0
    hits_df     = df[df["hit"] == 1]
    avg_rt      = hits_df["reaction_time"].mean() if len(hits_df) > 0 else 0
    std_rt      = hits_df["reaction_time"].std()  if len(hits_df) > 1 else 0
    min_rt      = hits_df["reaction_time"].min()  if len(hits_df) > 0 else 0
    max_rt      = hits_df["reaction_time"].max()  if len(hits_df) > 0 else 0
    avg_size    = hits_df["target_size"].mean()   if len(hits_df) > 0 else 0
    return {
        "total_shots":    int(total_shots),
        "total_hits":     int(total_hits),
        "accuracy":       round(accuracy, 4),
        "avg_reaction":   round(avg_rt, 4),
        "std_reaction":   round(std_rt, 4),
        "min_reaction":   round(min_rt, 4),
        "max_reaction":   round(max_rt, 4),
        "avg_target_size":round(avg_size, 2),
    }


def skill_color(label: str) -> str:
    return {"Beginner": "#50dc78", "Average": "#f5a623", "Pro": "#e03c5a"}.get(label, "#888")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI Aim Trainer")
    st.markdown("---")

    auto_refresh = st.toggle("Auto-refresh (3s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

    if st.button("🔄  Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Filter")
    df_all = load_data()
    sessions = ["All Sessions"]
    if df_all is not None and "session_id" in df_all.columns:
        sessions += sorted(df_all["session_id"].unique().tolist())
    selected_session = st.selectbox("Session", sessions)

    st.markdown("---")
    if st.button("🤖  Train ML Model", use_container_width=True):
        with st.spinner("Training models…"):
            clf = SkillClassifier()
            results = clf.train(verbose=False)
            if results:
                st.success("Model trained!")
                for name, r in results.items():
                    st.caption(f"{name}: CV={r['cv_mean']:.3f}")
            else:
                st.warning("Not enough data yet.")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "University AI Project  \n"
        "Supervised + Unsupervised Learning  \n"
        "Pattern Recognition  \n"
        "Real-Time Adaptive Difficulty"
    )


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# 🎯  AI Aim Trainer — Analytics Dashboard")

df = load_data()

if df is None:
    st.info("No gameplay data found yet.  \nRun `python main.py` to start the game and generate data.")
    st.stop()

stats = compute_stats(df, selected_session)

# ─── Row 1: Top metrics ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Player Stats</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, stats["total_hits"], "Total Hits")
metric_card(c2, f"{stats['accuracy']*100:.1f}%", "Accuracy")
metric_card(c3, f"{stats['avg_reaction']:.3f}s", "Avg Reaction")
metric_card(c4, f"{stats['min_reaction']:.3f}s", "Best Reaction")
metric_card(c5, stats["total_shots"], "Total Shots")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Row 2: ML Prediction + Clustering ───────────────────────────────────────
col_pred, col_cluster = st.columns([1, 1])

with col_pred:
    st.markdown('<div class="section-header">ML Skill Prediction (Supervised)</div>', unsafe_allow_html=True)
    clf = SkillClassifier()
    clf.train(verbose=False)
    skill = clf.predict(stats)
    color = skill_color(skill)
    st.markdown(
        f'<div style="text-align:center; margin:20px 0;">'
        f'<span class="skill-badge skill-{skill}">{skill}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    proba = clf.predict_proba(stats)
    if proba:
        for lbl, prob in proba.items():
            st.progress(int(prob * 100), text=f"{lbl}: {prob*100:.1f}%")

with col_cluster:
    st.markdown('<div class="section-header">Cluster Analysis (Unsupervised)</div>', unsafe_allow_html=True)
    pc = PlayerClustering()
    cluster = pc.predict_cluster(stats)
    st.markdown(
        f'<div style="text-align:center; margin:20px 0;">'
        f'<span class="skill-badge skill-{cluster}">{cluster}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    summary = pc.cluster_summary()
    if not summary.empty:
        st.dataframe(summary[["avg_reaction", "accuracy", "total_shots"]], use_container_width=True)

st.markdown("---")

# ─── Row 3: Performance Graphs ───────────────────────────────────────────────
st.markdown('<div class="section-header">Performance Graphs</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Reaction Time", "Accuracy", "Improvement", "Difficulty"])

session_arg = None if selected_session == "All Sessions" else selected_session

with tab1:
    p = plot_reaction_time_trend(session_arg)
    if p:
        st.image(p, use_column_width=True)

with tab2:
    p = plot_accuracy_over_time(session_arg)
    if p:
        st.image(p, use_column_width=True)

with tab3:
    p = plot_improvement(session_arg)
    if p:
        st.image(p, use_column_width=True)

with tab4:
    p = plot_difficulty_distribution(session_arg)
    if p:
        st.image(p, use_column_width=True)

st.markdown("---")

# ─── Row 4: Cluster scatter + Elbow ──────────────────────────────────────────
st.markdown('<div class="section-header">Cluster Visualisation</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    scatter_path = pc.plot_clusters()
    if scatter_path:
        st.image(scatter_path, caption="PCA Cluster Scatter", use_column_width=True)

with col_b:
    pc.elbow_analysis()
    elbow_path = PLOT_DIR / "elbow.png"
    if elbow_path.exists():
        st.image(str(elbow_path), caption="Elbow & Silhouette", use_column_width=True)

st.markdown("---")

# ─── Row 5: Raw data table ────────────────────────────────────────────────────
with st.expander("📄  Raw Dataset"):
    view_df = df if selected_session == "All Sessions" else df[df["session_id"] == selected_session]
    st.dataframe(view_df.tail(200), use_container_width=True)
    st.caption(f"Showing last 200 of {len(view_df)} rows for selected filter.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.75rem; margin-top:30px; font-family:monospace;">
  AI Aim Trainer  ·  University AI Engineering Project  ·  Powered by scikit-learn + Streamlit
</div>
""", unsafe_allow_html=True)
