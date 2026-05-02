"""
dashboard.py — Real-Time Analytics Dashboard

Role:
- Display game stats, ML prediction, and charts using Streamlit

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from data_collector import load_dataset
from model import predict_skill
from clustering import run_clustering, analyze_patterns
from visualization import plot_reaction_time, plot_accuracy, plot_clusters

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Aim Trainer Dashboard",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #e0e0f0; }

    [data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 14px;
    }

    h2, h3 { color: #7eb8f7; }

    .skill-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────────
def compute_stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "hits": 0,
            "misses": 0,
            "total": 0,
            "accuracy": 0.0,
            "avg_rt": 0.0,
            "avg_diff": 1.0,
        }

    hits = sum(r["hit"] for r in rows)
    total = len(rows)
    misses = total - hits

    avg_rt = round(np.mean([r["reaction_time"] for r in rows]), 3)
    avg_diff = round(np.mean([r["difficulty_level"] for r in rows]), 2)
    accuracy = round((hits / total) * 100, 1)

    return {
        "hits": hits,
        "misses": misses,
        "total": total,
        "accuracy": accuracy,
        "avg_rt": avg_rt,
        "avg_diff": avg_diff,
    }

def skill_color(skill: str) -> str:
    return {
        "Beginner": "#e74c3c",
        "Average": "#f39c12",
        "Pro": "#2ecc71"
    }.get(skill, "#aaaaaa")

def dark_fig(*axes_list):
    fig = axes_list[0].get_figure()
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes_list:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    return fig

# ── Layout ──────────────────────────────────────────────────────────────
st.title("🎯 AI Adaptive Aim Trainer — Dashboard")
st.caption("Reads from dataset.csv and model.pkl")

# Sidebar controls
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)

if auto_refresh:
    time.sleep(10)
    st.rerun()

if st.sidebar.button("Refresh Now"):
    st.rerun()

# ── Load Data ───────────────────────────────────────────────────────────
rows = load_dataset()
stats = compute_stats(rows)

# ── Section 1: Stats ────────────────────────────────────────────────────
st.markdown("## Session Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Hits", stats["hits"])
col2.metric("Misses", stats["misses"])
col3.metric("Accuracy", f"{stats['accuracy']}%")
col4.metric("Avg Reaction", f"{stats['avg_rt']} s")
col5.metric("Avg Difficulty", stats["avg_diff"])

st.divider()

# ── Section 2: ML Prediction ────────────────────────────────────────────
st.markdown("## Player Skill Prediction")

if not os.path.exists("model.pkl"):
    st.warning("No model found. Run model.py first.")
else:
    skill = predict_skill(
        avg_reaction_time=stats["avg_rt"],
        accuracy=stats["accuracy"] / 100,
        avg_difficulty=stats["avg_diff"],
    )

    color = skill_color(skill)

    st.markdown(
        f'<span class="skill-badge" style="background:{color}30; border:2px solid {color}; color:{color};">{skill}</span>',
        unsafe_allow_html=True
    )

st.divider()

# ── Section 3: Patterns ─────────────────────────────────────────────────
st.markdown("## Performance Patterns")

if len(rows) >= 10:
    patterns = analyze_patterns(rows)

    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "Reaction Time",
            f"{patterns['late_avg_rt']} s",
            delta=f"{patterns['rt_change']:+.3f} s",
            delta_color="inverse"
        )

    with p2:
        st.metric(
            "Accuracy",
            f"{patterns['late_accuracy']}%",
            delta=f"{patterns['acc_change']:+.1f}%"
        )
else:
    st.info("Play more to unlock analysis (≥ 10 clicks)")

st.divider()

# ── Section 4: Charts ───────────────────────────────────────────────────
st.markdown("## Charts")

c1, c2, c3 = st.columns(3)

with c1:
    fig1, ax1 = plt.subplots()
    plot_reaction_time(rows, ax1)
    dark_fig(ax1)
    st.pyplot(fig1)
    plt.close(fig1)

with c2:
    fig2, ax2 = plt.subplots()
    plot_accuracy(rows, ax2)
    dark_fig(ax2)
    st.pyplot(fig2)
    plt.close(fig2)

with c3:
    fig3, ax3 = plt.subplots()
    plot_clusters(rows, ax3)
    dark_fig(ax3)
    st.pyplot(fig3)
    plt.close(fig3)

# ── Footer ──────────────────────────────────────────────────────────────
st.divider()
st.caption(f"{len(rows)} rows loaded from dataset.csv")
