"""
visualization.py — AI Analyst / Visualization Module
Pattern recognition on player behavior + matplotlib graphs.

Generates:
  1. Reaction time trend (per click, rolling average)
  2. Accuracy over time (per 10-event windows)
  3. Reaction time heatmap (where on screen player is fast/slow)
  4. Skill improvement detection (linear regression on RT)
  5. Confusion matrix heatmap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

CSV_PATH = Path(__file__).parent / "dataset.csv"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BG    = "#0f0f19"
PANEL = "#15151f"
LINE1 = "#e03c5a"
LINE2 = "#50dc78"
LINE3 = "#f5a623"
TEXT  = "#dde"
DIM   = "#888"


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


def _save(fig, name: str) -> str:
    path = PLOT_DIR / name
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved → {path}")
    return str(path)


# ── 1. Reaction Time Trend ────────────────────────────────────────────────────
def plot_reaction_time_trend(session_id: str = None) -> str:
    """
    Line chart of reaction time per hit event,
    with rolling average overlay.
    """
    df = _load(session_id, hits_only=True)
    if df is None or len(df) < 3:
        return ""

    rt = df["reaction_time"].values
    x  = np.arange(len(rt))

    window = max(3, len(rt) // 8)
    roll   = pd.Series(rt).rolling(window, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.plot(x, rt,   alpha=0.35, color=LINE1, linewidth=1, label="Raw RT")
    ax.plot(x, roll, color=LINE2, linewidth=2.5, label=f"Rolling avg (w={window})")
    ax.fill_between(x, rt, roll, alpha=0.08, color=LINE1)

    # Detect trend direction
    slope = np.polyfit(x, rt, 1)[0]
    trend_txt = "📉 Improving" if slope < -0.002 else ("📈 Declining" if slope > 0.002 else "➡ Stable")
    ax.set_title(f"Reaction Time Trend  —  {trend_txt}", fontsize=13)
    ax.set_xlabel("Hit #")
    ax.set_ylabel("Reaction Time (s)")
    ax.legend(facecolor="#222", labelcolor=TEXT, edgecolor="#444")

    return _save(fig, "reaction_time_trend.png")


# ── 2. Accuracy Over Time ─────────────────────────────────────────────────────
def plot_accuracy_over_time(session_id: str = None, window: int = 10) -> str:
    """
    Rolling accuracy (hits / total shots) over a sliding window.
    """
    df = _load(session_id, hits_only=False)
    if df is None or len(df) < window:
        return ""

    acc = df["hit"].rolling(window, min_periods=1).mean().values * 100
    x   = np.arange(len(acc))

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.plot(x, acc, color=LINE3, linewidth=2.5)
    ax.fill_between(x, acc, alpha=0.15, color=LINE3)
    ax.axhline(y=80, color=LINE2, linestyle="--", linewidth=1, alpha=0.6, label="Pro threshold (80%)")
    ax.axhline(y=60, color=LINE1, linestyle="--", linewidth=1, alpha=0.6, label="Average threshold (60%)")

    ax.set_ylim(0, 105)
    ax.set_title("Accuracy Over Time", fontsize=13)
    ax.set_xlabel("Shot #")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(facecolor="#222", labelcolor=TEXT, edgecolor="#444")

    return _save(fig, "accuracy_over_time.png")


# ── 3. Reaction Time Heatmap ──────────────────────────────────────────────────
def plot_reaction_heatmap(session_id: str = None) -> str:
    """
    2D heatmap: average reaction time per screen region.
    Shows where on screen the player reacts fastest/slowest.
    """
    df = _load(session_id, hits_only=True)
    if df is None or len(df) < 5:
        return ""

    # Bin into 10×8 grid (800x600 → 80px × 75px cells)
    bins_x, bins_y = 10, 8
    df = df[(df["target_x"] > 0) & (df["target_y"] > 0)]
    df = df.copy()
    df["bx"] = pd.cut(df["target_x"], bins=bins_x, labels=False)
    df["by"] = pd.cut(df["target_y"], bins=bins_y, labels=False)

    grid = df.groupby(["by", "bx"])["reaction_time"].mean().unstack(fill_value=np.nan)
    # Fill NaN with column means
    grid = grid.T.fillna(grid.T.mean()).T

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    cmap = plt.get_cmap("RdYlGn_r")
    im   = ax.imshow(grid.values, cmap=cmap, aspect="auto", interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=DIM)
    cbar.set_label("Avg Reaction Time (s)", color=DIM)

    ax.set_title("Reaction Time Heatmap (screen regions)", fontsize=13)
    ax.set_xlabel("Screen X zone")
    ax.set_ylabel("Screen Y zone")

    return _save(fig, "reaction_heatmap.png")


# ── 4. Skill Improvement Detection ───────────────────────────────────────────
def plot_improvement(session_id: str = None) -> str:
    """
    Polynomial trend line on reaction times.
    Annotates whether the player is improving.
    """
    df = _load(session_id, hits_only=True)
    if df is None or len(df) < 5:
        return ""

    rt = df["reaction_time"].values
    x  = np.arange(len(rt))

    poly = np.polyfit(x, rt, 2)  # quadratic trend
    trend_y = np.polyval(poly, x)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.scatter(x, rt, color=LINE1, alpha=0.5, s=20, label="Reaction times")
    ax.plot(x, trend_y, color=LINE3, linewidth=2.5, label="Trend (poly deg-2)")

    slope_start = trend_y[0]
    slope_end   = trend_y[-1]
    if slope_end < slope_start * 0.95:
        note = "✅  Player is improving!"
    elif slope_end > slope_start * 1.05:
        note = "⚠️  Performance declining"
    else:
        note = "🔄  Performance is stable"

    ax.set_title(f"Performance Trend  —  {note}", fontsize=13)
    ax.set_xlabel("Hit #")
    ax.set_ylabel("Reaction Time (s)")
    ax.legend(facecolor="#222", labelcolor=TEXT, edgecolor="#444")

    return _save(fig, "improvement_trend.png")


# ── 5. Difficulty Distribution ────────────────────────────────────────────────
def plot_difficulty_distribution(session_id: str = None) -> str:
    """Bar chart of shots per difficulty level."""
    df = _load(session_id, hits_only=False)
    if df is None or "difficulty_level" not in df.columns:
        return ""

    counts = df["difficulty_level"].value_counts()
    colors = {"Beginner": LINE2, "Average": LINE3, "Pro": LINE1}
    bar_colors = [colors.get(d, "#aaa") for d in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="#333", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", color=TEXT, fontsize=11)

    ax.set_title("Shots per Difficulty Level", fontsize=13)
    ax.set_ylabel("Shot count")

    return _save(fig, "difficulty_dist.png")


# ── 6. Confusion Matrix Heatmap ───────────────────────────────────────────────
def plot_confusion_matrix(cm_array, class_names: list) -> str:
    """
    Render a styled confusion matrix.
    cm_array: numpy 2D array from sklearn.metrics.confusion_matrix
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    cmap = plt.get_cmap("Blues")
    im   = ax.imshow(cm_array, cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color=TEXT)
    ax.set_yticklabels(class_names, color=TEXT)
    ax.set_xlabel("Predicted", color=DIM)
    ax.set_ylabel("Actual", color=DIM)
    ax.set_title("Confusion Matrix", fontsize=13, color=TEXT)

    max_val = cm_array.max() if cm_array.max() > 0 else 1
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_array[i, j] > max_val * 0.5 else "#ccc"
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center", color=color, fontsize=14)

    return _save(fig, "confusion_matrix.png")


# ── Utility ───────────────────────────────────────────────────────────────────
def _load(session_id, hits_only: bool) -> pd.DataFrame | None:
    if not CSV_PATH.exists():
        print("[Visualization] dataset.csv not found.")
        return None
    df = pd.read_csv(CSV_PATH)
    if session_id:
        df = df[df["session_id"] == session_id]
    if hits_only:
        df = df[df["hit"] == 1]
    if len(df) == 0:
        return None
    return df.reset_index(drop=True)


def generate_all_plots(session_id: str = None) -> list[str]:
    """Generate every available plot and return list of file paths."""
    paths = [
        plot_reaction_time_trend(session_id),
        plot_accuracy_over_time(session_id),
        plot_reaction_heatmap(session_id),
        plot_improvement(session_id),
        plot_difficulty_distribution(session_id),
    ]
    return [p for p in paths if p]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    paths = generate_all_plots()
    print(f"\nGenerated {len(paths)} plots:")
    for p in paths:
        print(f"  {p}")
