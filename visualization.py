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

# Import numerical operations library
import numpy as np

# Import data manipulation library
import pandas as pd

# Import matplotlib base module
import matplotlib

# Use non-GUI backend (important for saving plots without display)
matplotlib.use("Agg")

# Import plotting interface
import matplotlib.pyplot as plt

# Import color utilities
import matplotlib.colors as mcolors

# Handle file paths easily
from pathlib import Path


# Path to dataset CSV file
CSV_PATH = Path(__file__).parent / "dataset.csv"

# Folder to store generated plots
PLOT_DIR = Path(__file__).parent / "plots"

# Create folder if it doesn't exist
PLOT_DIR.mkdir(exist_ok=True)


# ── Shared style ──────────────────────────────────────────────────────────────

# Background color for figures
BG    = "#0f0f19"

# Panel (axes) background
PANEL = "#15151f"

# Colors for different plot lines
LINE1 = "#e03c5a"
LINE2 = "#50dc78"
LINE3 = "#f5a623"

# Text colors
TEXT  = "#dde"
DIM   = "#888"


# Apply consistent styling to matplotlib axes
def _style_ax(ax):
    ax.set_facecolor(PANEL)  # Set background
    ax.tick_params(colors=DIM)  # Axis ticks color
    ax.xaxis.label.set_color(DIM)  # X label color
    ax.yaxis.label.set_color(DIM)  # Y label color
    ax.title.set_color(TEXT)  # Title color
    
    # Style borders
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


# Save plot to file and close figure
def _save(fig, name: str) -> str:
    path = PLOT_DIR / name  # Full path

    # Save figure with high quality
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")

    plt.close(fig)  # Free memory

    print(f"[Visualization] Saved → {path}")  # Debug print
    return str(path)  # Return file path


# ── 1. Reaction Time Trend ────────────────────────────────────────────────────

def plot_reaction_time_trend(session_id: str = None) -> str:
    """
    Line chart of reaction time per hit event,
    with rolling average overlay.
    """

    # Load data (hits only)
    df = _load(session_id, hits_only=True)

    # Check if data is valid
    if df is None or len(df) < 3:
        return ""

    # Extract reaction times
    rt = df["reaction_time"].values

    # X axis = event index
    x  = np.arange(len(rt))

    # Rolling window size (adaptive)
    window = max(3, len(rt) // 8)

    # Compute rolling average
    roll   = pd.Series(rt).rolling(window, min_periods=1).mean().values

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    # Plot raw reaction times
    ax.plot(x, rt, alpha=0.35, color=LINE1, linewidth=1, label="Raw RT")

    # Plot smoothed rolling average
    ax.plot(x, roll, color=LINE2, linewidth=2.5, label=f"Rolling avg (w={window})")

    # Fill area between raw and average
    ax.fill_between(x, rt, roll, alpha=0.08, color=LINE1)

    # Fit linear regression (degree 1 polynomial)
    slope = np.polyfit(x, rt, 1)[0]

    # Determine trend
    trend_txt = "📉 Improving" if slope < -0.002 else ("📈 Declining" if slope > 0.002 else "➡ Stable")

    # Titles and labels
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

    # Load all data (hits + misses)
    df = _load(session_id, hits_only=False)

    if df is None or len(df) < window:
        return ""

    # Compute rolling accuracy (%)
    acc = df["hit"].rolling(window, min_periods=1).mean().values * 100

    x   = np.arange(len(acc))

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    # Plot accuracy
    ax.plot(x, acc, color=LINE3, linewidth=2.5)

    # Fill under curve
    ax.fill_between(x, acc, alpha=0.15, color=LINE3)

    # Add reference thresholds
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
    """

    df = _load(session_id, hits_only=True)

    if df is None or len(df) < 5:
        return ""

    # Define grid resolution
    bins_x, bins_y = 10, 8

    # Remove invalid positions
    df = df[(df["target_x"] > 0) & (df["target_y"] > 0)]

    df = df.copy()

    # Assign each point to a bin
    df["bx"] = pd.cut(df["target_x"], bins=bins_x, labels=False)
    df["by"] = pd.cut(df["target_y"], bins=bins_y, labels=False)

    # Compute mean reaction time per grid cell
    grid = df.groupby(["by", "bx"])["reaction_time"].mean().unstack(fill_value=np.nan)

    # Fill missing values with column means
    grid = grid.T.fillna(grid.T.mean()).T

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    # Color map (red = slow, green = fast)
    cmap = plt.get_cmap("RdYlGn_r")

    # Draw heatmap
    im   = ax.imshow(grid.values, cmap=cmap, aspect="auto", interpolation="bilinear")

    # Add color bar
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
    Detect improvement using polynomial trend.
    """

    df = _load(session_id, hits_only=True)

    if df is None or len(df) < 5:
        return ""

    rt = df["reaction_time"].values
    x  = np.arange(len(rt))

    # Fit quadratic curve
    poly = np.polyfit(x, rt, 2)

    # Generate smooth curve
    trend_y = np.polyval(poly, x)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.scatter(x, rt, color=LINE1, alpha=0.5, s=20, label="Reaction times")
    ax.plot(x, trend_y, color=LINE3, linewidth=2.5, label="Trend")

    # Compare start vs end performance
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


# ── Utility Function ──────────────────────────────────────────────────────────

def _load(session_id, hits_only: bool) -> pd.DataFrame | None:
    # Check if dataset exists
    if not CSV_PATH.exists():
        print("[Visualization] dataset.csv not found.")
        return None

    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Filter by session if provided
    if session_id:
        df = df[df["session_id"] == session_id]

    # Filter only hits if needed
    if hits_only:
        df = df[df["hit"] == 1]

    # Return None if empty
    if len(df) == 0:
        return None

    return df.reset_index(drop=True)
