"""
visualization.py — AI Analyst Module (Part 2)

Role:
- Create 3 matplotlib charts:
  1. Reaction time over time
  2. Accuracy over time
  3. Clustering scatter plot

Run:
    python visualization.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from data_collector import load_dataset
from clustering import run_clustering

# ── Colors ──────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

# ── Chart 1: Reaction Time ──────────────────────────────────────────────
def plot_reaction_time(rows: list[dict], ax: plt.Axes):
    if not rows:
        ax.set_title("Reaction Time Over Time")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    times = list(range(1, len(rows) + 1))
    rts = [r["reaction_time"] for r in rows]

    window = min(10, len(rts))
    rolling = np.convolve(rts, np.ones(window) / window, mode="valid")
    rolling_x = list(range(window, len(rts) + 1))

    ax.scatter(times, rts, color="#aaaacc", alpha=0.4, s=10, label="Raw")
    ax.plot(rolling_x, rolling, color="#3498db", linewidth=2, label="Rolling avg")

    ax.set_title("Reaction Time Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Click #")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# ── Chart 2: Accuracy ───────────────────────────────────────────────────
def plot_accuracy(rows: list[dict], ax: plt.Axes):
    if not rows:
        ax.set_title("Accuracy Over Time")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    hits = [r["hit"] for r in rows]

    window = min(15, len(hits))
    rolling_acc = np.convolve(hits, np.ones(window) / window, mode="valid") * 100
    rolling_x = list(range(window, len(hits) + 1))

    ax.plot(rolling_x, rolling_acc, color="#2ecc71", linewidth=2)

    ax.axhline(y=80, color="#e74c3c", linestyle="--", linewidth=1, label="80% target")
    ax.axhline(y=60, color="#f39c12", linestyle="--", linewidth=1, label="60% floor")

    ax.set_title("Accuracy Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Click #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# ── Chart 3: Clustering ─────────────────────────────────────────────────
def plot_clusters(rows: list[dict], ax: plt.Axes):
    result = run_clustering(rows)

    if not result:
        ax.set_title("Player Behavior Clusters")
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center")
        return

    features = result["features"]
    labels = result["labels"]
    centers = result["centers"]

    cluster_names = ["Struggling", "Improving", "Skilled"]

    for cluster_id in range(3):
        mask = labels == cluster_id

        ax.scatter(
            features[mask, 0],
            features[mask, 1],
            c=CLUSTER_COLORS[cluster_id],
            label=cluster_names[cluster_id],
            alpha=0.6,
            s=25,
        )

    # Centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        marker="X",
        s=120,
        zorder=5,
        label="Centers"
    )

    ax.set_title("Player Behavior Clusters", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reaction Time (s)")
    ax.set_ylabel("Hit (0=Miss, 1=Hit)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# ── Generate All Charts ─────────────────────────────────────────────────
def generate_charts(rows: list[dict], show: bool = True):
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Dark styling
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    plot_reaction_time(rows, ax1)
    plot_accuracy(rows, ax2)
    plot_clusters(rows, ax3)

    fig.suptitle(
        "AI Aim Trainer — Analytics Dashboard",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.02
    )

    if show:
        plt.tight_layout()
        plt.show()

    return fig

# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_dataset()

    if rows:
        generate_charts(rows, show=True)
    else:
        print("[Visualization] No data found. Play the game first!")
