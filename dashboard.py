"""
clustering.py — AI Analyst Module (Part 1)

Role:
- Use KMeans clustering to group player behavior patterns
- Detect performance trends over time

Run directly:
    python clustering.py
"""

import numpy as np
from sklearn.cluster import KMeans
from data_collector import load_dataset

# ── Config ───────────────────────────────────────────────────────────────
N_CLUSTERS = 3

CLUSTER_LABELS = {
    0: "Struggling",
    1: "Improving",
    2: "Skilled"
}

# ── Feature Preparation ──────────────────────────────────────────────────
def prepare_cluster_features(rows: list[dict]) -> tuple:
    """
    Convert raw rows into feature matrix:
    [reaction_time, hit]
    """

    features = []
    timestamps = []

    for r in rows:
        features.append([r["reaction_time"], float(r["hit"])])
        timestamps.append(r["timestamp"])

    return np.array(features), timestamps

# ── KMeans Clustering ────────────────────────────────────────────────────
def run_clustering(rows: list[dict]) -> dict:
    """
    Run clustering and return results
    """

    if len(rows) < N_CLUSTERS:
        print(f"[Clustering] Need at least {N_CLUSTERS} data points.")
        return {}

    features, timestamps = prepare_cluster_features(rows)

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    # Train KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)

    # Convert centers back to original scale
    centers = kmeans.cluster_centers_ * std + mean

    print(f"[Clustering] Clustered {len(rows)} points into {N_CLUSTERS} groups.")

    for i in range(N_CLUSTERS):
        count = np.sum(labels == i)
        cx = round(centers[i][0], 3)
        cy = round(centers[i][1], 3)

        print(f"  Cluster {i}: {count} points | avg RT={cx}s, avg hit={cy:.2f}")

    return {
        "features": features,
        "labels": labels,
        "centers": centers,
        "timestamps": timestamps,
    }

# ── Pattern Recognition ──────────────────────────────────────────────────
def analyze_patterns(rows: list[dict]) -> dict:
    """
    Compare early vs late performance
    """

    if len(rows) < 10:
        return {"trend": "Not enough data for pattern analysis."}

    mid = len(rows) // 2
    early = rows[:mid]
    late = rows[mid:]

    early_rt = np.mean([r["reaction_time"] for r in early])
    late_rt = np.mean([r["reaction_time"] for r in late])

    early_acc = np.mean([r["hit"] for r in early])
    late_acc = np.mean([r["hit"] for r in late])

    rt_change = round(late_rt - early_rt, 4)
    acc_change = round(late_acc - early_acc, 4)

    rt_trend = "faster ✓" if rt_change < 0 else "slower ✗"
    acc_trend = "better ✓" if acc_change > 0 else "worse ✗"

    patterns = {
        "early_avg_rt": round(early_rt, 3),
        "late_avg_rt": round(late_rt, 3),
        "rt_change": rt_change,
        "rt_trend": rt_trend,
        "early_accuracy": round(early_acc * 100, 1),
        "late_accuracy": round(late_acc * 100, 1),
        "acc_change": round(acc_change * 100, 1),
        "acc_trend": acc_trend,
    }

    print("\n[Pattern Recognition]")
    print(f"  Reaction time : {patterns['early_avg_rt']}s → {patterns['late_avg_rt']}s ({rt_trend})")
    print(f"  Accuracy      : {patterns['early_accuracy']}% → {patterns['late_accuracy']}% ({acc_trend})")

    return patterns

# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_dataset()

    if rows:
        run_clustering(rows)
        analyze_patterns(rows)
    else:
        print("[Clustering] No data found. Play the game first!")
