"""
clustering.py — AI Analyst / Unsupervised Learning Module
K-Means clustering to discover player behavioral patterns without labels.

Steps:
  1. Load dataset.csv and engineer per-session features
  2. Apply K-Means (k=3 clusters ≈ Beginner / Average / Pro)
  3. Use Elbow Method to find optimal k
  4. Visualise cluster separation via PCA
  5. Return cluster assignments and centroids for the dashboard
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — safe when Pygame is also running
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = Path(__file__).parent / "dataset.csv"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "avg_reaction", "std_reaction", "accuracy",
    "total_shots", "avg_target_size"
]

# Human-readable cluster labels assigned after fitting
CLUSTER_NAMES = {0: "Beginner", 1: "Average", 2: "Pro"}


# ── Feature Engineering (mirrors model.py) ────────────────────────────────────
def _build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("session_id").agg(
        total_shots     = ("hit", "count"),
        total_hits      = ("hit", "sum"),
        avg_reaction    = ("reaction_time", lambda x: x[x > 0].mean()),
        std_reaction    = ("reaction_time", lambda x: x[x > 0].std()),
        avg_target_size = ("target_size",   lambda x: x[x > 0].mean()),
    ).reset_index()
    agg["accuracy"] = agg["total_hits"] / agg["total_shots"].replace(0, np.nan)
    agg = agg.fillna(0)
    return agg


# ── Main Clustering Class ──────────────────────────────────────────────────────
class PlayerClustering:

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.scaler     = StandardScaler()
        self.kmeans     = None
        self.feat_df    = None
        self.X_scaled   = None
        self.labels_    = None

    # ── Fit ────────────────────────────────────────────────────────────────────
    def fit(self) -> bool:
        """
        Load data, build features, and fit K-Means.
        Returns True if successful, False otherwise.
        """
        if not CSV_PATH.exists():
            print("[Clustering] No dataset found.")
            return False

        df = pd.read_csv(CSV_PATH)
        if len(df) < 6:
            print("[Clustering] Not enough data for clustering (need ≥6 rows).")
            return False

        self.feat_df = _build_session_features(df)
        if len(self.feat_df) < self.n_clusters:
            print(f"[Clustering] Need at least {self.n_clusters} sessions.")
            return False

        # Use only features present in data
        available = [c for c in FEATURE_COLS if c in self.feat_df.columns]
        X = self.feat_df[available].values
        self.X_scaled = self.scaler.fit_transform(X)

        self.kmeans  = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = self.kmeans.fit_predict(self.X_scaled)
        self.feat_df["cluster"] = self.labels_

        # Map cluster indices to skill names based on centroid avg_reaction
        self._assign_cluster_names()

        sil = silhouette_score(self.X_scaled, self.labels_) if len(set(self.labels_)) > 1 else 0
        print(f"[Clustering] Fitted K-Means (k={self.n_clusters})  Silhouette={sil:.3f}")
        return True

    def _assign_cluster_names(self):
        """
        Sort clusters by centroid avg_reaction: lowest RT → Pro,
        middle → Average, highest → Beginner.
        """
        rt_idx = FEATURE_COLS.index("avg_reaction") if "avg_reaction" in FEATURE_COLS else 0
        centroids = self.kmeans.cluster_centers_
        order = np.argsort(centroids[:, rt_idx])  # ascending RT
        name_map = {order[0]: "Pro", order[1]: "Average", order[2]: "Beginner"}
        if self.n_clusters == 2:
            name_map = {order[0]: "Pro", order[1]: "Beginner"}
        self.feat_df["cluster_name"] = self.feat_df["cluster"].map(name_map)

    # ── Elbow Method ───────────────────────────────────────────────────────────
    def elbow_analysis(self, max_k: int = 8) -> dict:
        """Compute inertia for k=2..max_k and plot the elbow curve."""
        if self.X_scaled is None:
            self.fit()
        if self.X_scaled is None:
            return {}

        ks      = range(2, min(max_k + 1, len(self.X_scaled)))
        inertias = []
        sils     = []

        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(self.X_scaled)
            inertias.append(km.inertia_)
            if k < len(self.X_scaled):
                sils.append(silhouette_score(self.X_scaled, km.labels_))
            else:
                sils.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0f0f19")
        for ax in (ax1, ax2):
            ax.set_facecolor("#15151f")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        ax1.plot(list(ks), inertias, "o-", color="#e03c5a", linewidth=2, markersize=7)
        ax1.set_title("Elbow Method — Inertia", color="#dde", fontsize=13)
        ax1.set_xlabel("k (clusters)", color="#aaa")
        ax1.set_ylabel("Inertia", color="#aaa")

        ax2.plot(list(ks), sils, "o-", color="#50dc78", linewidth=2, markersize=7)
        ax2.set_title("Silhouette Score vs k", color="#dde", fontsize=13)
        ax2.set_xlabel("k (clusters)", color="#aaa")
        ax2.set_ylabel("Silhouette Score", color="#aaa")

        plt.tight_layout()
        path = PLOT_DIR / "elbow.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Clustering] Elbow plot saved → {path}")
        return {"ks": list(ks), "inertias": inertias, "silhouettes": sils}

    # ── PCA Scatter ────────────────────────────────────────────────────────────
    def plot_clusters(self) -> str:
        """
        Reduce to 2D with PCA and plot cluster scatter.
        Returns path to saved PNG.
        """
        if self.X_scaled is None or self.labels_ is None:
            if not self.fit():
                return ""

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.X_scaled)
        var = pca.explained_variance_ratio_

        PALETTE = {"Pro": "#e03c5a", "Average": "#f5a623", "Beginner": "#50dc78"}
        DEFAULT = "#7777cc"

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f0f19")
        ax.set_facecolor("#15151f")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        for name in ["Beginner", "Average", "Pro"]:
            mask = self.feat_df.get("cluster_name", pd.Series()) == name
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=PALETTE.get(name, DEFAULT),
                    label=name, s=90, alpha=0.85, edgecolors="white", linewidths=0.5
                )

        # Centroids in PCA space
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c="white", marker="X", s=200, zorder=5, label="Centroid")

        ax.set_title("Player Clusters (PCA)", color="#dde", fontsize=14)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", color="#aaa")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", color="#aaa")
        legend = ax.legend(facecolor="#222", labelcolor="#dde", edgecolor="#444")
        plt.tight_layout()

        path = PLOT_DIR / "clusters.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Clustering] Cluster plot saved → {path}")
        return str(path)

    # ── Stats per cluster ──────────────────────────────────────────────────────
    def cluster_summary(self) -> pd.DataFrame:
        """Return a DataFrame with mean stats per cluster."""
        if self.feat_df is None:
            self.fit()
        if self.feat_df is None:
            return pd.DataFrame()

        cols = [c for c in FEATURE_COLS if c in self.feat_df.columns] + ["cluster_name"]
        summary = (
            self.feat_df[cols]
            .groupby("cluster_name")
            .mean()
            .round(3)
        )
        return summary

    # ── Predict cluster for live session ──────────────────────────────────────
    def predict_cluster(self, session_stats: dict) -> str:
        """Assign a new session to the nearest cluster."""
        if self.kmeans is None:
            if not self.fit():
                return "Unknown"
        available = [c for c in FEATURE_COLS if c in session_stats]
        X = np.array([[session_stats.get(c, 0) for c in available]])
        X_s = self.scaler.transform(X)
        idx = self.kmeans.predict(X_s)[0]
        # Map index to name
        rt_idx = FEATURE_COLS.index("avg_reaction") if "avg_reaction" in FEATURE_COLS else 0
        order  = np.argsort(self.kmeans.cluster_centers_[:, rt_idx])
        names  = {order[0]: "Pro", order[1]: "Average", order[2]: "Beginner"}
        return names.get(idx, "Unknown")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pc = PlayerClustering()
    ok = pc.fit()
    if ok:
        print(pc.cluster_summary())
        pc.plot_clusters()
        pc.elbow_analysis()
    else:
        print("Play the game first to generate data!")
