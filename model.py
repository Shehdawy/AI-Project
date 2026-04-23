"""
model.py — Machine Learning Engineer Module
Supervised learning: classify player skill level as Beginner / Average / Pro.

Pipeline:
  1. Load dataset.csv
  2. Engineer features per session
  3. Label sessions by skill tier
  4. Train KNN, Decision Tree, Random Forest
  5. Select best model by cross-validation accuracy
  6. Expose predict() for real-time difficulty adaptation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

CSV_PATH   = Path(__file__).parent / "dataset.csv"
MODEL_PATH = Path(__file__).parent / "trained_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# ── Skill labeling thresholds ─────────────────────────────────────────────────
# Based on: average reaction time and accuracy
# (These can be tuned after collecting real data)
def label_skill(avg_rt: float, accuracy: float) -> str:
    """
    Rule-based labeling used when we have too few sessions for
    clustering-based labeling.
    """
    if avg_rt <= 0.35 and accuracy >= 0.80:
        return "Pro"
    elif avg_rt <= 0.55 and accuracy >= 0.60:
        return "Average"
    else:
        return "Beginner"


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw event-level CSV data into per-session feature vectors.
    Returns a DataFrame with one row per session_id.
    """
    hits_df = df[df["hit"] == 1].copy()

    agg = df.groupby("session_id").agg(
        total_shots     = ("hit", "count"),
        total_hits      = ("hit", "sum"),
        avg_reaction    = ("reaction_time", lambda x: x[x > 0].mean()),
        std_reaction    = ("reaction_time", lambda x: x[x > 0].std()),
        min_reaction    = ("reaction_time", lambda x: x[x > 0].min()),
        max_reaction    = ("reaction_time", lambda x: x[x > 0].max()),
        avg_target_size = ("target_size",   lambda x: x[x > 0].mean()),
    ).reset_index()

    agg["accuracy"] = agg["total_hits"] / agg["total_shots"].replace(0, np.nan)
    agg = agg.fillna(0)

    # Label
    agg["skill_label"] = agg.apply(
        lambda r: label_skill(r["avg_reaction"], r["accuracy"]), axis=1
    )

    return agg


FEATURE_COLS = [
    "total_shots", "total_hits", "avg_reaction", "std_reaction",
    "min_reaction", "max_reaction", "avg_target_size", "accuracy"
]


# ── SkillClassifier ────────────────────────────────────────────────────────────
class SkillClassifier:

    def __init__(self):
        self.scaler      = StandardScaler()
        self.le          = LabelEncoder()
        self.best_model  = None
        self.best_name   = None
        self.is_trained  = False

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, verbose: bool = True) -> dict:
        """
        Load dataset.csv, engineer features, train 3 models, pick the best.
        Returns a results dict with accuracy scores.
        """
        if not CSV_PATH.exists():
            print("[Model] No dataset.csv found. Please play the game first.")
            return {}

        df = pd.read_csv(CSV_PATH)
        if len(df) < 10:
            print("[Model] Not enough data to train. Play more rounds first.")
            return {}

        feat_df = build_features(df)
        if len(feat_df) < 3:
            print("[Model] Need at least 3 sessions to train. Keep playing!")
            return {}

        X = feat_df[FEATURE_COLS].values
        y = self.le.fit_transform(feat_df["skill_label"])

        X_scaled = self.scaler.fit_transform(X)

        # ── Models ─────────────────────────────────────────────────────────────
        models = {
            "KNN (k=5)":        KNeighborsClassifier(n_neighbors=5),
            "Decision Tree":    DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=42),
        }

        results = {}
        best_score = -1

        for name, clf in models.items():
            # Cross-validation — use min(5, min_class_count) to avoid StratifiedKFold errors
            min_class_count = min(np.bincount(y)) if len(set(y)) > 1 else 0
            n_folds = min(5, min_class_count)
            if n_folds >= 2:
                cv_scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")
                mean_cv = cv_scores.mean()
            else:
                mean_cv = 0.0

            # Train on all data for deployment
            clf.fit(X_scaled, y)
            train_acc = accuracy_score(y, clf.predict(X_scaled))
            results[name] = {"cv_mean": round(mean_cv, 4), "train_acc": round(train_acc, 4)}

            if mean_cv > best_score or (mean_cv == 0 and train_acc > best_score):
                best_score = max(mean_cv, train_acc)
                self.best_model = clf
                self.best_name  = name

        self.is_trained = True

        # Train/test split report (if enough data)
        if len(X_scaled) >= 6:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=None
            )
            self.best_model.fit(X_tr, y_tr)
            y_pred = self.best_model.predict(X_te)
            if verbose:
                self._print_report(y_te, y_pred, results)

        # Save to disk
        joblib.dump(self.best_model, MODEL_PATH)
        joblib.dump(self.scaler,     SCALER_PATH)
        if verbose:
            print(f"\n[Model] ✅  Best model: {self.best_name}")
            print(f"[Model]    Saved to {MODEL_PATH}")

        return results

    # ── Prediction ─────────────────────────────────────────────────────────────
    def predict(self, session_stats: dict) -> str:
        """
        Predict skill label from a dict of session stats.

        session_stats keys (all floats):
          total_shots, total_hits, avg_reaction, std_reaction,
          min_reaction, max_reaction, avg_target_size, accuracy
        """
        if not self.is_trained:
            self._try_load()

        if not self.is_trained:
            # Fallback to rule-based
            return label_skill(
                session_stats.get("avg_reaction", 1.0),
                session_stats.get("accuracy", 0.0)
            )

        X = np.array([[session_stats.get(c, 0) for c in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X)
        label_idx = self.best_model.predict(X_scaled)[0]
        return self.le.inverse_transform([label_idx])[0]

    def predict_proba(self, session_stats: dict) -> dict:
        """Return probability dict {label: probability}."""
        if not self.is_trained:
            self._try_load()
        if not self.is_trained or not hasattr(self.best_model, "predict_proba"):
            return {}

        X = np.array([[session_stats.get(c, 0) for c in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X)
        proba = self.best_model.predict_proba(X_scaled)[0]
        return {self.le.classes_[i]: round(p, 3) for i, p in enumerate(proba)}

    # ── Utils ───────────────────────────────────────────────────────────────────
    def _try_load(self):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self.best_model = joblib.load(MODEL_PATH)
                self.scaler     = joblib.load(SCALER_PATH)
                self.is_trained = True
                print("[Model] Loaded saved model from disk.")
            except Exception as e:
                print(f"[Model] Could not load saved model: {e}")

    def _print_report(self, y_true, y_pred, results):
        print("\n" + "═" * 50)
        print("  ML MODEL EVALUATION REPORT")
        print("═" * 50)
        for name, r in results.items():
            marker = "★" if name == self.best_name else " "
            print(f"  {marker} {name:<22} CV={r['cv_mean']:.3f}  Train={r['train_acc']:.3f}")
        print()
        present = sorted(set(y_true) | set(y_pred))
        labels  = [self.le.classes_[i] for i in present if i < len(self.le.classes_)]
        print(classification_report(y_true, y_pred, labels=present, target_names=labels, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("═" * 50)

    def get_confusion_matrix(self):
        """Return confusion matrix data for visualization."""
        if not CSV_PATH.exists():
            return None, None
        df  = pd.read_csv(CSV_PATH)
        if len(df) < 10:
            return None, None
        feat_df = build_features(df)
        if len(feat_df) < 3:
            return None, None
        X = feat_df[FEATURE_COLS].values
        y = self.le.fit_transform(feat_df["skill_label"])
        X_scaled = self.scaler.fit_transform(X)
        if not self.is_trained:
            self.train(verbose=False)
        y_pred = self.best_model.predict(X_scaled)
        return confusion_matrix(y, y_pred), list(self.le.classes_)

    def get_feature_importances(self):
        """Return feature importance dict (Random Forest only)."""
        if self.is_trained and hasattr(self.best_model, "feature_importances_"):
            return dict(zip(FEATURE_COLS, self.best_model.feature_importances_))
        return {}


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = SkillClassifier()
    results = clf.train(verbose=True)

    # Simulate a live prediction
    sample = {
        "total_shots": 40, "total_hits": 32,
        "avg_reaction": 0.42, "std_reaction": 0.08,
        "min_reaction": 0.25, "max_reaction": 0.70,
        "avg_target_size": 32, "accuracy": 0.80,
    }
    label = clf.predict(sample)
    print(f"\n[Model] Sample prediction: {label}")
