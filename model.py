"""
model.py — Machine Learning Engineer Module

Role:
- Train a classifier to predict player skill level (offline)
- Expose a simple predict() function for the dashboard

Skill labels:
0 → Beginner
1 → Average
2 → Pro

Run this file directly to train and save the model:
    python model.py
"""

import os
import numpy as np
import joblib
from data_collector import load_dataset

MODEL_PATH = "model.pkl"

# ── Label mapping ─────────────────────────────────────────────────────────────
SKILL_LABELS = {
    0: "Beginner",
    1: "Average",
    2: "Pro"
}

# ── Feature Engineering ───────────────────────────────────────────────────────
def build_features(rows: list[dict]):
    """
    Convert raw rows into (X, y) for training.

    We group rows into windows of 20 clicks and compute:
    - average reaction time
    - accuracy (hits / total clicks)
    - average difficulty level

    Labeling rules:
    - reaction_time < 0.5 AND accuracy > 0.80 → Pro
    - reaction_time > 1.2 OR  accuracy < 0.50 → Beginner
    - otherwise → Average
    """

    if len(rows) < 5:
        return None, None

    X, y = [], []
    window = 20  # one session = 20 clicks

    for i in range(0, len(rows) - window + 1, window):
        chunk = rows[i:i + window]

        avg_rt = np.mean([r["reaction_time"] for r in chunk])
        accuracy = np.mean([r["hit"] for r in chunk])
        avg_diff = np.mean([r["difficulty_level"] for r in chunk])

        # Rule-based labeling
        if avg_rt < 0.5 and accuracy > 0.80:
            label = 2  # Pro
        elif avg_rt > 1.2 or accuracy < 0.50:
            label = 0  # Beginner
        else:
            label = 1  # Average

        X.append([avg_rt, accuracy, avg_diff])
        y.append(label)

    return np.array(X), np.array(y)

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    """
    Load data, train a Random Forest classifier, and save to model.pkl
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("[Model] Loading dataset...")
    rows = load_dataset()

    if len(rows) < 20:
        print("[Model] Not enough data to train (need at least 20 rows).")
        print("[Model] Using a dummy model instead.")
        _save_dummy_model()
        return

    X, y = build_features(rows)

    if X is None or len(X) < 4:
        print("[Model] Not enough windows to train. Using dummy model.")
        _save_dummy_model()
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("[Model] Training complete. Test results:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=list(SKILL_LABELS.values())
    ))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[Model] Model saved to {MODEL_PATH}")

def _save_dummy_model():
    """
    Save a fallback model when there isn't enough data
    """

    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0 , 0, 0]], [0])

    joblib.dump(dummy, MODEL_PATH)
    print(f"[Model] Dummy model saved to {MODEL_PATH}")

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_skill(avg_reaction_time: float, accuracy: float, avg_difficulty: float) -> str:
    """
    Predict the player's skill level

    Args:
        avg_reaction_time: float (seconds)
        accuracy: float (0–1)
        avg_difficulty: float (1–3)

    Returns:
        "Beginner", "Average", or "Pro"
    """

    if not os.path.exists(MODEL_PATH):
        return "No model found — run: python model.py"

    model = joblib.load(MODEL_PATH)
    X = np.array([[avg_reaction_time, accuracy, avg_difficulty]])

    label_id = model.predict(X)[0]
    return SKILL_LABELS.get(int(label_id), "Unknown")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
