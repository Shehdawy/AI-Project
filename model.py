"""
model.py — Machine Learning Engineer Module

Role:
- Train a classifier to predict player skill level offline
- Expose predict_skill() for the dashboard
- Show predicted skill level in the terminal

Skill labels:
0 → Beginner
1 → Average
2 → Pro

Run this file directly:
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
    Convert raw rows into X and y for training.
    Each 5 clicks become one training sample.
    """

    window = 5  # Number of clicks per training sample

    if len(rows) < window:
        return None, None

    X, y = [], []

    for i in range(0, len(rows) - window + 1, window):
        chunk = rows[i:i + window]

        avg_rt = np.mean([float(r["reaction_time"]) for r in chunk])
        accuracy = np.mean([int(r["hit"]) for r in chunk])
        avg_diff = np.mean([float(r["difficulty_level"]) for r in chunk])

        # Rule-based labeling
        # You can tune these numbers depending on your game data.
        if avg_rt < 0.7 and accuracy >= 0.70:
            label = 2  # Pro
        elif avg_rt > 1.0 or accuracy < 0.60:
            label = 0  # Beginner
        else:
            label = 1  # Average

        X.append([avg_rt, accuracy, avg_diff])
        y.append(label)

    return np.array(X), np.array(y)


# ── Dummy Model ───────────────────────────────────────────────────────────────
def _save_dummy_model(constant: int = 0):
    """
    Save a fallback model when there is not enough useful data.
    """

    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="constant", constant=constant)
    dummy.fit([[0, 0, 0]], [constant])

    joblib.dump(dummy, MODEL_PATH)
    print(f"[Model] Dummy model saved to {MODEL_PATH}")


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    """
    Load data, train a Random Forest classifier, and save it to model.pkl.
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("[Model] Loading dataset...")
    rows = load_dataset()

    if len(rows) < 20:
        print("[Model] Not enough data to train. Need at least 20 rows.")
        print("[Model] Using dummy model instead.")
        _save_dummy_model()
        return

    X, y = build_features(rows)

    if X is None or y is None or len(X) < 4:
        print("[Model] Not enough training windows.")
        print("[Model] Using dummy model instead.")
        _save_dummy_model()
        return

    print(f"[Model] Training samples created: {len(X)}")

    # Show label distribution
    unique, counts = np.unique(y, return_counts=True)

    print("[Model] Label distribution:")
    for label, count in zip(unique, counts):
        print(f"  {SKILL_LABELS[int(label)]}: {count}")

    # If only one class exists, a normal classifier will always predict that class
    if len(unique) < 2:
        only_label = int(unique[0])
        print("[Model] Only one skill class found in dataset.")
        print(f"[Model] Model will always predict: {SKILL_LABELS[only_label]}")
        _save_dummy_model(constant=only_label)
        return

    # Stratify only if every class has at least 2 samples
    if np.min(counts) >= 2:
        stratify_value = y
    else:
        stratify_value = None
        print("[Model] Warning: Some classes have fewer than 2 samples.")
        print("[Model] Training without stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("[Model] Training complete. Test results:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Beginner", "Average", "Pro"],
        zero_division=0
    ))

    joblib.dump(model, MODEL_PATH)
    print(f"[Model] Model saved to {MODEL_PATH}")


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_skill(
    avg_reaction_time: float,
    accuracy: float,
    avg_difficulty: float
) -> str:
    """
    Predict the player's skill level.
    """

    if not os.path.exists(MODEL_PATH):
        return "No model found — run: python model.py"

    model = joblib.load(MODEL_PATH)

    X = np.array([[
        avg_reaction_time,
        accuracy,
        avg_difficulty
    ]])

    label_id = int(model.predict(X)[0])

    return SKILL_LABELS.get(label_id, "Unknown")


def predict_latest_session():
    """
    Predict skill level from the latest 20 clicks in dataset.csv
    and show the result in the terminal.
    """

    print("\n[Model] Predicting latest player skill...")

    rows = load_dataset()
    window = 5

    if len(rows) < window:
        print("[Model] Not enough data to predict. Need at least 20 rows.")
        return

    latest_chunk = rows[-window:]

    avg_rt = np.mean([float(r["reaction_time"]) for r in latest_chunk])
    accuracy = np.mean([int(r["hit"]) for r in latest_chunk])
    avg_diff = np.mean([float(r["difficulty_level"]) for r in latest_chunk])

    predicted_level = predict_skill(
        avg_reaction_time=avg_rt,
        accuracy=accuracy,
        avg_difficulty=avg_diff
    )

    print("[Model] Latest session stats:")
    print(f"  Average reaction time: {avg_rt:.3f}s")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Average difficulty: {avg_diff:.2f}")

    print(f"\n[Model] Predicted player level: {predicted_level}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
    predict_latest_session()
