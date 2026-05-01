"""
data_collector.py — Data Engineer Module

Role:
- Collect and store gameplay data to CSV
- Provide a loader for ML training
"""

import csv
import os
import time

# ── Dataset Path ─────────────────────────────────────────────────────────
DATASET_PATH = "dataset.csv"

# ── CSV Headers ──────────────────────────────────────────────────────────
CSV_HEADERS = [
    "reaction_time",
    "target_x",
    "target_y",
    "target_size",
    "hit",
    "timestamp",
    "difficulty_level",
]

# ── Data Collector Class ─────────────────────────────────────────────────
class DataCollector:
    """Handles writing gameplay events to dataset.csv."""

    def __init__(self):
        # Create file with headers if it doesn't exist
        if not os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
            print(f"[DataCollector] Created {DATASET_PATH}")
        else:
            print(f"[DataCollector] Appending to existing {DATASET_PATH}")

    def record(
        self,
        reaction_time: float,
        target_x: int,
        target_y: int,
        target_size: int,
        hit: int,
        difficulty: int,
    ):
        """
        Append one data row to the CSV.
        Called after every click (hit or miss).
        """

        # Basic validation
        if reaction_time < 0 or reaction_time > 30:
            return
        if hit not in (0, 1):
            return

        row = {
            "reaction_time": round(reaction_time, 4),
            "target_x": int(target_x),
            "target_y": int(target_y),
            "target_size": int(target_size),
            "hit": int(hit),
            "timestamp": round(time.time(), 2),
            "difficulty_level": int(difficulty),
        }

        with open(DATASET_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)

# ── Utility: Load Dataset ────────────────────────────────────────────────
def load_dataset() -> list[dict]:
    """
    Load dataset.csv into a list of dictionaries.
    """

    if not os.path.exists(DATASET_PATH):
        print("[DataCollector] dataset.csv not found.")
        return []

    rows = []

    with open(DATASET_PATH, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                rows.append({
                    "reaction_time": float(row["reaction_time"]),
                    "target_x": int(row["target_x"]),
                    "target_y": int(row["target_y"]),
                    "target_size": int(row["target_size"]),
                    "hit": int(row["hit"]),
                    "timestamp": float(row["timestamp"]),
                    "difficulty_level": int(row["difficulty_level"]),
                })
            except (ValueError, KeyError):
                continue  # skip bad rows

    print(f"[DataCollector] Loaded {len(rows)} rows from {DATASET_PATH}")
    return rows
