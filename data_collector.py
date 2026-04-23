"""
data_collector.py — Data Engineer Module
Collects and stores gameplay data in real-time to a CSV file.

Features captured per event:
  - reaction_time   : seconds from target spawn to click (0 if miss)
  - target_x        : x position of target (or click x on miss)
  - target_y        : y position of target (or click y on miss)
  - target_size     : radius of target (0 if miss)
  - hit             : 1 = hit, 0 = miss
  - timestamp       : Unix epoch float
  - difficulty_level: Beginner / Average / Pro
"""

import csv
import os
import time
from pathlib import Path

CSV_PATH = Path(__file__).parent / "dataset.csv"

FIELDNAMES = [
    "session_id",
    "timestamp",
    "reaction_time",
    "target_x",
    "target_y",
    "target_size",
    "hit",
    "difficulty_level",
]


class DataCollector:
    """
    Buffers gameplay events and flushes them to dataset.csv.
    Call record() after every click; call save() at session end.
    """

    def __init__(self, session_id: str = "s001"):
        self.session_id = session_id
        self.buffer: list[dict] = []

        # Create CSV with header if it doesn't exist
        if not CSV_PATH.exists():
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            print(f"[DataCollector] Created new dataset at {CSV_PATH}")
        else:
            print(f"[DataCollector] Appending to existing dataset at {CSV_PATH}")

    # ── Public API ─────────────────────────────────────────────────────────────
    def record(
        self,
        reaction_time: float,
        target_x: int,
        target_y: int,
        target_size: int,
        hit: int,
        difficulty: str,
    ):
        """
        Buffer one gameplay event.
        Parameters match the CSV column names exactly.
        """
        row = {
            "session_id":       self.session_id,
            "timestamp":        round(time.time(), 4),
            "reaction_time":    reaction_time,
            "target_x":         target_x,
            "target_y":         target_y,
            "target_size":      target_size,
            "hit":              hit,
            "difficulty_level": difficulty,
        }
        self.buffer.append(row)

        # Auto-flush every 20 events so we don't lose data on crash
        if len(self.buffer) >= 20:
            self._flush()

    def save(self):
        """Flush remaining buffer and confirm save."""
        self._flush()
        total = self._count_rows()
        print(f"[DataCollector] Dataset saved. Total rows in file: {total}")

    # ── Internal ───────────────────────────────────────────────────────────────
    def _flush(self):
        if not self.buffer:
            return
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerows(self.buffer)
        self.buffer.clear()

    def _count_rows(self) -> int:
        with open(CSV_PATH, "r") as f:
            return max(0, sum(1 for _ in f) - 1)  # subtract header

    # ── Utility ────────────────────────────────────────────────────────────────
    @staticmethod
    def load_dataframe():
        """
        Load the full dataset as a pandas DataFrame.
        Returns None if pandas is not installed or file is empty.
        """
        try:
            import pandas as pd
            if CSV_PATH.exists():
                df = pd.read_csv(CSV_PATH)
                if len(df) == 0:
                    return None
                return df
        except ImportError:
            print("[DataCollector] pandas not installed.")
        return None

    @staticmethod
    def clear_dataset():
        """Delete the CSV file (start fresh). Use with caution."""
        if CSV_PATH.exists():
            os.remove(CSV_PATH)
            print("[DataCollector] Dataset cleared.")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dc = DataCollector("test_session")
    for i in range(5):
        dc.record(
            reaction_time=round(0.3 + i * 0.05, 4),
            target_x=100 + i * 30,
            target_y=200 + i * 20,
            target_size=40,
            hit=1 if i % 2 == 0 else 0,
            difficulty="Beginner",
        )
    dc.save()
    df = DataCollector.load_dataframe()
    if df is not None:
        print(df.tail(5))
