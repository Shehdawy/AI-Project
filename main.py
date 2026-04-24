"""
main.py — Project Orchestrator

This file is the MAIN CONTROLLER of the whole project.
It connects:
Game ↔ DataCollector ↔ ML Model ↔ Clustering ↔ Visualization

Usage:
  python main.py          → Run full system (game + AI + analysis)
  python main.py --demo   → Generate fake data (no game needed)
  python main.py --train  → Train ML model only
  python main.py --viz    → Generate plots only
"""

# ─────────────────────────────────────────────────────────────
# Standard Libraries
# ─────────────────────────────────────────────────────────────
import argparse        # For command line arguments (--demo, --train, etc.)
import sys             # System-level operations (not heavily used here)
import time            # Used to generate unique session ID
import threading       # Used to run background AI loop
import random          # Random values for demo data
import numpy as np     # Math & statistics (mean, std, etc.)
from pathlib import Path  # File path handling

# ─────────────────────────────────────────────────────────────
#  Project Modules (Your files)
# ─────────────────────────────────────────────────────────────
from data_collector import DataCollector, CSV_PATH
from model import SkillClassifier, build_features, FEATURE_COLS
from clustering import PlayerClustering
from visualization import generate_all_plots

# ─────────────────────────────────────────────────────────────
# Unique Session ID
# ─────────────────────────────────────────────────────────────
# Create a unique ID based on current time
SESSION_ID = f"session_{int(time.time())}"


# ─────────────────────────────────────────────────────────────
#  Demo Data Generator
# ─────────────────────────────────────────────────────────────
def generate_demo_data(n_sessions: int = 6, events_per_session: int = 50):
    """
    Generate fake gameplay data so we can test AI without running the game.
    """

    print(f"[Demo] Generating {n_sessions} sessions × {events_per_session} events…")

    # Different types of players (simulated)
    skill_profiles = {
        "pro_player":     {"rt_mean": 0.28, "rt_std": 0.05, "hit_rate": 0.92},
        "average_player": {"rt_mean": 0.50, "rt_std": 0.10, "hit_rate": 0.70},
        "beginner_1":     {"rt_mean": 0.80, "rt_std": 0.20, "hit_rate": 0.45},
        "beginner_2":     {"rt_mean": 0.95, "rt_std": 0.25, "hit_rate": 0.38},
        "avg_improving":  {"rt_mean": 0.55, "rt_std": 0.12, "hit_rate": 0.65},
        "semi_pro":       {"rt_mean": 0.35, "rt_std": 0.07, "hit_rate": 0.85},
    }

    # Difficulty levels
    difficulties = ["Beginner", "Average", "Pro"]

    # Target sizes per difficulty
    diff_sizes = {"Beginner": 50, "Average": 32, "Pro": 18}

    # Loop through players
    for sid, profile in list(skill_profiles.items())[:n_sessions]:
        dc = DataCollector(sid)  # Create data collector

        rt_m = profile["rt_mean"]  # reaction time mean
        rt_s = profile["rt_std"]   # reaction time std
        hr   = profile["hit_rate"] # accuracy

        # Generate events
        for i in range(events_per_session):

            # Simulate improvement over time
            improvement = i / events_per_session * 0.15

            # Generate reaction time
            rt = max(0.1, np.random.normal(rt_m - improvement, rt_s))

            # Hit or miss
            hit = 1 if random.random() < hr else 0

            # Random difficulty
            diff = random.choice(difficulties)

            # Record data
            dc.record(
                reaction_time = round(rt, 4) if hit else 0,
                target_x      = random.randint(60, 740),
                target_y      = random.randint(80, 560),
                target_size   = diff_sizes[diff] if hit else 0,
                hit           = hit,
                difficulty    = diff,
            )

        # Save session
        dc.save()

    print(f"[Demo] ✅ Data saved to {CSV_PATH}")


# ─────────────────────────────────────────────────────────────
#  Adaptive Difficulty (Background Thread)
# ─────────────────────────────────────────────────────────────
def adaptive_difficulty_loop(game, clf: SkillClassifier, interval: float = 5.0):
    """
    Runs in background:
    Every few seconds → analyze player → change difficulty
    """

    while game.running:
        time.sleep(interval)  # wait

        # Skip if not enough data
        if game.score < 5:
            continue

        shots = game.total_shots
        hits  = game.score
        rts   = game.reaction_times

        if not rts:
            continue

        # Build stats dictionary
        stats = {
            "total_shots": shots,
            "total_hits": hits,
            "avg_reaction": round(np.mean(rts), 4),
            "std_reaction": round(np.std(rts), 4),
            "min_reaction": round(min(rts), 4),
            "max_reaction": round(max(rts), 4),
            "avg_target_size": 32.0,
            "accuracy": hits / shots if shots > 0 else 0,
        }

        # Predict skill
        skill = clf.predict(stats)

        # Update game difficulty
        game.set_difficulty(skill)

        print(f"[Adaptive] Skill: {skill} → difficulty updated")


# ─────────────────────────────────────────────────────────────
#  Run Game with AI
# ─────────────────────────────────────────────────────────────
def run_game_with_ai():
    """Run full pipeline: game + AI"""

    print("\n[Main] Loading ML model…")

    clf = SkillClassifier()  # create model
    clf.train(verbose=True)  # train model

    print("[Main] Starting game...")

    try:
        from game import AimTrainerGame
        game = AimTrainerGame(session_id=SESSION_ID)

        # Start adaptive AI in background thread
        t = threading.Thread(
            target=adaptive_difficulty_loop,
            args=(game, clf, 5.0),
            daemon=True
        )
        t.start()

        # Run the game
        game.run()

    except ImportError:
        print("Install pygame first: pip install pygame")
        return

    # After game ends → analysis
    post_session_analysis(SESSION_ID)


# ─────────────────────────────────────────────────────────────
# Post-Game Analysis
# ─────────────────────────────────────────────────────────────
def post_session_analysis(session_id: str = None):

    print("\n[Analysis] Training model...")
    clf = SkillClassifier()
    clf.train(verbose=True)

    print("\n[Analysis] Clustering...")
    pc = PlayerClustering()

    if pc.fit():
        print(pc.cluster_summary())
        pc.plot_clusters()
        pc.elbow_analysis()

    print("\n[Analysis] Generating plots...")
    plots = generate_all_plots(session_id)

    print(f"Generated {len(plots)} plots")

    print("\n[Analysis] Final Prediction:")

    try:
        df = DataCollector.load_dataframe()

        if df is not None:
            sess_df = df[df["session_id"] == session_id]

            hits = sess_df[sess_df["hit"] == 1]

            if len(hits) > 0:
                stats = {
                    "total_shots": len(sess_df),
                    "total_hits": int(sess_df["hit"].sum()),
                    "avg_reaction": hits["reaction_time"].mean(),
                    "std_reaction": hits["reaction_time"].std(),
                    "min_reaction": hits["reaction_time"].min(),
                    "max_reaction": hits["reaction_time"].max(),
                    "avg_target_size": hits["target_size"].mean(),
                    "accuracy": sess_df["hit"].mean(),
                }

                skill = clf.predict(stats)
                proba = clf.predict_proba(stats)

                print("Skill:", skill)
                print("Probabilities:", proba)

    except Exception as e:
        print("Error:", e)

    print("\nRun dashboard: streamlit run dashboard.py")


# ─────────────────────────────────────────────────────────────
#  CLI (Command Line Interface)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--viz",   action="store_true")
    parser.add_argument("--sessions", type=int, default=6)

    args = parser.parse_args()

    # Choose mode
    if args.demo:
        generate_demo_data(n_sessions=args.sessions)
        post_session_analysis()

    elif args.train:
        clf = SkillClassifier()
        clf.train(verbose=True)

    elif args.viz:
        plots = generate_all_plots()
        print("Plots generated:", len(plots))

    else:
        run_game_with_ai()
