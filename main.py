"""
main.py — Project Orchestrator
Ties together: Game ↔ DataCollector ↔ ML Model ↔ Dashboard

Usage:
  python main.py          → Play game, then show analysis
  python main.py --demo   → Generate synthetic data first (no Pygame needed)
  python main.py --train  → Train ML model only
  python main.py --viz    → Generate all plots only
"""

import argparse
import sys
import time
import threading
import random
import numpy as np
from pathlib import Path

# ── Project imports ────────────────────────────────────────────────────────────
from data_collector import DataCollector, CSV_PATH
from model import SkillClassifier, build_features, FEATURE_COLS
from clustering import PlayerClustering
from visualization import generate_all_plots

# ── Session ID ────────────────────────────────────────────────────────────────
SESSION_ID = f"session_{int(time.time())}"


# ── Demo data generator ───────────────────────────────────────────────────────
def generate_demo_data(n_sessions: int = 6, events_per_session: int = 50):
    """
    Generate realistic synthetic gameplay data so the ML and clustering
    components can be tested without running Pygame.
    """
    print(f"[Demo] Generating {n_sessions} sessions × {events_per_session} events…")

    skill_profiles = {
        "pro_player":     {"rt_mean": 0.28, "rt_std": 0.05, "hit_rate": 0.92},
        "average_player": {"rt_mean": 0.50, "rt_std": 0.10, "hit_rate": 0.70},
        "beginner_1":     {"rt_mean": 0.80, "rt_std": 0.20, "hit_rate": 0.45},
        "beginner_2":     {"rt_mean": 0.95, "rt_std": 0.25, "hit_rate": 0.38},
        "avg_improving":  {"rt_mean": 0.55, "rt_std": 0.12, "hit_rate": 0.65},
        "semi_pro":       {"rt_mean": 0.35, "rt_std": 0.07, "hit_rate": 0.85},
    }

    difficulties = ["Beginner", "Average", "Pro"]
    diff_sizes   = {"Beginner": 50, "Average": 32, "Pro": 18}

    for sid, profile in list(skill_profiles.items())[:n_sessions]:
        dc = DataCollector(sid)
        rt_m = profile["rt_mean"]
        rt_s = profile["rt_std"]
        hr   = profile["hit_rate"]

        # Simulate improvement over the session
        for i in range(events_per_session):
            improvement = i / events_per_session * 0.15  # slight RT improvement
            rt = max(0.1, np.random.normal(rt_m - improvement, rt_s))
            hit = 1 if random.random() < hr else 0
            diff = random.choice(difficulties)
            dc.record(
                reaction_time = round(rt, 4) if hit else 0,
                target_x      = random.randint(60, 740),
                target_y      = random.randint(80, 560),
                target_size   = diff_sizes[diff] if hit else 0,
                hit           = hit,
                difficulty    = diff,
            )
        dc.save()

    print(f"[Demo] ✅  Demo data written to {CSV_PATH}")


# ── Adaptive difficulty thread ────────────────────────────────────────────────
def adaptive_difficulty_loop(game, clf: SkillClassifier, interval: float = 5.0):
    """
    Background thread: every `interval` seconds, compute current session stats
    and update the game's difficulty via model prediction.
    """
    while game.running:
        time.sleep(interval)
        if game.score < 5:
            continue  # not enough data yet

        # Build stats from current session
        shots = game.total_shots
        hits  = game.score
        rts   = game.reaction_times
        if not rts:
            continue

        stats = {
            "total_shots":     shots,
            "total_hits":      hits,
            "avg_reaction":    round(np.mean(rts), 4),
            "std_reaction":    round(np.std(rts), 4),
            "min_reaction":    round(min(rts), 4),
            "max_reaction":    round(max(rts), 4),
            "avg_target_size": 32.0,
            "accuracy":        hits / shots if shots > 0 else 0,
        }

        skill = clf.predict(stats)
        game.set_difficulty(skill)
        print(f"[Adaptive] Predicted skill: {skill} → difficulty updated")


# ── Main flow ─────────────────────────────────────────────────────────────────
def run_game_with_ai():
    """Full pipeline: play → collect → classify → adapt."""

    # 1. Train or load the model first
    print("\n[Main] Loading ML model…")
    clf = SkillClassifier()
    clf.train(verbose=True)

    # 2. Start the game
    print("[Main] Starting Aim Trainer… (ESC to quit)")
    try:
        from game import AimTrainerGame
        game = AimTrainerGame(session_id=SESSION_ID)

        # Start adaptive difficulty in background
        t = threading.Thread(
            target=adaptive_difficulty_loop,
            args=(game, clf, 5.0),
            daemon=True
        )
        t.start()

        game.run()

    except ImportError:
        print("[Main] pygame not found. Install with: pip install pygame")
        return

    # 3. Post-session analysis
    print("\n[Main] Session complete. Running post-game analysis…")
    post_session_analysis(SESSION_ID)


def post_session_analysis(session_id: str = None):
    """ML training, clustering, and plot generation."""

    print("\n[Analysis] Training ML model on updated data…")
    clf = SkillClassifier()
    results = clf.train(verbose=True)

    print("\n[Analysis] Running K-Means clustering…")
    pc = PlayerClustering()
    ok = pc.fit()
    if ok:
        print(pc.cluster_summary().to_string())
        pc.plot_clusters()
        pc.elbow_analysis()

    print("\n[Analysis] Generating visualization plots…")
    plots = generate_all_plots(session_id)
    print(f"[Analysis] {len(plots)} plots saved in ./plots/")

    print("\n[Analysis] ─── Final session prediction ─── ")
    try:
        from data_collector import DataCollector
        df = DataCollector.load_dataframe()
        if df is not None:
            sess_df = df[df["session_id"] == session_id] if session_id else df
            hits    = sess_df[sess_df["hit"] == 1]
            if len(hits) > 0:
                stats = {
                    "total_shots":     len(sess_df),
                    "total_hits":      int(sess_df["hit"].sum()),
                    "avg_reaction":    hits["reaction_time"].mean(),
                    "std_reaction":    hits["reaction_time"].std(),
                    "min_reaction":    hits["reaction_time"].min(),
                    "max_reaction":    hits["reaction_time"].max(),
                    "avg_target_size": hits["target_size"].mean(),
                    "accuracy":        sess_df["hit"].mean(),
                }
                skill = clf.predict(stats)
                proba = clf.predict_proba(stats)
                print(f"   Skill Level : {skill}")
                if proba:
                    print(f"   Probabilities: {proba}")
    except Exception as e:
        print(f"[Analysis] Could not compute final prediction: {e}")

    print("\n[Main] ✅  All done! Run the dashboard with:")
    print("        streamlit run dashboard.py\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Aim Trainer — Main Entry Point")
    parser.add_argument("--demo",  action="store_true", help="Generate synthetic demo data")
    parser.add_argument("--train", action="store_true", help="Train ML model only")
    parser.add_argument("--viz",   action="store_true", help="Generate plots only")
    parser.add_argument("--sessions", type=int, default=6, help="Demo sessions to generate")
    args = parser.parse_args()

    if args.demo:
        generate_demo_data(n_sessions=args.sessions)
        post_session_analysis()

    elif args.train:
        clf = SkillClassifier()
        clf.train(verbose=True)

    elif args.viz:
        plots = generate_all_plots()
        print(f"Generated {len(plots)} plots.")

    else:
        run_game_with_ai()
