"""
main.py — Project Entry Point

Provides a simple menu to run each component of the project.

Usage:
    python main.py
"""

import os
import sys

# ── Menu Display ─────────────────────────────────────────────────────────
def print_menu():
    print("\n" + "=" * 50)
    print("    AI Adaptive Aim Trainer")
    print("=" * 50)
    print("  1. Play the game          (game.py)")
    print("  2. Train ML model         (model.py)")
    print("  3. Run clustering         (clustering.py)")
    print("  4. Show charts            (visualization.py)")
    print("  5. Open dashboard         (dashboard.py)")
    print("  6. Exit")
    print("=" * 50)

# ── Main Loop ────────────────────────────────────────────────────────────
def main():
    while True:
        print_menu()
        choice = input("  Choose an option (1–6): ").strip()

        if choice == "1":
            print("\n[main] Starting game...")
            os.system("python game.py")

        elif choice == "2":
            print("\n[main] Training ML model...")
            from model import train
            train()

        elif choice == "3":
            print("\n[main] Running clustering analysis...")
            from data_collector import load_dataset
            from clustering import run_clustering, analyze_patterns

            rows = load_dataset()
            if rows:
                run_clustering(rows)
                analyze_patterns(rows)
            else:
                print("No data found. Play the game first!")

        elif choice == "4":
            print("\n[main] Generating charts...")
            from data_collector import load_dataset
            from visualization import generate_charts

            rows = load_dataset()
            if rows:
                generate_charts(rows, show=True)
            else:
                print("No data found. Play the game first!")

        elif choice == "5":
            print("\n[main] Opening Streamlit dashboard...")
            os.system("streamlit run dashboard.py")

        elif choice == "6":
            print("\nGoodbye!")
            sys.exit(0)

        else:
            print("\nInvalid choice. Please enter 1–6.")

# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
