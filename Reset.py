import os, shutil

files = ["dataset.csv", "trained_model.pkl", "scaler.pkl"]
for f in files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")

if os.path.exists("plots"):
    shutil.rmtree("plots")
    print("Deleted: plots/")

print("\nReset complete. Ready for a fresh start.")