# 🎯 AI Adaptive Aim Trainer

**University AI Engineering Project**  
*Machine Learning · Pattern Recognition · Real-Time Dashboard*


## 👥 Team

| Role | Responsibility |
|---|---|
| 🎮 Game Developer | `game.py` — Pygame game loop, adaptive difficulty |
| 📊 Data Engineer | `data_collector.py` — CSV collection and loading |
| 🤖 ML Engineer | `model.py` — Random Forest classifier, skill prediction |
| 📈 AI Analyst | `clustering.py` + `visualization.py` — KMeans, charts |

---

## 📁 Project Structure

```
/project
├── game.py             ← Aim trainer game (Pygame)
├── data_collector.py   ← Saves gameplay data to CSV
├── model.py            ← Trains & loads the ML model
├── clustering.py       ← KMeans clustering + pattern analysis
├── visualization.py    ← 3 matplotlib charts
├── dashboard.py        ← Streamlit analytics dashboard
├── main.py             ← CLI menu launcher
├── reset.py            ← Reset all data and models
├── dataset.csv         ← Auto-generated during gameplay
├── model.pkl           ← Auto-generated after training
├── requirements.txt    ← Python dependencies
└── report.html         ← Full project report (open in browser)
```

---

## ⚙️ Requirements

- Python **3.10** or higher
- pip

Install all dependencies with:

```bash
pip install pygame numpy streamlit matplotlib sklearn
```

---

## 🚀 How to Run — Step by Step

### Step 1 — Play the game

```bash
python game.py
```

- Click on the red targets as fast and accurately as you can
- Data is automatically saved to `dataset.csv` after every click
- Difficulty adjusts every **60 seconds** based on your accuracy:
  - Accuracy > 80% → difficulty increases
  - Accuracy < 60% → difficulty decreases

### Step 2 — Train the ML model

> ⚠️ Do this **after** playing, not during. Training is offline only.

```bash
python model.py
```

- Reads `dataset.csv`
- Trains a Random Forest classifier
- Saves the model to `model.pkl`
- Prints a classification report in the terminal

### Step 3 — Open the dashboard

```bash
python -m streamlit run dashboard.py
```

- Opens in your browser at `http://localhost:8501`
- Shows: total hits/misses, accuracy, avg reaction time, skill prediction, pattern trends, and all 3 charts
- Use the **Refresh** button or enable **Auto-refresh** in the sidebar

### Step 4 — (Optional) View charts only

```bash
python visualization.py
```

### Step 5 — (Optional) View clustering analysis in terminal

```bash
python clustering.py
```

---

## 🖥️ Alternative: Use the Main Menu

```bash
python main.py
```

Launches an interactive CLI menu with options to run every component.

---

## 🔄 Reset Everything

To wipe all data and start fresh:

```bash
python reset.py
```

This deletes `dataset.csv` and `model.pkl`. See [reset.py](#) for details.

---

## 🎮 Game Controls

| Action | Control |
|---|---|
| Click a target | Left mouse button |
| Quit the game | Close the window or press `Ctrl+C` in terminal |

---

## 🧠 How the ML Works

### Supervised Learning — Skill Classification (`model.py`)

- **Algorithm:** Random Forest (50 decision trees)
- **Input features:** avg reaction time, accuracy, avg difficulty level
- **Output:** `Beginner` / `Average` / `Pro`
- **Labels** are generated automatically using a rule-based system on windows of 20 clicks

### Unsupervised Learning — Behavior Clustering (`clustering.py`)

- **Algorithm:** KMeans with 3 clusters
- **Features:** reaction time + hit/miss per click
- **Clusters:** Struggling / Improving / Skilled
- No labels needed — patterns emerge from the data

---

## 📊 The 3 Charts (`visualization.py`)

1. **Reaction Time Over Time** — raw data + rolling average
2. **Accuracy Over Time** — rolling hit rate with 80% / 60% reference lines
3. **Clustering Scatter Plot** — colored by KMeans cluster assignment

---

## 📈 Adaptive Difficulty System

Every 60 seconds the game checks your accuracy and adjusts:

| Condition | Action | Effect |
|---|---|---|
| Accuracy > 80% | Increase difficulty | Smaller targets, faster movement |
| Accuracy < 60% | Decrease difficulty | Larger targets, no movement |
| 60%–80% | No change | Stay at current level |

Difficulty levels:

| Level | Target Size | Target Speed |
|---|---|---|
| 1 — Beginner | 45 px | Stationary |
| 2 — Medium | 35 px | Slow movement |
| 3 — Hard | 25 px | Faster movement |

---

## 🗂️ Dataset Columns

| Column | Type | Description |
|---|---|---|
| `reaction_time` | float | Seconds from target spawn to click |
| `target_x` | int | Target X position (pixels) |
| `target_y` | int | Target Y position (pixels) |
| `target_size` | int | Target radius (pixels) |
| `hit` | 0 / 1 | 1 = hit, 0 = miss |
| `timestamp` | float | Unix timestamp of click |
| `difficulty_level` | 1/2/3 | Difficulty at time of click |

---

## 🛠️ Troubleshooting

**Pygame window doesn't open**
→ Make sure `pygame` is installed: `pip install pygame`

**"Not enough data to train"**
→ Play the game for at least 2–3 minutes before running `model.py`

**Dashboard shows empty charts**
→ Play the game first so `dataset.csv` has data

**`model.pkl` not found warning**
→ Run `python model.py` to train and save the model

---

