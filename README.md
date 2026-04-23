# 🎯 AI Adaptive Aim Trainer

**University AI Engineering Project**  
*Machine Learning · Pattern Recognition · Real-Time Dashboard*

---

## 📁 Project Structure

```
/project
  ├── game.py            → Pygame aim trainer game (Game Developer)
  ├── data_collector.py  → Real-time CSV data collection (Data Engineer)
  ├── model.py           → Supervised ML classification (ML Engineer)
  ├── clustering.py      → K-Means unsupervised learning (AI Analyst)
  ├── visualization.py   → Pattern recognition & graphs (AI Analyst)
  ├── dashboard.py       → Streamlit real-time dashboard (AI Analyst)
  ├── main.py            → Orchestrator / entry point
  ├── dataset.csv        → Auto-generated gameplay data
  ├── plots/             → Auto-generated PNG charts
  └── README.md          → This file
```

---

## ⚙️ Installation

```bash
pip install pygame scikit-learn pandas numpy matplotlib streamlit joblib
```

## 🚀 How to Run

### Option 1 — Full Game + AI Pipeline
```bash
python -m main.py
```
Starts the game. After you quit (ESC), it trains the ML model, runs clustering, and generates plots.

### Option 2 — Demo Mode (no Pygame needed, great for testing)
```bash
python -m main.py --demo
```
Generates 6 synthetic sessions of realistic gameplay data, then runs the full AI pipeline.

### Option 3 — Launch the Dashboard
```bash
python -m streamlit run dashboard.py
```
Opens in your browser at `http://localhost:8501`

### Option 4 — Train model only
```bash
python -m main.py --train
```

### Option 5 — Generate plots only
```bash
python -m main.py --viz
```

---

## 🎮 Game Controls

| Key / Action | Effect |
|---|---|
| Left click | Shoot a target |
| ESC | Quit game |
| R | Reset session |

**Difficulty levels** are applied automatically by the AI based on your skill:
- **Beginner** → Large (r=50px), slow targets, 2s interval
- **Average**  → Medium (r=32px), 1.2s interval
- **Pro**      → Small (r=18px), fast, 0.7s interval

---

## 🧠 AI Architecture

### 1. Supervised Learning (model.py)

**Goal:** Classify player as Beginner / Average / Pro

**Features per session:**
| Feature | Description |
|---|---|
| avg_reaction | Mean reaction time on hits |
| std_reaction | Variability of reaction times |
| accuracy | Hits / total shots |
| total_shots | Volume of play |
| avg_target_size | Difficulty of targets hit |

**Models trained:**
- K-Nearest Neighbors (k=5)
- Decision Tree (max_depth=5)
- Random Forest (100 estimators)

**Selection:** Best model chosen by 5-fold cross-validation.

---

### 2. Unsupervised Learning (clustering.py)

**Goal:** Discover behavioral groups without labels

**Algorithm:** K-Means (k=3)
- Elbow method used to validate k
- Silhouette score measures cluster quality
- PCA reduces features to 2D for visualization
- Clusters auto-labeled by reaction time centroids

---

### 3. Pattern Recognition (visualization.py)

| Plot | What it detects |
|---|---|
| Reaction Time Trend | Rolling average; improvement/decline slope |
| Accuracy Over Time | Rolling hit rate; performance stability |
| Reaction Heatmap | Screen zones where player is fast vs slow |
| Improvement Trend | Polynomial regression on RT — improving? |
| Difficulty Distribution | Shot volume per difficulty level |

---

### 4. Adaptive Difficulty (main.py + game.py)

Every 5 seconds during gameplay, a background thread:
1. Computes current session stats
2. Calls `SkillClassifier.predict()`
3. Updates `game.set_difficulty(label)`

The game immediately adjusts target size and spawn speed.

---

## 📊 System Flow

```
Player plays game
      ↓
game.py  ──► data_collector.py  ──► dataset.csv
                                        ↓
                              model.py (KNN / DT / RF)
                                        ↓
                            Skill: Beginner / Average / Pro
                                        ↓
                              clustering.py (K-Means)
                                        ↓
                              visualization.py (graphs)
                                        ↓
                              dashboard.py (Streamlit)
                                        ↓
                         game.py ← adaptive difficulty update
```

---

## 📈 Dashboard Features

- **Player Stats** — Hits, accuracy, average/best reaction time
- **ML Prediction** — Live skill label with probability bars
- **Cluster Analysis** — K-Means cluster assignment + summary table
- **4 Performance Graphs** — Reaction trend, accuracy, improvement, difficulty
- **Cluster Scatter** — PCA 2D visualization of all sessions
- **Elbow Chart** — Optimal k validation
- **Raw Data Table** — Browse CSV directly in browser
- **Auto-refresh toggle** — Live updates every 3 seconds

---

## 🔬 Supervised vs Unsupervised Learning

### Supervised Learning
- **Requires labeled data** — we define what "Pro", "Average", "Beginner" means via threshold rules
- **Learns a mapping** from features → label
- **Used for:** real-time prediction during gameplay
- **Real-world analogy:** spam email detection (labeled spam/not-spam)

### Unsupervised Learning
- **No labels needed** — algorithm finds natural groupings in raw data
- **K-Means** partitions sessions into k groups by minimizing intra-cluster distance
- **Used for:** discovering player archetypes researchers didn't pre-define
- **Real-world analogy:** customer segmentation in marketing

### When both are used together:
Unsupervised clustering can **validate** or **refine** supervised labels. If our rule-based "Pro" boundary is wrong, K-Means clusters might reveal a different natural boundary.

---

## 👥 Team Roles

| Member | Module | Responsibility |
|---|---|---|
| Game Developer | game.py | Pygame engine, targets, scoring, difficulty scaling |
| Data Engineer | data_collector.py | Real-time CSV collection, feature schema |
| ML Engineer | model.py | KNN, Decision Tree, Random Forest, evaluation |
| AI Analyst | clustering.py, visualization.py, dashboard.py | K-Means, pattern recognition, Streamlit dashboard |
