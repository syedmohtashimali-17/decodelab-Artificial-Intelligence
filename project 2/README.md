# 🌸 Iris KNN Classifier — Production Pipeline & Web UI

End-to-end supervised classification using **K-Nearest Neighbours** on the Iris benchmark dataset, built with scikit-learn and Streamlit.

## 🚀 One-Click Run Launchers (Windows)

Simply **double-click** any of these files directly from Windows File Explorer:

- 🖱️ **`run_web_ui.bat`** — Launches the interactive Web Dashboard in your browser!
- 🖱️ **`run.bat`** — Runs the interactive Console CLI pipeline.
- 🖱️ **`run_automated.bat`** — Runs the automated test pipeline and generates visual plots.

---

## Project Structure

```
project 2/
├── app.py                  # Streamlit Web UI Application
├── main.py                 # CLI entry point — runs the full 5-step pipeline
├── run.bat                 # One-click launcher for Interactive CLI mode
├── run_automated.bat       # One-click launcher for Automated CLI mode
├── run_web_ui.bat          # One-click launcher for Streamlit Web App
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── outputs/                # Auto-generated diagnostic plots
│   ├── feature_distributions.png
│   ├── elbow_curve.png
│   ├── confusion_matrix.png
│   └── decision_boundary.png
└── src/
    ├── __init__.py          # Package metadata
    ├── data_loader.py       # Data ingestion, StandardScaler, stratified split
    ├── model.py             # KNN training & hyperparameter tuning (K=1…20)
    ├── evaluator.py         # Confusion matrix, classification report, TP/FP/FN/TN
    ├── visualizer.py        # Pairplot, elbow curve, heatmap, decision boundary
    └── predictor.py         # Interactive console predictor with probability bars
```

---

## Quick Start (Terminal Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Web UI (Interactive Dashboard)
streamlit run app.py

# 3. Run full CLI pipeline (Interactive Console)
python main.py

# 4. Run automated CLI pipeline (Headless Mode)
python main.py --no-interactive
```

---

## Key Design Decisions

- **StandardScaler** (Z-score normalization) ensures all 4 features contribute equally to Euclidean distance.
- **Stratified splitting** preserves class balance in both train/test partitions.
- **Elbow detection** selects the K with minimum test error (largest K on ties to prefer smoother boundaries).
- **PCA 2-D projection** for decision boundary visualization captures ~95% of variance.

---

## Dependencies

- Python 3.10+
- scikit-learn ≥ 1.3
- pandas ≥ 2.0
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- streamlit ≥ 1.25
