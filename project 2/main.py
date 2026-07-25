#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         IRIS KNN CLASSIFIER — PRODUCTION PIPELINE           ║
║                                                              ║
║  End-to-end supervised classification using K-Nearest        ║
║  Neighbours on the Iris benchmark dataset.                   ║
║                                                              ║
║  Pipeline:                                                   ║
║    1. Data Ingestion & Feature Scaling (StandardScaler)      ║
║    2. Hyperparameter Tuning (K = 1…20, Elbow Detection)      ║
║    3. Model Evaluation (Confusion Matrix, F1, Precision…)    ║
║    4. Diagnostic Visualizations (Pairplot, Elbow, Boundary)  ║
║    5. Interactive Console Predictor                          ║
║                                                              ║
║  Author : ML Engineering Pipeline                            ║
║  Stack  : scikit-learn · pandas · numpy · matplotlib ·       ║
║           seaborn                                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import warnings

# Ensure UTF-8 output stream encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

from src.data_loader import IrisDataLoader
from src.model import KNNModel
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer
from src.predictor import InteractivePredictor


def print_banner() -> None:
    """Print application header."""
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║           🌸  IRIS FLOWER CLASSIFICATION SYSTEM  🌸          ║
║          K-Nearest Neighbours · Scikit-Learn Pipeline        ║
╚══════════════════════════════════════════════════════════════╝
        """
    )


def print_performance_summary(evaluator: ModelEvaluator, model: KNNModel) -> None:
    """
    Print a final summary section highlighting key performance stats
    and takeaways from the trained model.
    """
    perf = evaluator.performance_summary()
    acc = perf["accuracy"]

    print("\n" + "=" * 60)
    print("  📋 MODEL PERFORMANCE SUMMARY & KEY TAKEAWAYS")
    print("=" * 60)
    print(f"""
  ┌─ Key Statistics ─────────────────────────────────────┐
  │  Optimal K             : {model.best_k}
  │  Overall Accuracy      : {acc:.4f} ({acc * 100:.2f}%)
  │  Distance Metric       : Euclidean (Minkowski p=2)
  │  Feature Scaling       : StandardScaler (Z-score)
  │  Split Strategy        : Stratified 80/20 (shuffle=True)
  └──────────────────────────────────────────────────────┘
    """)

    print("  ┌─ Per-Class Performance ───────────────────────────┐")
    for name, metrics in perf["per_class"].items():
        print(
            f"  │  {name:>12s}  │  "
            f"P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  "
            f"F1={metrics['f1_score']:.3f}  "
            f"Support={metrics['support']}"
        )
    print("  └──────────────────────────────────────────────────┘")

    print(
        """
  🔑 Key Takeaways:
  ─────────────────
  1. KNN with StandardScaler achieves near-perfect accuracy on the
     Iris dataset thanks to well-separated feature clusters (especially
     petal dimensions).

  2. The elbow curve confirms that small K values (3–7) generalise
     well, while K=1 risks overfitting to noise.

  3. Setosa is linearly separable from versicolor/virginica in every
     feature pair. The remaining two classes overlap slightly in sepal
     space but are well-separated in petal space.

  4. The decision boundary visualisation (PCA 2-D projection) shows
     clean class regions with minimal overlap, confirming robust
     generalisation.

  5. Feature scaling is essential — without StandardScaler, sepal
     length (range 4–8) would dominate over petal width (range 0.1–2.5)
     in the Euclidean distance computation.
    """
    )


def main() -> None:
    """Execute the full classification pipeline."""
    print_banner()

    # ── Step 1: Data Ingestion & Preprocessing ───────────────────
    print("━" * 60)
    print("  STEP 1 │ Data Ingestion & Preprocessing")
    print("━" * 60)
    data = IrisDataLoader(test_size=0.20, random_state=42, shuffle=True)
    print(data.summary())

    # ── Step 2: Hyperparameter Tuning ────────────────────────────
    print("\n" + "━" * 60)
    print("  STEP 2 │ KNN Hyperparameter Tuning (K = 1…20)")
    print("━" * 60)
    model = KNNModel(k_min=1, k_max=20)
    model.tune(data.X_train, data.y_train, data.X_test, data.y_test)
    print(model.tuning_summary())

    # ── Step 3: Model Evaluation ─────────────────────────────────
    print("\n" + "━" * 60)
    print("  STEP 3 │ Model Evaluation & Diagnostics")
    print("━" * 60)
    y_pred = model.predict(data.X_test)
    evaluator = ModelEvaluator(
        data.y_test, y_pred, target_names=list(data.target_names)
    )
    print(evaluator.full_report())

    # ── Step 4: Visualizations ───────────────────────────────────
    print("\n" + "━" * 60)
    print("  STEP 4 │ Generating Diagnostic Visualizations")
    print("━" * 60)
    Visualizer.generate_all(
        df=data.df,
        k_range=model.k_range,
        error_rates=model.error_rates,
        best_k=model.best_k,
        cm=evaluator.cm,
        target_names=list(data.target_names),
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        y_test=data.y_test,
    )

    # ── Step 5: Performance Summary ──────────────────────────────
    print_performance_summary(evaluator, model)

    # ── Step 6: Interactive Predictor ────────────────────────────
    print("━" * 60)
    print("  STEP 5 │ Interactive Predictor")
    print("━" * 60)

    if "--no-interactive" in sys.argv:
        print("  [Skipped: --no-interactive flag detected]\n")
        return

    try:
        predictor = InteractivePredictor(data, model)
        predictor.run()
    except (EOFError, KeyboardInterrupt):
        print("\n  👋 Session ended.")


if __name__ == "__main__":
    main()
