"""
Interactive Predictor Module
============================
Console-based CLI that accepts custom Iris measurements and returns
the predicted species with per-class probability scores.

Usage:
    Called from main.py after training completes.
    The user can enter 4 float values (sepal/petal length/width)
    and receive an instant prediction with confidence.
"""

import numpy as np
from src.data_loader import IrisDataLoader
from src.model import KNNModel


class InteractivePredictor:
    """
    Interactive console loop for live KNN predictions.

    Parameters
    ----------
    data_loader : IrisDataLoader
        Fitted loader (provides scaler and target names).
    model : KNNModel
        Trained KNN model (provides predict / predict_proba).
    """

    FEATURE_PROMPTS = [
        ("Sepal Length (cm)", 4.0, 8.0),
        ("Sepal Width  (cm)", 2.0, 4.5),
        ("Petal Length (cm)", 1.0, 7.0),
        ("Petal Width  (cm)", 0.1, 2.5),
    ]

    def __init__(self, data_loader: IrisDataLoader, model: KNNModel):
        self.data_loader = data_loader
        self.model = model
        self.target_names = list(data_loader.target_names)

    def _read_feature(self, name: str, low: float, high: float) -> float:
        """Prompt for a single numeric feature with range hint."""
        while True:
            raw = input(f"    {name} [{low}–{high}]: ").strip()
            if raw == "":
                return (low + high) / 2  # Default to midpoint
            try:
                val = float(raw)
                return val
            except ValueError:
                print(f"    ⚠ Invalid number. Please enter a decimal value.")

    def run(self) -> None:
        """
        Start the interactive prediction loop.

        Type 'q' or 'quit' at any prompt to exit.
        Press Enter on a feature prompt to use the midpoint default.
        """
        print("\n" + "=" * 60)
        print("  🔮 INTERACTIVE IRIS CLASSIFIER")
        print("=" * 60)
        print("  Enter flower measurements to predict the species.")
        print("  Type 'q' or 'quit' to exit.")
        print("  Press Enter on any feature to use the default (midpoint).")
        print("=" * 60)

        while True:
            print("\n  ── New Prediction ──")
            features = []

            for name, low, high in self.FEATURE_PROMPTS:
                # Check for quit command
                try:
                    val = self._read_feature(name, low, high)
                except (EOFError, KeyboardInterrupt):
                    print("\n  👋 Exiting predictor.")
                    return
                features.append(val)

            # Shape into (1, 4) array and scale
            raw = np.array(features).reshape(1, -1)
            scaled = self.data_loader.transform_input(raw)

            # Predict
            prediction = self.model.predict(scaled)[0]
            probabilities = self.model.predict_proba(scaled)[0]

            species = self.target_names[prediction]

            # Display results
            print("\n  ┌─────────────────────────────────────┐")
            print(f"  │  Predicted Species: {species:>15s}  │")
            print("  ├─────────────────────────────────────┤")
            print("  │  Class Probabilities:               │")
            for i, name in enumerate(self.target_names):
                bar_len = int(probabilities[i] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(
                    f"  │   {name:>12s}: {probabilities[i]:>6.2%} "
                    f"{bar}  │"
                )
            print("  └─────────────────────────────────────┘")

            # Ask to continue
            again = input("\n  Predict another? [Y/n/q]: ").strip().lower()
            if again in ("n", "q", "quit", "exit"):
                print("\n  👋 Exiting predictor. Goodbye!")
                return
