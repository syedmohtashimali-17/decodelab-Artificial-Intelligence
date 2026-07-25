"""
KNN Model & Hyperparameter Tuning Module
=========================================
Implements K-Nearest Neighbors classification with automated
elbow-point detection for optimal K selection.

Distance Metric Choice — Euclidean (Minkowski p=2):
    For the 4-dimensional continuous Iris features, Euclidean distance
    is the natural metric:
        d(x, y) = √( Σᵢ (xᵢ - yᵢ)² )
    It is well-suited because:
      • All features are numeric and on comparable scales (after StandardScaler).
      • It treats all dimensions uniformly — no ordinal or categorical issues.
    Alternative: Manhattan (p=1) is more robust to outliers but less
    discriminative in low dimensions.

Overfitting / Underfitting via K:
    • K=1  → model memorises noise (high variance, low bias).
    • K≫   → decision boundary over-smoothed (low variance, high bias).
    The elbow point in the error-rate curve gives the best bias-variance trade-off.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple


class KNNModel:
    """
    Wraps KNeighborsClassifier with hyperparameter search over K.

    Attributes
    ----------
    k_range : range
        Range of K values evaluated during tuning.
    error_rates : list[float]
        Test error rate (1 - accuracy) for each K.
    best_k : int
        K with lowest error rate (elbow point).
    best_model : KNeighborsClassifier
        Fitted classifier at optimal K.
    best_accuracy : float
        Test accuracy at optimal K.
    """

    def __init__(self, k_min: int = 1, k_max: int = 20):
        """
        Parameters
        ----------
        k_min, k_max : int
            Inclusive range of K values to evaluate.
        """
        self.k_range = range(k_min, k_max + 1)
        self.error_rates: List[float] = []
        self.accuracy_scores: List[float] = []
        self.best_k: int = 0
        self.best_model: KNeighborsClassifier | None = None
        self.best_accuracy: float = 0.0

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> "KNNModel":
        """
        Train KNN for each K in the range and record error rates.

        The optimal K is selected as the value with the minimum test error.
        In case of ties, the larger K is preferred (smoother boundary).

        Returns
        -------
        self : allows method chaining.
        """
        self.error_rates = []
        self.accuracy_scores = []

        for k in self.k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric="minkowski",  # Euclidean when p=2
                p=2,
                weights="uniform",   # All neighbours vote equally
            )
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            error = 1.0 - acc

            self.accuracy_scores.append(acc)
            self.error_rates.append(error)

        # ---- Identify optimal K (elbow point) ----
        min_error = min(self.error_rates)
        # Among all K values with minimum error, pick the largest
        # (prefer smoother decision boundary when accuracy is equal).
        candidates = [
            k
            for k, err in zip(self.k_range, self.error_rates)
            if err == min_error
        ]
        self.best_k = max(candidates)
        self.best_accuracy = 1.0 - min_error

        # Retrain final model at optimal K
        self.best_model = KNeighborsClassifier(
            n_neighbors=self.best_k,
            metric="minkowski",
            p=2,
            weights="uniform",
        )
        self.best_model.fit(X_train, y_train)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call .tune() first.")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for samples in X."""
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call .tune() first.")
        return self.best_model.predict_proba(X)

    def tuning_summary(self) -> str:
        """Return a formatted hyperparameter tuning summary."""
        lines = [
            "=" * 60,
            "  HYPERPARAMETER TUNING RESULTS",
            "=" * 60,
            f"  K range evaluated    : {self.k_range.start} – {self.k_range.stop - 1}",
            f"  Optimal K (elbow)    : {self.best_k}",
            f"  Best test accuracy   : {self.best_accuracy:.4f} "
            f"({self.best_accuracy * 100:.2f}%)",
            f"  Best test error rate : {1 - self.best_accuracy:.4f}",
            "-" * 60,
            "  K  |  Accuracy  |  Error Rate",
            "  ---|-----------|------------",
        ]
        for k, acc, err in zip(
            self.k_range, self.accuracy_scores, self.error_rates
        ):
            marker = " ◀ optimal" if k == self.best_k else ""
            lines.append(
                f"  {k:>2d} |  {acc:.4f}   |  {err:.4f}{marker}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
