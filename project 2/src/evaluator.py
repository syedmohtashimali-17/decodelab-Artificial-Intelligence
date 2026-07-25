"""
Model Evaluation & Diagnostic Suite
====================================
Generates comprehensive classification diagnostics beyond plain accuracy.

Metrics Explained:
    • Precision  = TP / (TP + FP) — Of all predicted positives, how many are correct?
    • Recall     = TP / (TP + FN) — Of all actual positives, how many did we find?
    • F1-Score   = 2 · (P · R) / (P + R) — Harmonic mean balancing precision & recall.
    • Support    = Number of true instances per class.
    • Confusion Matrix — Rows = actual class, Columns = predicted class.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from typing import Optional


class ModelEvaluator:
    """
    Generates diagnostic reports for a trained classifier.

    Attributes
    ----------
    y_true : ndarray
        Ground-truth labels.
    y_pred : ndarray
        Predicted labels.
    target_names : list[str]
        Human-readable class names.
    cm : ndarray
        Confusion matrix (n_classes × n_classes).
    report_str : str
        Formatted classification report.
    accuracy : float
        Overall accuracy.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = (
            list(target_names) if target_names is not None else None
        )

        # Compute diagnostics eagerly
        self.cm = confusion_matrix(y_true, y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.report_str = classification_report(
            y_true,
            y_pred,
            target_names=self.target_names,
            digits=4,
        )
        self.precision, self.recall, self.f1, self.support = (
            precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
        )

    def confusion_matrix_detail(self) -> str:
        """
        Return a per-class breakdown of TP, FP, FN, TN.

        For multi-class problems, TP/FP/FN/TN are computed in a
        one-vs-rest fashion for each class.
        """
        n_classes = len(self.cm)
        names = self.target_names or [str(i) for i in range(n_classes)]

        lines = [
            "=" * 60,
            "  CONFUSION MATRIX — PER-CLASS BREAKDOWN",
            "=" * 60,
        ]

        # Pretty-print the raw matrix
        header = "           " + "  ".join(f"{n:>10s}" for n in names)
        lines.append(f"  Predicted → {header}")
        lines.append(f"  Actual ↓")
        for i, name in enumerate(names):
            row = "  ".join(f"{val:>10d}" for val in self.cm[i])
            lines.append(f"    {name:>12s}  {row}")

        lines.append("-" * 60)
        lines.append("  One-vs-Rest TP / FP / FN / TN:")
        lines.append("  " + "-" * 50)

        for i, name in enumerate(names):
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp
            tn = self.cm.sum() - tp - fp - fn
            lines.append(
                f"    {name:>12s}  TP={tp:>3d}  FP={fp:>3d}  "
                f"FN={fn:>3d}  TN={tn:>3d}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def full_report(self) -> str:
        """Return the complete diagnostic report as a formatted string."""
        lines = [
            self.confusion_matrix_detail(),
            "",
            "=" * 60,
            "  CLASSIFICATION REPORT",
            "=" * 60,
            self.report_str,
            "=" * 60,
            f"  Overall Accuracy: {self.accuracy:.4f} "
            f"({self.accuracy * 100:.2f}%)",
            "=" * 60,
        ]
        return "\n".join(lines)

    def performance_summary(self) -> dict:
        """Return key metrics as a dictionary for programmatic use."""
        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.cm.tolist(),
            "per_class": {
                name: {
                    "precision": float(self.precision[i]),
                    "recall": float(self.recall[i]),
                    "f1_score": float(self.f1[i]),
                    "support": int(self.support[i]),
                }
                for i, name in enumerate(
                    self.target_names or range(len(self.cm))
                )
            },
        }
