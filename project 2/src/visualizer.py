"""
Visualization Module
====================
Generates publication-quality plots for model diagnostics.

Plots:
    1. Feature Distribution  — Pairwise scatter + histograms (seaborn pairplot).
    2. Elbow Curve           — Error rate vs K to visualise optimal hyperparameter.
    3. Confusion Matrix      — Heatmap with per-cell counts and colour intensity.
    4. Decision Boundary     — 2D projection (top-2 PCA components) with KNN regions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file-based output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Optional


# ── Consistent theme ──────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="deep",
    font_scale=1.1,
    rc={
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.titleweight": "bold",
    },
)

OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


class Visualizer:
    """Static-method collection for all diagnostic visualizations."""

    # ------------------------------------------------------------------
    # 1. Feature Distribution (Pairplot)
    # ------------------------------------------------------------------
    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
        """
        Pairwise scatter matrix coloured by species.

        This reveals class separability per feature pair — e.g., petal
        dimensions cleanly separate setosa from versicolor/virginica.
        """
        g = sns.pairplot(
            df,
            hue="species",
            palette="husl",
            diag_kind="kde",
            plot_kws={"alpha": 0.7, "s": 40, "edgecolor": "white", "linewidth": 0.5},
            diag_kws={"fill": True, "alpha": 0.5},
            corner=False,
        )
        g.figure.suptitle(
            "Iris Feature Distributions by Species",
            y=1.02,
            fontsize=16,
            fontweight="bold",
        )
        if save:
            path = _ensure_output_dir() / "feature_distributions.png"
            g.savefig(path, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # 2. Elbow Curve (Error Rate vs K)
    # ------------------------------------------------------------------
    @staticmethod
    def plot_elbow_curve(
        k_range: range,
        error_rates: List[float],
        best_k: int,
        save: bool = True,
    ) -> None:
        """
        Plot error rate across K values with the optimal K highlighted.

        The 'elbow' is the K where error flattens — beyond it, increasing
        K adds bias without reducing variance.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            list(k_range),
            error_rates,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color="#2196F3",
            markerfacecolor="#1565C0",
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Error Rate",
        )

        # Highlight optimal K
        best_idx = list(k_range).index(best_k)
        ax.axvline(
            x=best_k,
            color="#F44336",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Optimal K = {best_k}",
        )
        ax.scatter(
            [best_k],
            [error_rates[best_idx]],
            s=200,
            color="#F44336",
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )
        ax.annotate(
            f"  K={best_k}\n  Error={error_rates[best_idx]:.4f}",
            xy=(best_k, error_rates[best_idx]),
            fontsize=11,
            fontweight="bold",
            color="#D32F2F",
        )

        ax.set_xlabel("K (Number of Neighbours)", fontsize=13)
        ax.set_ylabel("Error Rate (1 − Accuracy)", fontsize=13)
        ax.set_title(
            "KNN Elbow Curve — Optimal K Selection",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xticks(list(k_range))
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = _ensure_output_dir() / "elbow_curve.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # 3. Confusion Matrix Heatmap
    # ------------------------------------------------------------------
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        target_names: list,
        save: bool = True,
    ) -> None:
        """Annotated heatmap of the confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            linewidths=1.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 16, "weight": "bold"},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title(
            "Confusion Matrix",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )
        fig.tight_layout()

        if save:
            path = _ensure_output_dir() / "confusion_matrix.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # 4. Decision Boundary (PCA 2-D Projection)
    # ------------------------------------------------------------------
    @staticmethod
    def plot_decision_boundary(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        best_k: int,
        target_names: list,
        save: bool = True,
    ) -> None:
        """
        Visualize KNN decision regions on a 2-D PCA projection.

        Since the Iris feature space is 4-D, we project onto the first
        two principal components (capturing ~95% variance) to create
        an interpretable 2-D boundary map.
        """
        # Project to 2-D via PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_test_2d = pca.transform(X_test)

        # Fit a KNN in the reduced space for boundary visualisation
        knn_2d = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2)
        knn_2d.fit(X_train_2d, y_train)

        # Create mesh grid
        h = 0.02  # Step size
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h),
        )

        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Colour maps
        bg_cmap = ListedColormap(["#FFCDD2", "#C8E6C9", "#BBDEFB"])
        point_cmap = ListedColormap(["#D32F2F", "#388E3C", "#1565C0"])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=bg_cmap)
        ax.contour(xx, yy, Z, colors="grey", linewidths=0.5, alpha=0.5)

        # Plot training points
        scatter_train = ax.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=y_train,
            cmap=point_cmap,
            edgecolors="white",
            linewidths=1,
            s=70,
            alpha=0.9,
            label="Train",
        )
        # Plot test points with different marker
        ax.scatter(
            X_test_2d[:, 0],
            X_test_2d[:, 1],
            c=y_test,
            cmap=point_cmap,
            edgecolors="black",
            linewidths=1.5,
            s=100,
            marker="D",
            alpha=0.95,
            label="Test",
        )

        # Legend
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=point_cmap(i),
                markersize=10,
                label=name,
            )
            for i, name in enumerate(target_names)
        ]
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                       markersize=10, label="Train")
        )
        handles.append(
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
                       markersize=10, markeredgecolor="black", label="Test")
        )
        ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

        explained = pca.explained_variance_ratio_
        ax.set_xlabel(
            f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=13
        )
        ax.set_ylabel(
            f"PC2 ({explained[1] * 100:.1f}% variance)", fontsize=13
        )
        ax.set_title(
            f"KNN Decision Boundary (K={best_k}) — PCA Projection",
            fontsize=15,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if save:
            path = _ensure_output_dir() / "decision_boundary.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # Generate all plots at once
    # ------------------------------------------------------------------
    @staticmethod
    def generate_all(
        df: pd.DataFrame,
        k_range: range,
        error_rates: List[float],
        best_k: int,
        cm: np.ndarray,
        target_names: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Convenience method to render the full diagnostic suite."""
        print("\n📊 Generating Visualizations...")
        print("-" * 40)
        Visualizer.plot_feature_distributions(df)
        Visualizer.plot_elbow_curve(k_range, error_rates, best_k)
        Visualizer.plot_confusion_matrix(cm, target_names)
        Visualizer.plot_decision_boundary(
            X_train, y_train, X_test, y_test, best_k, target_names
        )
        print("-" * 40)
        print("📊 All visualizations saved to ./outputs/\n")
