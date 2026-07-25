"""
Data Ingestion & Preprocessing Module
======================================
Handles loading the Iris benchmark dataset and preparing it for model training.

Mathematical Justification — Feature Scaling:
    KNN relies on distance metrics (default: Euclidean distance).
    If features have different scales (e.g., petal length 1–7 cm vs sepal width 2–4 cm),
    larger-scale features dominate the distance calculation, biasing the classifier.

    StandardScaler applies the Z-score transformation:
        z = (x - μ) / σ
    This centres each feature at mean=0 and scales to unit variance (σ=1),
    ensuring each dimension contributes equally to the distance computation.

Stratification Justification:
    Although Iris has perfectly balanced classes (50 each), stratified splitting
    is a best-practice guard that preserves class proportions in both partitions,
    which is critical when deploying on imbalanced real-world datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class IrisDataLoader:
    """
    Encapsulates the full data pipeline: load → inspect → scale → split.

    Attributes
    ----------
    feature_names : list[str]
        Human-readable feature column names.
    target_names : ndarray
        Class label strings (setosa, versicolor, virginica).
    df : pd.DataFrame
        Raw dataset as a labelled DataFrame for EDA.
    X_train, X_test : ndarray
        Scaled feature matrices for training and evaluation.
    y_train, y_test : ndarray
        Target vectors preserving stratified class balance.
    scaler : StandardScaler
        Fitted scaler instance (retained for transforming new predictions).
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        """
        Parameters
        ----------
        test_size : float
            Fraction of data reserved for evaluation (default 20%).
        random_state : int
            Seed for reproducible shuffled splits.
        shuffle : bool
            Whether to shuffle before splitting (recommended for KNN).
        """
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

        # Pipeline execution
        self._raw = load_iris()
        self.feature_names: list[str] = self._raw.feature_names
        self.target_names: np.ndarray = self._raw.target_names

        self.df = self._build_dataframe()
        self.scaler = StandardScaler()
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self._prepare()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataframe(self) -> pd.DataFrame:
        """Create a tidy DataFrame for exploratory analysis."""
        df = pd.DataFrame(self._raw.data, columns=self.feature_names)
        df["target"] = self._raw.target
        df["species"] = df["target"].map(
            {i: name for i, name in enumerate(self.target_names)}
        )
        return df

    def _prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features and perform stratified train/test split.

        Returns
        -------
        X_train_scaled, X_test_scaled, y_train, y_test
        """
        X = self._raw.data
        y = self._raw.target

        # Stratified split preserves class ratios in both partitions
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=y,
        )

        # Fit scaler on training data ONLY to prevent data leakage,
        # then transform both partitions with the same statistics.
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted dataset summary string."""
        lines = [
            "=" * 60,
            "  IRIS DATASET SUMMARY",
            "=" * 60,
            f"  Total samples        : {len(self.df)}",
            f"  Features             : {len(self.feature_names)}",
            f"  Classes              : {len(self.target_names)} "
            f"({', '.join(self.target_names)})",
            f"  Training samples     : {len(self.X_train)} "
            f"({(1 - self.test_size) * 100:.0f}%)",
            f"  Test samples         : {len(self.X_test)} "
            f"({self.test_size * 100:.0f}%)",
            "-" * 60,
            "  Class Distribution (full dataset):",
        ]
        for i, name in enumerate(self.target_names):
            count = (self.df["target"] == i).sum()
            lines.append(f"    {name:>12s} : {count}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def transform_input(self, features: np.ndarray) -> np.ndarray:
        """
        Scale a new observation using the fitted training scaler.

        Parameters
        ----------
        features : ndarray of shape (1, 4)
            Raw feature values [sepal_length, sepal_width, petal_length, petal_width].

        Returns
        -------
        Scaled feature array ready for prediction.
        """
        return self.scaler.transform(features)
