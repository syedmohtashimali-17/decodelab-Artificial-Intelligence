"""
Iris KNN Classifier — Streamlit Interactive Web Application
===========================================================
A modern Web Dashboard with interactive sliders, real-time prediction buttons,
class probability breakdowns, and dynamic diagnostic visualization tabs.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import IrisDataLoader
from src.model import KNNModel
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="Iris KNN Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F0F4F8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data_and_model():
    """Load data and train initial model once."""
    loader = IrisDataLoader(test_size=0.20, random_state=42, shuffle=True)
    model = KNNModel(k_min=1, k_max=20)
    model.tune(loader.X_train, loader.y_train, loader.X_test, loader.y_test)
    return loader, model


def main():
    st.markdown('<div class="main-title">🌸 Iris Flower Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Supervised Learning via K-Nearest Neighbors (Scikit-Learn)</div>', unsafe_allow_html=True)

    loader, model = load_data_and_model()
    target_names = list(loader.target_names)

    # ── Sidebar Configuration ──────────────────────────────────────
    st.sidebar.header("⚙️ Model Controls")
    
    selected_k = st.sidebar.slider(
        "Select K (Neighbours)",
        min_value=1,
        max_value=20,
        value=model.best_k,
        help="Adjust K to see live effect on decision boundaries and accuracy.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Feature Inputs")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35, 0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

    predict_btn = st.sidebar.button("🔮 Predict Species", type="primary", use_container_width=True)

    # ── Main Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Live Prediction",
        "📈 Diagnostic Visualizations",
        "📊 Model Evaluation",
        "ℹ️ Dataset Summary",
    ])

    # ── Tab 1: Live Prediction ────────────────────────────────────
    with tab1:
        st.subheader("Live Prediction & Probability Breakdown")
        
        # Prepare input features
        raw_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled_features = loader.transform_input(raw_features)

        # Get prediction using optimal or user-selected K
        from sklearn.neighbors import KNeighborsClassifier
        active_knn = KNeighborsClassifier(n_neighbors=selected_k, metric="minkowski", p=2)
        active_knn.fit(loader.X_train, loader.y_train)
        
        pred_idx = active_knn.predict(scaled_features)[0]
        probs = active_knn.predict_proba(scaled_features)[0]
        predicted_species = target_names[pred_idx]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Selected Features")
            st.dataframe(
                pd.DataFrame(
                    raw_features,
                    columns=loader.feature_names,
                ),
                use_container_width=True,
            )

            st.success(f"### 🏷️ Predicted Class: **{predicted_species.upper()}**")
            st.caption(f"Evaluated using K = {selected_k} Euclidean Neighbours")

        with col2:
            st.markdown("##### Prediction Confidence")
            for i, name in enumerate(target_names):
                st.write(f"**{name.title()}**")
                st.progress(float(probs[i]), text=f"{probs[i] * 100:.1f}% confidence")

    # ── Tab 2: Visualizations ─────────────────────────────────────
    with tab2:
        st.subheader("Diagnostic Plots & Boundary Maps")
        plot_choice = st.selectbox(
            "Select Diagnostic Plot",
            [
                "Elbow Curve (Error vs K)",
                "Confusion Matrix Heatmap",
                "2D PCA Decision Boundaries",
                "Pairwise Feature Distributions",
            ],
        )

        output_dir = Path("outputs")
        
        if plot_choice == "Elbow Curve (Error vs K)":
            st.image(str(output_dir / "elbow_curve.png"), use_container_width=True)
        elif plot_choice == "Confusion Matrix Heatmap":
            st.image(str(output_dir / "confusion_matrix.png"), use_container_width=True)
        elif plot_choice == "2D PCA Decision Boundaries":
            st.image(str(output_dir / "decision_boundary.png"), use_container_width=True)
        elif plot_choice == "Pairwise Feature Distributions":
            st.image(str(output_dir / "feature_distributions.png"), use_container_width=True)

    # ── Tab 3: Model Evaluation ───────────────────────────────────
    with tab3:
        st.subheader("Classification Performance Metrics")
        
        y_pred = active_knn.predict(loader.X_test)
        evaluator = ModelEvaluator(loader.y_test, y_pred, target_names=target_names)
        perf = evaluator.performance_summary()

        col_a, col_b = st.columns(2)
        col_a.metric("Test Accuracy", f"{perf['accuracy'] * 100:.2f}%")
        col_b.metric("Optimal K (Elbow)", model.best_k)

        st.markdown("##### Detailed Classification Report")
        st.text(evaluator.report_str)

        st.markdown("##### Per-Class Confusion Breakdown (TP / FP / FN / TN)")
        st.text(evaluator.confusion_matrix_detail())

    # ── Tab 4: Dataset Summary ────────────────────────────────────
    with tab4:
        st.subheader("Iris Benchmark Dataset Overview")
        st.text(loader.summary())
        st.markdown("##### Raw Dataset Preview")
        st.dataframe(loader.df, use_container_width=True)


if __name__ == "__main__":
    main()
