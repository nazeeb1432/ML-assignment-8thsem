"""
Evaluation helpers: metrics table, confusion matrices, and ROC curves.

Because the DiaBD dataset is heavily imbalanced (~94% non-diabetic), we
never rely on accuracy alone. Every model is scored on accuracy,
precision, recall, F1-score, and ROC-AUC, with particular attention to
recall for the diabetic (positive) class - in a screening context, a
false negative (telling a diabetic patient they are healthy) is far more
costly than a false positive.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import FIGURES_DIR


def evaluate_model(name, fitted_pipeline, X_test, y_test) -> dict:
    """
    Compute the standard set of metrics for one fitted pipeline on the
    held-out test set. Returns a flat dict suitable for building a
    comparison DataFrame.
    """
    y_pred = fitted_pipeline.predict(X_test)
    y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics


def build_comparison_table(fitted_pipelines: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every fitted pipeline and return a comparison DataFrame."""
    rows = [evaluate_model(name, pipe, X_test, y_test) for name, pipe in fitted_pipelines.items()]
    results = pd.DataFrame(rows).set_index("Model")
    return results


def print_classification_reports(fitted_pipelines: dict, X_test, y_test):
    """Print the full sklearn classification_report for each model."""
    for name, pipeline in fitted_pipelines.items():
        y_pred = pipeline.predict(X_test)
        print(f"\n--- Classification Report: {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))


def plot_confusion_matrices(fitted_pipelines: dict, X_test, y_test, filename="confusion_matrices.png"):
    """Plot a grid of confusion matrices, one per model, and save to disk."""
    n_models = len(fitted_pipelines)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, pipeline) in zip(axes, fitted_pipelines.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic", "Diabetic"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)

    fig.suptitle("Confusion Matrices by Model", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(fitted_pipelines: dict, X_test, y_test, filename="roc_curves.png"):
    """Plot ROC curves for all models on one figure and save to disk."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, pipeline in fitted_pipelines.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison_bar(comparison_df: pd.DataFrame, filename="model_comparison.png"):
    """Bar chart comparing Accuracy/Precision/Recall/F1/ROC-AUC across models."""
    plot_df = comparison_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison Across Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_xticklabels(plot_df.index, rotation=0)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Feature importance / interpretability
# ---------------------------------------------------------------------------
# NOTE: these show which features the model relies on for its predictions.
# This is association, not causation - we report "associated with model
# predictions", never "causes diabetes".


def get_logistic_regression_importance(fitted_pipeline) -> pd.DataFrame:
    """
    Standardized coefficient magnitudes for Logistic Regression.
    Because the numerical features were scaled with StandardScaler before
    fitting, the coefficients are directly comparable to each other -
    a larger |coefficient| means a stronger association with the
    predicted log-odds of being diabetic.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    coefs = classifier.coef_[0]
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    df["Abs_Coefficient"] = df["Coefficient"].abs()
    return df.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)


def get_tree_importance(fitted_pipeline) -> pd.DataFrame:
    """
    Built-in feature_importances_ for Random Forest / XGBoost.
    Both estimators expose the same attribute, so one function covers both.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame, value_col: str, title: str, filename: str, top_n: int = 14):
    """Horizontal bar chart of the top_n most important features."""
    plot_df = importance_df.head(top_n).sort_values(value_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["Feature"], plot_df[value_col], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
