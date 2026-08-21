"""
Exploratory Data Analysis (EDA) plotting functions.

Every function here takes the cleaned DataFrame (target already mapped to
0/1 unless noted) and saves one figure to outputs/figures/. These are used
both by main.py (script pipeline) and by the Jupyter notebook, so the EDA
story is identical in both places.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import FIGURES_DIR, NUMERICAL_FEATURES

sns.set_style("whitegrid")


def _savefig(fig, filename):
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_distribution(df, target_col="diabetic_label", filename="class_distribution.png"):
    """
    Bar chart of diabetic vs non-diabetic counts, with percentage labels.
    Expects a human-readable target column (e.g. 'No'/'Yes' strings) so
    the axis labels are self-explanatory.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    for bar, pct in zip(bars, percentages.reindex(counts.index)):
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax.set_title("Diabetic vs Non-Diabetic Class Distribution")
    ax.set_xlabel("Diabetes Status")
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return _savefig(fig, filename)


def plot_numeric_distribution(df, column, target_col="diabetic_label", filename=None):
    """
    For one numerical column, plot a histogram (overall) and a class-wise
    boxplot side by side, so we can see both the overall distribution and
    how it differs between diabetic and non-diabetic patients.
    """
    if filename is None:
        filename = f"{column}_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(data=df, x=column, hue=target_col, kde=True, ax=axes[0], element="step")
    axes[0].set_title(f"{column} Distribution (Histogram)")
    axes[0].set_xlabel(column)

    sns.boxplot(data=df, x=target_col, y=column, ax=axes[1])
    axes[1].set_title(f"{column} by Diabetes Status (Boxplot)")
    axes[1].set_xlabel("Diabetes Status")
    axes[1].set_ylabel(column)

    fig.tight_layout()
    return _savefig(fig, filename)


def plot_categorical_vs_target(df, column, target_col="diabetic_label", filename=None):
    """
    Stacked/grouped bar chart showing how a categorical/binary feature
    relates to the diabetes target (counts by group).
    """
    if filename is None:
        filename = f"{column}_vs_target.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ct = df.groupby([column, target_col]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title(f"{column} vs Diabetes Status")
    ax.set_xlabel(column)
    ax.set_ylabel("Number of Patients")
    ax.legend(title="Diabetes Status")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return _savefig(fig, filename)


def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):
    """
    Correlation heatmap for the numerical clinical measurements only
    (age, pulse_rate, systolic_bp, diastolic_bp, glucose, height, weight,
    bmi). Encoded flags/identifiers are excluded since Pearson correlation
    is not meaningful for them in the same way.
    """
    corr = df[NUMERICAL_FEATURES].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap of Numerical Features")
    fig.tight_layout()
    return _savefig(fig, filename)


def plot_boxplots_outliers(df, filename="outlier_boxplots.png"):
    """Grid of boxplots for all numerical features, used for outlier inspection."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, NUMERICAL_FEATURES):
        sns.boxplot(y=df[col], ax=ax, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("Boxplots of Numerical Features (Outlier Inspection)", fontsize=14)
    fig.tight_layout()
    return _savefig(fig, filename)
