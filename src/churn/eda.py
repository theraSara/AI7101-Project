import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def set_plot_theme(style: str = "whitegrid", context: str="talk", palette: str="Set2", font_scale: float=1.0) -> None:
    sns.set_theme(style=style, context=context, palette=palette)
    sns.set(rc={
        "figure.autolayout": False, 
        "axes.titlesize": "x-large",
        "axes.labelsize": "large",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "legend.title_fontsize": "medium",
        "legend.fontsize": "medium",
    })
    colors = sns.color_palette(palette)

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 10 * font_scale
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    mpl.rcParams["axes.titlesize"] = "x-large"
    mpl.rcParams["axes.labelsize"] = "large"
    mpl.rcParams["xtick.labelsize"] = "medium"
    mpl.rcParams["ytick.labelsize"] = "medium"
    mpl.rcParams["legend.title_fontsize"] = "medium"
    mpl.rcParams["legend.fontsize"] = "medium"

def plot_missing_values(df: pd.DataFrame, top_k: int | None = None):
    missing_values = df.isnull().mean().sort_values(ascending=True)
    if top_k:
        missing_values = missing_values.tail(top_k)
    
    ax = missing_values.plot.barh(figsize=(12, 6), color='skyblue')
    ax.set_title("Missing Values Percentage")
    ax.set_xlabel("Fraction missing")
    ax.set_ylabel("Feature")
    plt.tight_layout()

def plot_distribution(df: pd.DataFrame, target: str):
    ax = df[target].value_counts().plot.bar(figsize=(6, 4), color='skyblue')
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    plt.tight_layout()

def plot_categorical_by_target(df: pd.DataFrame, target: str, categorical: list[str]):
    n = len(categorical)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical):
        sns.countplot(data=df, x=col, hue=target, ax=axes[i],
                      order=df[col].value_counts(dropna=False).index)
        axes[i].set_title(f"Distribution of {col} by {target}")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

        if axes[i].legend_:
            axes[i].legend_.set_title(target)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

def plot_numerical_box(df: pd.DataFrame, target: str, numerical: list[str]):
    n = len(numerical)
    fig, axes = plt.subplots(n, 2, figsize=(16, 6 * n))
    # axes = axes.flatten()
    for i, col in enumerate(numerical):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i, 0])
        axes[i, 0].set_title(f"Distribution of {col}")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Count")

        sns.boxplot(x=target, y=col, data=df, ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} by {target}")
        axes[i, 1].set_xlabel(target)
        axes[i, 1].set_ylabel(col)
    plt.tight_layout()

def plot_pairs(df: pd.DataFrame, target: str, features: list[str]):
    pairs = list(combinations(features, 2))
    n = len(pairs)
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
    axes = axes.flatten()

    for i, (x, y) in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(data=df, x=x, y=y, hue=target, alpha=0.6, ax=ax)
        ax.set_title(f"{y} vs. {x} by {target}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if ax.legend_:
            ax.legend_.set_title(target)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Variables", fontsize=16, fontweight='bold')
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")
    plt.tight_layout()
