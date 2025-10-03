import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def set_plot_theme(style="whitegrid", context="talk", font_scale=1.2):
    sns.set_theme(style=style, context=context, font_scale=font_scale)
    # define consistent binary palette
    colors = ["#4C72B0", "#DD8452"]  # blue, orange
    import matplotlib as mpl
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    return colors


def plot_missing_values(df: pd.DataFrame, top_k: int | None = None, colors=None):
    missing_values = df.isnull().mean().sort_values(ascending=False) * 100
    if top_k:
        missing_values = missing_values.head(top_k)
        
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=missing_values.values, y=missing_values.index, palette=colors, ax=ax)
    ax.set_title("Missing Values (%)")
    ax.set_xlabel("Percentage missing")
    ax.set_ylabel("Feature")
    plt.tight_layout()

def plot_distribution(df: pd.DataFrame, target: str, use_pie=False, colors=None):
    counts = df[target].value_counts()
    if use_pie:
        plt.figure(figsize=(5,5))
        plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors)
        plt.title(f"Distribution of {target}")
    else:
        plt.figure(figsize=(6,4))
        sns.barplot(x=counts.index, y=counts.values, palette=colors)
        plt.title(f"Distribution of {target}")
        plt.xlabel(target)
        plt.ylabel("Count")
    plt.tight_layout()

def plot_categoricals_by_target(df: pd.DataFrame, cat_features: list[str], target: str, colors=None):
    n = len(cat_features)
    cols = 4
    valid_features = [c for c in cat_features if c in df.columns]
    if not valid_features:
        print("No valid categorical columns to plot.")
        return

    rows = math.ceil(len(valid_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_features):
        sns.countplot(
            data=df, x=col, hue=target, palette=colors, ax=axes[i],
            order=df[col].value_counts(dropna=False).index
        )
        axes[i].set_title(f"{col} by {target}")
        axes[i].tick_params(axis="x", rotation=45)
        if axes[i].legend_:
            axes[i].legend_.set_title(target)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_numerical_box(df, target: str, numerical: list[str], colors=None):
    fig, axes = plt.subplots(len(numerical), 2, figsize=(16, 4*len(numerical)))
    for i, col in enumerate(numerical):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i,0], color=colors[0])
        axes[i,0].set_title(f"Distribution of {col}")
        sns.boxplot(x=target, y=col, data=df, ax=axes[i,1], palette=colors)
        axes[i,1].set_title(f"{col} by {target}")
    plt.tight_layout()

def plot_pairs(df: pd.DataFrame, target: str, features: list[str], colors=None):
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