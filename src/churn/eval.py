import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_fscore_support, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

from utils import set_seed

set_seed(42)

def set_eval_theme(style: str = "whitegrid", context: str = "notebook", font_scale: float = 1.0):
    sns.set_theme(style=style, context=context, font_scale=font_scale)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 10
    
    
def load_artifact(path: str):
    
    obj = joblib.load(path)
    
    if isinstance(obj, dict):
        model = obj.get("model")
        features = obj.get("features")
        if model is None:
            raise ValueError("Artifact dict must contain 'model' key.")
        return model, features
    
    features = getattr(obj, "feature_names_in_", None)
    if features is not None:
        features = list(features)
    return obj, features


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return X.reindex(columns=feature_names, fill_value=0)


def get_predictions(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model
    
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        return estimator.decision_function(X).ravel()
    else:
        raise AttributeError("Model must have predict_proba or decision_function")


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold)
    }
    
    return metrics

def plot_evaluation_figures(y_true: np.ndarray, scores: np.ndarray, threshold: float, title_prefix: str = "Validation"):
    set_eval_theme()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, scores, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
    axes[0].set_title(f"ROC Curve ({title_prefix})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    
    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=axes[1])
    axes[1].set_title(f"Precision-Recall Curve ({title_prefix})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    
    # Confusion Matrix
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Blues", ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion Matrix ({title_prefix})\nThreshold={threshold:.2f}", fontsize=12, fontweight='bold')
    axes[2].grid(False)
    
    for text in disp.text_.ravel():
        text.set_fontsize(11)
    
    plt.tight_layout()
    plt.show()
    return fig

def get_feature_importance(model, feature_names: list[str], 
                          plot: bool = False, top: int = 20) -> pd.DataFrame:
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if plot:
        set_eval_theme()
        top_features = df_importance.head(top)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top_features, y="feature", x="importance", color="#4C72B0", ax=ax)
        
        ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.tick_params(axis='y', labelsize=10, pad=8)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.show()
    
    return df_importance


def evaluate_model(model_path: str, 
                  X_val: pd.DataFrame, y_val: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  threshold: str = "f1"):

    model, features = load_artifact(model_path)
    if features is None:
        raise RuntimeError("Cannot determine feature names. Save model with features.")

    X_val_aligned = align_features(X_val, features)
    X_test_aligned = align_features(X_test, features)

    val_scores = get_predictions(model, X_val_aligned)
    test_scores = get_predictions(model, X_test_aligned)

    if threshold == "f1":
        threshold = find_best_threshold(y_val.values, val_scores)
    else:
        threshold = float(threshold)

    val_metrics = compute_metrics(y_val.values, val_scores, threshold)
    test_metrics = compute_metrics(y_test.values, test_scores, threshold)

    val_predictions = (val_scores >= threshold).astype(int)
    test_predictions = (test_scores >= threshold).astype(int)
    val_report = classification_report(y_val, val_predictions, zero_division=0)
    test_report = classification_report(y_test, test_predictions, zero_division=0)

    fig_val = plot_evaluation_figures(y_val, val_scores, threshold, "Validation")
    fig_test = plot_evaluation_figures(y_test, test_scores, threshold, "Test")
    
    figures = {
        "validation": fig_val,
        "test": fig_test
    }
    
    importances = get_feature_importance(model, features)
    
    return {
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_report": val_report,
        "test_report": test_report,
        "figures": figures,
        "importances": importances if not importances.empty else None,
        "features": features,
        "model": model
    }

def evaluate_many(model_paths: dict[str, str],
                 X_val: pd.DataFrame, y_val: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: str = "f1"):
    results = {}
    
    for name, path in model_paths.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print('='*50)
        
        results[name] = evaluate_model(path, X_val, y_val, X_test, y_test, threshold)
        
        val_metrics = results[name]["val_metrics"]
        test_metrics = results[name]["test_metrics"]
        
        print(f"Threshold: {results[name]['threshold']:.3f}")
        print(f"\nValidation Metrics:")
        print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  PR AUC:  {val_metrics['pr_auc']:.4f}")
        print(f"  F1:      {val_metrics['f1']:.4f}")
        print(f"\nTest Metrics:")
        print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  PR AUC:  {test_metrics['pr_auc']:.4f}")
        print(f"  F1:      {test_metrics['f1']:.4f}")
    
    return results