import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


SEED = 42
TARGET = "CHURN"
DATA_DIR = "data/processed"
MODELS_DIR = "models"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
def load_splits(data_dir: str = DATA_DIR, target: str = TARGET):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv")) 
    val   = pd.read_csv(os.path.join(data_dir, "val_processed.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

    X_train, y_train = train.drop(columns=[target]), train[target].astype(int)
    X_val,   y_val   = val.drop(columns=[target]),   val[target].astype(int)
    X_test,  y_test  = test.drop(columns=[target]),  test[target].astype(int)
    return X_train, y_train, X_val, y_val, X_test, y_test

def choose_threshold_by_f1(y_true, probs):
    grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr

def make_cv(n_splits: int, seed: int = SEED):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def evaluate_probs(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    roc = roc_auc_score(y_true, probs)
    pr = average_precision_score(y_true, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"roc_auc": float(roc), "pr_auc": float(pr), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def save_artifacts(name: str, model, feature_names, metrics: dict, out_dir: str = MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": list(feature_names)}, os.path.join(out_dir, f"{name}.joblib"))
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {name}.joblib and {name}_metrics.json in {out_dir}")
