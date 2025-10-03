from __future__ import annotations
import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

SEED = 42
TARGET = "CHURN"
DATA_DIR = "data/processed"
MODELS_DIR = "models"

# utils 
def load_splits(data_dir: str = DATA_DIR, target: str = TARGET):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv")) 
    val   = pd.read_csv(os.path.join(data_dir, "val_processed.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

    X_train, y_train = train.drop(columns=[target]), train[target].astype(int)
    X_val,   y_val   = val.drop(columns=[target]), val[target].astype(int)
    X_test,  y_test  = test.drop(columns=[target]), test[target].astype(int)
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_probs(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    roc = roc_auc_score(y_true, probs)
    pr = average_precision_score(y_true, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"roc_auc": float(roc), "pr_auc": float(pr), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def choose_threshold_by_f1(y_true, probs):
    grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr

def make_cv(n_splits, seed=SEED):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def save_artifacts(name: str, model, feature_names, metrics: dict, out_dir: str = MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": list(feature_names)}, os.path.join(out_dir, f"{name}.joblib"))
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {name}.joblib and {name}_metrics.json in {out_dir}")

def build_logreg():
    model = LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        max_iter=1000,
    )
    return model

def exp_logreg_tuned(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(
            random_state=SEED,
            max_iter=1000
        )),
    ])
    
    param_grid = [
        {
            "logreg__solver": ["liblinear"], 
            "logreg__penalty": ["l1", "l2"], 
            "logreg__C": [0.1, 0.5, 1.0, 2.0, 5.0], 
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}]
        },
        {
            "logreg__solver": ["lbfgs"],
            "logreg__penalty": ["l2"],
            "logreg__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}],
            "logreg__n_jobs": [-1]  
        },
        {
            "logreg__solver": ["saga"],
            "logreg__penalty": ["l1", "l2"],
            "logreg__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}],
            "logreg__n_jobs": [-1]
        }
    ]
    
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=make_cv(cv_folds),
        n_jobs=n_jobs_grid,
        verbose=2,
        pre_dispatch="2*n_jobs"
    )
    
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    val_proba = best.predict_proba(X_val)[:,1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    
    best_param = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }
    return best, best_param

# ----------------------------- feature selection experiments -----------------------------
def exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("logreg", build_logreg()),
    ])
    
    n_all = X_train.shape[1]
    param_grid = [
        {"select__k": [min(10, n_all), 20, 25, "all"]},
        {
            "select__k": [min(10, n_all), 20, 25, "all"],
            "logreg__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}],
        },
    ]
    
    grid = GridSearchCV(
        pipe, 
        param_grid=param_grid, 
        scoring="roc_auc",
        cv=make_cv(cv_folds), 
        n_jobs=n_jobs_grid, 
        verbose=2, 
        pre_dispatch="1*n_jobs"
    )
    
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    best_param = {
        "search_best_params": grid.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }
    
    return best, best_param

def exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):  
    rfe_base = LogisticRegression(
        solver = "liblinear",
        penalty="l2",
        class_weight="balanced",
        random_state=SEED,
        max_iter=1000
    )
    
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rfe",  RFE(estimator=rfe_base, n_features_to_select=25, step=0.2)),
        ("logreg",   build_logreg()),
    ])
    
    n_all = X_train.shape[1]
    param_grid = [
        {"rfe__n_features_to_select": [10, 20, n_all]}, 
        {  
            "rfe__n_features_to_select": [10, 20, n_all],
            "logreg__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}],
        },
    ]
    
    grid = GridSearchCV(
        pipe, 
        param_grid=param_grid, 
        scoring="roc_auc",
        cv=make_cv(cv_folds), 
        n_jobs=n_jobs_grid, 
        verbose=2,
        pre_dispatch="1*n_jobs"
    )
    
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    best_param = {
        "search_best_params": grid.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }
    
    return best, best_param

def exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    selector = LogisticRegression(
        penalty="l1", solver="liblinear", class_weight="balanced",
        random_state=SEED, max_iter=1000, C=1.0
    )
    
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("sfm", SelectFromModel(selector, threshold="median")),
        ("logreg",  build_logreg()),
    ])
    
    param_grid = [
        {"sfm__threshold": ["median", "mean", 0.0]},
        {
            "sfm__threshold": ["median", "mean", 0.0],
            "logreg__C": [0.01, 0.1, 1.0, 2.0],
            "logreg__class_weight": [None, "balanced", {0:1, 1:3}, {0:1, 1:5}],
        },
    ]
    
    grid = GridSearchCV(
        pipe, 
        param_grid=param_grid, 
        scoring="roc_auc",
        cv=make_cv(cv_folds), 
        n_jobs=n_jobs_grid, 
        verbose=2, 
        pre_dispatch="1*n_jobs"
    )
    
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    best_param = {
        "search_best_params": grid.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }
    
    return best, best_param

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run feature selection experiments with a fixed Logistic Regression.")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to processed CSVs")
    parser.add_argument("--models_dir", default=MODELS_DIR, help="Where to save models/metrics")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--jobs", type=int, default=2, help="n_jobs for GridSearch")
    parser.add_argument("--run_filter",   action="store_true", help="Run SelectKBest filter experiment")
    parser.add_argument("--run_wrapper",  action="store_true", help="Run RFE wrapper experiment")
    parser.add_argument("--run_embedded", action="store_true", help="Run SelectFromModel embedded experiment")
    parser.add_argument("--run_l1",       action="store_true", help="Run L1 Logistic Regression baseline")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip fitting the fixed LR baseline")
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    results = {}

    # Fixed LR baseline (no tuning) – quick & useful for comparison
    if not args.skip_baseline:
        print("\n=== Fixed LR (no hyperparam search) ===")
        LR_fixed, info = exp_logreg_tuned(X_train, y_train, X_val, y_val)
        info["test_metrics"] = evaluate_probs(y_test, LR_fixed.predict_proba(X_test)[:, 1], info["threshold"])
        print("[LR-tuned] val:", info["val_metrics"], "\n[LR-tuned] test:", info["test_metrics"])
        save_artifacts("LR_tuned", LR_fixed, X_train.columns, info, out_dir=args.models_dir)
        results["LR_tuned"] = info

    # Filter
    if args.run_filter:
        print("\n=== Filter: SelectKBest(f_classif) + LR ===")
        m, info = exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Filter] val:", info["val_metrics"], "\n[Filter] test:", info["test_metrics"])
        save_artifacts("LR_filter_selectk", m, X_train.columns, info, out_dir=args.models_dir)
        results["LR_filter_selectk"] = info

    # Wrapper
    if args.run_wrapper:
        print("\n=== Wrapper: RFE + LR ===")
        m, info = exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Wrapper] val:", info["val_metrics"], "\n[Wrapper] test:", info["test_metrics"])
        save_artifacts("LR_wrapper_rfe", m, X_train.columns, info, out_dir=args.models_dir)
        results["LR_wrapper_rfe"] = info

    # Embedded
    if args.run_embedded:
        print("\n=== Embedded: SelectFromModel(LR) + LR ===")
        m, info = exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Embedded+LR] val:", info["val_metrics"], "\n[Embedded+LR] test:", info["test_metrics"])
        save_artifacts("LR_embedded_sfm", m, X_train.columns, info, out_dir=args.models_dir)
        results["LR_embedded_sfm"] = info

    # Save summary
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "logreg_experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[summary] written logreg_experiments_summary.json")

if __name__ == "__main__":
    main()
