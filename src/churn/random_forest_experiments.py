from __future__ import annotations
import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

SEED = 42
TARGET = "CHURN"
DATA_DIR = "data/processed"
MODELS_DIR = "models"

# >>> Freeze the best RF params 
BEST_RF_PARAMS = {
    "max_depth": 10,
    "min_samples_leaf": 2,
    "min_samples_split": 10,
    "n_estimators": 500,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# ----------------------------- utils -----------------------------
def load_splits(data_dir: str = DATA_DIR, target: str = TARGET):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val_processed.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

    X_train, y_train = train.drop(columns=[target]), train[target].astype(int)
    X_val,   y_val   = val.drop(columns=[target]),   val[target].astype(int)
    X_test,  y_test  = test.drop(columns=[target]),  test[target].astype(int)
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_probs(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    roc  = roc_auc_score(y_true, probs)
    pr   = average_precision_score(y_true, probs)
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

def make_cv(n_splits: int, seed: int = SEED):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def save_artifacts(name: str, model, feature_names, metrics: dict, out_dir: str = MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": list(feature_names)}, os.path.join(out_dir, f"{name}.joblib"))
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {name}.joblib and {name}_metrics.json in {out_dir}")

def build_fixed_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**BEST_RF_PARAMS)

# ----------------------------- baseline (fixed RF) -----------------------------
def exp_fixed_rf_baseline(X_train, y_train, X_val, y_val):
    """No hyperparam search; fit the frozen RF once to report baseline numbers."""
    rf = build_fixed_rf()
    rf.fit(X_train, y_train)
    val_proba = rf.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    val_metrics = evaluate_probs(y_val, val_proba, thr)
    return rf, {"val_metrics": val_metrics, "threshold": thr}

# ----------------------------- feature selection experiments -----------------------------
def exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    """
    Pipeline: SelectKBest(f_classif) -> Fixed RF
    Grid over ONLY the selector (k). RF params are frozen.
    """
    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=30)),
        ("rf",    build_fixed_rf()),
    ])
    param_grid = {
        "select__k": [20, 30, 39], 
    }
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc",
        cv=make_cv(cv_folds), n_jobs=n_jobs_grid, verbose=1, pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }

def exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    """
    Pipeline: RFE(fixed RF) -> Fixed RF
    Grid over ONLY n_features_to_select in RFE.
    """
    base_rf_for_rfe = build_fixed_rf()
    pipe = Pipeline([
        ("rfe",  RFE(estimator=base_rf_for_rfe, n_features_to_select=30, step=0.2)),
        ("rf",   build_fixed_rf()),
    ])
    param_grid = {
        "rfe__n_features_to_select": [20, 30, 39],
    }
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc",
        cv=make_cv(cv_folds), n_jobs=n_jobs_grid, verbose=1, pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }

def exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    """
    Pipeline: SelectFromModel(fixed RF importance) -> Fixed RF
    Grid over ONLY the threshold in SelectFromModel.
    """
    pipe = Pipeline([
        ("sfm", SelectFromModel(build_fixed_rf(), threshold="median")),
        ("rf",  build_fixed_rf()),
    ])
    param_grid = {
        "sfm__threshold": ["median", "mean", 0.001],
    }
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc",
        cv=make_cv(cv_folds), n_jobs=n_jobs_grid, verbose=1, pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }

def exp_logreg_l1(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    """Optional: L1 Logistic Regression (no FS), handy as a compact embedded selector baseline."""
    pipe = Pipeline([
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", class_weight="balanced", random_state=SEED, max_iter=2000
        ))
    ])
    param_grid = {"clf__C": [0.1, 0.5, 1.0, 2.0]}
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc",
        cv=make_cv(cv_folds), n_jobs=n_jobs_grid, verbose=1, pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
    }

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run feature selection experiments with a fixed RandomForest.")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to processed CSVs")
    parser.add_argument("--models_dir", default=MODELS_DIR, help="Where to save models/metrics")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--jobs", type=int, default=2, help="n_jobs for GridSearch")
    parser.add_argument("--run_filter",   action="store_true", help="Run SelectKBest filter experiment")
    parser.add_argument("--run_wrapper",  action="store_true", help="Run RFE wrapper experiment")
    parser.add_argument("--run_embedded", action="store_true", help="Run SelectFromModel embedded experiment")
    parser.add_argument("--run_l1",       action="store_true", help="Run L1 Logistic Regression baseline")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip fitting the fixed RF baseline")
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    results = {}

    # Fixed RF baseline (no tuning) – quick & useful for comparison
    if not args.skip_baseline:
        print("\n=== Fixed RF (no hyperparam search) ===")
        rf_fixed, info = exp_fixed_rf_baseline(X_train, y_train, X_val, y_val)
        info["test_metrics"] = evaluate_probs(y_test, rf_fixed.predict_proba(X_test)[:, 1], info["threshold"])
        print("[RF-fixed] val:", info["val_metrics"], "\n[RF-fixed] test:", info["test_metrics"])
        save_artifacts("rf_fixed", rf_fixed, X_train.columns, info, out_dir=args.models_dir)
        results["rf_fixed"] = info

    # Filter
    if args.run_filter:
        print("\n=== Filter: SelectKBest(f_classif) + Fixed RF ===")
        m, info = exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Filter] val:", info["val_metrics"], "\n[Filter] test:", info["test_metrics"])
        save_artifacts("rf_filter_selectk", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_filter_selectk"] = info

    # Wrapper
    if args.run_wrapper:
        print("\n=== Wrapper: RFE + Fixed RF ===")
        m, info = exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Wrapper] val:", info["val_metrics"], "\n[Wrapper] test:", info["test_metrics"])
        save_artifacts("rf_wrapper_rfe", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_wrapper_rfe"] = info

    # Embedded
    if args.run_embedded:
        print("\n=== Embedded: SelectFromModel(RF) + Fixed RF ===")
        m, info = exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Embedded] val:", info["val_metrics"], "\n[Embedded] test:", info["test_metrics"])
        save_artifacts("rf_embedded_sfm", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_embedded_sfm"] = info

    # Optional: L1 Logistic
    if args.run_l1:
        print("\n=== L1 Logistic Regression ===")
        m, info = exp_logreg_l1(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[L1] val:", info["val_metrics"], "\n[L1] test:", info["test_metrics"])
        save_artifacts("logreg_l1", m, X_train.columns, info, out_dir=args.models_dir)
        results["logreg_l1"] = info

    # Save summary
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "fs_experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[summary] written fs_experiments_summary.json")

if __name__ == "__main__":
    main()
