from __future__ import annotations
import os
import json
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

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
BEST_XGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "min_child_weight": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
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

def make_cv(n_splits: int, seed: int = SEED):
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def save_artifacts(name: str, model, feature_names, metrics: dict, out_dir: str = MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "features": list(feature_names)}, os.path.join(out_dir, f"{name}.joblib"))
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {name}.joblib and {name}_metrics.json in {out_dir}")

def build_fixed_xgb(overrides: dict | None=None) -> XGBClassifier:
    params = BEST_XGB_PARAMS.copy()
    if overrides:
        cleaned = {k.split("__")[-1]: v for k, v in overrides.items()}
        params.update(cleaned)
    return XGBClassifier(**params)

# ----------------------------- baseline (fixed XGB) -----------------------------
def exp_fixed_xgb_baseline(X_train, y_train, X_val, y_val, overrides=None):
    t0 = time.perf_counter()
    xgb = build_fixed_xgb(overrides)
    xgb.fit(X_train, y_train)
    val_proba = xgb.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    val_metrics = evaluate_probs(y_val, val_proba, thr)
    run_s = time.perf_counter() - t0
    return xgb, {"val_metrics": val_metrics, "threshold": thr, "run_seconds": run_s}


# ----------------------------- feature selection experiments -----------------------------
def exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2, overrides=None):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=25)),
        ("xgb",    build_fixed_xgb(overrides)),
    ])
    param_grid = {
        "select__k": [20, 25, "all"],
    }
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=make_cv(cv_folds),
        n_jobs=n_jobs_grid,
        verbose=2,
        pre_dispatch="1*n_jobs",
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s,
    }


def exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2, overrides=None):
    t0 = time.perf_counter()
    base_for_rfe = build_fixed_xgb(overrides)
    pipe = Pipeline([
        ("rfe",  RFE(estimator=base_for_rfe, n_features_to_select=25, step=0.2)),
        ("xgb",  build_fixed_xgb(overrides)),
    ])
    n_all = X_train.shape[1]
    param_grid = {
        "rfe__n_features_to_select": [20, 30, n_all],
    }
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=make_cv(cv_folds),
        n_jobs=n_jobs_grid,
        verbose=2,
        pre_dispatch="1*n_jobs",
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s,
    }

def exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2, overrides=None):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("sfm", SelectFromModel(build_fixed_xgb(overrides), threshold="median")),
        ("xgb", build_fixed_xgb(overrides)),
    ])
    param_grid = {
        "sfm__threshold": ["median", "mean", 0.0],
    }
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc",
        cv=make_cv(cv_folds), n_jobs=n_jobs_grid, verbose=2, pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    return best, {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s,
    }

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run feature selection experiments with a fixed XGBoost.")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to processed CSVs")
    parser.add_argument("--models_dir", default=MODELS_DIR, help="Where to save models/metrics")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--jobs", type=int, default=2, help="n_jobs for GridSearch")
    parser.add_argument("--run_filter",   action="store_true", help="Run SelectKBest filter experiment")
    parser.add_argument("--run_wrapper",  action="store_true", help="Run RFE wrapper experiment")
    parser.add_argument("--run_embedded", action="store_true", help="Run SelectFromModel embedded experiment")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip fitting the fixed XGB baseline")
    parser.add_argument("--best_json", default="", help="Optional path to tuned JSON with best_params to override")
    args = parser.parse_args()

    overrides = None
    if args.best_json and os.path.exists(args.best_json):
        try:
            with open(args.best_json, "r") as f:
                blob = json.load(f)
            overrides = blob.get("best_params", None)
            if overrides:
                print("[xgb] overriding fixed params from JSON best_params")
        except Exception as e:
            print(f"[warn] failed to read best_json: {e}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    wall_t0 = time.perf_counter()
    results = {}

    # Fixed XGB baseline
    if not args.skip_baseline:
        print("\n=== Fixed XGB (no hyperparam search) ===")
        xgb_fixed, info = exp_fixed_xgb_baseline(X_train, y_train, X_val, y_val, overrides=overrides)
        info["test_metrics"] = evaluate_probs(y_test, xgb_fixed.predict_proba(X_test)[:, 1], info["threshold"])
        print("[XGB-fixed] val:", info["val_metrics"], "\n[XGB-fixed] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("xgb_fixed", xgb_fixed, X_train.columns, info, out_dir=args.models_dir)
        results["xgb_fixed"] = info

    # Filter
    if args.run_filter:
        print("\n=== Filter: SelectKBest(f_classif) + Fixed XGB ===")
        m, info = exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs, overrides=overrides)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Filter] val:", info["val_metrics"], "\n[Filter] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("xgb_filter_selectk", m, X_train.columns, info, out_dir=args.models_dir)
        results["xgb_filter_selectk"] = info

    # Wrapper
    if args.run_wrapper:
        print("\n=== Wrapper: RFE + Fixed XGB ===")
        m, info = exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs, overrides=overrides)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Wrapper] val:", info["val_metrics"], "\n[Wrapper] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("xgb_wrapper_rfe", m, X_train.columns, info, out_dir=args.models_dir)
        results["xgb_wrapper_rfe"] = info

    # Embedded
    if args.run_embedded:
        print("\n=== Embedded: SelectFromModel(XGB) + Fixed XGB ===")
        m, info = exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs, overrides=overrides)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Embedded] val:", info["val_metrics"], "\n[Embedded] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("xgb_embedded_sfm", m, X_train.columns, info, out_dir=args.models_dir)
        results["xgb_embedded_sfm"] = info

    results["_total_wall_seconds"] = time.perf_counter() - wall_t0
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "xgb_fs_experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[summary] written xgb_fs_experiments_summary.json\n[total time] {results['_total_wall_seconds']:.2f}s")

if __name__ == "__main__":
    main()