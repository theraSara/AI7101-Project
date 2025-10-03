from __future__ import annotations
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.utils import set_seed, load_splits, choose_threshold_by_f1, make_cv, evaluate_probs, save_artifacts

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel


SEED = 42
set_seed(SEED)

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


def build_fixed_rf():
    return RandomForestClassifier(**BEST_RF_PARAMS)

# ----------------------------- baseline (fixed RF) -----------------------------
def exp_fixed_rf_baseline(X_train, y_train, X_val, y_val):
    t0 = time.perf_counter()
    rf = build_fixed_rf()
    rf.fit(X_train, y_train)
    val_proba = rf.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    val_metrics = evaluate_probs(y_val, val_proba, thr)
    run_s = time.perf_counter() - t0
    rf_info = {
        "val_metrics": val_metrics, 
        "threshold": thr, 
        "run_seconds":run_s
    }
    return rf, rf_info

# ----------------------------- feature selection experiments -----------------------------
def exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=25)),
        ("rf",    build_fixed_rf()),
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
        pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    rf_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s
    }
    return best, rf_info

def exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):  
    t0 = time.perf_counter()
    base_rf_for_rfe = build_fixed_rf()
    pipe = Pipeline([
        ("rfe",  RFE(estimator=base_rf_for_rfe, n_features_to_select=25, step=0.2)),
        ("rf",   build_fixed_rf()),
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
        pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    rf_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s
    }
    return best, rf_info

def exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("sfm", SelectFromModel(build_fixed_rf(), threshold="median")),
        ("rf",  build_fixed_rf()),
    ])
    param_grid = {
        "sfm__threshold": ["median", "mean", 0.001],
    }
    gs = GridSearchCV(
        pipe, 
        param_grid=param_grid, 
        scoring="roc_auc",
        cv=make_cv(cv_folds), 
        n_jobs=n_jobs_grid, 
        verbose=2, 
        pre_dispatch="1*n_jobs"
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    val_proba = best.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    rf_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s
    }
    return best, rf_info

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
    parser.add_argument("--skip_baseline", action="store_true", help="Skip fitting the fixed RF baseline")
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    wall_t0 = time.perf_counter()
    results = {}

    # Fixed RF baseline
    if not args.skip_baseline:
        print("\n=== Fixed RF (no hyperparam search) ===")
        rf_fixed, info = exp_fixed_rf_baseline(X_train, y_train, X_val, y_val)
        info["test_metrics"] = evaluate_probs(y_test, rf_fixed.predict_proba(X_test)[:, 1], info["threshold"])
        print("[RF-fixed] val:", info["val_metrics"], "\n[RF-fixed] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("rf_fixed", rf_fixed, X_train.columns, info, out_dir=args.models_dir)
        results["rf_fixed"] = info

    # Filter
    if args.run_filter:
        print("\n=== Filter: SelectKBest(f_classif) + Fixed RF ===")
        m, info = exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Filter] val:", info["val_metrics"], "\n[Filter] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("rf_filter_selectk", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_filter_selectk"] = info

    # Wrapper
    if args.run_wrapper:
        print("\n=== Wrapper: RFE + Fixed RF ===")
        m, info = exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Wrapper] val:", info["val_metrics"], "\n[Wrapper] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("rf_wrapper_rfe", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_wrapper_rfe"] = info

    # Embedded
    if args.run_embedded:
        print("\n=== Embedded: SelectFromModel(RF) + Fixed RF ===")
        m, info = exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Embedded] val:", info["val_metrics"], "\n[Embedded] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("rf_embedded_sfm", m, X_train.columns, info, out_dir=args.models_dir)
        results["rf_embedded_sfm"] = info

    results["_total_wall_seconds"] = time.perf_counter() - wall_t0
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "fs_experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[summary] written fs_experiments_summary.json\n[total time] {results['_total_wall_seconds']:.2f}s")

if __name__ == "__main__":
    main()
