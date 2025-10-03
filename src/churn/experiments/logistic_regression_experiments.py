from __future__ import annotations
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.utils import set_seed, load_splits, choose_threshold_by_f1, make_cv, evaluate_probs, save_artifacts

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel


SEED = 42
set_seed(SEED)

TARGET = "CHURN"
DATA_DIR = "data/processed"
MODELS_DIR = "models"


BEST_LR_PARAMS = {
    "class_weight": None,
    "solver": "liblinear",
    "penalty":"l1",
    "C":1.0,
    "max_iter":1000,
    "random_state":SEED
}
def build_fixed_logreg():
    return LogisticRegression(**BEST_LR_PARAMS)

# ----------------------------- baseline (fixed XGB) -----------------------------
def exp_lr_fixed_baseline(X_train, y_train, X_val, y_val):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", build_fixed_logreg()),
    ])
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    thr = choose_threshold_by_f1(y_val, val_proba)
    run_s = time.perf_counter() - t0
    lr_info = {
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s,
        "note": "Fixed LR with your best params (class_weight=None)."
    }
    return pipe, lr_info

# ----------------------------- feature selection experiments -----------------------------
def exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("logreg", build_fixed_logreg()),
    ])
    n_all = X_train.shape[1]
    param_grid = {"select__k": [min(10, n_all), 20, 25, "all"]}

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
    lr_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s,
    }
    return best, lr_info

def exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    t0 = time.perf_counter()
    rfe_base = build_fixed_logreg()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rfe",  RFE(estimator=rfe_base, n_features_to_select=25, step=0.2)),
        ("logreg", build_fixed_logreg()),
    ])
    n_all = X_train.shape[1]
    param_grid = {
        "rfe__n_features_to_select": [20, 30, n_all]
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
    lr_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s
    }
    return best, lr_info

def exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=5, n_jobs_grid=2):
    t0 = time.perf_counter()
    selector = build_fixed_logreg()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("sfm", SelectFromModel(selector, threshold="median")),
        ("logreg", build_fixed_logreg()),
    ])
    param_grid = {
        "sfm__threshold": ["median", "mean", 0.0]
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
    lr_info = {
        "search_best_params": gs.best_params_,
        "val_metrics": evaluate_probs(y_val, val_proba, thr),
        "threshold": thr,
        "run_seconds": run_s
    }
    return best, lr_info

# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature selection on top of a fixed Logistic Regression.")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to processed CSVs")
    parser.add_argument("--models_dir", default=MODELS_DIR, help="Where to save models/metrics")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--jobs", type=int, default=2, help="n_jobs for GridSearch")
    parser.add_argument("--run_filter",   action="store_true", help="Run SelectKBest filter experiment")
    parser.add_argument("--run_wrapper",  action="store_true", help="Run RFE wrapper experiment")
    parser.add_argument("--run_embedded", action="store_true", help="Run SelectFromModel embedded experiment")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip fitting the fixed LR baseline")
    args = parser.parse_args()

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(args.data_dir)
    print(f"[data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    wall_t0 = time.perf_counter()
    results = {}

    # Fixed XGB baseline
    if not args.skip_baseline:
        print("\n=== Fixed LR baseline (no hyperparam search) ===")
        lr_fixed, info = exp_lr_fixed_baseline(X_train, y_train, X_val, y_val)
        info["test_metrics"] = evaluate_probs(y_test, lr_fixed.predict_proba(X_test)[:, 1], info["threshold"])
        print("[LR-fixed] val:", info["val_metrics"], "\n[LR-fixed] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("lr_fixed", lr_fixed, X_train.columns, info, out_dir=args.models_dir)
        results["lr_fixed"] = info

    # Filter
    if args.run_filter:
        print("\n=== Filter: SelectKBest(f_classif) + Fixed LR ===")
        m, info = exp_filter_selectk(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Filter] val:", info["val_metrics"], "\n[Filter] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("lr_filter_selectk", m, X_train.columns, info, out_dir=args.models_dir)
        results["lr_filter_selectk"] = info

    # Wrapper
    if args.run_wrapper:
        print("\n=== Wrapper: RFE + Fixed LR ===")
        m, info = exp_wrapper_rfe(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Wrapper] val:", info["val_metrics"], "\n[Wrapper] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("lr_wrapper_rfe", m, X_train.columns, info, out_dir=args.models_dir)
        results["lr_wrapper_rfe"] = info

    # Embedded
    if args.run_embedded:
        print("\n=== Embedded: SelectFromModel(L1-LR) + Fixed LR ===")
        m, info = exp_embedded_sfm(X_train, y_train, X_val, y_val, cv_folds=args.cv, n_jobs_grid=args.jobs)
        info["test_metrics"] = evaluate_probs(y_test, m.predict_proba(X_test)[:, 1], info["threshold"])
        print("[Embedded] val:", info["val_metrics"], "\n[Embedded] test:", info["test_metrics"], f"\n[time] {info['run_seconds']:.2f}s")
        save_artifacts("lr_embedded_sfm", m, X_train.columns, info, out_dir=args.models_dir)
        results["lr_embedded_sfm"] = info

    results["_total_wall_seconds"] = time.perf_counter() - wall_t0
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "logreg_fs_fixed_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[summary] written logreg_fs_fixed_summary.json\n[total time] {results['_total_wall_seconds']:.2f}s")

if __name__ == "__main__":
    main()
