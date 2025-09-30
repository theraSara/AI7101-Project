import os, json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, average_precision_score
from .utils import set_seed

SEED = 42
set_seed(SEED)

TARGET = 'CHURN'
DATA_DIR = 'data/processed'

def load_data(file_path=DATA_DIR, target=TARGET):
    train = pd.read_csv(os.path.join(file_path, 'train_processed.csv'))
    val = pd.read_csv(os.path.join(file_path, 'val_processed.csv'))
    test = pd.read_csv(os.path.join(file_path, 'test_processed.csv'))
    return train, val, test

def train_random_forest():
    train, val, test = load_data()
    
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )
    
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    val_probas = best_rf.predict_proba(X_val)[:, 1]
    val_preds = (val_probas >= 0.5).astype(int)
    
    metrics = {
        "best_params": grid.best_params_,
        "val": {
            "roc_auc": float(roc_auc_score(y_val, val_probas)),
            "average_precision": float(average_precision_score(y_val, val_probas)),
            "f1_score": float(f1_score(y_val, val_preds)),
            "confusion_matrix": confusion_matrix(y_val, val_preds).tolist(),
            "classification_report": classification_report(y_val, val_preds, output_dict=True)
        }
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/random_forest_model.joblib')
    with open('models/random_forest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Best Hyperparameters:", grid.best_params_)
    print("Validation ROC AUC:", metrics["val"]["roc_auc"])
    print("Validation Average Precision:", metrics["val"]["average_precision"])
    print("Validation F1 Score:", metrics["val"]["f1_score"])
    print("Validation Confusion Matrix:\n", metrics["val"]["confusion_matrix"])
    print("Validation Classification Report:\n", json.dumps(metrics["val"]["classification_report"], indent=2))
    
if __name__ == "__main__":
    train_random_forest()
    