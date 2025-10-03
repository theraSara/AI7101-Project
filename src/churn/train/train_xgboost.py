import os, json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, average_precision_score
from ..utils import set_seed

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier


SEED = 42
set_seed(SEED)

TARGET = 'CHURN'
DATA_DIR = 'data/processed'

def load_data(file_path=DATA_DIR, target=TARGET):
    train = pd.read_csv(os.path.join(file_path, 'train_processed.csv'))
    val = pd.read_csv(os.path.join(file_path, 'val_processed.csv'))
    test = pd.read_csv(os.path.join(file_path, 'test_processed.csv'))
    return train, val, test

def train_xgboost():
    train, val, test = load_data()
    
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

    # pipeline
    pipe = Pipeline([
        ('select', SelectKBest(score_func=f_classif, k=39)),
        ('xgb', XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',
            scale_pos_weight=4,
            n_jobs=-1,
            random_state=42
        ))
    ])


    param_grid = {
        'xgb__learning_rate': [0.1, 0.2],
        'xgb__n_estimators': [100, 200, 500],
        'xgb__max_depth': [3, 5, 7],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=2
    )


    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    val_probas = best_xgb.predict_proba(X_val)[:, 1]
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
    joblib.dump(best_xgb, 'models/xgboost_model.joblib')
    with open('models/xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Best Hyperparameters:", grid.best_params_)
    print("Validation ROC AUC:", metrics["val"]["roc_auc"])
    print("Validation Average Precision:", metrics["val"]["average_precision"])
    print("Validation F1 Score:", metrics["val"]["f1_score"])
    print("Validation Confusion Matrix:\n", metrics["val"]["confusion_matrix"])
    print("Validation Classification Report:\n", json.dumps(metrics["val"]["classification_report"], indent=2))
    
if __name__ == "__main__":
    train_xgboost()
    