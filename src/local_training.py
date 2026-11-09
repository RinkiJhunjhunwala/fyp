"""
Local Hospital Training Module (Baseline Isolated Training)
Trains an independent Logistic Regression model for each hospital (Hospital A, B, C)
using ONLY its local partition of training data.
Evaluates each local model on the shared held-out global test set.
Demonstrates the empirical value of collaborative Federated Learning vs isolated training.
"""

from pathlib import Path
import sys
import time
import json
import joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_binary_classifier, format_metrics_table

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate_local_hospitals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    scaler,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Trains independent local models for 3 hospitals and evaluates each on the global test set.
    """
    print("=" * 70)
    print("PHASE 4: INDEPENDENT LOCAL HOSPITAL TRAINING (ISOLATED BASELINE)")
    print("=" * 70)
    
    np.random.seed(seed)
    
    # Stratified split into 3 hospital cohorts
    pos_df = train_df[train_df["ami_during_admission"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg_df = train_df[train_df["ami_during_admission"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    pos_splits = np.array_split(pos_df, 3)
    neg_splits = np.array_split(neg_df, 3)
    
    hospitals = [
        ("Hospital A", "hosp_a"),
        ("Hospital B", "hosp_b"),
        ("Hospital C", "hosp_c")
    ]
    
    # Prepared held-out test data
    X_test_raw = test_df[feature_cols].values.astype(float)
    X_test_scaled = scaler.transform(X_test_raw)
    y_test = test_df["ami_during_admission"].values.astype(int)
    
    local_metrics = []
    
    for i, (name, hid) in enumerate(hospitals):
        h_data = pd.concat([pos_splits[i], neg_splits[i]], ignore_index=True)
        h_data = h_data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        X_local_raw = h_data[feature_cols].values.astype(float)
        X_local_scaled = scaler.transform(X_local_raw)
        y_local = h_data["ami_during_admission"].values.astype(int)
        
        print(f"\n--- Training {name} ---")
        print(f"Local training cohort: {len(h_data)} patients ({np.sum(y_local==1)} AMI, {np.sum(y_local==0)} Control)")
        
        local_model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        t_start = time.perf_counter()
        local_model.fit(X_local_scaled, y_local)
        train_time = time.perf_counter() - t_start
        
        # Save local model artifact
        joblib.dump(local_model, MODELS_DIR / f"local_{hid}.joblib")
        
        # Evaluate on the global held-out test set
        t_inf_start = time.perf_counter()
        y_pred = local_model.predict(X_test_scaled)
        inf_time = time.perf_counter() - t_inf_start
        y_prob = local_model.predict_proba(X_test_scaled)[:, 1]
        
        m = evaluate_binary_classifier(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            train_time_sec=train_time,
            inference_time_sec=inf_time,
            model_name=f"Local {name} Only"
        )
        local_metrics.append(m)
        print(f"Test Accuracy on Global Test Set: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {m.get('roc_auc')}")
        
    summary_table = format_metrics_table(local_metrics)
    print("\n" + "=" * 70)
    print("LOCAL HOSPITAL PERFORMANCE SUMMARY ON GLOBAL TEST SET")
    print("=" * 70)
    print(summary_table.to_string(index=False))
    
    summary_table.to_csv(METRICS_DIR / "local_hospitals_metrics.csv", index=False)
    with open(METRICS_DIR / "local_hospitals_metrics.json", "w") as f:
        json.dump(local_metrics, f, indent=2)
        
    return local_metrics
