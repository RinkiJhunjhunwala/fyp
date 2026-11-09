"""
Model Evaluation Module
Computes comprehensive classification metrics, confusion matrices, ROC curves,
and runtime profiling for centralized, local, and federated models.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes all standard academic evaluation metrics for a binary classification model.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    auc = np.nan
    fpr_curve, tpr_curve = None, None
    if y_prob is not None:
        try:
            # If 2D array with 2 columns, take probability of positive class (col 1)
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            auc = float(roc_auc_score(y_true, y_prob))
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
        except Exception:
            auc = np.nan
            
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fpr_rate),
        "false_negative_rate": float(fnr_rate),
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": float(inference_time_sec),
        "confusion_matrix": cm.tolist(),
        "roc_curve_fpr": fpr_curve.tolist() if fpr_curve is not None else None,
        "roc_curve_tpr": tpr_curve.tolist() if tpr_curve is not None else None,
    }
    return metrics

def format_metrics_table(metrics_list: list) -> pd.DataFrame:
    """Formats a list of metric dictionaries into a clean summary table."""
    records = []
    for m in metrics_list:
        records.append({
            "Model": m["model_name"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall (Sens.)": f"{m['recall_sensitivity']:.4f}",
            "Specificity": f"{m['specificity']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A",
            "TN": m["true_negatives"],
            "FP": m["false_positives"],
            "FN": m["false_negatives"],
            "TP": m["true_positives"],
            "Train Time (s)": f"{m['train_time_sec']:.4f}",
        })
    return pd.DataFrame(records)
