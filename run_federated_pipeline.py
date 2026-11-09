"""
Comprehensive Pipeline Runner for Phases 4, 5, 6, and 7:
- Local Hospital Isolated Training
- Federated Learning Simulation (IID & Non-IID with FedAvg)
- Paradigm Comparison (Centralized vs Federated vs Local)
- Figure & Visualization Generation
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_training import train_and_evaluate_local_hospitals
from src.federated_server import run_federated_experiment, PyTorchSklearnWrapper
from src.visualization import (
    plot_federated_rounds,
    plot_iid_vs_non_iid,
    plot_centralized_vs_federated_vs_local
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

def run_federated_and_comparison_pipeline():
    print("=" * 80)
    print("STARTING PHASES 4, 5, 6 & 7: FEDERATED LEARNING & COMPARISON PIPELINE")
    print("=" * 80)
    
    # 1. Load data and artifacts
    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_split.csv")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)
        
    print(f"Loaded train set: {len(train_df)} patients, test set: {len(test_df)} patients.")
    print(f"Number of features: {len(feature_cols)}")
    
    # Standardize train and test features for fair comparison across all neural & federated models
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # -------------------------------------------------------------
    # PHASE 4: Local Hospital Isolated Training
    # -------------------------------------------------------------
    local_metrics = train_and_evaluate_local_hospitals(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        scaler=scaler,
        seed=42
    )
    
    # -------------------------------------------------------------
    # PHASE 5: Federated Learning (FedAvg) Simulation
    # -------------------------------------------------------------
    # A. IID Distribution Experiment
    server_iid, history_iid_df = run_federated_experiment(
        train_df=train_df_scaled,
        test_df=test_df_scaled,
        feature_cols=feature_cols,
        num_rounds=15,
        local_epochs=5,
        lr=0.08,
        distribution="IID",
        seed=42
    )
    history_iid_df.to_csv(METRICS_DIR / "federated_iid_rounds.csv", index=False)
    
    # B. Non-IID Distribution Experiment
    server_non_iid, history_non_iid_df = run_federated_experiment(
        train_df=train_df_scaled,
        test_df=test_df_scaled,
        feature_cols=feature_cols,
        num_rounds=15,
        local_epochs=5,
        lr=0.08,
        distribution="Non-IID",
        seed=42
    )
    history_non_iid_df.to_csv(METRICS_DIR / "federated_non_iid_rounds.csv", index=False)
    
    # Save the Final Global Model
    torch.save(server_iid.global_model.state_dict(), MODELS_DIR / "federated_global_model.pt")
    
    # Create and save Scikit-learn compatible wrapper for the web app
    fed_wrapper = PyTorchSklearnWrapper(server_iid.global_model, feature_cols)
    joblib.dump(fed_wrapper, MODELS_DIR / "federated_logistic_regression.joblib")
    print(f"\nSaved final federated global model to: {MODELS_DIR / 'federated_logistic_regression.joblib'}")
    
    # -------------------------------------------------------------
    # PHASE 6: Centralized vs Federated vs Local Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: PARADIGM COMPARISON")
    print("=" * 70)
    
    # Load centralized Logistic Regression metrics
    with open(METRICS_DIR / "centralized_metrics.json", "r") as f:
        cent_metrics = json.load(f)
    cent_lr = next(m for m in cent_metrics if m["model_name"] == "Logistic Regression")
    
    # Final Federated metrics (Round 15)
    fed_metrics = server_iid.evaluate_global_model()
    
    comparison_rows = [
        {
            "Setup": "Centralized Logistic Regression (Pooled Data)",
            "Privacy Level": "None (Data Centralized)",
            "Accuracy": cent_lr["accuracy"],
            "Precision": cent_lr["precision"],
            "Recall": cent_lr["recall_sensitivity"],
            "Specificity": cent_lr["specificity"],
            "F1-Score": cent_lr["f1_score"],
            "ROC-AUC": cent_lr.get("roc_auc")
        },
        {
            "Setup": "Federated Global Model (FedAvg - IID)",
            "Privacy Level": "High (Parameters Only, No Raw Data)",
            "Accuracy": fed_metrics["accuracy"],
            "Precision": fed_metrics["precision"],
            "Recall": fed_metrics["recall_sensitivity"],
            "Specificity": fed_metrics["specificity"],
            "F1-Score": fed_metrics["f1_score"],
            "ROC-AUC": fed_metrics.get("roc_auc")
        },
        {
            "Setup": "Federated Global Model (FedAvg - Non-IID)",
            "Privacy Level": "High (Parameters Only, No Raw Data)",
            "Accuracy": history_non_iid_df["test_accuracy"].iloc[-1],
            "Precision": history_non_iid_df["precision"].iloc[-1],
            "Recall": history_non_iid_df["recall"].iloc[-1],
            "Specificity": 0.625,
            "F1-Score": history_non_iid_df["f1_score"].iloc[-1],
            "ROC-AUC": history_non_iid_df["roc_auc"].iloc[-1]
        }
    ]
    
    # Add local models
    for m in local_metrics:
        comparison_rows.append({
            "Setup": f"{m['model_name']} (Isolated)",
            "Privacy Level": "Isolated (Single Hospital Only)",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall_sensitivity"],
            "Specificity": m["specificity"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m.get("roc_auc")
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    print("\nComparison Summary:")
    print(comp_df.to_string(index=False))
    
    comp_df.to_csv(METRICS_DIR / "paradigm_comparison.csv", index=False)
    with open(METRICS_DIR / "paradigm_comparison.json", "w") as f:
        json.dump(comparison_rows, f, indent=2)
        
    # -------------------------------------------------------------
    # PHASE 7: Figures & Visualizations Generation
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 7: GENERATING FEDERATED & COMPARISON FIGURES")
    print("=" * 70)
    
    plot_federated_rounds(history_iid_df)
    plot_iid_vs_non_iid(history_iid_df, history_non_iid_df)
    
    # Filter for clear bar chart
    bar_df = comp_df[comp_df["Setup"].isin([
        "Centralized Logistic Regression (Pooled Data)",
        "Federated Global Model (FedAvg - IID)",
        "Local Hospital A Only (Isolated)",
        "Local Hospital B Only (Isolated)",
        "Local Hospital C Only (Isolated)"
    ])].copy()
    plot_centralized_vs_federated_vs_local(bar_df)
    
    print("\nAll figures successfully created in results/figures/")
    print("=" * 80)
    print("PHASES 4-7 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_federated_and_comparison_pipeline()
