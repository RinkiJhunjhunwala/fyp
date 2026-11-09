"""
Federated Server Module (FedAvg Orchestration)
Implements Federated Averaging (FedAvg) algorithm across simulated hospitals.
Coordinates communication rounds, aggregates client parameter updates,
evaluates global convergence on held-out test data, and manages IID vs Non-IID experiments.
"""

from pathlib import Path
import sys
import time
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated_client import PyTorchLogisticRegression, FederatedHospitalClient
from src.evaluation import evaluate_binary_classifier
from src.visualization import (
    plot_federated_rounds,
    plot_iid_vs_non_iid,
    plot_centralized_vs_federated_vs_local
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

class FederatedServer:
    """
    Central Coordinator Server for Federated Learning (FedAvg).
    """
    def __init__(
        self,
        input_dim: int,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.global_model = PyTorchLogisticRegression(input_dim)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        self.y_test_np = y_test
        self.criterion = nn.BCELoss()
        
        # Initialize global model weights (small Gaussian initialization)
        nn.init.normal_(self.global_model.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.global_model.linear.bias)
        
    def get_global_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns copies of current global weights and bias."""
        return (
            self.global_model.linear.weight.data.clone(),
            self.global_model.linear.bias.data.clone()
        )
        
    def aggregate_fedavg(self, client_updates: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Federated Averaging (FedAvg) parameter aggregation.
        Computes sample-weighted average of client weight tensors and biases:
        
        W_global = sum_k (n_k / N) * W_k
        b_global = sum_k (n_k / N) * b_k
        where N = sum_k n_k.
        """
        total_samples = sum(u["num_samples"] for u in client_updates)
        if total_samples == 0:
            return self.get_global_parameters()
            
        avg_weights = torch.zeros_like(self.global_model.linear.weight.data)
        avg_bias = torch.zeros_like(self.global_model.linear.bias.data)
        
        for u in client_updates:
            weight_factor = u["num_samples"] / total_samples
            avg_weights += u["weights"] * weight_factor
            avg_bias += u["bias"] * weight_factor
            
        with torch.no_grad():
            self.global_model.linear.weight.copy_(avg_weights)
            self.global_model.linear.bias.copy_(avg_bias)
            
        return avg_weights, avg_bias

    def evaluate_global_model(self) -> Dict[str, Any]:
        """Evaluates global model on the strictly held-out global test set."""
        self.global_model.eval()
        with torch.no_grad():
            preds_prob = self.global_model(self.X_test).squeeze().numpy()
            loss = self.criterion(
                torch.tensor(preds_prob).unsqueeze(1),
                self.y_test
            ).item()
            
        preds_binary = (preds_prob >= 0.5).astype(int)
        
        metrics = evaluate_binary_classifier(
            y_true=self.y_test_np,
            y_pred=preds_binary,
            y_prob=preds_prob,
            model_name="Federated Global Logistic Regression (FedAvg)"
        )
        metrics["test_loss"] = float(loss)
        return metrics

def partition_data_iid(
    train_df: pd.DataFrame,
    feature_cols: list,
    num_clients: int = 3,
    seed: int = 42
) -> List[FederatedHospitalClient]:
    """
    Partitions training patients into 3 hospitals with balanced (IID) distributions.
    Patient integrity is strictly preserved: each patient's row belongs to exactly one client.
    """
    np.random.seed(seed)
    hospitals = [
        ("hosp_a", "Hospital A (Urban Medical Center)"),
        ("hosp_b", "Hospital B (University Hospital)"),
        ("hosp_c", "Hospital C (Regional Health Center)")
    ]
    
    pos_df = train_df[train_df["ami_during_admission"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg_df = train_df[train_df["ami_during_admission"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    pos_splits = np.array_split(pos_df, num_clients)
    neg_splits = np.array_split(neg_df, num_clients)
    
    clients = []
    for i in range(num_clients):
        hosp_data = pd.concat([pos_splits[i], neg_splits[i]], ignore_index=True)
        hosp_data = hosp_data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        X_local = hosp_data[feature_cols].values.astype(float)
        y_local = hosp_data["ami_during_admission"].values.astype(int)
        p_ids = hosp_data["subject_id"].tolist()
        
        client = FederatedHospitalClient(
            hospital_id=hospitals[i][0],
            hospital_name=hospitals[i][1],
            X_local=X_local,
            y_local=y_local,
            patient_ids=p_ids
        )
        clients.append(client)
        
    return clients

def partition_data_non_iid(
    train_df: pd.DataFrame,
    feature_cols: list,
    seed: int = 42
) -> List[FederatedHospitalClient]:
    """
    Partitions training patients with deliberate label skew (Non-IID):
    - Hospital A (Cardiac Specialty): High positive ratio (e.g. ~75% AMI)
    - Hospital B (General Community): Low positive ratio (e.g. ~20% AMI)
    - Hospital C (Suburban Clinic): Balanced cohort (~50% AMI)
    """
    np.random.seed(seed)
    pos_df = train_df[train_df["ami_during_admission"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg_df = train_df[train_df["ami_during_admission"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # 32 pos, 32 neg total in train set
    # Hospital A: 16 pos, 5 neg (21 patients, 76% pos)
    # Hospital B: 5 pos, 17 neg (22 patients, 23% pos)
    # Hospital C: 11 pos, 10 neg (21 patients, 52% pos)
    
    hosp_a_df = pd.concat([pos_df.iloc[:16], neg_df.iloc[:5]], ignore_index=True)
    hosp_b_df = pd.concat([pos_df.iloc[16:21], neg_df.iloc[5:22]], ignore_index=True)
    hosp_c_df = pd.concat([pos_df.iloc[21:32], neg_df.iloc[22:32]], ignore_index=True)
    
    dfs = [
        ("hosp_a", "Hospital A (Specialty Cardiac - 76% AMI)", hosp_a_df),
        ("hosp_b", "Hospital B (Community Hospital - 23% AMI)", hosp_b_df),
        ("hosp_c", "Hospital C (Regional Clinic - 52% AMI)", hosp_c_df)
    ]
    
    clients = []
    for h_id, h_name, df_h in dfs:
        df_h = df_h.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        X_local = df_h[feature_cols].values.astype(float)
        y_local = df_h["ami_during_admission"].values.astype(int)
        
        client = FederatedHospitalClient(
            hospital_id=h_id,
            hospital_name=h_name,
            X_local=X_local,
            y_local=y_local,
            patient_ids=df_h["subject_id"].tolist()
        )
        clients.append(client)
        
    return clients

def run_federated_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    num_rounds: int = 15,
    local_epochs: int = 5,
    lr: float = 0.05,
    distribution: str = "IID",
    seed: int = 42
) -> Tuple[FederatedServer, pd.DataFrame]:
    """
    Runs a full Federated Learning simulation across multiple communication rounds.
    """
    print(f"\n=======================================================")
    print(f"RUNNING FEDERATED SIMULATION ({distribution.upper()} DISTRIBUTION)")
    print(f"=======================================================")
    
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["ami_during_admission"].values.astype(int)
    
    # Partition clients
    if distribution.upper() == "IID":
        clients = partition_data_iid(train_df, feature_cols, num_clients=3, seed=seed)
    else:
        clients = partition_data_non_iid(train_df, feature_cols, seed=seed)
        
    print("\nClient Partitioning Summary:")
    for c in clients:
        info = c.get_summary()
        print(f"- {info['hospital_name']}: {info['num_patients']} patients ({info['ami_positive']} AMI, {info['ami_control']} Control, Prevalence: {info['prevalence']:.1%})")
        
    server = FederatedServer(
        input_dim=len(feature_cols),
        X_test=X_test,
        y_test=y_test,
        seed=seed
    )
    
    # Evaluate initial round 0
    init_metrics = server.evaluate_global_model()
    print(f"\nRound 0 (Initial Global Model) -> Test Acc: {init_metrics['accuracy']:.4f}, Test Loss: {init_metrics['test_loss']:.4f}")
    
    history_records = [{
        "round": 0,
        "test_accuracy": init_metrics["accuracy"],
        "test_loss": init_metrics["test_loss"],
        "precision": init_metrics["precision"],
        "recall": init_metrics["recall_sensitivity"],
        "f1_score": init_metrics["f1_score"],
        "roc_auc": init_metrics["roc_auc"] or 0.5,
        "round_time_sec": 0.0,
        "comm_overhead_bytes": 0
    }]
    
    total_start_time = time.perf_counter()
    
    for r in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        
        # 1. Server broadcasts global parameters
        global_w, global_b = server.get_global_parameters()
        
        # 2. Local client training
        client_updates = []
        round_bytes = 0
        for client in clients:
            update = client.train_one_round(
                global_weights=global_w,
                global_bias=global_b,
                local_epochs=local_epochs,
                lr=lr,
                batch_size=8
            )
            client_updates.append(update)
            # Both downlink (broadcast) and uplink (update)
            round_bytes += update["update_size_bytes"] * 2
            
        # 3. Server aggregates updates via FedAvg
        server.aggregate_fedavg(client_updates)
        round_time = time.perf_counter() - round_start
        
        # 4. Evaluate new global model on held-out global test set
        eval_metrics = server.evaluate_global_model()
        
        history_records.append({
            "round": r,
            "test_accuracy": eval_metrics["accuracy"],
            "test_loss": eval_metrics["test_loss"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall_sensitivity"],
            "f1_score": eval_metrics["f1_score"],
            "roc_auc": eval_metrics["roc_auc"] or 0.5,
            "round_time_sec": round_time,
            "comm_overhead_bytes": round_bytes
        })
        
        if r % 3 == 0 or r == num_rounds:
            print(f"Round {r:02d}/{num_rounds:02d} -> Test Acc: {eval_metrics['accuracy']:.4f} | F1: {eval_metrics['f1_score']:.4f} | Loss: {eval_metrics['test_loss']:.4f} | Round Time: {round_time:.3f}s")
            
    total_time = time.perf_counter() - total_start_time
    print(f"\nFederated training finished in {total_time:.2f} seconds.")
    
    history_df = pd.DataFrame(history_records)
    return server, history_df

class PyTorchSklearnWrapper:
    """
    Wraps PyTorch Logistic Regression model with standard Scikit-learn interface
    (predict, predict_proba) so that it can be dropped directly into the web application.
    """
    def __init__(self, pytorch_model: PyTorchLogisticRegression, feature_cols: list):
        self.weights = pytorch_model.linear.weight.detach().cpu().numpy().flatten()
        self.bias = float(pytorch_model.linear.bias.detach().cpu().numpy()[0])
        self.feature_cols = feature_cols
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.weights) + self.bias
        prob_1 = 1.0 / (1.0 + np.exp(-logits))
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
