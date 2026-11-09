"""
Federated Hospital Client Module
Simulates local hospital environments (Hospital A, Hospital B, Hospital C).
Trains local Logistic Regression model on strictly private patient data.
Returns only parameter updates (weights and bias) to the central server,
ensuring raw patient records NEVER leave the hospital.
"""

from pathlib import Path
import sys
import copy
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PyTorchLogisticRegression(nn.Module):
    """
    Logistic Regression implemented in PyTorch for mathematically valid parameter averaging.
    Input dimension: d features
    Output: Sigmoid probability in [0, 1]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

class FederatedHospitalClient:
    """
    Simulated Hospital client node participating in Federated Learning.
    Each client owns its local patient records.
    """
    def __init__(
        self,
        hospital_id: str,
        hospital_name: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        patient_ids: list = None
    ):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name
        self.X_local = X_local
        self.y_local = y_local
        self.patient_ids = patient_ids or []
        self.num_samples = len(y_local)
        self.input_dim = X_local.shape[1]
        
        # Local PyTorch dataset and model
        self.tensor_x = torch.tensor(X_local, dtype=torch.float32)
        self.tensor_y = torch.tensor(y_local, dtype=torch.float32).unsqueeze(1)
        self.dataset = TensorDataset(self.tensor_x, self.tensor_y)
        
        self.model = PyTorchLogisticRegression(self.input_dim)
        self.criterion = nn.BCELoss()
        
    def get_summary(self) -> Dict[str, Any]:
        """Returns non-sensitive metadata about the client cohort."""
        pos = int(np.sum(self.y_local == 1))
        neg = int(np.sum(self.y_local == 0))
        return {
            "hospital_id": self.hospital_id,
            "hospital_name": self.hospital_name,
            "num_patients": self.num_samples,
            "ami_positive": pos,
            "ami_control": neg,
            "prevalence": float(pos / self.num_samples) if self.num_samples > 0 else 0.0
        }
        
    def train_one_round(
        self,
        global_weights: torch.Tensor,
        global_bias: torch.Tensor,
        local_epochs: int = 5,
        lr: float = 0.05,
        batch_size: int = 8,
        weight_decay: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Executes local training for one federated communication round.
        
        1. Receive global parameters from central coordinator.
        2. Set local model weights and bias to global values.
        3. Train for E local epochs using local Stochastic Gradient Descent.
        4. Return updated weights, bias, sample count, and training loss.
        
        PRIVACY GUARANTEE:
        Only numerical tensors (weights, bias) and scalar metadata leave this method.
        Raw records remain strictly within self.X_local and self.y_local.
        """
        # Load global model state
        with torch.no_grad():
            self.model.linear.weight.copy_(global_weights)
            self.model.linear.bias.copy_(global_bias)
            
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        epoch_losses = []
        
        for epoch in range(local_epochs):
            batch_losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses) if batch_losses else 0.0)
            
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        
        # Package updates (cloned to prevent in-place memory aliasing)
        return {
            "hospital_id": self.hospital_id,
            "weights": self.model.linear.weight.data.clone(),
            "bias": self.model.linear.bias.data.clone(),
            "num_samples": self.num_samples,
            "local_loss": avg_loss,
            # Update size in bytes (weight tensor + bias tensor)
            "update_size_bytes": (self.model.linear.weight.nelement() + self.model.linear.bias.nelement()) * 4
        }
