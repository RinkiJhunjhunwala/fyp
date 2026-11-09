"""
Visualization Module for AMI Classification & Federated Learning
Generates high-resolution academic figures:
1. Centralized model performance comparison (Accuracy, Precision, Recall, F1)
2. ROC curves for all models
3. Confusion matrices
4. Feature importances for tree-based models
5. Federated learning convergence across communication rounds
6. IID vs Non-IID comparison
7. Centralized vs Federated vs Local comparison
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for server/script generation
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def plot_centralized_model_comparison(metrics_df: pd.DataFrame, save_path: Path = None):
    """Generates grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    if save_path is None:
        save_path = FIGURES_DIR / "centralized_models_comparison.png"
        
    df = metrics_df.copy()
    metric_cols = ["Accuracy", "Precision", "Recall_Sensitivity", "F1_Score", "ROC_AUC"]
    # Ensure float types
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    melted = df.melt(
        id_vars=["Model"],
        value_vars=[c for c in metric_cols if c in df.columns],
        var_name="Metric",
        value_name="Score"
    )
    # Clean metric names
    melted["Metric"] = melted["Metric"].str.replace("_", " ")
    
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Centralized Machine Learning Model Comparison on Held-Out Test Set", pad=15, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.xlabel("Model Architecture")
    plt.xticks(rotation=25, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curves(models_metrics: List[Dict[str, Any]], save_path: Path = None):
    """Plots combined ROC curves for all models that produced probability outputs."""
    if save_path is None:
        save_path = FIGURES_DIR / "roc_curves_comparison.png"
        
    plt.figure(figsize=(8, 7))
    palette = sns.color_palette("tab10", len(models_metrics))
    
    for i, m in enumerate(models_metrics):
        fpr = m.get("roc_curve_fpr")
        tpr = m.get("roc_curve_tpr")
        auc = m.get("roc_auc")
        name = m.get("model_name", f"Model {i}")
        if fpr is not None and tpr is not None and auc is not None:
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=palette[i], lw=2)
            
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", pad=15, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrices(models_metrics: List[Dict[str, Any]], save_path: Path = None):
    """Plots a multi-panel grid of confusion matrices for evaluated models."""
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrices_grid.png"
        
    n = len(models_metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    
    for idx, m in enumerate(models_metrics):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Control (0)", "AMI (1)"],
            yticklabels=["Control (0)", "AMI (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{m['model_name']}\n(Acc: {m['accuracy']:.1%}, F1: {m['f1_score']:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    # Hide empty subplots if any
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])
        
    plt.suptitle("Confusion Matrices on Held-Out Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_feature_importances(feature_names: list, importances: np.ndarray, model_name: str, top_k: int = 15, save_path: Path = None):
    """Plots top-k feature importances for tree-based models."""
    if save_path is None:
        save_path = FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=top_scores, y=top_features, palette="viridis_r")
    plt.title(f"Top {top_k} Feature Importances — {model_name}", pad=15, fontweight="bold")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Extracted Vital Sign Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_federated_rounds(round_history: pd.DataFrame, save_path: Path = None):
    """Plots Global Test Accuracy and Loss across Federated Communication Rounds."""
    if save_path is None:
        save_path = FIGURES_DIR / "federated_learning_rounds.png"
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color_acc = "#1f77b4"
    ax1.set_xlabel("Federated Communication Round", labelpad=10)
    ax1.set_ylabel("Global Test Accuracy", color=color_acc, labelpad=10)
    line1 = ax1.plot(round_history["round"], round_history["test_accuracy"], marker="o", color=color_acc, lw=2.2, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim([0.0, 1.05])
    
    ax2 = ax1.twinx()
    color_loss = "#d62728"
    ax2.set_ylabel("Global Test Loss (BCE)", color=color_loss, labelpad=10)
    line2 = ax2.plot(round_history["round"], round_history["test_loss"], marker="s", linestyle="--", color=color_loss, lw=2.2, label="Test Loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    
    plt.title("Federated Learning (FedAvg) Convergence Across Rounds", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_iid_vs_non_iid(iid_history: pd.DataFrame, non_iid_history: pd.DataFrame, save_path: Path = None):
    """Compares Federated Convergence under IID vs Non-IID Hospital Data Distributions."""
    if save_path is None:
        save_path = FIGURES_DIR / "federated_iid_vs_non_iid.png"
        
    plt.figure(figsize=(9, 5.5))
    plt.plot(iid_history["round"], iid_history["test_accuracy"], marker="o", color="#2ca02c", lw=2.5, label="IID Client Distribution")
    plt.plot(non_iid_history["round"], non_iid_history["test_accuracy"], marker="^", linestyle="--", color="#ff7f0e", lw=2.5, label="Non-IID Client Distribution (Skewed)")
    
    plt.title("Federated Global Model Accuracy: IID vs Non-IID Hospital Distributions", pad=15, fontweight="bold")
    plt.xlabel("Communication Round")
    plt.ylabel("Held-Out Global Test Accuracy")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_centralized_vs_federated_vs_local(comparison_df: pd.DataFrame, save_path: Path = None):
    """
    Bar chart comparing:
    - Centralized Logistic Regression
    - Federated Global Logistic Regression (FedAvg)
    - Independent Local Models (Hospital A, B, C)
    """
    if save_path is None:
        save_path = FIGURES_DIR / "centralized_vs_federated_vs_local.png"
        
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=comparison_df, x="Setup", y="Accuracy", palette="mako")
    plt.title("Paradigm Comparison: Centralized vs Federated (FedAvg) vs Independent Local Hospitals", pad=15, fontweight="bold")
    plt.ylabel("Accuracy on Held-Out Global Test Set")
    plt.xlabel("Training Strategy")
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), ha="center", fontweight="bold", fontsize=10)
        
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
