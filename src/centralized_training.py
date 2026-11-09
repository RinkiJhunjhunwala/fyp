"""
Centralized Machine Learning Training & Evaluation Module
Trains, tunes, and evaluates 6 traditional ML models:
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Support Vector Machine (SVM)
5. K-Nearest Neighbours (KNN)
6. Gradient Boosting

Enforces strict patient-level stratified splitting, feature scaling fitted only on training data,
stratified cross-validation for hyperparameter tuning, and comprehensive metric evaluation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation import evaluate_binary_classifier, format_metrics_table
from src.visualization import (
    plot_centralized_model_comparison,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importances
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "patient_vital_features_24h.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed random seed for complete reproducibility
RANDOM_SEED = 42

def load_and_split_data(test_size: float = 0.20, seed: int = RANDOM_SEED):
    """
    Loads patient-level features, excludes identifiers, and performs stratified splitting.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Target and identifier separation
    target_col = "ami_during_admission"
    id_cols = ["subject_id", "hadm_id"]
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values
    patient_ids = df["subject_id"].values
    
    # Patient-level stratified split (since 1 row = 1 patient, patient-level is preserved)
    X_train_df, X_test_df, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids, test_size=test_size, stratify=y, random_state=seed
    )
    
    print(f"Data split (seed={seed}):")
    print(f"- Training set: {len(X_train_df)} patients (Pos: {np.sum(y_train==1)}, Neg: {np.sum(y_train==0)})")
    print(f"- Held-out test set: {len(X_test_df)} patients (Pos: {np.sum(y_test==1)}, Neg: {np.sum(y_test==0)})")
    
    # Fit scaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    # Save scaler and feature column metadata
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
        
    # Save the exact train and test split sets for federated and local experiments
    train_df = pd.DataFrame(X_train_df, columns=feature_cols)
    train_df["subject_id"] = ids_train
    train_df[target_col] = y_train
    train_df.to_csv(PROJECT_ROOT / "data" / "processed" / "train_split.csv", index=False)
    
    test_df = pd.DataFrame(X_test_df, columns=feature_cols)
    test_df["subject_id"] = ids_test
    test_df[target_col] = y_test
    test_df.to_csv(PROJECT_ROOT / "data" / "processed" / "test_split.csv", index=False)
    
    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler
    }

def get_model_hyperparameter_grids():
    """Defines candidate models and hyperparameter search grids for Stratified CV."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            {
                "C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "max_depth": [2, 3, 4, 6],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            }
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_SEED),
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
        ),
        "K-Nearest Neighbours": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [30, 50, 100],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0]
            }
        )
    }

def train_and_evaluate_all():
    print("=" * 70)
    print("PHASE 3: CENTRALIZED MACHINE LEARNING TRAINING & EVALUATION")
    print("=" * 70)
    
    split_data = load_and_split_data(test_size=0.20, seed=RANDOM_SEED)
    X_train = split_data["X_train_scaled"]
    X_test = split_data["X_test_scaled"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    feature_cols = split_data["feature_cols"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model_grids = get_model_hyperparameter_grids()
    
    all_metrics = []
    trained_models = {}
    
    print("\nStarting Hyperparameter Tuning and Evaluation across 6 algorithms...")
    
    for name, (base_model, param_grid) in model_grids.items():
        print(f"\n--- Training {name} ---")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        
        t_start = time.perf_counter()
        grid_search.fit(X_train, y_train)
        train_time = time.perf_counter() - t_start
        
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
        
        # Save model artifact
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best_model, MODELS_DIR / filename)
        
        # Inference timing and evaluation on held-out test set
        t_inf_start = time.perf_counter()
        y_pred = best_model.predict(X_test)
        inf_time = time.perf_counter() - t_inf_start
        
        y_prob = None
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)
        elif hasattr(best_model, "decision_function"):
            decision = best_model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision))
            
        metrics = evaluate_binary_classifier(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            train_time_sec=train_time,
            inference_time_sec=inf_time,
            model_name=name
        )
        all_metrics.append(metrics)
        print(f"Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics.get('roc_auc')}")
        
    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_summary_table = format_metrics_table(all_metrics)
    print("\n" + "=" * 70)
    print("FINAL TEST METRICS SUMMARY")
    print("=" * 70)
    print(metrics_summary_table.to_string(index=False))
    
    # Save CSV and JSON
    metrics_summary_table.to_csv(METRICS_DIR / "centralized_metrics.csv", index=False)
    with open(METRICS_DIR / "centralized_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"\nMetrics saved to: {METRICS_DIR / 'centralized_metrics.csv'}")
    
    # Generate Figures
    print("\nGenerating figures...")
    plot_centralized_model_comparison(metrics_summary_table)
    plot_roc_curves(all_metrics)
    plot_confusion_matrices(all_metrics)
    
    # Feature importances for Random Forest and Gradient Boosting
    rf_model = trained_models.get("Random Forest")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        plot_feature_importances(feature_cols, rf_model.feature_importances_, "Random Forest")
        
    gb_model = trained_models.get("Gradient Boosting")
    if gb_model is not None and hasattr(gb_model, "feature_importances_"):
        plot_feature_importances(feature_cols, gb_model.feature_importances_, "Gradient Boosting")
        
    print("=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    return all_metrics

if __name__ == "__main__":
    train_and_evaluate_all()
