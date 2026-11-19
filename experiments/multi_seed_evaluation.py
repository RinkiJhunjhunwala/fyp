import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import contextlib
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation import evaluate_binary_classifier
from src.local_training import train_and_evaluate_local_hospitals
from src.federated_server import run_federated_experiment

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "patient_vital_features_24h.csv"
SEEDS = [42, 101, 123, 256, 512, 777, 888, 999, 1234, 2026]

def run_multi_seed_experiments():
    df = pd.read_csv(DATA_PATH)
    target_col = "ami_during_admission"
    id_cols = ["subject_id", "hadm_id"]
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    
    X_full = df[feature_cols].copy()
    y_full = df[target_col].astype(int).values
    p_ids_full = df["subject_id"].values
    
    # Storage for results across seeds
    # models to track
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Decision Tree": DecisionTreeClassifier(max_depth=3),
        "Support Vector Machine": SVC(kernel="rbf", probability=True),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3)
    }
    
    cent_results = {m: [] for m in models_dict}
    local_results = {"Hospital A": [], "Hospital B": [], "Hospital C": []}
    fed_iid_results = []
    fed_non_iid_results = []
    
    print(f"Running multi-seed experiments across {len(SEEDS)} seeds: {SEEDS}")
    
    for seed in SEEDS:
        print(f"\n---> Evaluating Seed {seed}...")
        
        # Split 80% train (64), 20% test (16)
        X_train_df, X_test_df, y_train, y_test, ids_train, ids_test = train_test_split(
            X_full, y_full, p_ids_full, test_size=0.20, stratify=y_full, random_state=seed
        )
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_df)
        X_test_s = scaler.transform(X_test_df)
        
        # 1. Centralized models
        for m_name, model in models_dict.items():
            model.set_params(random_state=seed) if hasattr(model, "random_state") else None
            # Quick tuning for Logistic Regression C if needed, or fit standard
            if m_name == "Logistic Regression":
                model.set_params(C=1.0, penalty="l2", solver="lbfgs")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred
            metrics = evaluate_binary_classifier(y_test, y_pred, y_prob, model_name=m_name)
            cent_results[m_name].append(metrics)
            
        # Build scaled train & test DataFrames for local & federated
        train_df_scaled = pd.DataFrame(X_train_s, columns=feature_cols)
        train_df_scaled["subject_id"] = ids_train
        train_df_scaled["ami_during_admission"] = y_train
        
        test_df_scaled = pd.DataFrame(X_test_s, columns=feature_cols)
        test_df_scaled["subject_id"] = ids_test
        test_df_scaled["ami_during_admission"] = y_test
        
        # 2. Local Hospital Training
        with contextlib.redirect_stdout(io.StringIO()):
            local_evals = train_and_evaluate_local_hospitals(
                train_df=train_df_scaled,
                test_df=test_df_scaled,
                feature_cols=feature_cols,
                scaler=scaler,
                seed=seed
            )
            for h_eval in local_evals:
                h_id = h_eval["model_name"]
                if "Hospital A" in h_id:
                    local_results["Hospital A"].append(h_eval)
                elif "Hospital B" in h_id:
                    local_results["Hospital B"].append(h_eval)
                elif "Hospital C" in h_id:
                    local_results["Hospital C"].append(h_eval)
                    
            # 3. Federated IID
            server_iid, hist_iid = run_federated_experiment(
                train_df=train_df_scaled,
                test_df=test_df_scaled,
                feature_cols=feature_cols,
                num_rounds=15,
                local_epochs=5,
                lr=0.05,
                distribution="IID",
                seed=seed
            )
            m_iid = server_iid.evaluate_global_model()
            fed_iid_results.append(m_iid)
            
            # 4. Federated Non-IID
            server_non_iid, hist_non_iid = run_federated_experiment(
                train_df=train_df_scaled,
                test_df=test_df_scaled,
                feature_cols=feature_cols,
                num_rounds=15,
                local_epochs=5,
                lr=0.05,
                distribution="Non-IID",
                seed=seed
            )
            m_non_iid = server_non_iid.evaluate_global_model()
            fed_non_iid_results.append(m_non_iid)

    # Summarize metrics
    def summarize(metric_list, name):
        df_m = pd.DataFrame(metric_list)
        summary = {"Model": name}
        for col in ["accuracy", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]:
            vals = df_m[col].values * 100 if col != "roc_auc" else df_m[col].values
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if col == "roc_auc":
                summary[col] = f"{mean_v:.4f} ± {std_v:.4f}"
            else:
                summary[col] = f"{mean_v:.2f}% ± {std_v:.2f}%"
        # Rename for cleaner output
        summary["sensitivity"] = summary.pop("recall_sensitivity")
        return summary

    print("\n" + "="*80)
    print("FINAL MULTI-SEED SUMMARY (10 SEEDS: Mean ± Std)")
    print("="*80)
    
    rows = []
    print("\n--- CENTRALIZED MODELS ---")
    for m_name in cent_results:
        s = summarize(cent_results[m_name], m_name)
        rows.append(s)
        print(f"{m_name:25s} | Acc: {s['accuracy']:16s} | Sens: {s['sensitivity']:16s} | Spec: {s['specificity']:16s} | F1: {s['f1_score']:16s} | AUC: {s['roc_auc']}")
        
    print("\n--- LOCAL ISOLATED HOSPITALS ---")
    for h_name in local_results:
        s = summarize(local_results[h_name], f"Local {h_name}")
        rows.append(s)
        print(f"{h_name:25s} | Acc: {s['accuracy']:16s} | Sens: {s['sensitivity']:16s} | Spec: {s['specificity']:16s} | F1: {s['f1_score']:16s} | AUC: {s['roc_auc']}")
        
    print("\n--- FEDERATED LEARNING (FedAvg) ---")
    s_iid = summarize(fed_iid_results, "FedAvg (IID)")
    rows.append(s_iid)
    print(f"{'FedAvg (IID)':25s} | Acc: {s_iid['accuracy']:16s} | Sens: {s_iid['sensitivity']:16s} | Spec: {s_iid['specificity']:16s} | F1: {s_iid['f1_score']:16s} | AUC: {s_iid['roc_auc']}")
    
    s_non_iid = summarize(fed_non_iid_results, "FedAvg (Non-IID)")
    rows.append(s_non_iid)
    print(f"{'FedAvg (Non-IID)':25s} | Acc: {s_non_iid['accuracy']:16s} | Sens: {s_non_iid['sensitivity']:16s} | Spec: {s_non_iid['specificity']:16s} | F1: {s_non_iid['f1_score']:16s} | AUC: {s_non_iid['roc_auc']}")

    # Save summary table
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(PROJECT_ROOT / "results" / "metrics" / "multi_seed_summary_10_seeds.csv", index=False)
    print("\nSaved summary to results/metrics/multi_seed_summary_10_seeds.csv")

if __name__ == "__main__":
    run_multi_seed_experiments()
