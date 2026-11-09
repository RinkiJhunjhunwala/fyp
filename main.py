"""
Main Entry Point for Privacy-Preserving Cardiovascular Disease Classification Project
Orchestrates all project stages:
1. Dataset inspection
2. Preprocessing & 24h window feature engineering
3. Centralized ML model training & 5-fold CV
4. Local hospital baseline training
5. Federated Learning (FedAvg) simulation
6. Model evaluation & visualization generation
7. Web application launch
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_inspection():
    print("\n>>> EXECUTING PHASE 1: DATA INSPECTION <<<")
    from src.data_inspection import full_analysis_selected_cohort, full_analysis_vitals, inspect_other_files
    full_analysis_selected_cohort()
    full_analysis_vitals()
    inspect_other_files()

def run_preprocessing():
    print("\n>>> EXECUTING PHASE 2: PREPROCESSING & FEATURE ENGINEERING <<<")
    from run_preprocessing_pipeline import run_phase_2
    run_phase_2()

def run_centralized():
    print("\n>>> EXECUTING PHASE 3: CENTRALIZED MODEL TRAINING <<<")
    from src.centralized_training import train_and_evaluate_all
    train_and_evaluate_all()

def run_federated():
    print("\n>>> EXECUTING PHASES 4-7: FEDERATED & COMPARISON PIPELINE <<<")
    from run_federated_pipeline import run_federated_and_comparison_pipeline
    run_federated_and_comparison_pipeline()

def run_web_app(host: str = "127.0.0.1", port: int = 5000):
    print(f"\n>>> LAUNCHING WEB APPLICATION at http://{host}:{port} <<<")
    from app.app import app
    app.run(host=host, port=port, debug=False)

def main():
    parser = argparse.ArgumentParser(description="FedAMI: Privacy-Preserving CVD Classification Pipeline")
    parser.add_argument("--inspect", action="store_true", help="Run Phase 1: Dataset Inspection")
    parser.add_argument("--preprocess", action="store_true", help="Run Phase 2: Preprocessing & Feature Engineering")
    parser.add_argument("--train-centralized", action="store_true", help="Run Phase 3: Train Centralized Models")
    parser.add_argument("--train-federated", action="store_true", help="Run Phases 4-7: Local & Federated Training")
    parser.add_argument("--run-all", action="store_true", help="Run complete end-to-end pipeline")
    parser.add_argument("--app", action="store_true", help="Launch the Flask Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web app host")
    parser.add_argument("--port", type=int, default=5000, help="Web app port")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.run_all:
        print("=" * 80)
        print("RUNNING COMPLETE REPRODUCIBLE PIPELINE (END-TO-END)")
        print("=" * 80)
        run_inspection()
        run_preprocessing()
        run_centralized()
        run_federated()
        print("\nComplete pipeline successfully executed!")
        print("To start the web application, execute: python main.py --app")
    elif args.inspect:
        run_inspection()
    elif args.preprocess:
        run_preprocessing()
    elif args.train_centralized:
        run_centralized()
    elif args.train_federated:
        run_federated()
    elif args.app:
        run_web_app(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
