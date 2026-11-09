"""
Flask Web Application Backend for Privacy-Preserving AMI Classification
Serves web interface, handles prediction requests using saved pipelines and models,
and provides API endpoints for model comparison and federated simulation data.
"""

from pathlib import Path
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import extract_features_from_single_reading, extract_features_from_windowed_df
from src.preprocessing import preprocess_vitals
from src.federated_server import PyTorchSklearnWrapper

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "app" / "templates"),
    static_folder=str(PROJECT_ROOT / "app" / "static")
)
app.config["SECRET_KEY"] = "academic-ami-federated-learning-2026"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
STATIC_FIGURES = PROJECT_ROOT / "app" / "static" / "figures"
STATIC_FIGURES.mkdir(parents=True, exist_ok=True)

# Copy figures to static directory for serving
for fig in FIGURES_DIR.glob("*.png"):
    shutil.copy(fig, STATIC_FIGURES / fig.name)

# Cache loaded models and preprocessing pipeline
SCALER = joblib.load(MODELS_DIR / "scaler.joblib")
with open(MODELS_DIR / "feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

AVAILABLE_MODELS = {
    "federated_lr": {
        "name": "Federated Logistic Regression (FedAvg)",
        "file": "federated_logistic_regression.joblib",
        "description": "Collaboratively trained across 3 hospital nodes without pooling patient records.",
        "type": "Federated Learning"
    },
    "logistic_regression": {
        "name": "Centralized Logistic Regression",
        "file": "logistic_regression.joblib",
        "description": "Traditional centralized model trained on pooled data.",
        "type": "Centralized ML"
    },
    "random_forest": {
        "name": "Random Forest",
        "file": "random_forest.joblib",
        "description": "Ensemble of 50 decision trees with feature bagging.",
        "type": "Centralized ML"
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "file": "gradient_boosting.joblib",
        "description": "Iteratively boosted decision stumps minimizing logistic loss.",
        "type": "Centralized ML"
    },
    "decision_tree": {
        "name": "Decision Tree",
        "file": "decision_tree.joblib",
        "description": "Interpretable rule-based decision tree.",
        "type": "Centralized ML"
    },
    "support_vector_machine": {
        "name": "Support Vector Machine (Linear)",
        "file": "support_vector_machine.joblib",
        "description": "Maximal margin hyperplane separator with Platt scaling.",
        "type": "Centralized ML"
    },
    "k-nearest_neighbours": {
        "name": "K-Nearest Neighbours",
        "file": "k-nearest_neighbours.joblib",
        "description": "Instance-based non-parametric classifier (k=5).",
        "type": "Centralized ML"
    }
}

LOADED_MODELS = {}
for key, info in AVAILABLE_MODELS.items():
    model_path = MODELS_DIR / info["file"]
    if model_path.exists():
        LOADED_MODELS[key] = joblib.load(model_path)

@app.route("/")
def home():
    """Home / Project Overview Page."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Vital Signs Prediction Page."""
    if request.method == "GET":
        return render_template(
            "predict.html",
            available_models=AVAILABLE_MODELS,
            default_model="federated_lr"
        )
        
    try:
        model_key = request.form.get("model_choice", "federated_lr")
        if model_key not in LOADED_MODELS:
            model_key = "federated_lr"
        model = LOADED_MODELS[model_key]
        
        # Check if CSV file was uploaded
        file = request.files.get("vitals_file")
        if file and file.filename != "":
            raw_user_df = pd.read_csv(file)
            cleaned_df = preprocess_vitals(raw_user_df, observation_window_hours=24.0, filter_implausible=True)
            feature_df = extract_features_from_windowed_df(cleaned_df)
            input_summary = f"Batch uploaded CSV: {len(raw_user_df)} time-series readings across {feature_df['subject_id'].nunique()} patient(s)."
        else:
            # Single manual vital entry
            hr = float(request.form.get("hr", 80))
            sbp = float(request.form.get("sbp", 120))
            dbp = float(request.form.get("dbp", 80))
            spo2 = float(request.form.get("spo2", 98))
            
            # Validation
            if not (20 <= hr <= 240):
                raise ValueError("Heart Rate must be between 20 and 240 bpm.")
            if not (30 <= sbp <= 260):
                raise ValueError("Systolic BP must be between 30 and 260 mmHg.")
            if not (20 <= dbp <= 160):
                raise ValueError("Diastolic BP must be between 20 and 160 mmHg.")
            if sbp <= dbp:
                raise ValueError("Systolic BP must be greater than Diastolic BP.")
            if not (50 <= spo2 <= 100):
                raise ValueError("SpO2 must be between 50% and 100%.")
                
            feature_df = extract_features_from_single_reading(hr=hr, sbp=sbp, dbp=dbp, spo2=spo2)
            input_summary = f"Single observation: HR = {hr} bpm, SBP = {sbp} mmHg, DBP = {dbp} mmHg, SpO2 = {spo2}%"
            
        # Ensure all feature columns match training schema
        for c in FEATURE_COLS:
            if c not in feature_df.columns:
                feature_df[c] = 0.0
                
        X_raw = feature_df[FEATURE_COLS].values
        X_scaled = SCALER.transform(X_raw)
        
        # Inference
        probs = model.predict_proba(X_scaled)
        preds = model.predict(X_scaled)
        
        positive_prob = float(probs[0, 1])
        prediction_label = "AMI Positive (High Risk)" if preds[0] == 1 else "Negative / Control (Low Risk)"
        risk_level = "High" if positive_prob >= 0.50 else "Low"
        confidence = positive_prob if preds[0] == 1 else (1.0 - positive_prob)
        
        result = {
            "prediction_label": prediction_label,
            "ami_probability": positive_prob,
            "risk_level": risk_level,
            "confidence_percent": round(confidence * 100, 1),
            "model_used": AVAILABLE_MODELS[model_key]["name"],
            "model_type": AVAILABLE_MODELS[model_key]["type"],
            "input_summary": input_summary,
            "shock_index": round(float(feature_df["shock_index_mean"].iloc[0]), 2),
            "pulse_pressure": round(float(feature_df["pulse_pressure_mean"].iloc[0]), 1)
        }
        
        return render_template(
            "predict.html",
            available_models=AVAILABLE_MODELS,
            default_model=model_key,
            result=result
        )
    except Exception as e:
        return render_template(
            "predict.html",
            available_models=AVAILABLE_MODELS,
            default_model="federated_lr",
            error=str(e)
        )

@app.route("/comparison")
def comparison():
    """Model Comparison Page."""
    # Load comparison table and centralized table
    comp_df = pd.read_csv(METRICS_DIR / "paradigm_comparison.csv")
    cent_df = pd.read_csv(METRICS_DIR / "centralized_metrics.csv")
    
    return render_template(
        "comparison.html",
        paradigm_data=comp_df.to_dict(orient="records"),
        centralized_data=cent_df.to_dict(orient="records")
    )

@app.route("/federated")
def federated():
    """Federated Learning Simulation & Convergence Page."""
    # Load round histories
    iid_df = pd.read_csv(METRICS_DIR / "federated_iid_rounds.csv")
    non_iid_df = pd.read_csv(METRICS_DIR / "federated_non_iid_rounds.csv")
    
    return render_template(
        "federated.html",
        iid_rounds=iid_df.to_dict(orient="records"),
        non_iid_rounds=non_iid_df.to_dict(orient="records")
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST API Endpoint for programmatic inference."""
    try:
        data = request.get_json(force=True)
        hr = float(data.get("hr", 80))
        sbp = float(data.get("sbp", 120))
        dbp = float(data.get("dbp", 80))
        spo2 = float(data.get("spo2", 98))
        model_key = data.get("model", "federated_lr")
        
        model = LOADED_MODELS.get(model_key, LOADED_MODELS["federated_lr"])
        
        feat_df = extract_features_from_single_reading(hr=hr, sbp=sbp, dbp=dbp, spo2=spo2)
        for c in FEATURE_COLS:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        X_scaled = SCALER.transform(feat_df[FEATURE_COLS].values)
        
        prob = float(model.predict_proba(X_scaled)[0, 1])
        pred = int(model.predict(X_scaled)[0])
        
        return jsonify({
            "status": "success",
            "prediction": pred,
            "prediction_label": "AMI Positive" if pred == 1 else "Control / Negative",
            "probability": prob,
            "model": AVAILABLE_MODELS.get(model_key, {}).get("name", model_key),
            "disclaimer": "Academic research prototype. Not intended for medical diagnosis."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
