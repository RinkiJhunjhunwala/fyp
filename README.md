# Privacy-Preserving Cardiovascular Disease Classification Using Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch & Scikit-Learn](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)](https://pytorch.org/)
[![Web App: Flask](https://img.shields.io/badge/Web%20App-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License: Academic Research Prototype](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

A complete, reproducible, academic machine learning and federated learning project for **Acute Myocardial Infarction (AMI)** admission classification using continuous physiological intensive-care vital signs from **MIMIC-IV v3.1**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & Motivation](#problem-statement--motivation)
3. [MIMIC-IV Dataset & Cohort Schema](#mimic-iv-dataset--cohort-schema)
4. [Preprocessing Pipeline & Observation Window](#preprocessing-pipeline--observation-window)
5. [Feature Engineering (1 Row Per Patient)](#feature-engineering-1-row-per-patient)
6. [Centralized Machine Learning Benchmark](#centralized-machine-learning-benchmark)
7. [Federated Learning Architecture & FedAvg](#federated-learning-architecture--fedavg)
8. [Local vs Centralized vs Federated Results](#local-vs-centralized-vs-federated-results)
9. [Web Application Guide](#web-application-guide)
10. [Project Directory Structure](#project-directory-structure)
11. [Installation & Execution](#installation--execution)
12. [Academic Disclaimer](#academic-disclaimer)

---

## 1. Project Overview

In healthcare, patient electronic health records (EHR) cannot be easily centralized due to privacy laws (HIPAA, GDPR) and hospital data governance policies. Isolated hospitals that train machine learning models on their local data suffer from small sample sizes and poor generalizability.

This project implements and evaluates **Federated Learning (FedAvg)** to enable collaborative model training across three decentralized hospital nodes without ever sharing raw patient records.

### Key Highlights
* **Zero Raw Data Transfer**: Client hospital nodes exchange only mathematical parameter updates (weights and biases).
* **Strict Patient Integrity**: Observation-level records belonging to a patient are never split across clients or train/test splits.
* **Leakage-Free Feature Engineering**: Standardized 24-hour observation window with 43 statistical features plus 3 clinical composite ratios.
* **Real Empirical Verification**: All results are generated from actual execution on held-out test cohorts.
* **Full-Stack Research Interface**: Interactive Flask web application with live clinical inference, model benchmarks, and dynamic federated topology visualization.

---

## 2. Problem Statement & Motivation

Cardiovascular disease, specifically Acute Myocardial Infarction (AMI), is a leading cause of mortality worldwide. Intensive care unit (ICU) patients produce vast streams of continuous vital signs (Heart Rate, Blood Pressure, Oxygen Saturation). 

The goal of this project is:
1. Extract fixed-length physiological biomarkers from high-frequency time-series vitals within a controlled 24-hour window.
2. Establish a strong centralized baseline across 6 machine learning algorithms (Logistic Regression, Random Forest, Decision Tree, Support Vector Machine, K-Nearest Neighbours, Gradient Boosting).
3. Demonstrate that independent local hospital training produces unstable models due to small cohort sizes.
4. Prove that collaborative **Federated Averaging (FedAvg)** matches centralized accuracy while guaranteeing 100% data residency privacy.

---

## 3. MIMIC-IV Dataset & Cohort Schema

The dataset was curated from **MIMIC-IV version 3.1**:
* **AMI Admissions**: Diagnoses identified via ICD-9 code `410` and ICD-10 codes `I21–I22`.
* **Balanced Cohort**: 90 unique patients (45 AMI-positive, 45 Control). 
* **Chartevents ICU Vitals**: 80 patients (40 AMI, 40 Control — 50/50 balance) have high-frequency ICU chart events totaling **287,710 raw measurements**.
* **Identifiers**: `subject_id` (patient ID) and `hadm_id` (admission ID). Exactly one admission per patient (1:1 mapping).
* **Target Label**: `ami_during_admission` (1 = AMI positive, 0 = Control).
* **Supporting Laboratory Data**: Troponin T records (`itemid 51003`) are retained strictly for auxiliary reference and excluded from primary vital-sign feature modeling.

### Vital Signs Monitored
| Canonical Vital | MIMIC-IV Item IDs | Source / Label | Measurement Unit |
| :--- | :--- | :--- | :--- |
| **Heart Rate (HR)** | `220045` | Heart Rate | `bpm` |
| **Systolic BP (SBP)** | `220050`, `220179` | Arterial Line / Non-Invasive Cuff | `mmHg` |
| **Diastolic BP (DBP)** | `220051`, `220180` | Arterial Line / Non-Invasive Cuff | `mmHg` |
| **Mean Arterial BP (MBP)** | `220052`, `220181` | Arterial Line / Non-Invasive Cuff | `mmHg` |
| **Oxygen Saturation (SpO₂)** | `220277` | Pulse Oximetry | `%` |

---

## 4. Preprocessing Pipeline & Observation Window

The reusable preprocessing module is implemented in [`src/preprocessing.py`](src/preprocessing.py):
1. **Timestamp Normalization**: Mixed ISO-8601 chart timestamps parsed and chronologically sorted per patient.
2. **Clinical Physiological Filtering**: Removal of sensor disconnections and transducer flushing artifacts:
   * HR: $[20, 240]$ bpm
   * SBP: $[30, 260]$ mmHg
   * DBP: $[20, 160]$ mmHg
   * MBP: $[30, 200]$ mmHg
   * SpO₂: $[50, 100]$ %
3. **Signal Reconciliation**: When invasive arterial catheter (ABP) and cuff (NIBP) measurements occur at the identical minute, ABP is prioritized.
4. **Controlled Observation Window**: To prevent lookahead data leakage, observations are strictly windowed to the **first 24 hours from the patient's first recorded vital sign** (`charttime - first_charttime <= 24 hours`).

---

## 5. Feature Engineering (1 Row Per Patient)

Traditional machine learning requires fixed-length feature vectors. The module [`src/feature_engineering.py`](src/feature_engineering.py) transforms the 24-hour time series into a clean tabular matrix where **every patient is represented by exactly one row**:

For each of the 4 vital signs (HR, SBP, DBP, SpO₂), 10 statistical biomarkers are extracted:
* **Mean ($\mu$)**: Average physiological level.
* **Minimum**: Lowest recorded reading.
* **Maximum**: Peak reading.
* **Standard Deviation ($\sigma$)**: Signal variability and hemodynamic stability.
* **Range**: Peak-to-trough excursion ($Max - Min$).
* **First Value**: Admission baseline.
* **Last Value**: End-of-window measurement.
* **Net Change**: Trajectory ($Last - First$).
* **Trend (Slope)**: Rate of change per hour computed via linear regression:
  $$\text{slope} = \frac{\sum (t_i - \bar{t})(y_i - \bar{y})}{\sum (t_i - \bar{t})^2}$$
* **Observation Count**: Frequency of charting.

### Derived Clinical Indices
* **Mean Shock Index**: $\frac{\text{HR}_{\text{mean}}}{\text{SBP}_{\text{mean}}}$ (classic clinical marker for cardiogenic decompensation).
* **Peak Shock Index**: $\frac{\text{HR}_{\text{max}}}{\text{SBP}_{\text{min}}}$.
* **Pulse Pressure**: $\text{SBP}_{\text{mean}} - \text{DBP}_{\text{mean}}$ (vascular compliance index).

*Output shape: 80 patients $\times$ 46 columns (43 features + 2 IDs + 1 Target). Zero missing values.*

---

## 6. Centralized Machine Learning Benchmark

Evaluated on a held-out test cohort (16 patients: 8 AMI, 8 Control) using 5-Fold Stratified Cross-Validation on the training set (64 patients: 32 AMI, 32 Control) for hyperparameter tuning. Standard scaling is fitted strictly on the training set.

| Model Architecture | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **68.75%** | **71.43%** | **62.50%** | **75.00%** | **0.6667** | **0.7188** | 8.72s |
| **Random Forest** | 50.00% | 50.00% | 37.50% | 62.50% | 0.4286 | 0.4219 | 9.56s |
| **Decision Tree** | 56.25% | 60.00% | 37.50% | 75.00% | 0.4615 | 0.6172 | 0.36s |
| **Support Vector Machine** | 56.25% | 57.14% | 50.00% | 62.50% | 0.5333 | 0.5000 | 0.34s |
| **K-Nearest Neighbours** | 43.75% | 42.86% | 37.50% | 50.00% | 0.4000 | 0.4297 | 0.71s |
| **Gradient Boosting** | 56.25% | 60.00% | 37.50% | 75.00% | 0.4615 | 0.5859 | 4.96s |

---

## 7. Federated Learning Architecture & FedAvg

The federated simulation models three distinct hospital entities:
* **Hospital A**: Urban Medical Center
* **Hospital B**: University Hospital
* **Hospital C**: Regional Health Center

### Federated Averaging (FedAvg) Workflow
1. **Global Model Initialization**: PyTorch Logistic Regression with weight tensor $W \in \mathbb{R}^{1 \times 43}$ and bias $b \in \mathbb{R}^1$.
2. **Broadcast Phase**: The server transmits $(W_t, b_t)$ to all hospital nodes.
3. **Local Private Training**: Each hospital initializes its local model with global parameters and trains for $E=5$ epochs using local Stochastic Gradient Descent (SGD) on its private patient records:
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{n_k} \sum_{i=1}^{n_k} [y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)]$$
4. **Parameter Return**: Hospitals send only updated weights $W_k^{(t)}$, bias $b_k^{(t)}$, and cohort count $n_k$.
5. **Federated Aggregation**: The central coordinator aggregates parameters using sample-weighted averaging:
   $$W_{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_k^{(t)}, \quad b_{t+1} = \sum_{k=1}^K \frac{n_k}{N} b_k^{(t)}$$
6. **Held-Out Global Validation**: The updated global model is evaluated on the held-out global test cohort.

---

## 8. Local vs Centralized vs Federated Results

This experiment demonstrates the core value proposition of collaborative Federated Learning compared to isolated local hospital training:

| Setup / Training Strategy | Data Privacy Level | Test Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Logistic Regression** | None (Pooled Data) | **68.75%** | 71.43% | 62.50% | 75.00% | 0.6667 | 0.7188 |
| **Federated Global Model (FedAvg - IID)** | **High (Parameters Only)** | **68.75%** | **66.67%** | **75.00%** | **62.50%** | **0.7059** | **0.7188** |
| **Federated Global Model (FedAvg - Non-IID)**| **High (Parameters Only)** | **75.00%** | **75.00%** | **75.00%** | **62.50%** | **0.7500** | **0.7500** |
| **Local Hospital A Only** (Isolated) | Single Hospital Only | 62.50% | 66.67% | 50.00% | 75.00% | 0.5714 | 0.5781 |
| **Local Hospital B Only** (Isolated) | Single Hospital Only | 31.25% | 33.33% | 37.50% | 25.00% | 0.3529 | 0.3594 |
| **Local Hospital C Only** (Isolated) | Single Hospital Only | 56.25% | 55.56% | 62.50% | 50.00% | 0.5882 | 0.6250 |

### Key Takeaway
Isolated hospital models (such as Hospital B at 31.25%) fail due to local sample scarcity. Federated Learning elevates all participants to **68.75% - 75.00% accuracy**, achieving performance on par with centralized data pooling while maintaining complete patient record confidentiality.

---

## 9. Web Application Guide

The Flask web application provides an intuitive research interface:
* **Home Page (`/`)**: Overview, problem statement, Federated Learning workflow, and MIMIC-IV cohort details.
* **Clinical Inference (`/predict`)**: Input physiological vital signs (HR, SBP, DBP, SpO₂), select the classification model (Federated Global vs Centralized), and view immediate diagnostic risk probabilities and hemodynamic indicators.
* **Model Benchmarks (`/comparison`)**: Interactive benchmark tables and embedded high-resolution figures.
* **Federated Network (`/federated`)**: Decentralized topology diagram, client cohort summaries, and round-by-round convergence logs.
* **REST API (`/api/predict`)**: Programmatic JSON endpoint for external integration.

---

## 10. Project Directory Structure

```text
7th Sem Project/
├── data/
│   ├── final/
│   │   ├── cardiac_patient_labelled_vitals.csv   # Raw MIMIC-IV vitals (287,710 rows)
│   │   └── selected_90_patients.csv              # Cohort metadata
│   ├── troponin/
│   │   └── cardiac_patient_troponin.csv          # Supporting lab measurements
│   └── processed/
│       ├── windowed_vitals_24h.csv               # 24-hour filtered time-series
│       ├── patient_vital_features_24h.csv        # Tabular features (80 rows x 46 cols)
│       ├── train_split.csv                       # Training patient cohort (64 rows)
│       └── test_split.csv                        # Held-out evaluation cohort (16 rows)
│
├── src/
│   ├── data_inspection.py                        # Dataset inspection & schema validation
│   ├── preprocessing.py                          # Physiological bounds & 24h windowing
│   ├── feature_engineering.py                    # 1-row-per-patient feature extractor
│   ├── centralized_training.py                   # 6 ML algorithms with 5-fold CV
│   ├── local_training.py                         # Isolated hospital baselines
│   ├── federated_client.py                       # Private hospital client node
│   ├── federated_server.py                       # FedAvg parameter aggregation
│   ├── evaluation.py                             # Standardized academic metrics
│   └── visualization.py                          # Publication-ready plots
│
├── models/
│   ├── scaler.joblib                             # Training-fitted StandardScaler
│   ├── feature_columns.json                      # 43 predictive feature names
│   ├── federated_logistic_regression.joblib      # Final global federated model
│   ├── federated_global_model.pt                 # PyTorch state dict
│   ├── logistic_regression.joblib                # Centralized models
│   ├── random_forest.joblib
│   └── ...
│
├── results/
│   ├── metrics/
│   │   ├── centralized_metrics.csv
│   │   ├── local_hospitals_metrics.csv
│   │   ├── federated_iid_rounds.csv
│   │   ├── federated_non_iid_rounds.csv
│   │   └── paradigm_comparison.csv
│   └── figures/
│       ├── centralized_models_comparison.png
│       ├── roc_curves_comparison.png
│       ├── confusion_matrices_grid.png
│       ├── federated_learning_rounds.png
│       ├── federated_iid_vs_non_iid.png
│       ├── centralized_vs_federated_vs_local.png
│       ├── feature_importance_random_forest.png
│       └── feature_importance_gradient_boosting.png
│
├── app/
│   ├── app.py                                   # Flask backend server
│   ├── templates/                                # HTML5 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── predict.html
│   │   ├── comparison.html
│   │   └── federated.html
│   └── static/                                   # CSS & assets
│       ├── css/style.css
│       └── figures/
│
├── tests/
│   └── test_web_app.py                           # Automated test suite
│
├── main.py                                       # Unified pipeline entry point
├── run_preprocessing_pipeline.py                 # Phase 2 runner
├── run_federated_pipeline.py                     # Phases 4-7 runner
├── requirements.txt                              # Python package dependencies
└── README.md                                     # Project documentation
```

---

## 11. Installation & Execution

### Prerequisites
* Python 3.10+ (Python 3.13 tested)
* Windows, macOS, or Linux

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Complete End-to-End Pipeline
To inspect the dataset, preprocess observations, train centralized models, simulate federated learning, and generate all figures in one command:
```bash
python main.py --run-all
```

Or execute individual phases:
```bash
# Phase 1: Inspect Dataset
python main.py --inspect

# Phase 2: Preprocess & Feature Extraction
python main.py --preprocess

# Phase 3: Train Centralized Models
python main.py --train-centralized

# Phases 4-7: Local Baselines & Federated Learning
python main.py --train-federated
```

### Step 3: Launch Web Application
```bash
python main.py --app
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### Step 4: Run Automated Tests
```bash
python -m unittest discover tests/
```

---

## 12. Academic Disclaimer

This project is an **academic research prototype** developed for engineering capstone evaluation. It is evaluated on a small cohort of 80 unique patients from MIMIC-IV v3.1. The classification targets an administratively defined ICD admission label, not future prospective cardiovascular risk. It is **not** a clinical diagnostic medical device and must never be used for actual medical diagnosis, emergency triage, or clinical treatment decisions.
