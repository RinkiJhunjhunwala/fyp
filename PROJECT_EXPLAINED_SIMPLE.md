# The Complete Project Explained in Simple Words

> **Title:** Privacy-Preserving Cardiovascular Disease Classification Using Federated Learning  
> **Target Audience:** College students, viva examiners, software engineers, and anyone who wants to understand the project from the ground up without complex medical or mathematical jargon.

---

## Table of Contents
1. [The Big Picture: What is this project about?](#1-the-big-picture-what-is-this-project-about)
2. [The Real-World Problem: Why can't hospitals just share data?](#2-the-real-world-problem-why-cant-hospitals-just-share-data)
3. [The Dataset: What data did we actually have?](#3-the-dataset-what-data-did-we-actually-have)
4. [Step-by-Step: Everything we did and WHY we did it](#4-step-by-step-everything-we-did-and-why-we-did-it)
5. [The Models: Why did we pick them and why did some win while others lost?](#5-the-models-why-did-we-pick-them-and-why-did-some-win-while-others-lost)
6. [Federated Learning Explained with a Simple Analogy](#6-federated-learning-explained-with-a-simple-analogy)
7. [The Results: What actually happened and WHY?](#7-the-results-what-actually-happened-and-why)
8. [The Web Application: How does it work?](#8-the-web-application-how-does-it-work)
9. [Viva / Defense Cheat Sheet: 10 Questions Your Examiner Will Ask](#9-viva--defense-cheat-sheet-10-questions-your-examiner-will-ask)

---

## 1. The Big Picture: What is this project about?

Imagine a person arrives in the Intensive Care Unit (ICU). Doctors hook them up to monitors measuring their **Heart Rate**, **Blood Pressure**, and **Oxygen Saturation (SpO₂)**.

We want to answer a critical question:  
**"Based only on these basic vital signs recorded in the first 24 hours, did this patient suffer an Acute Myocardial Infarction (AMI — commonly known as a Heart Attack)?"**

AND we want to train an Artificial Intelligence (AI) model across **multiple hospitals** WITHOUT any hospital ever sharing their confidential patient records with anyone else.

---

## 2. The Real-World Problem: Why can't hospitals just share data?

In traditional machine learning:
1. Hospital A, Hospital B, and Hospital C send all their patient files to one big cloud database.
2. A data scientist trains a model on all the pooled data.

### Why this is impossible in real life:
* **Strict Laws:** Laws like **HIPAA** (in the US) and **GDPR** (in Europe) make sharing raw patient medical records illegal. If a hospital leaks patient records, they face millions of dollars in fines.
* **Hospital Competition & Security:** Hospitals do not want their proprietary patient data copied to external servers.

### The Problem with Single Hospitals:
If Hospital A decides: *"Fine, I won't share data, I will just train an AI on my own 20 patients,"* the model will be terrible. With only 20 patients, the AI memorizes those 20 specific people (overfitting) and fails completely when used on new patients. (In our experiment, Hospital B alone got only **31.25% accuracy**!)

### The Solution: Federated Learning
Instead of sending **patient data** to the AI, we send the **AI to the patient data**!
Each hospital trains the model inside its own building. Only the **math formulas (weights)** are sent to a central server, which averages them together.

---

## 3. The Dataset: What data did we actually have?

We used real-world hospital data from **MIMIC-IV (version 3.1)**, an internationally recognized database of ICU patients from Beth Israel Deaconess Medical Center in Boston.

* **Selected Cohort:** 80 unique ICU patients.
* **Balanced 50/50:**
  * **40 AMI Patients (Label = 1):** Patients diagnosed with heart attacks (ICD-9 code 410, ICD-10 codes I21–I22).
  * **40 Control Patients (Label = 0):** Patients admitted to the cardiac unit without those heart attack codes.
* **Raw Measurements:** Over **287,000 raw readings** of vital signs.
* **The 4 Vital Signs Used:**
  1. **Heart Rate (HR):** How fast the heart beats (beats per minute).
  2. **Systolic Blood Pressure (SBP):** Peak blood pressure when the heart contracts (mmHg).
  3. **Diastolic Blood Pressure (DBP):** Resting blood pressure between beats (mmHg).
  4. **Oxygen Saturation (SpO₂):** Percentage of oxygen in the blood (%).
* **What about Troponin?** Troponin is a blood test marker for heart damage. As per project guidelines, Troponin was kept as supporting lab information, but **excluded from the model input**. We wanted to see what the vital signs alone can tell us!

---

## 4. Step-by-Step: Everything we did and WHY we did it

Here is the entire engineering pipeline broken down simply:

```text
Raw ICU Vitals (287,710 rows)
       │
       ▼
[1. Clinical Filtering] ──► Remove sensor errors (e.g. negative blood pressure)
       │
       ▼
[2. 24-Hour Windowing] ───► Take ONLY the first 24 hours of each patient
       │
       ▼
[3. Feature Engineering] ──► Turn thousands of readings into 1 ROW PER PATIENT (43 features)
       │
       ▼
[4. Stratified Split] ────► 64 Training Patients (32 Pos / 32 Neg)
                            16 Testing Patients  (8 Pos / 8 Neg)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Centralized ML Benchmark]      [Federated Learning Network]
- Logistic Regression (68.75%)  - Split 64 patients into 3 Hospitals
- Random Forest (50.00%)        - Train locally (5 epochs)
- Decision Tree (56.25%)        - Server averages weights (FedAvg)
- SVM (56.25%)                  - 15 Communication Rounds
- KNN (43.75%)                  - Final Global Accuracy: 68.75% - 75.00%!
- Gradient Boosting (56.25%)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
             [Interactive Web Application]
             - Test live patient vitals
             - Compare all models
             - See animated hospital nodes
```

### Step 1: Data Inspection
* **What we did:** Wrote a script to look at every file, find column names, check data types, and check for missing values.
* **Why:** If you assume columns exist without checking, your code will crash or give fake results. We found that 10 patients from the hospital cohort were never transferred to the ICU, leaving exactly 80 ICU patients.

### Step 2: Cleaning & Filtering Implausible Values
* **What we did:** In raw ICU data, sensors glitch. We found minimum blood pressure was `-12.0` (impossible!) and maximum mean pressure was `7469.0` (impossible!). These happen when nurses zero or flush the arterial catheter lines.
* **Why:** Garbage in = Garbage out. We applied clinical bounds:
  * HR: 20 to 240 bpm
  * SBP: 30 to 260 mmHg
  * DBP: 20 to 160 mmHg
  * SpO₂: 50% to 100%

### Step 3: Enforcing the 24-Hour Observation Window
* **What we did:** For every patient, we found their very first vital sign measurement, and we threw away everything recorded after 24 hours.
* **Why (Data Leakage Prevention):** If a patient stays in the hospital for 3 weeks, looking at week 3 vitals to "predict" an admission condition is cheating (lookahead bias). Doctors need to make decisions in the first 24 hours.

### Step 4: Feature Engineering (1 Row Per Patient)
* **The Problem:** Patient 1 has 5,000 readings over 24 hours. Patient 2 has 800 readings. Machine learning models cannot take a variable-length list; they require a fixed table where every patient is **one row**.
* **What we did:** For every vital sign (HR, SBP, DBP, SpO₂), we calculated 10 summary numbers:
  1. `mean`: Average value
  2. `min`: Lowest value
  3. `max`: Highest value
  4. `std`: How much the signal fluctuated (variability)
  5. `range`: Max minus Min
  6. `first`: What the reading was when they arrived
  7. `last`: What the reading was at hour 24
  8. `change`: Last minus First
  9. `trend`: Was the vital going up or down over time? (Calculated as a linear regression slope in units/hour)
  10. `count`: How many times was it measured?
* **Clinical Composite Ratios:**
  * **Shock Index:** $\frac{\text{HR}}{\text{SBP}}$. If heart rate is high but blood pressure is low ($> 0.9$), the heart is failing to pump blood effectively!
  * **Pulse Pressure:** $\text{SBP} - \text{DBP}$. Measures vascular stiffness.
* **Result:** A clean table of 80 rows and 43 features. Zero missing values!

### Step 5: Patient-Level Train/Test Split
* **What we did:** Split the 80 patients into:
  * 64 Training Patients (32 AMI, 32 Control)
  * 16 Held-Out Testing Patients (8 AMI, 8 Control)
* **Crucial Rule:** All records for Patient X stay strictly in either Training or Testing. A patient's data was NEVER split across sets!

---

## 5. The Models: Why did we pick them and why did some win while others lost?

We trained 6 standard centralized algorithms, tuned their hyperparameters with 5-fold cross-validation on the training set, and tested them on the held-out test set:

| Model | Test Accuracy | F1-Score | ROC-AUC | What this model does |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **68.75%** | **0.6667** | **0.7188** | Draws a smooth linear boundary between classes |
| **Random Forest** | 50.00% | 0.4286 | 0.4219 | Builds an ensemble of 50 decision trees |
| **Decision Tree** | 56.25% | 0.4615 | 0.6172 | A single flowchart of IF/ELSE questions |
| **Support Vector Machine (SVM)**| 56.25% | 0.5333 | 0.5000 | Finds the widest margin dividing positive from negative |
| **K-Nearest Neighbours (KNN)** | 43.75% | 0.4000 | 0.4297 | Looks at the 5 most similar patients in memory |
| **Gradient Boosting** | 56.25% | 0.4615 | 0.5859 | Builds small trees sequentially to correct past errors |

---

### The Big Question: Why did simple Logistic Regression beat complex Random Forest & Gradient Boosting?

This is the #1 question professors love to ask:  
*"Why did a simple 1960s linear model beat modern powerful tree ensembles?"*

#### The Answer:
1. **Small Sample Size (80 Patients):** Complex models like Random Forest and Gradient Boosting have hundreds of parameters. When you have only 64 training patients and 43 features, complex tree algorithms have too much freedom. They "memorize" the random noise in those 64 people instead of learning true generalizable cardiovascular patterns (this is called **overfitting**).
2. **Occam's Razor / The Curse of Dimensionality:** In medical data with limited samples, simpler models with strong regularization ($L_2$ penalty) perform far better. Logistic Regression fits a single smooth slope for each vital sign (e.g. "higher heart rate and lower blood pressure steadily increases risk"). It doesn't create convoluted branch splits.
3. **Continuous Physiological Relationships:** Heart rate and blood pressure changes are continuous, gradual physiological gradients. Linear decision surfaces naturally capture these hemodynamic trends better than step-function axis-aligned tree splits.

---

## 6. Federated Learning Explained with a Simple Analogy

### The Analogy:
Imagine 3 students (Hospital A, Hospital B, Hospital C) studying for a final exam in different cities.
* Their textbooks contain secret proprietary questions that their schools won't let them share with each other.
* If Student B studies completely alone, they only see 20 questions and fail the exam (**31.25% accuracy**).
* **The Federated Teacher (Central Server):**
  1. The teacher creates an initial study guide (global weights $W$).
  2. The teacher sends the study guide to all 3 students.
  3. Each student studies at their own desk and notes corrections based on their local practice questions.
  4. The students do **not** send their private practice questions to the teacher. They only send a list of suggested bullet-point changes ($\Delta W$).
  5. The teacher averages all the suggestions together (**FedAvg**) to update the master study guide.
  6. The teacher sends the updated guide back out.
  7. After 15 rounds, all 3 students have a master study guide that scores **68.75% - 75.00%** on the test!

### Why FedAvg MUST use Logistic Regression (and NOT Random Forest):
* **Why Logistic Regression works:** A Logistic Regression model consists of a weight vector $W = [w_1, w_2, \dots, w_{43}]$ and a bias $b$. These are simple numbers. If Hospital A says $w_1 = 0.4$ and Hospital B says $w_1 = 0.6$, the server averages them: $\frac{0.4 + 0.6}{2} = 0.5$. That is mathematically valid!
* **Why Random Forest FAILS in FedAvg:** A Random Forest is a forest of decision trees (flowcharts). You cannot "average" the sentence `"IF Heart Rate > 95"` with `"IF Blood Pressure < 80"`. Tree structures cannot be averaged into a single tree without losing meaning or sharing the actual data points.

---

## 7. The Results: What actually happened and WHY?

Let's look at the real empirical comparison from our experiments:

### 1. Isolated Hospitals (No Sharing):
* **Hospital A alone:** 62.50%
* **Hospital B alone:** **31.25%** (disaster!)
* **Hospital C alone:** 56.25%

*Why did Hospital B do so badly?*  
Hospital B only had 22 training patients. By pure chance, its small cohort had slightly different distributions. The model overfit completely and flunked the external test set. This proves that **single hospitals working alone cannot build safe clinical AI**.

### 2. Federated Learning (Collaborative without Sharing Data):
* **Federated Global Model (IID):** **68.75% accuracy (ROC-AUC 0.7188)**.
* **Federated Global Model (Non-IID):** **75.00% accuracy (ROC-AUC 0.7500)**.

### What is IID vs Non-IID?
* **IID (Independent and Identically Distributed):** Every hospital gets an equal, balanced mix of 50% AMI and 50% Control patients.
* **Non-IID (Real-World Skew):** Real hospitals are not identical!
  * Hospital A is a cardiac emergency center $\rightarrow$ 76% of its patients have heart attacks.
  * Hospital B is a community clinic $\rightarrow$ only 23% have heart attacks.
  * Hospital C is a general clinic $\rightarrow$ 52% have heart attacks.
* **Our Finding:** FedAvg still converged smoothly under Non-IID skew, reaching 75% accuracy! The collaborative model learned from the specialty hospital's positive examples AND the community hospital's control examples.

---

## 8. The Web Application: How does it work?

We built a clean Flask web application to make this research interactive:

1. **Live Clinical Inference (`/predict`):**
   * You can type in any patient's vitals (e.g. HR = 112, SBP = 92, DBP = 60, SpO₂ = 91).
   * Or click the preset buttons (`High-Risk AMI Profile`, `Stable Control Profile`).
   * The backend takes those readings, calculates the exact same derived features (Shock Index, Pulse Pressure), normalizes them using the saved `scaler.joblib`, runs the Federated model, and returns:
     * Diagnostic Label (AMI Positive vs Control)
     * Exact estimated probability percentage
     * Visual color-coded risk meter (Red for High Risk, Green for Low Risk)
     * Clinical indicators
2. **Model Benchmarks (`/comparison`):**
   * Displays all comparison tables.
   * Renders the publication-ready figures (ROC curves, confusion matrices, feature importances).
3. **Federated Network Visualizer (`/federated`):**
   * Visualizes the central coordinator server and the 3 hospital client nodes.
   * Shows the mathematical formulas and the step-by-step communication round history.
4. **REST API (`/api/predict`):**
   * Any external system can send a `POST` request with JSON vitals and receive back a predictive score.

---

## 9. Viva / Defense Cheat Sheet: 10 Questions Your Examiner Will Ask

### Q1: What is the primary objective of your project?
**Answer:** To classify whether an intensive care admission is associated with Acute Myocardial Infarction (AMI) using only the first 24 hours of non-invasive vital signs, while preserving patient privacy across distributed hospitals using Federated Learning (FedAvg).

### Q2: What dataset did you use, and how did you define the target label?
**Answer:** We used MIMIC-IV version 3.1. AMI-positive patients (label = 1) were identified by ICD-9 code `410` and ICD-10 codes `I21–I22`. Controls (label = 0) were patients admitted to cardiac care without those diagnosis codes. We had 80 unique ICU patients with complete vital sign chart events.

### Q3: Why did you limit the observation window to the first 24 hours?
**Answer:** To prevent lookahead data leakage. In an actual hospital emergency, a physician needs an assessment within the first day of arrival. Using vitals recorded days or weeks later would introduce hindsight bias into the model.

### Q4: How did you transform continuous time-series vitals into tabular features?
**Answer:** We engineered 10 statistical biomarkers for each of the 4 vital signs (Mean, Min, Max, Standard Deviation, Range, First reading, Last reading, Net change, OLS linear regression slope/trend, and Observation count), plus composite hemodynamic markers (Shock Index and Pulse Pressure), producing a fixed 43-feature vector representing each patient as exactly one row.

### Q5: Why did simple Logistic Regression outperform Random Forest and Gradient Boosting?
**Answer:** Because our dataset contains 80 patients. In small-sample regimes with 43 features, complex tree-based ensembles suffer from high variance and overfit to noise. Logistic Regression with $L_2$ regularization has low variance, smooth linear decision boundaries, and adheres to Occam's Razor, generalizing better on held-out test data.

### Q6: What is Federated Learning and how does FedAvg work?
**Answer:** Federated Learning is a decentralized machine learning technique where multiple clients collaboratively train a shared model without exchanging raw data. In FedAvg (Federated Averaging), each hospital trains a local model on its private data for $E$ epochs, sends only its weight updates to the central server, and the server computes a sample-weighted average of those weights to produce the next global model.

### Q7: Why did you implement Federated Logistic Regression in PyTorch instead of Scikit-Learn?
**Answer:** Scikit-learn's Logistic Regression optimizer does not expose intermediate weight tensors for round-by-round parameter aggregation. PyTorch explicitly exposes the weight tensor $W$ and bias $b$, allowing us to implement exact mathematical Federated Averaging and perform local SGD epochs at each hospital node.

### Q8: What did your local hospital experiment demonstrate?
**Answer:** It demonstrated that isolated training fails when data is scarce. Hospital B trained alone achieved only 31.25% accuracy. However, when Hospital B collaborated with Hospitals A and C via FedAvg, the resulting global model achieved 68.75%–75.00% accuracy on the exact same global test set.

### Q9: What is the difference between your IID and Non-IID experiments?
**Answer:** In the IID setting, patients were randomly distributed so every hospital had an identical 50% positive class balance. In the Non-IID setting, we simulated real-world institutional specialization: Hospital A had 76% positive AMI cases (cardiac specialty center), Hospital B had 23% positive cases (community hospital), and Hospital C was balanced (52%). FedAvg successfully converged in both settings.

### Q10: Is this system ready to be used in a real hospital today?
**Answer:** No. It is an academic research prototype. It demonstrates the technical feasibility and privacy preservation of Federated Learning on vital signs. However, 80 patients is a small proof-of-concept cohort, and administrative ICD diagnosis codes are not the same as prospective clinical bedside validation. Before clinical deployment, it would require large multi-center clinical trials and regulatory approval.
