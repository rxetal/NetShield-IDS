```markdown
# NetShield AI — Hierarchical Intrusion Detection System

Machine Learning model and interactive web application for detecting network intrusions, classifying threat severity, and analyzing traffic behavior using real-world network data.

🎓 **National Telecommunications Institute (NTI)** — Capstone Project 2026

---

## Project Overview

NetShield AI addresses malicious network activity using a **Hierarchical AI Pipeline** built on the **UNSW-NB15 dataset**. Instead of a single model, the system routes traffic through a two-tier classification architecture and an unsupervised analysis engine:

* **Tier 1 (Binary):** Instantly screens network flows as Normal or Attack traffic (**99.25% Accuracy**).
* **Tier 2 (Multiclass):** Categorizes detected threats into 9 specific attack vectors (**81.04% F1-Score**).
* **Risk Score Engine:** Calculates dynamic severity levels (Medium, High, Critical) based on classification confidence.
* **Unsupervised Clustering:** Uses PCA (2-Component) and K-Means to discover behavioral patterns in unlabelled traffic.

The project demonstrates a complete machine learning workflow, including exploratory data analysis, strict data leakage prevention, model benchmarking, hierarchical inference, and interactive dashboard deployment.

---

## System Architecture

```text
                  [ Incoming Network Traffic Flow ]
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   Tier 1: Binary Classifier  │
                   │    (Normal vs. Attack Flow)  │
                   └──────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ 🟢 Normal Traffic ]            [ 🚨 Threat Detected ]
     (End-to-End Clearance)                       │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │  Tier 2: Multiclass Model    │
                                   │   (9 Attack Categorizations) │
                                   └──────────────┬───────────────┘
                                                  │
                                                  ▼
                                   [ Categorized Threat Vector ]
                                (Generic, Exploits, DoS, Fuzzers, etc.)

```

---

## System Performance & Metrics

| Evaluation Phase | Target Objective | Accuracy / Performance |
| --- | --- | --- |
| **Tier 1 Model** | Binary Traffic Screening (Normal vs Attack) | **99.25%** |
| **Tier 2 Model** | 9-Class Threat Categorization | **81.04%** |
| **End-to-End Pipeline** | Full Hierarchical Evaluation | **97.89%** |

---

## Application Demo

An interactive dashboard for NetShield was developed using Streamlit to allow real-time network flow inspection, risk scoring, model benchmarking, and cluster analysis.

* **Page 1 — 📊 System Dashboard & Metrics:** High-level statistical KPI cards and interactive confusion matrices.
* **Page 2 — 🔍 Interactive Traffic Inspector:** Live inspection of UNSW-NB15 flows with risk scores (0–100), alert banners, and 49-feature telemetry breakdowns.
* **Page 3 — 🔬 Model Selection & Unsupervised Analysis:** Benchmarks comparing XGBoost against Random Forest, Decision Trees, Logistic Regression, and SVM, alongside 2D PCA threat clustering.

You can run the web app locally using:

```bash
streamlit run app/main.py

```

---

## Dataset

The **UNSW-NB15 dataset** is used for training and evaluating the models.

Due to file size constraints, the dataset is not included directly in this repository. To run the pipeline, obtain the processed dataset and place it in the following path:
`data/processed/cleaned_unsw_nb15.parquet`

---

## Project Workflow

### 1. Exploratory Data Analysis & Leakage Prevention

Network flow attributes (`proto`, `service`, `state`, flow durations, byte rates) were inspected. Categorical encoders and scalers were fitted strictly on the training set before transforming test data to eliminate data leakage.

### 2. Feature Engineering & Preprocessing

Categorical features were transformed using `OrdinalEncoder` configured to handle unseen categories safely during inference. Stratified train/test splits (80/20) preserved rare attack distributions.

### 3. Model Development & Benchmarking

Multiple machine learning algorithms (Logistic Regression, Decision Trees, Random Forests, SVM, and XGBoost) were benchmarked. **XGBoost** was selected as the top performer for both Tier 1 and Tier 2 models.

### 4. Unsupervised Behavior Analysis

A combined **PCA + K-Means** pipeline reduces high-dimensional flow attributes to 2 components, uncovering spatial cluster anomalies without relying on target labels.

---

## Technologies and Libraries

* **Core & Processing:** Python, pandas, numpy, joblib
* **Machine Learning:** scikit-learn, xgboost
* **Visualization & Interface:** streamlit, matplotlib, seaborn

---

## Repository Structure

```text
NetShield-IDS/
├── app/
│   └── main.py                     # Streamlit Multi-Page Web Application
├── data/                           # Processed UNSW-NB15 Parquet Data (not tracked)
├── models/                         # Trained Artifacts (.joblib Models & Encoders)
├── results/
│   └── figures/                    # Confusion Matrices, Benchmarks & Cluster Plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data Ingestion & Memory Optimization
│   ├── models.py                   # Model Architectures
│   ├── pipeline.py                 # Hierarchical Inference Engine
│   └── visualization.py            # Plot & Chart Generators
├── train.py                        # Unified Model Training Script
├── run_model_selection.py          # Baseline Model Benchmarking
├── run_unsupervised.py             # PCA + K-Means Clustering Script
├── requirements.txt
└── README.md

```

---

## How to Run

1. **Clone this repository:**
```bash
git clone [https://github.com/rxetal/NetShield-IDS.git](https://github.com/rxetal/NetShield-IDS.git)
cd NetShield-IDS

```


2. **Install required libraries:**
```bash
pip install -r requirements.txt

```


3. **Set up the dataset:**
Place `cleaned_unsw_nb15.parquet` inside the `data/processed/` directory.
4. **Run the execution pipeline:**
```bash
# Step 1: Train Tier 1 & Tier 2 models
python train.py

# Step 2: Generate model selection comparison benchmarks
python run_model_selection.py

# Step 3: Run PCA & K-Means unsupervised threat clustering
python run_unsupervised.py

# Step 4: Launch the interactive Streamlit Web App
streamlit run app/main.py

```



---

## Project Goal

The goal of this project is to build an end-to-end, leak-free machine learning system capable of real-time network intrusion detection, risk assessment, and traffic analysis. The project demonstrates a complete machine learning lifecycle including benchmarking, hierarchical classification, unsupervised clustering, and interactive deployment.

```

```