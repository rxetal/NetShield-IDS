```markdown
# 🛡️ NetShield AI — Two-Tier Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (IDS) designed to detect malicious network traffic, classify detected attacks into specific threat categories, and provide an additional unsupervised view of network behavior.

🎓 **National Telecommunications Institute (NTI)** — Capstone Project 2026

---

## 📌 Project Overview

Network attacks are becoming increasingly diverse and difficult to detect using traditional rule-based security systems.

**NetShield AI** addresses this problem using a **Hierarchical AI Pipeline** built on the **UNSW-NB15 dataset**. Instead of passing every network flow through a single complex multiclass model, the system follows a structured multi-tiered approach:

* **Tier 1 (Binary Detection):** Instantly screens incoming traffic as Normal or Attack.
* **Tier 2 (Multiclass Classification):** Categorizes detected threats into specific attack categories.
* **Risk Score Engine:** Calculates an instant threat severity rating paired with alert levels (Medium, High, Critical).
* **Unsupervised Clustering:** Employs PCA and K-Means to discover latent traffic behavior and anomalies in 2D space.

---

## 🧠 System Architecture

NetShield follows a hierarchical machine learning architecture:

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
                                   │   (Attack Categorization)    │
                                   └──────────────┬───────────────┘
                                                  │
                                                  ▼
                                   [ Categorized Threat Vector ]
                                (Generic, Exploits, DoS, Fuzzers, etc.)

```

### Tier 1 — Binary Classification

The first model determines whether a network flow is **Normal** or an **Attack**. Tier 1 uses an XGBoost binary classifier for high-speed initial screening.

### Tier 2 — Multiclass Attack Classification

If Tier 1 identifies traffic as malicious, the sample is passed to Tier 2. Tier 2 uses XGBoost multiclass classification to isolate the specific attack vector.

### Unsupervised Behavior Analysis

Runs alongside the supervised models using **PCA** and **K-Means** to group network flows by feature-space behavior without needing explicit attack labels.

---

## 📊 System Performance & Metrics

| Evaluation Phase | Target Objective | Metric / Performance |
| --- | --- | --- |
| **Tier 1 Model** | Binary Traffic Screening (Normal vs Attack) | *Check `results/metrics/tier1_report.txt*` |
| **Tier 2 Model** | Multiclass Threat Categorization | *Check `results/metrics/tier2_report.txt*` |
| **End-to-End Pipeline** | Full Hierarchical Evaluation | *Check `results/metrics/end_to_end_report.txt*` |

---

## 🖥️ Application Demo & Features

An interactive multi-page dashboard for NetShield was developed using Streamlit to allow real-time network flow inspection, risk scoring, model benchmarking, and cluster analysis.

* **Page 1 — 📊 System Dashboard & Metrics:** High-level statistical KPI cards and interactive confusion matrices.
* **Page 2 — 🔍 Interactive Traffic Inspector:** Live inspection of UNSW-NB15 flows with risk scores (0–100), alert banners, and 49-feature telemetry breakdowns.
* **Page 3 — 🔬 Model Selection & Unsupervised Analysis:** Benchmarks comparing XGBoost against Random Forest, Decision Trees, Logistic Regression, and SVM, alongside 2D PCA threat clustering.

You can launch the web app locally using:

```bash
streamlit run app/main.py

```

---

## 📊 Dataset

The **UNSW-NB15 dataset** is used for training and evaluating the models.

Due to file size constraints, the dataset is not included directly in this repository. To run the pipeline, place the processed dataset in:
`data/processed/cleaned_unsw_nb15.parquet`

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis & Leakage Prevention

Network flow attributes (`proto`, `service`, `state`, flow durations, byte rates) were inspected. Categorical encoders and scalers were fitted strictly on the training set before transforming test data to eliminate data leakage.

### 2. Feature Engineering & Preprocessing

Categorical features were transformed using `OrdinalEncoder` configured to handle unseen categories safely during inference. Stratified train/test splits (80/20) preserved rare attack distributions.

### 3. Model Development & Benchmarking

Multiple machine learning algorithms (Logistic Regression, Decision Trees, Random Forests, and XGBoost) were benchmarked. **XGBoost** was selected as the top performer for both Tier 1 and Tier 2 models.

### 4. Unsupervised Behavior Analysis

A combined **PCA + K-Means** pipeline reduces high-dimensional flow attributes to 2 components, uncovering spatial cluster anomalies without relying on target labels.

---

## 🛠️ Technologies & Libraries

* **Core & Data:** Python, Pandas, NumPy, Joblib, PyArrow
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization & Web App:** Streamlit, Matplotlib, Seaborn

---

## 📁 Repository Structure

```text
NetShield-IDS/
├── app/
│   └── main.py                     # Streamlit Multi-Page Web Application
├── data/                           # Processed UNSW-NB15 Parquet Data (not tracked)
├── models/                         # Trained Artifacts (.joblib Models & Encoders)
├── results/
│   ├── figures/                    # Confusion Matrices, Benchmarks & Cluster Plots
│   └── metrics/                    # Text & CSV Evaluation Reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data Ingestion & Memory Optimization
│   ├── models.py                   # Model Architectures
│   ├── pipeline.py                 # Hierarchical Inference Engine
│   └── visualization.py            # Plot & Chart Generators
├── train.py                        # Unified Model Training Script
├── run_model_selection.py          # Baseline Model Benchmarking
├── run_unsupervised.py             # PCA + K-Means Clustering Script
├── run_phase7.py                   # End-to-End Evaluation Script
├── requirements.txt
└── README.md

```

---

## 🚀 How to Run

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

# Step 4: Run batch end-to-end evaluation
python run_phase7.py

# Step 5: Launch the interactive Streamlit Web App
streamlit run app/main.py

```



---

## 🎯 Project Goal

The goal of this project is to build an end-to-end, leak-free machine learning system capable of real-time network intrusion detection, risk assessment, and traffic analysis. The project demonstrates a complete machine learning lifecycle including benchmarking, hierarchical classification, unsupervised clustering, and interactive deployment.

```

```