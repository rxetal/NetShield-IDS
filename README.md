# NetShield — Network Intrusion Detection System

> **Hierarchical Machine Learning Pipeline** for real-time network intrusion screening, threat severity scoring, and unsupervised behavior analysis built on the UNSW-NB15 dataset.

*National Telecommunications Institute (NTI) — Capstone Project 2026*

---

## Key Features

* **Tier 1 Binary Screening:** High-speed binary classifier filtering incoming network flows as *Normal* or *Attack*.
* **Tier 2 Multiclass Classification:** Automated threat taxonomy identifying specific attack vectors (Exploits, DoS, Fuzzers, Generic, etc.).
* **Dynamic Severity Engine:** 0–100 risk scoring algorithm generating instant alert levels (*Medium*, *High*, *Critical*).
* **Unsupervised Anomaly Discovery:** 2D PCA and K-Means clustering pipeline exposing latent traffic behavior without ground-truth labels.

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
┌──────────────┴───────────────┐
▼                               ▼
[ Normal Traffic ]              [ Threat Detected ]
(End-to-End Clearance)          │
                                ▼
                 ┌──────────────────────────────┐
                 │  Tier 2: Multiclass Model    │
                 │   (Attack Categorization)    │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 [ Categorized Threat Vector ]
              (Generic, Exploits, DoS, Fuzzers, etc.)


## Stack & Technologies

| Layer | Tools & Libraries |
| --- | --- |
| **Core Runtime** | Python 3.10+, NumPy, Pandas, PyArrow, Joblib |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Dashboard & UI** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |

---

## Dashboard Overview

The included multi-page **Streamlit** application provides complete insight into model analytics and network monitoring:

1. **System Dashboard:** High-level statistical KPI cards and interactive confusion matrices.
2. **Traffic Inspector:** Live UNSW-NB15 flow analysis with 0–100 risk scores, dynamic alert badges, and full 49-feature telemetry.
3. **Model Selection & Clustering:** Comparative benchmark suite (XGBoost vs. Random Forest, Decision Trees, Logistic Regression, SVM) alongside interactive 2D PCA cluster visualizations.

---

## Repository Layout

```text
NetShield-IDS/
├── app/
│   └── main.py                     # Streamlit Multi-Page Interface
├── data/                           # Processed Data Store (Not Tracked)
├── models/                         # Trained Artifacts (.joblib Models & Encoders)
├── results/
│   ├── figures/                    # Confusion Matrices & Clustering Plots
│   └── metrics/                    # Quantitative Evaluation Reports
├── src/
│   ├── data_loader.py              # Ingestion & Memory Optimization
│   ├── models.py                   # Model Architectures
│   ├── pipeline.py                 # Hierarchical Inference Engine
│   └── visualization.py            # Rendering Helpers
├── train.py                        # Primary Model Training Script
├── run_model_selection.py          # Benchmark Execution Script
├── run_unsupervised.py             # PCA + K-Means Pipeline
├── run_phase7.py                   # End-to-End Evaluation Pipeline
├── requirements.txt
└── README.md

```

---

## Quickstart Guide

### 1. Environment Setup

```bash
# Clone repository
git clone [https://github.com/rxetal/NetShield-IDS.git](https://github.com/rxetal/NetShield-IDS.git)
cd NetShield-IDS

# Install dependencies
pip install -r requirements.txt

```

### 2. Dataset Configuration

Place the cleaned UNSW-NB15 dataset file inside the processed data directory:
`data/processed/cleaned_unsw_nb15.parquet`

> *Note: Dataset files are excluded from Git tracking due to size constraints.*

### 3. Execution Pipeline

```bash
# Train models
python train.py

# Run benchmarks & unsupervised analysis
python run_model_selection.py
python run_unsupervised.py
python run_phase7.py

# Launch web dashboard
streamlit run app/main.py

```

---

## Project Goal

The primary goal of NetShield is to build an end-to-end, leak-free machine learning system capable of real-time network intrusion detection, risk assessment, and traffic analysis. The project demonstrates a complete machine learning engineering lifecycle—from data preprocessing and baseline model benchmarking to multi-tier hierarchical classification, unsupervised clustering, and an interactive deployment dashboard.

```

```