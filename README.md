# 🛡️ NetShield AI — Hierarchical Intrusion Detection System

> A premium, award-worthy web application and machine learning engine built for network threat classification, anomaly detection, and real-time security inspection.

🎓 **NTI (National Telecommunications Institute) — Capstone Project 2026**

---

## 🎨 DESIGN & SYSTEM ARCHITECTURE

### 🌌 Color Palette & Visual Style
* **Background:** `#0A0E1A` (Deep Navy Black)
* **Surface Cards:** `#111827` with subtle border `#1F2937`
* **Primary Accent:** `#3B82F6` (Electric Blue) — Active states, primary buttons, indicators
* **Secondary Accent:** `#6366F1` (Indigo) — Gradients, hover states, badges
* **Threat Levels:**
  * 🟢 **Normal / Healthy Flow:** `#10B981` (Emerald Green) with subtle glow
  * 🟡 **Medium / High Risk:** `#F59E0B` (Amber Gold) with glow
  * 🚨 **Critical Severity:** `#EF4444` (Vivid Red) with pulsing glow
* **Text:** Primary `#F9FAFB` | Secondary `#9CA3AF`
* **Typography:** `Inter` (Clean, modern, technical)
* **UI Elements:** Glassmorphism (`rgba(255,255,255,0.04)`, backdrop blur `12px`), responsive metric cards, interactive data inspection tables, dynamic confusion matrix plots.

---

## 🏗️ MULTI-TIERED ML ARCHITECTURE

NetShield utilizes a **Hierarchical AI Pipeline** designed to optimize threat detection speed and accuracy across tabular UNSW-NB15 network traffic flows:


```

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

```

1. **Tier 1 (Binary Classification):** Employs an optimized XGBoost model to instantly isolate normal traffic from malicious packets.
2. **Tier 2 (Multiclass Classification):** Passes flagged attack traffic into a secondary XGBoost model to categorize the threat into 9 distinct attack categories.
3. **Risk Score Engine:** Calculates an instant threat severity rating (0–100) paired with actionable alert statuses (Healthy, Medium, High, Critical).
4. **Unsupervised Anomaly Detection:** Applies **PCA (Principal Component Analysis)** dimensionality reduction combined with **K-Means Clustering** to map network flow behavior in 2D space.

---

## 📊 SYSTEM PERFORMANCE & METRICS

| Evaluation Phase | Target Objective | Accuracy / F1-Score |
| :--- | :--- | :--- |
| **Tier 1 Model** | Binary Traffic Screening (Normal vs Attack) | **99.25%** |
| **Tier 2 Model** | 9-Class Threat Categorization | **81.04%** |
| **End-to-End Pipeline** | Full Hierarchical Evaluation | **97.89%** |

---

## 🌐 APPLICATION PAGES & NAVIGATION

### PAGE 1 — 📊 System Dashboard & Metrics
* **Key Metrics Header:** High-level statistical cards displaying Tier 1 Accuracy, Tier 2 Accuracy, and System End-to-End Performance.
* **Confusion Matrices:** Side-by-side interactive visual inspection for Binary and Multiclass classification outputs.

### PAGE 2 — 🔍 Interactive Traffic Inspector
* **Traffic Flow Sampler:** Live inspection trigger evaluating random UNSW-NB15 network flows using the NetShield Pipeline.
* **Threat Telemetry Display:** Instant status metrics showing Traffic Status, Detected Attack Vector, and Risk Score (0–100).
* **Alert Banners:** Color-coded severity callouts (Critical Red, Warning Amber, Healthy Green).
* **Feature Inspection:** Structured view highlighting key network parameters (`proto`, `service`, `state`, `dur`, `sbytes`, `dbytes`) with an expandable raw view for all 49 packet features.

### PAGE 3 — 🔬 Model Selection & Unsupervised Analysis
* **Tab 1 — Baseline Model Selection:** Comparative benchmarks justifying XGBoost against Random Forest, Decision Trees, Logistic Regression, and SVM.
* **Tab 2 — Unsupervised Threat Clustering:** High-dimensional spatial visualization using 2-Component PCA and K-Means clustering to uncover unlabelled network flow anomalies.

---

## 📂 REPOSITORY STRUCTURE

```text
NetShield-IDS/
├── app/
│   └── main.py                     # Streamlit Multi-Page Web Application
├── data/                           # Processed & Cleaned UNSW-NB15 Parquet Data
├── models/                         # Trained Artifacts (.joblib Models & Ordinal Encoders)
├── results/
│   └── figures/                    # Confusion Matrices, Model Benchmarks & Cluster Plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data Ingestion & Memory Optimization
│   ├── models.py                   # XGBoost & Pipeline Architectures
│   ├── pipeline.py                 # NetShield Hierarchical Inference Engine
│   └── visualization.py            # Plot & Chart Generators
├── train.py                        # Unified Model Training & Artifact Persistence Script
├── run_model_selection.py          # Benchmark Model Selection Engine
├── run_unsupervised.py             # PCA + K-Means Clustering Script
├── requirements.txt                # Python Dependencies
└── README.md                       # Project Documentation

```

---

## 🚀 QUICK START & LOCAL DEVELOPMENT

### 1. Environment Setup

Ensure you have Python 3.9+ installed, then clone this repository and install dependencies:

```sh
git clone [https://github.com/YOUR_USERNAME/NetShield-IDS.git](https://github.com/YOUR_USERNAME/NetShield-IDS.git)
cd NetShield-IDS
pip install -r requirements.txt

```

### 2. Execution Order

```sh
# Step 1: Train Tier 1/2 models and persist Ordinal Encoders
python train.py

# Step 2: Generate baseline model selection comparison benchmarks
python run_model_selection.py

# Step 3: Run PCA & K-Means unsupervised threat clustering
python run_unsupervised.py

# Step 4: Launch the interactive Streamlit Web App
streamlit run app/main.py

```

---

🎓 **National Telecommunications Institute (NTI) — Graduation Project 2026**

```

```