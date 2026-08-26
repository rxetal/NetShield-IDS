import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# إضافة المسار الرئيسي للمشروع لاستيراد الـ Pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import NetShieldPipeline

# إعدادات الصفحة
st.set_page_config(
    page_title="NetShield | AI Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# تحميل الـ Pipeline
@st.cache_resource
def load_pipeline():
    return NetShieldPipeline()

pipeline = load_pipeline()

# العنوان الرئيسي
st.title("🛡️ NetShield: Hierarchical AI Intrusion Detection System")
st.markdown("An end-to-end multi-tiered threat detection platform for network traffic evaluation.")

# القائمة الجانبية المحدثة
st.sidebar.header("🕹️ Navigation & Controls")
menu = st.sidebar.radio("Go to:", [
    "📊 System Dashboard & Metrics", 
    "🔍 Interactive Traffic Inspector",
    "🔬 Model Selection & Unsupervised Analysis"
])

# تحميل بيانات الاختبار المجهزة
@st.cache_data
def load_test_data():
    paths = [
        "data/processed/cleaned_unsw_nb15.parquet",
        "data/processed/clean_unsw_nb15.parquet",
        "data/processed/test_set.parquet"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_parquet(p)
    return None

test_df = load_test_data()


# ==========================================
# PAGE 1: DASHBOARD & METRICS
# ==========================================
if menu == "📊 System Dashboard & Metrics":
    st.header("📈 Architecture & Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tier 1 Accuracy", "99.25%", "Normal vs Attack")
    col2.metric("Tier 2 Accuracy", "81.04%", "9 Attack Classes")
    col3.metric("System End-to-End Accuracy", "97.89%", "Full Pipeline")
    
    st.divider()
    
    # ------------------------------------------
    # 1. Confusion Matrices
    # ------------------------------------------
    st.subheader("🖼️ Model Confusion Matrices")
    c1, c2 = st.columns(2)
    
    t1_cm_path = "results/figures/tier1_confusion_matrix.png"
    t2_cm_path = "results/figures/tier2_confusion_matrix.png"
    
    with c1:
        st.write("**Tier 1 Binary Confusion Matrix**")
        if os.path.exists(t1_cm_path):
            st.image(t1_cm_path, use_container_width=True)
        else:
            st.info("Run Phase 5 to generate Tier 1 plot.")
            
    with c2:
        st.write("**Tier 2 Multiclass Confusion Matrix**")
        if os.path.exists(t2_cm_path):
            st.image(t2_cm_path, use_container_width=True)
        else:
            st.info("Run Phase 6 to generate Tier 2 plot.")

    st.divider()

    # ------------------------------------------
    # 2. Target Class Distributions (التعديل الجديد)
    # ------------------------------------------
    st.subheader("📊 Dataset Class Imbalance Analysis")
    st.markdown("Visualizing the traffic class distribution across Tier 1 (Binary) and Tier 2 (Multiclass) targets.")
    
    d1, d2 = st.columns(2)
    
    t1_dist_path = "results/figures/tier1_distribution.png"
    t2_dist_path = "results/figures/tier2_distribution.png"
    
    with d1:
        st.write("**Tier 1: Normal vs Attack Distribution**")
        if os.path.exists(t1_dist_path):
            st.image(t1_dist_path, use_container_width=True)
        else:
            st.info("Tier 1 distribution plot not found in `results/figures/`.")
            
    with d2:
        st.write("**Tier 2: 9 Attack Categories Breakdown**")
        if os.path.exists(t2_dist_path):
            st.image(t2_dist_path, use_container_width=True)
        else:
            st.info("Tier 2 distribution plot not found in `results/figures/`.")


# ==========================================
# PAGE 2: INTERACTIVE TRAFFIC INSPECTOR
# ==========================================
elif menu == "🔍 Interactive Traffic Inspector":
    st.header("📊 Interactive Network Traffic Sample Inspector")
    st.markdown("Inspect network flow samples from UNSW-NB15 dataset using the NetShield Pipeline.")
    
    if test_df is not None:
        if st.button("Inspect Random Network Sample"):
            sample = test_df.sample(1).reset_index(drop=True)
            result = pipeline.predict_single(sample)
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Traffic Status", result["tier1_label"])
            col2.metric("Detected Attack Vector", result["attack_type"])
            col3.metric("Risk Score", f"{result['risk_score']}/100")
            
            st.markdown("---")
            
            if result["severity"] == "Critical":
                st.error(f"🚨 Threat Level: CRITICAL (Confidence: {result['attack_prob']*100:.1f}%)")
            elif result["severity"] in ["High", "Medium"]:
                st.warning(f"⚠️ Threat Level: {result['severity'].upper()} (Confidence: {result['attack_prob']*100:.1f}%)")
            else:
                st.success("✅ Healthy Traffic Flow (No Threat Detected)")
                
            st.subheader("📋 Inspect Key Network Features")
            key_cols = [c for c in ['proto', 'service', 'state', 'dur', 'sbytes', 'dbytes'] if c in sample.columns]
            if key_cols:
                st.dataframe(sample[key_cols], use_container_width=True)
            
            with st.expander("🔍 View All 49 Raw Features"):
                st.dataframe(sample, use_container_width=True)
    else:
        st.error("Cleaned dataset missing in 'data/processed/'. Please make sure your parquet file exists.")

# ==========================================
# PAGE 3: MODEL SELECTION & UNSUPERVISED
# ==========================================
elif menu == "🔬 Model Selection & Unsupervised Analysis":
    st.header("🔬 Model Selection & Unsupervised Anomaly Detection")
    st.markdown("Comprehensive evaluation justifying baseline selection and unsupervised cluster analysis.")
    
    tab1, tab2 = st.tabs(["📊 Baseline Model Selection", "🌀 Unsupervised PCA & K-Means"])
    
    with tab1:
        st.subheader("Model Comparison Justification (Tier 1 Binary Classification)")
        ms_chart = "results/figures/model_selection_comparison.png"
        if os.path.exists(ms_chart):
            st.image(ms_chart, use_container_width=True)
            st.success("✅ XGBoost was selected as the core architecture due to superior F1-score performance and efficiency on tabular network features.")
        else:
            st.info("Run `python run_model_selection.py` to generate model comparison metrics.")
            
    with tab2:
        st.subheader("Unsupervised Threat Clustering (K-Means & PCA)")
        unsup_chart = "results/figures/unsupervised_clusters.png"
        if os.path.exists(unsup_chart):
            st.image(unsup_chart, use_container_width=True)
            st.caption("Visualizing high-dimensional UNSW-NB15 flow data projected onto 2 principal components.")
        else:
            st.info("Run `python run_unsupervised.py` to generate cluster visualizations.")