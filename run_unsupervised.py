import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("[INFO] Loading cleaned dataset for Unsupervised Learning...")
dataset_paths = [
    'data/processed/clean_unsw_nb15.parquet',
    'data/processed/cleaned_unsw_nb15.parquet',
    'data/processed/test_set.parquet'
]

data_path = next((p for p in dataset_paths if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("Cleaned dataset missing!")

df = pd.read_parquet(data_path)

# تحويل الأعمدة النصية
cat_cols = ['proto', 'service', 'state']
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols])

X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
y = df['label'] if 'label' in df.columns else None

# 1. Scaling البيانات (ضروري جداً للـ PCA والـ K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. تطبيق PCA لتقليل الأبعاد لـ 2 Components
print("[INFO] Applying PCA (Dimensionality Reduction)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"[INFO] PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

# 3. تطبيق K-Means Clustering (k=2: Normal Cluster vs Anomaly Cluster)
print("[INFO] Fitting K-Means Clustering model...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# حفظ نماذج الـ Unsupervised
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca_model.joblib")
joblib.dump(kmeans, "models/kmeans_model.joblib")

# 4. رسم النتائج بيانيًا
os.makedirs("results/figures", exist_ok=True)

plt.figure(figsize=(12, 5))

# Plot 1: K-Means Clusters
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=10)
plt.title("K-Means Clustering (PCA Reduced)", fontweight='bold')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Plot 2: Actual Ground Truth (لو متوفرة)
if y is not None:
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
    plt.title("Actual Labels (0: Normal, 1: Attack)", fontweight='bold')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

plt.tight_layout()
chart_path = "results/figures/unsupervised_clusters.png"
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"[SUCCESS] Unsupervised models saved to 'models/' and plot saved to: {chart_path}")