import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("     UNSUPERVISED ANOMALY CLUSTERING PIPELINE    ")
print("==================================================")

train_path = "data/processed/train_set.parquet"
if not os.path.exists(train_path):
    train_path = "data/processed/clean_unsw_nb15.parquet"

df_train = pd.read_parquet(train_path)

cat_cols = ['proto', 'service', 'state']
encoder = joblib.load("models/categorical_encoder.joblib")

df_encoded = df_train.copy()
df_encoded[cat_cols] = encoder.transform(df_encoded[cat_cols])
X = df_encoded.drop(columns=['label', 'attack_cat'], errors='ignore')

# 1. Fit & Save StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit & Save PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. Fit & Save KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/unsupervised_scaler.joblib")
joblib.dump(pca, "models/unsupervised_pca.joblib")
joblib.dump(kmeans, "models/unsupervised_kmeans.joblib")

# 4. Visualization
os.makedirs("results/figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', alpha=0.5)
plt.title("Unsupervised Anomaly Clustering (PCA + K-Means)", fontsize=12, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("results/figures/unsupervised_clusters.png", dpi=300)
plt.close()

print("✅ Unsupervised Pipeline Executed & Artifacts Saved Successfully.")