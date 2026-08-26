import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.models import transform_categorical_features

print("==================================================")
print("     RUNNING UNSUPERVISED ANALYSIS (LEAK-FREE)    ")
print("==================================================")

train_path = "data/processed/train_set.parquet"
encoder_path = "models/categorical_encoder.joblib"

if not os.path.exists(train_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("🚨 Please run train.py first to generate training set and encoder.")

train_df = pd.read_parquet(train_path)
cat_encoder = joblib.load(encoder_path)

# 1. Encoding & Scaling على Train Set
df_encoded = transform_categorical_features(train_df, cat_encoder)
X = df_encoded.drop(columns=['label', 'attack_cat'], errors='ignore')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# 4. حفظ النماذج لاستخدامها مستقبلاً
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/unsupervised_scaler.joblib")
joblib.dump(pca, "models/unsupervised_pca.joblib")
joblib.dump(kmeans, "models/unsupervised_kmeans.joblib")

# 5. الرسم والحفظ
os.makedirs("results/figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', alpha=0.6, s=15)
plt.title("Unsupervised Threat Clusters (PCA + K-Means)", fontsize=12, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("results/figures/unsupervised_clusters.png", dpi=300)
plt.close()

print("✅ Unsupervised analysis completed successfully. Results saved to results/figures/")