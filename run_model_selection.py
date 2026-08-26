import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("[INFO] Loading cleaned dataset for Model Selection...")
# البحث عن ملف Parquet المتاح
dataset_paths = [
    'data/processed/clean_unsw_nb15.parquet',
    'data/processed/cleaned_unsw_nb15.parquet',
    'data/processed/test_set.parquet'
]

data_path = None
for path in dataset_paths:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    raise FileNotFoundError("Cleaned dataset not found in data/processed/")

df = pd.read_parquet(data_path)

# إعداد Features و Target
cat_cols = ['proto', 'service', 'state']
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols])

X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
y = df['label']

# تقسيم البيانات 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# تعريف النماذج المقارنة
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost (Proposed)": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
}

results = []

print("\n==================================================")
print("          STARTING MODEL SELECTION EVALUATION     ")
print("==================================================")

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Training Time (s)": round(train_time, 2)
    })
    print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")

# إنشاء DataFrame للنتائج
results_df = pd.DataFrame(results)

# حفظ النتائج في مجلد results/figures
os.makedirs("results/figures", exist_ok=True)

# رسم بياني للمقارنة
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="F1-Score", palette="Blues_d")
plt.title("Tier 1 Baseline Comparison (F1-Score Evaluation)", fontsize=14, fontweight='bold')
plt.ylim(0.8, 1.0)
plt.ylabel("F1-Score")
plt.xlabel("Classification Models")
plt.tight_layout()

chart_path = "results/figures/model_selection_comparison.png"
plt.savefig(chart_path, dpi=300)
plt.close()

print("\n==================================================")
print("             MODEL SELECTION SUMMARY TABLE        ")
print("==================================================")
print(results_df.to_string(index=False))
print(f"\n[SUCCESS] Comparison chart saved to: {chart_path}")