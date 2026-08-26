import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models import prepare_tier1_data

print("==================================================")
print("     RUNNING MODEL SELECTION (LEAK-FREE)          ")
print("==================================================")

# 1. تحميل مجموعات البيانات المعزولة
train_path = "data/processed/train_set.parquet"
test_path = "data/processed/test_set.parquet"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("🚨 Processed datasets missing! Please run train.py first.")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. تحميل الـ Encoder المعتمد في التدريب الرئيسي
encoder_path = "models/categorical_encoder.joblib"
if not os.path.exists(encoder_path):
    raise FileNotFoundError("🚨 Categorical encoder missing! Please run train.py first.")

cat_encoder = joblib.load(encoder_path)

# 3. تجهيز الميزات لـ Tier 1
X_train, y_train = prepare_tier1_data(train_df, cat_encoder)
X_test, y_test = prepare_tier1_data(test_df, cat_encoder)

# 4. تعريف الخوارزميات للمقارنة
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
}

results = []

for name, model in models.items():
    print(f"[BENCHMARK] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })

# 5. حفظ وحرض التقرير والرسم البياني
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\n", results_df.to_string(index=False))

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

results_df.to_csv("results/metrics/model_selection_comparison.csv", index=False)

# رسم بياني للمقارنة
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="F1-Score", palette="Blues_d")
plt.title("Tier 1 Algorithm Selection (F1-Score Comparison)", fontsize=12, fontweight='bold')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("results/figures/model_selection_f1.png", dpi=300)
plt.close()

print("\n✅ Model Selection benchmark complete. Results saved to results/metrics/")