import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline import NetShieldPipeline

print("==================================================")
print("     RUNNING PHASE 7: END-TO-END PIPELINE EVAL    ")
print("==================================================")

# 1. تحميل Held-out Test Set المحدد سابقاً في Master Split
test_path = "data/processed/test_set.parquet"
if not os.path.exists(test_path):
    raise FileNotFoundError("🚨 test_set.parquet not found. Please run train.py first!")

print(f"[INFO] Loading Held-out Test Set: {test_path}")
test_df = pd.read_parquet(test_path)

# 2. تحميل الـ Pipeline الشامل
pipeline = NetShieldPipeline()

# 3. إحداث التوقعات على Held-Out Test Data
print("[INFO] Evaluating End-to-End Pipeline...")
final_preds, t1_preds, t1_probas = pipeline.predict(test_df)

# إعداد القيم الحقيقية (Ground Truth)
y_true = []
for _, row in test_df.iterrows():
    if row['label'] == 0:
        y_true.append("Normal")
    else:
        y_true.append(row['attack_cat'])

# 4. حساب دقة الـ Pipeline ككل
e2e_acc = accuracy_score(y_true, final_preds)

print(f"\n==========================================")
print(f" END-TO-END PIPELINE ACCURACY : {e2e_acc * 100:.2f}%")
print(f"==========================================")

# حفظ التقرير والتنفيذ
os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/end_to_end_report.txt", "w") as f:
    f.write(f"End-to-End Pipeline Overall Accuracy: {e2e_acc * 100:.2f}%\n\n")
    f.write(classification_report(y_true, final_preds))

print("✅ Phase 7 Evaluation Complete. Report saved to results/metrics/end_to_end_report.txt")