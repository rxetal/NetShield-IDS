import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.pipeline import NetShieldPipeline

print("==================================================")
print("     EVALUATING END-TO-END NETSHIELD PIPELINE     ")
print("==================================================")

test_path = "data/processed/test_set.parquet"

if not os.path.exists(test_path):
    raise FileNotFoundError("🚨 Test set missing! Please run train.py first.")

test_df = pd.read_parquet(test_path)
pipeline = NetShieldPipeline()

print(f"[EVALUATION] Running FAST batch evaluation on {len(test_df):,} test samples...")

# 1. التوقع دفعة واحدة (استغراق ثوانٍ معدودة)
y_pred_tier1, y_pred_final = pipeline.predict_batch(test_df)

# 2. تجهيز القيم الحقيقية
y_true_tier1 = test_df["label"].values
y_true_final = test_df.apply(
    lambda row: "Normal" if row["label"] == 0 else str(row["attack_cat"]), 
    axis=1
).values

# ==========================================
# TIER 1 RESULTS
# ==========================================
tier1_acc = accuracy_score(y_true_tier1, y_pred_tier1)

tier1_report = (
    "TIER 1 END-TO-END RESULTS\n"
    "=========================\n"
    f"Accuracy: {tier1_acc * 100:.2f}%\n\n"
    + classification_report(
        y_true_tier1,
        y_pred_tier1,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)

print("\n" + tier1_report)

# ==========================================
# FULL TWO-TIER RESULTS
# ==========================================
final_acc = accuracy_score(y_true_final, y_pred_final)

final_report = (
    "FULL TWO-TIER END-TO-END RESULTS\n"
    "================================\n"
    f"Accuracy: {final_acc * 100:.2f}%\n\n"
    + classification_report(
        y_true_final,
        y_pred_final,
        zero_division=0
    )
)

print("\n" + final_report)

# ==========================================
# SAVE RESULTS
# ==========================================
os.makedirs("results/metrics", exist_ok=True)
report_file_path = "results/metrics/end_to_end_report.txt"

with open(report_file_path, "w") as f:
    f.write(tier1_report)
    f.write("\n\n" + "=" * 50 + "\n\n")
    f.write(final_report)

print("\n✅ Full two-tier End-to-End evaluation completed successfully.")
print(f"📁 Report saved to {report_file_path}")