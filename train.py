import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print("==================================================")
print("       NETSHIELD UNIFIED TRAINING PIPELINE        ")
print("==================================================")

# 1. تحميل البيانات النظيفة
dataset_paths = [
    'data/processed/clean_unsw_nb15.parquet',
    'data/processed/cleaned_unsw_nb15.parquet',
    'data/processed/test_set.parquet'
]

data_path = next((p for p in dataset_paths if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("🚨 Cleaned dataset missing in data/processed/")

print(f"[INFO] Loading dataset from: {data_path}")
df = pd.read_parquet(data_path)

# ==========================================
# 🔍 حساب وتوثيق الـ Deduplication (إضافة تحت القراءة مباشرة)
# ==========================================
before = len(df)
print(f"Total Rows Before: {before}")

df = df.drop_duplicates()

after = len(df)
print(f"Total Rows After: {after}")
print(f"Duplicates Removed: {before - after}")
# ==========================================

# إثبات وتوثيق الـ Deduplication لو محتاجة
print(f"[INFO] Total dataset records: {len(df)}")

# 2. تجهيز وتدريب الـ Categorical Encoder (المحفظ بشكل موحد)
cat_cols = ['proto', 'service', 'state']
print(f"[INFO] Fitting OrdinalEncoder on categorical features: {cat_cols}")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])

# حفظ الـ Encoder لاستخدامه في الـ Inference وفي Streamlit
os.makedirs("models", exist_ok=True)
encoder_path = "models/categorical_encoder.joblib"
joblib.dump(encoder, encoder_path)
print(f"[SUCCESS] Saved categorical encoder to: {encoder_path}")

# 3. تدريب Tier 1 (Binary Classification: Normal vs Attack)
print("\n--- Training Tier 1 Model (Binary: Normal vs Attack) ---")
X_t1 = df.drop(columns=['label', 'attack_cat'], errors='ignore')
y_t1 = df['label']

X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_t1, y_t1, test_size=0.2, random_state=42, stratify=y_t1
)

model_t1 = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
model_t1.fit(X_train_t1, y_train_t1)

pred_t1 = model_t1.predict(X_test_t1)
acc_t1 = accuracy_score(y_test_t1, pred_t1)
print(f"✅ Tier 1 Accuracy: {acc_t1 * 100:.2f}%")

joblib.dump(model_t1, "models/tier1_xgb.joblib")
print("[SUCCESS] Saved Tier 1 model to models/tier1_xgb.joblib")

# 4. تدريب Tier 2 (Multiclass Classification: Attack Types)
print("\n--- Training Tier 2 Model (Multiclass: Attack Categories) ---")
# تدريب Tier 2 فقط على بيانات الهجمات (Attacks Only)
attacks_df = df[df['label'] == 1].copy()

if 'attack_cat' in attacks_df.columns:
    X_t2 = attacks_df.drop(columns=['label', 'attack_cat'], errors='ignore')
    
    # تحويل target الـ attack_cat إلى أرقام وحفظ الـ Mapping
    attacks_df['attack_cat_code'] = attacks_df['attack_cat'].astype('category').cat.codes
    y_t2 = attacks_df['attack_cat_code']
    
    # حفظ خريطة فئات الهجوم (Attack Mapping)
    attack_mapping = dict(enumerate(attacks_df['attack_cat'].astype('category').cat.categories))
    joblib.dump(attack_mapping, "models/attack_mapping.joblib")
    
    X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
        X_t2, y_t2, test_size=0.2, random_state=42, stratify=y_t2
    )

    model_t2 = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss', n_jobs=-1)
    model_t2.fit(X_train_t2, y_train_t2)

    pred_t2 = model_t2.predict(X_test_t2)
    acc_t2 = accuracy_score(y_test_t2, pred_t2)
    print(f"✅ Tier 2 Accuracy: {acc_t2 * 100:.2f}%")

    joblib.dump(model_t2, "models/tier2_xgb.joblib")
    print("[SUCCESS] Saved Tier 2 model to models/tier2_xgb.joblib")

print("\n==================================================")
print("       TRAINING COMPLETE & ALL MODELS SAVED        ")
print("==================================================")